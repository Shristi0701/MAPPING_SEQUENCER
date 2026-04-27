from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np
import json
import os
import pickle

# Use the same model as similarity.py (already cached)
model = SentenceTransformer("all-mpnet-base-v2")
# Global classifier — trained once, reused for all requests
_classifier = None

# ── Optimization 3: Pre-compute PO embeddings once and cache them ──────────
# These are the fixed 12 NBA POs — they never change, so we embed once.
_po_emb_cache: dict[str, np.ndarray] = {}


def get_po_embedding(po_text: str) -> np.ndarray:
    """Return cached PO embedding, computing it only once per process."""
    if po_text not in _po_emb_cache:
        _po_emb_cache[po_text] = model.encode([po_text])[0]
    return _po_emb_cache[po_text]


def _build_feature(co_text, po_text, co_vec=None):
    """
    Convert a CO-PO pair into a feature vector for the classifier.
    Uses cached PO embedding for speed.
    """
    if co_vec is None:
        co_vec = model.encode([co_text])[0]
    po_vec = get_po_embedding(po_text)

    # Concatenate + element-wise difference + product
    diff    = np.abs(co_vec - po_vec)
    product = co_vec * po_vec
    return np.concatenate([co_vec, po_vec, diff, product])


def _get_pickle_path(labeled_pairs_path: str) -> str:
    """Derive the .pkl path adjacent to the labeled_pairs.json file."""
    base = os.path.dirname(os.path.abspath(labeled_pairs_path))
    return os.path.join(base, "classifier.pkl")


def train_classifier(labeled_pairs_path="data/labeled_pairs.json"):
    """
    Train an SVM classifier on labeled CO-PO pairs.
    Saves the trained model to classifier.pkl so it can be reloaded
    instantly on the next server restart without re-training.
    """
    global _classifier

    pkl_path = _get_pickle_path(labeled_pairs_path)

    # ── Optimization 1: Load persisted classifier if it exists ──────────────
    if os.path.exists(pkl_path):
        pairs_mtime = os.path.getmtime(labeled_pairs_path)
        pkl_mtime   = os.path.getmtime(pkl_path)

        if pkl_mtime >= pairs_mtime:
            print(f"Loading classifier from cache: {pkl_path}")
            with open(pkl_path, "rb") as f:
                _classifier = pickle.load(f)
            print("Classifier loaded from cache (instant).")
            return None

    # Load labeled data
    with open(labeled_pairs_path) as f:
        pairs = json.load(f)

    print(f"Training classifier on {len(pairs)} labeled pairs...")

    # ── Optimization 2: Batch-encode all CO and PO texts at once ────────────
    co_texts = [p["co"] for p in pairs]
    po_texts = [p["po"] for p in pairs]

    # Encode all unique texts in two big batches (much faster than one-by-one)
    all_texts   = list(set(co_texts + po_texts))
    all_embs    = model.encode(all_texts, batch_size=64, show_progress_bar=False)
    emb_lookup  = {text: emb for text, emb in zip(all_texts, all_embs)}

    def build_feature_fast(co_t, po_t):
        co_vec  = emb_lookup[co_t]
        po_vec  = emb_lookup[po_t]
        diff    = np.abs(co_vec - po_vec)
        product = co_vec * po_vec
        return np.concatenate([co_vec, po_vec, diff, product])

    X = np.array([build_feature_fast(p["co"], p["po"]) for p in pairs])
    y = np.array([p["label"] for p in pairs])

    # Split into train and test
    if len(pairs) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        X_train, X_test = X, X
        y_train, y_test = y, y

    # Train SVM classifier
    clf = SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=42,
    )

    # Stratified 5-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=skf, scoring="f1_weighted")
    print(f"CV F1: {scores.mean():.3f} ± {scores.std():.3f}")

    clf.fit(X_train, y_train)

    # Report accuracy
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Classifier trained. Test accuracy: {round(acc * 100, 1)}%")

    _classifier = clf

    # Persist trained model so next startup loads instantly
    with open(pkl_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"Classifier saved to cache: {pkl_path}")

    return acc


def classifier_score(co_text, po_text, co_emb=None):
    """
    Returns probability (0.0 to 1.0) that co_text maps to po_text.
    Trains the classifier on first call if not already trained.
    """
    global _classifier

    if _classifier is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, "data", "labeled_pairs.json")
        train_classifier(path)

    features = _build_feature(co_text, po_text, co_vec=co_emb).reshape(1, -1)
    prob = _classifier.predict_proba(features)[0][1]
    return round(float(prob), 3)