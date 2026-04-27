from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from mapping.classifier import classifier_score, get_po_embedding
import numpy as np

model = SentenceTransformer("all-mpnet-base-v2")

def preprocess_co(text):
    lower = text.lower()
    verbs = {"apply", "analyze", "evaluate", "create", "understand", "remember", "implement", "design", "develop", "know", "demonstrate", "illustrate", "discuss", "explain", "describe", "identify", "calculate", "assess", "compare", "differentiate", "solve", "formulate", "construct", "conduct"}
    tokens = text.split()
    if tokens and tokens[0].lower() in verbs:
        return " ".join(tokens[1:])
    return text


def similarity_to_level(score: float) -> int:
    if score < 0.2:   return 0
    elif score < 0.4: return 1
    elif score < 0.6: return 2
    else:             return 3

def get_keywords(text1, text2):
    stopwords = {"a","an","the","to","of","in","and","for","is","are",
                 "using","with","by","be","can","will","that","this"}
    def stem(word):
        for suffix in ["ing","tion","ity","ies","s","ed","al","ical"]:
            if word.endswith(suffix) and len(word) - len(suffix) > 3:
                return word[:-len(suffix)]
        return word
    w1 = {stem(w) for w in text1.lower().split() if w not in stopwords}
    w2 = {stem(w) for w in text2.lower().split() if w not in stopwords}
    return list(w1 & w2)


def keyword_score(co_text, po_text):
    stopwords = {"a", "an", "the", "to", "of", "in", "and", "for", "is", "are"}
    co_words = {w.strip(".,;:()[]") for w in co_text.lower().split() if w not in stopwords}
    po_words = {w.strip(".,;:()[]") for w in po_text.lower().split() if w not in stopwords}
    overlap = co_words & po_words
    return min(len(overlap) / 3.0, 1.0)


# Per-pair result cache — avoids recomputing for repeated CO text
_score_cache = {}


def score_one_pair(co_emb: np.ndarray, co_text: str, po_text: str) -> dict:
    """Score a single CO-PO pair using pre-computed CO embedding."""
    key = (co_text.strip().lower(), po_text.strip().lower())
    if key in _score_cache:
        return _score_cache[key].copy()

    # Use the pre-cached PO embedding from classifier module
    po_emb = get_po_embedding(po_text)
    # Preprocess CO text to strip Bloom's taxonomy verbs for embedding and classification
    clean_co = preprocess_co(co_text)

    bert_sim = float(cosine_similarity([co_emb], [po_emb])[0][0])
    clf_prob = classifier_score(clean_co, po_text, co_emb=co_emb)
    kw_score = keyword_score(clean_co, po_text)
    
    # Tuned weights reflecting the improved embedding model (mpnet) and less keyword reliance
    hybrid = round(0.55 * bert_sim + 0.35 * clf_prob + 0.10 * kw_score, 3)
    level = similarity_to_level(hybrid)
    kw = get_keywords(clean_co, po_text)

    justification = f"Level {level} alignment. "
    if level == 0:
        justification += "No significant semantic overlap found between the course outcome and the program outcome."
    elif kw:
        justification += f"High relevance found in shared technical concepts: {', '.join(kw)}."
    elif bert_sim > 0.4:
        justification += "Strong semantic relationship and conceptual alignment identified by BERT model."
    else:
        justification += "Partial conceptual overlap identified through context analysis."

    result = {
        "similarity": round(bert_sim, 3),
        "classifier_prob": round(clf_prob, 3),
        "keyword_score": round(kw_score, 3),
        "hybrid_score": hybrid,
        "level": level,
        "keywords": kw,
        "explanation": justification,
    }

    _score_cache[key] = result
    return result.copy()


def compute_similarity(cos, pos, top_k=3):
    """
    Compute CO-PO similarity for all pairs.

    Optimization: Batch-encodes ALL CO texts in a single model.encode() call,
    then scores each CO against all POs using pre-cached PO embeddings.
    """
    # ── Batch encode all COs that are not already fully cached ──────────────
    uncached_cos = [
        co for co in cos
        if not all(
            (co["text"].strip().lower(), po["text"].strip().lower()) in _score_cache
            for po in pos
        )
    ]
    if uncached_cos:
        co_texts  = [preprocess_co(co["text"]) for co in uncached_cos]
        co_embeddings = model.encode(co_texts, batch_size=32, show_progress_bar=False)
        co_emb_map = {co["text"]: emb for co, emb in zip(uncached_cos, co_embeddings)}
    else:
        co_emb_map = {}

    # ── Pre-warm all PO embeddings (no-op if already cached) ────────────────
    for po in pos:
        get_po_embedding(po["text"])

    # ── Score every CO × PO pair ─────────────────────────────────────────────
    results = []
    for co in cos:
        co_emb = co_emb_map.get(co["text"])
        candidates = []
        for po in pos:
            # Check if key is already in cache
            key = (co["text"].strip().lower(), po["text"].strip().lower())
            if key in _score_cache:
                scored = _score_cache[key].copy()
            else:
                if co_emb is None:
                    co_emb = model.encode([preprocess_co(co["text"])])[0]
                    co_emb_map[co["text"]] = co_emb
                scored = score_one_pair(co_emb, co["text"], po["text"])
                
            if scored["hybrid_score"] >= 0.2:
                scored["po_id"] = po["id"]
                candidates.append(scored)

        candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)

        results.append({
            "co_id":      co["id"],
            "co_text":    co["text"],
            "candidates": candidates[:top_k]
        })

    return results