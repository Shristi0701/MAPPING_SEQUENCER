import json
import os
from mapping.similarity import score_one_pair, model, preprocess_co


def load_labeled_pairs(path="data/labeled_pairs.json"):
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(base, path)
    with open(full_path) as f:
        return json.load(f)
    

def evaluate_precision_at_k(k=3):
    """
    For every positive pair (label=1) in labeled_pairs.json:
    - Take the CO text
    - Find all unique POs from the dataset
    - Run similarity + classifier for each CO-PO combination
    - Check if the correct PO appears in top-K results

    Returns precision@1, precision@3, and per-pair details
    """
    pairs = load_labeled_pairs()

    # separate positives (correct mappings) and get all unique POs
    positives = [p for p in pairs if p["label"] == 1]
    all_pos_texts = list({p["po"] for p in pairs})

    # -- Optimization: Batch encode all unique CO texts --
    unique_co_texts = list({p["co"] for p in positives})
    preprocessed_cos = [preprocess_co(co) for co in unique_co_texts]
    print(f"Batch encoding {len(unique_co_texts)} unique COs for evaluation...")
    co_embs = model.encode(preprocessed_cos, batch_size=32, show_progress_bar=False)
    co_emb_map = {co: emb for co, emb in zip(unique_co_texts, co_embs)}


    hits_at_1 = 0
    hits_at_3 = 0
    details = []

    for pair in positives:
        co_text     = pair["co"]
        correct_po  = pair["po"]
        co_emb      = co_emb_map[co_text]

        # compute hybrid score for each PO
        scored = []
        for po_text in all_pos_texts:
            res = score_one_pair(co_emb, co_text, po_text)
            scored.append({
                "po_text":      po_text,
                "bert_sim":     res["similarity"],
                "clf_prob":     res["classifier_prob"],
                "hybrid_score": res["hybrid_score"],
            })

         # rank by hybrid score descending
        scored.sort(key=lambda x: x["hybrid_score"], reverse=True)

        top_1_texts = [scored[0]["po_text"]]
        top_3_texts = [s["po_text"] for s in scored[:3]]

        hit1 = correct_po in top_1_texts
        hit3 = correct_po in top_3_texts

        if hit1: hits_at_1 += 1
        if hit3: hits_at_3 += 1

        details.append({
            "co":          co_text,
            "correct_po":  correct_po,
            "top_1":       scored[0]["po_text"],
            "top_3":       top_3_texts,
            "hit@1":       hit1,
            "hit@3":       hit3,
            "top_score":   scored[0]["hybrid_score"],
        })

    total = len(positives)
    return {
        "total_pairs_evaluated": total,
        "precision_at_1": round(hits_at_1 / total, 3),
        "precision_at_3": round(hits_at_3 / total, 3),
        "precision_at_1_pct": f"{round(hits_at_1 / total * 100, 1)}%",
        "precision_at_3_pct": f"{round(hits_at_3 / total * 100, 1)}%",
        "hits_at_1": hits_at_1,
        "hits_at_3": hits_at_3,
        "details": details
    }


