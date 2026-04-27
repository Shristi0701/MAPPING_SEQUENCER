import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# All 12 NBA POs
NBA_POS = [
    {"id": "PO1",  "text": "Apply knowledge of mathematics science and engineering fundamentals"},
    {"id": "PO2",  "text": "Identify formulate and solve complex engineering problems"},
    {"id": "PO3",  "text": "Design solutions for complex problems meeting societal and environmental needs"},
    {"id": "PO4",  "text": "Conduct investigations of complex problems using research-based knowledge"},
    {"id": "PO5",  "text": "Use modern engineering and IT tools for complex engineering activities"},
    {"id": "PO6",  "text": "Understand the impact of engineering solutions in society and environment"},
    {"id": "PO7",  "text": "Apply reasoning to assess societal health safety and legal issues"},
    {"id": "PO8",  "text": "Apply ethical principles and commit to professional ethics"},
    {"id": "PO9",  "text": "Function effectively as an individual or in multidisciplinary teams"},
    {"id": "PO10", "text": "Communicate effectively on complex engineering activities"},
    {"id": "PO11", "text": "Manage projects and finances in multidisciplinary environments"},
    {"id": "PO12", "text": "Recognize the need for and engage in lifelong learning"},
]

def find_best_po(co_text, pos):
    """Find the best matching PO for a CO using TF-IDF similarity."""
    all_texts = [co_text] + [p["text"] for p in pos]
    vec       = TfidfVectorizer(ngram_range=(1, 2))
    tfidf     = vec.fit_transform(all_texts)
    co_vec    = tfidf[0]
    po_vecs   = tfidf[1:]
    sims      = cosine_similarity(co_vec, po_vecs)[0]
    best_idx  = int(np.argmax(sims))
    return pos[best_idx]["id"], pos[best_idx]["text"], float(sims[best_idx])


def generate_pairs(syllabus_path, existing_pairs_path, output_path):
    # Load syllabus
    with open(syllabus_path, encoding="utf-8") as f:
        syllabus = json.load(f)

    # Load existing pairs so we don't duplicate them
    with open(existing_pairs_path, encoding="utf-8") as f:
        existing = json.load(f)

    existing_cos = {p["co"].strip().lower() for p in existing}
    print(f"Existing pairs: {len(existing)}")
    print(f"Existing unique COs: {len(existing_cos)}")

    new_pairs = []
    skipped   = 0
    total_cos = 0

    for branch_key, branch in syllabus["branches"].items():
        for sem_num, sem_data in branch["semesters"].items():
            for subj in sem_data["subjects"]:
                for co in subj["cos"]:
                    co_text = co["text"].strip()
                    total_cos += 1

                    # Skip if already in existing pairs
                    if co_text.lower() in existing_cos:
                        skipped += 1
                        continue

                    # Find best matching PO (positive pair)
                    best_po_id, best_po_text, best_score = find_best_po(
                        co_text, NBA_POS
                    )

                    # Only add as positive if confidence is reasonable
                    if best_score >= 0.05:
                        new_pairs.append({
                            "co":    co_text,
                            "po":    best_po_text,
                            "label": 1
                        })

                    # Find worst matching PO (negative pair)
                    all_texts = [co_text] + [p["text"] for p in NBA_POS]
                    vec       = TfidfVectorizer(ngram_range=(1, 2))
                    tfidf     = vec.fit_transform(all_texts)
                    sims      = cosine_similarity(tfidf[0], tfidf[1:])[0]
                    worst_idx = int(np.argmin(sims))

                    new_pairs.append({
                        "co":    co_text,
                        "po":    NBA_POS[worst_idx]["text"],
                        "label": 0
                    })

    print(f"Total COs in syllabus: {total_cos}")
    print(f"Skipped (already exist): {skipped}")
    print(f"New pairs generated: {len(new_pairs)}")

    # Combine existing + new
    combined = existing + new_pairs
    print(f"Total combined pairs: {len(combined)}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to: {output_path}")
    return len(combined)


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))

    syllabus_path       = os.path.join(base, "data", "syllabus_data.json")
    existing_pairs_path = os.path.join(base, "data", "labeled_pairs.json")
    output_path         = os.path.join(base, "data", "labeled_pairs.json")

    if not os.path.exists(syllabus_path):
        print("Warning: syllabus_data.json not found. Falling back to mock_los.json.")
        mock_los_path = os.path.join(base, "data", "mock_los.json")
        if os.path.exists(mock_los_path):
            with open(mock_los_path, encoding="utf-8") as f:
                mock_los = json.load(f)

            syllabus = {
                "branches": {
                    "mock": {
                        "semesters": {
                            "1": {
                                "subjects": [
                                    {
                                        "cos": [{"text": lo["text"]} for lo in mock_los]
                                    }
                                ]
                            }
                        }
                    }
                }
            }

            with open(syllabus_path, "w", encoding="utf-8") as f:
                json.dump(syllabus, f, indent=2, ensure_ascii=False)
        else:
            raise FileNotFoundError("Neither syllabus_data.json nor mock_los.json found. Add one to proceed.")

    total = generate_pairs(syllabus_path, existing_pairs_path, output_path)
    print(f"\nDone! Your classifier will now train on {total} pairs.")
    print("Restart your server — it will retrain automatically.")
