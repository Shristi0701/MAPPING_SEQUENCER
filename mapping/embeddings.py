from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load model once (important!)
model = SentenceTransformer("all-MiniLM-L6-v2")


def compute_embedding_similarity(cos, pos):

    co_texts = [co["text"] for co in cos]
    po_texts = [po["text"] for po in pos]

    # Generate embeddings
    co_embeddings = model.encode(co_texts)
    po_embeddings = model.encode(po_texts)

    similarity_matrix = cosine_similarity(co_embeddings, po_embeddings)

    results = []

    for i, co in enumerate(cos):
        best_index = np.argmax(similarity_matrix[i])
        best_score = float(similarity_matrix[i][best_index])

        results.append({
            "co_id": co["id"],
            "best_po_id": pos[best_index]["id"],
            "similarity_score": round(best_score, 3)
        })

    return results
