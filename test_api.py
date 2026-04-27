import json
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

cos = [
    {"id": "CO1", "text": "Apply Evolutionary Computation Methods to find solutions to complex problems"},
    {"id": "CO2", "text": "Implement neural networks using backpropagation"}
]
pos = [
    {"id": "PO1", "text": "Apply knowledge of mathematics, science and engineering fundamentals"},
    {"id": "PO2", "text": "Identify, formulate and solve complex engineering problems"}
]

req = {
    "cos": cos,
    "pos": pos,
    "top_k": 3
}

mapping_resp = client.post("/map/auto", json=req).json()
matrix_resp = client.post("/map/matrix", json=req).json()

print("--- MAPPING OUTPUT ---")
for co in mapping_resp["mappings"]:
    for cand in co["candidates"]:
        print(f"{co['co_id']} -> {cand['po_id']}: Level {cand['level']}")

print("\n--- MATRIX OUTPUT ---")
for co_id, row in matrix_resp["matrix"].items():
    for po_id, level in row.items():
        print(f"{co_id} -> {po_id}: Level {level}")
