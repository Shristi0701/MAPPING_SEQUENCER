import json
import sys
from mapping.similarity import score_one_pair, compute_similarity
from api.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

co = "Apply Evolutionary Computation Methods to find solutions to complex problems"
po = "Apply knowledge of mathematics, science and engineering fundamentals"

print("Evaluating...")

cos = [{"id": "CO1", "text": co}]
pos = [{"id": "PO1", "text": po}]
req = {"cos": cos, "pos": pos, "top_k": 3}

# Mapping API
map_res = client.post("/map/auto", json=req).json()
map_level = map_res["mappings"][0]["candidates"][0]["level"] if map_res["mappings"][0]["candidates"] else 0

# Matrix API
mat_res = client.post("/map/matrix", json=req).json()
mat_level = mat_res["matrix"]["CO1"]["PO1"]

print(f"Mapping Level: {map_level}")
print(f"Matrix Level: {mat_level}")

if (map_level == mat_level):
    print("MATCH!")
else:
    print("MISMATCH!")
