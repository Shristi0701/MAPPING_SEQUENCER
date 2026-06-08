import sys
sys.path.append('.')
import json
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

# Setup sample payload
payload = {
    "cos": ["CO1", "CO2"],
    "pos": ["PO1", "PO2"],
    "psos": ["PSO1"],
    "matrix": {
        "CO1": {"PO1": 3, "PO2": 2, "PSO1": 1},
        "CO2": {"PO1": 1, "PO2": 3, "PSO1": 2}
    },
    "questions": [
        {"id": "Q1", "name": "Q1", "max_marks": 10, "co_id": "CO1"},
        {"id": "Q2", "name": "Q2", "max_marks": 20, "co_id": "CO2"}
    ],
    "students": [
        {"student_id": "S1", "student_name": "Alice", "marks": {"Q1": 8, "Q2": 15}},  # Q1: 80% (meets target), Q2: 75% (meets target)
        {"student_id": "S2", "student_name": "Bob", "marks": {"Q1": 5, "Q2": 8}},     # Q1: 50% (fails target), Q2: 40% (fails target)
        {"student_id": "S3", "student_name": "Charlie", "marks": {"Q1": 9, "Q2": 18}} # Q1: 90% (meets target), Q2: 90% (meets target)
    ],
    "target_score_percent": 60.0,
    "threshold_l1": 50.0,
    "threshold_l2: ": 60.0,
    "threshold_l3": 70.0
}

# Run calculate endpoint
response = client.post("/attainment/calculate", json=payload)
print(f"Status Code: {response.status_code}")
res_json = response.json()

print("\n--- QUESTION STATS ---")
print(json.dumps(res_json["question_stats"], indent=2))

print("\n--- CO STATS ---")
print(json.dumps(res_json["co_stats"], indent=2))

print("\n--- PO STATS ---")
print(json.dumps(res_json["po_stats"], indent=2))

print("\n--- PSO STATS ---")
print(json.dumps(res_json["pso_stats"], indent=2))
