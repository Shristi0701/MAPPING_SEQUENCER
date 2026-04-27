import json
import requests

API_URL = "http://127.0.0.1:8000"

cos = [{"id": "CO1", "text": "Apply sorting algorithms to solve problems"}]
pos = [{"id": "PO1", "text": "Apply knowledge of mathematics, science and engineering fundamentals"}]
psos = [{"id": "PSO1", "text": "Design computer applications with AI"}]
peos = [{"id": "PEO1", "text": "Professionals in software industry"}]

payload = {
    "cos": cos,
    "pos": pos,
    "psos": psos,
    "peos": peos,
    "top_k": 3,
    "subject": "CS Test",
    "semester": "III"
}

# Test PDF Export
print("Testing PDF Export...")
try:
    resp = requests.post(
        f"{API_URL}/export/pdf", 
        data={"payload": json.dumps(payload)},
        timeout=60
    )
    print(f"PDF Status: {resp.status_code}")
    if resp.status_code != 200:
        print(f"Error: {resp.text}")
except Exception as e:
    print(f"PDF Request Exception: {e}")

# Test Excel Export
print("\nTesting Excel Export...")
try:
    resp = requests.post(
        f"{API_URL}/export/excel", 
        data={"payload": json.dumps(payload)},
        timeout=60
    )
    print(f"Excel Status: {resp.status_code}")
    if resp.status_code != 200:
        print(f"Error: {resp.text}")
except Exception as e:
    print(f"Excel Request Exception: {e}")
