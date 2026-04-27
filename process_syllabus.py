import json
import os
import re
import sys

co_texts = []
with open("raw_cos.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line.startswith("- "):
            text = line[2:].strip()
            text = re.sub(r'^CO-\d+[:\s]*', '', text).strip()
            if text:
                co_texts.append(text)
        elif line.startswith("CO-"):
            text = re.sub(r'^CO-\d+[:\s]*', '', line).strip()
            if text:
                co_texts.append(text)

mock_los = [{"id": f"LO-{i+1}", "text": text} for i, text in enumerate(co_texts)]

with open("data/mock_los.json", "w", encoding="utf-8") as f:
    json.dump(mock_los, f, indent=2)

if os.path.exists("data/syllabus_data.json"):
    os.remove("data/syllabus_data.json")

print(f"Created data/mock_los.json with {len(mock_los)} COs.")

print("Running generate_labeled_pairs.py...")
os.system(f'"{sys.executable}" generate_labeled_pairs.py')

print("Reloading API...")
os.utime("api/main.py", None)

print("Running test_api.py...")
os.system(f'"{sys.executable}" test_api.py')
