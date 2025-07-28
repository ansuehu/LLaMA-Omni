import json
from pathlib import Path

merged = []
for i in range(10):
    with open(f"/path/to/output/translated_{i}.json") as f:
        for line in f:
            merged.append(json.loads(line))

with open("translated_all.json", "w") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)