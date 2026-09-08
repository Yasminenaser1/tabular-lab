"""IDS_mapping.csv (three stacked tables) -> models/labels.json
{"admission_type_id": {"1": "Emergency", ...}, "discharge_disposition_id": {...}, "admission_source_id": {...}}
"""
import csv, json
from pathlib import Path

labels, current = {}, None
with open("data/IDS_mapping.csv", newline="") as f:
    for row in csv.reader(f):
        if not row or not row[0].strip():
            continue                       # blank separator line
        if row[1].strip().lower() == "description":
            current = row[0].strip()       # a new table's header row
            labels[current] = {}
            continue
        code, desc = row[0].strip(), row[1].strip()
        if desc.upper() in ("NULL", "NOT AVAILABLE", "NOT MAPPED", "UNKNOWN/INVALID"):
            desc = "Not recorded"
        labels[current][code] = desc

Path("models/labels.json").write_text(json.dumps(labels, indent=2))
for k, v in labels.items():
    print(f"{k}: {len(v)} codes, e.g. 1 -> {v.get('1')}")
