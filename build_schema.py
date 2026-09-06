import json
from pathlib import Path
from features import load, NUMERIC, CATEGORICAL

df, y = load()

schema = {"numeric": {}, "categorical": {}}

for col in NUMERIC:
    s = df[col].dropna()
    schema["numeric"][col] = {
        "min": float(s.min()),
        "max": float(s.max()),
        "median": float(s.median()),
    }

for col in CATEGORICAL:
    vals = sorted(df[col].dropna().astype(str).unique().tolist())
    schema["categorical"][col] = {
        "options": vals,
        "default": str(df[col].mode().iloc[0]),
        "n_options": len(vals),
    }

Path("models").mkdir(exist_ok=True)
Path("models/schema.json").write_text(json.dumps(schema, indent=2))
print(f"{len(NUMERIC)} numeric, {len(CATEGORICAL)} categorical -> models/schema.json")
for col, info in schema["categorical"].items():
    print(f"  {col}: {info['n_options']} options")
