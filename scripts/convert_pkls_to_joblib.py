# scripts/convert_pkls_to_joblib.py
import os
from pathlib import Path
import joblib
import cloudpickle

MODELS_DIR = Path("models")
PKL_FILES = [
    "wafer_pipeline.pkl",
    "soil_moisture_pipeline.pkl",
    "gas_pipeline.pkl",
    "temperature_pipeline.pkl",
    "ldr_pipeline.pkl"
]

def load_any(path):
    try:
        return joblib.load(path)
    except Exception as e1:
        try:
            with open(path, "rb") as f:
                return cloudpickle.load(f)
        except Exception as e2:
            raise RuntimeError(f"Could not load {path}: joblib error: {e1}; cloudpickle error: {e2}")

def convert_all():
    for fname in PKL_FILES:
        p = MODELS_DIR / fname
        if not p.exists():
            print(f"[WARN] {p} not found — skip")
            continue
        print(f"[INFO] Loading {p} ...")
        try:
            model = load_any(p)
        except Exception as e:
            print(f"[ERROR] Failed to load {p}: {e}")
            continue
        out = p.with_suffix(".joblib")
        joblib.dump(model, out)
        print(f"[OK] Saved joblib: {out}")

if __name__ == "__main__":
    convert_all()
