import joblib
import cloudpickle

def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e1:
        try:
            with open(path, "rb") as f:
                return cloudpickle.load(f)
        except Exception as e2:
            raise RuntimeError(f"Failed to load model {path}. joblib error: {e1}; cloudpickle error: {e2}")
