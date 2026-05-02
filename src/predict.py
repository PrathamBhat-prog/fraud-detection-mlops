import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pickle

from src.preprocessing import transform_input

# ================================
# PATH SETUP
# ================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# ================================
# LOAD ARTIFACTS
# ================================
with open(MODELS_DIR / "fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(MODELS_DIR / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open(MODELS_DIR / "threshold.pkl", "rb") as f:
    threshold = pickle.load(f)

# ================================
# PREDICTION FUNCTION
# ================================
def predict_transaction(data: dict):
    try:
        # Preprocess input
        df_input = transform_input(data, scaler)

        # Predict probability
        prob = model.predict_proba(df_input)[0][1]

        # Apply threshold
        prediction = int(prob > threshold)

        return {
            "fraud_probability": float(prob),
            "prediction": prediction
        }

    except Exception as e:
        return {"error": str(e)}


# ================================
# TEST BLOCK
# ================================
if __name__ == "__main__":
    sample_input = {
        **{f"V{i}": 0.0 for i in range(1, 29)},
        "Amount": 1000.0
    }

    result = predict_transaction(sample_input)

    print("Test Prediction Output:")
    print(result)