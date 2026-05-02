import pickle
import pandas as pd
from pathlib import Path

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
# FEATURE ORDER (TRAINING SCHEMA)
# ================================
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled"]

# ================================
# PREDICTION FUNCTION
# ================================
def predict_transaction(data: dict):
    """
    data should contain:
    V1 ... V28, Amount (RAW amount)
    """

    try:
        # Step 1: Convert input to DataFrame
        df_input = pd.DataFrame([data])

        # Step 2: Validate required columns
        required_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
        for col in required_cols:
            if col not in df_input.columns:
                return {"error": f"Missing column: {col}"}

        # Step 3: Scale Amount → Amount_scaled
        df_input["Amount_scaled"] = scaler.transform(df_input[["Amount"]])

        # Step 4: Drop raw Amount
        df_input = df_input.drop(columns=["Amount"])

        # Step 5: Ensure correct column order
        df_input = df_input[FEATURE_COLUMNS]

        # Step 6: Predict probability
        prob = model.predict_proba(df_input)[0][1]

        # Step 7: Apply threshold
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