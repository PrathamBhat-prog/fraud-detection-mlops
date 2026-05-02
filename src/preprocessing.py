import pandas as pd
from sklearn.preprocessing import StandardScaler

# ================================
# FEATURE CONFIG
# ================================
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled"]


# ================================
# TRAIN PREPROCESSING
# ================================
def fit_preprocessor(df: pd.DataFrame):
    """
    Fit scaler on training data
    """
    scaler = StandardScaler()

    df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
    df = df.drop(["Amount", "Time"], axis=1)

    return df, scaler


# ================================
# INFERENCE PREPROCESSING
# ================================
def transform_input(data: dict, scaler):
    """
    Transform raw input into model-ready format
    """
    df = pd.DataFrame([data])

    # Scale Amount
    df["Amount_scaled"] = scaler.transform(df[["Amount"]])

    # Drop raw Amount
    df = df.drop(columns=["Amount"])

    # Ensure correct order
    df = df[FEATURE_COLUMNS]

    return df