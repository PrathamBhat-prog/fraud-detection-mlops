import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import fit_preprocessor

# ================================
# PATH SETUP
# ================================
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"
MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(exist_ok=True)

# ================================
# LOAD DATA
# ================================
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

# ================================
# TRAIN MODEL
# ================================
def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

# ================================
# MAIN PIPELINE
# ================================
def run_training():
    print("Starting training pipeline...")

    df = load_data()

    # Preprocessing
    df, scaler = fit_preprocessor(df)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Train model
    model = train_model(X_train, y_train)

    # Save artifacts
    with open(MODELS_DIR / "fraud_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(MODELS_DIR / "threshold.pkl", "wb") as f:
        pickle.dump(0.2, f)

    print("Training completed and model saved!")


# ================================
# RUN SCRIPT
# ================================
if __name__ == "__main__":
    run_training()