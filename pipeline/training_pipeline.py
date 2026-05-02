import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_ingestion import load_data
from src.preprocessing import fit_preprocessor
from src.train import train_model
from src.evaluate import evaluate_model
from sklearn.model_selection import train_test_split
import pickle

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

def run_pipeline():
    print("Running full training pipeline...")

    # Step 1: Load data
    df = load_data()

    # Step 2: Preprocess
    df, scaler = fit_preprocessor(df)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Step 3: Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Step 4: Train
    model = train_model(X_train, y_train)

    # Step 5: Evaluate
    evaluate_model(model, X_test, y_test)

    # Step 6: Save artifacts
    with open(MODELS_DIR / "fraud_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(MODELS_DIR / "threshold.pkl", "wb") as f:
        pickle.dump(0.2, f)

    print("Pipeline completed successfully!")

if __name__ == "__main__":
    run_pipeline()