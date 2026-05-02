import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"

def load_data():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    return df