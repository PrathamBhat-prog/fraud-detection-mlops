# 💳 Fraud Detection System (End-to-End MLOps Project)

## 🚀 Overview
This project implements a **real-time fraud detection system** using machine learning and MLOps principles.

The system predicts whether a financial transaction is fraudulent using a trained model and exposes it via a REST API.

---

## 🔥 Key Features

- ✅ End-to-end ML pipeline (data → model → API)
- ✅ Handles class imbalance using class weighting
- ✅ Threshold tuning for business optimization
- ✅ Model comparison (Random Forest, XGBoost, SMOTE)
- ✅ Modular code structure (train, preprocess, predict)
- ✅ FastAPI-based real-time prediction API

---

## 🧠 Problem Statement

Fraud detection is a highly imbalanced classification problem where:
- Fraud cases are extremely rare
- Missing fraud is costly
- False positives affect user experience

This project focuses on **optimizing recall while controlling false positives**.

---

## ⚙️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- FastAPI
- Uvicorn

---

## 📊 Model Details

- **Final Model**: Random Forest
- **Imbalance Handling**: `class_weight='balanced'`
- **Threshold**: 0.2 (tuned for optimal recall)
- **Recall (Fraud)**: ~0.86
- **Approach**:
  - Compared multiple techniques:
    - Undersampling
    - SMOTE
    - XGBoost
  - Selected best based on trade-off between recall & false positives

---
## 📁 Project Structure

```text
fraud-detection/
│
├── app/
│ └── main.py # FastAPI app
│
├── src/
│ ├── train.py # Training pipeline
│ ├── preprocessing.py # Data preprocessing
│ ├── predict.py # Inference logic
│
├── models/ # Saved model artifacts
│
├── data/
│ └── raw/ # Dataset (not included)
│
├── notebooks/
│ └── EDA.ipynb # Exploration
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train model
```bash
python src/train.py
```

### 3. Run API
```bash
uvicorn app.main:app --reload
```

### 4. Launch frontend (Gradio UI)
```bash
python app/gradio_app.py
```
