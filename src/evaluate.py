from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test, threshold=0.2):
    print(f"Evaluating model with threshold={threshold}...")

    # Use predict_proba to get probabilities for the positive class (1)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Apply custom threshold
    y_pred = (y_prob >= threshold).astype(int)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))