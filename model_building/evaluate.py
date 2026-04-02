
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from datasets import load_dataset

# Load test data from Hugging Face
dataset = load_dataset(
    "tushar77more/engine_dataset",
    data_files={
        "Xtest": "data/processed/Xtest.csv",
        "ytest": "data/processed/ytest.csv"
    }
)

Xtest = dataset["Xtest"].to_pandas()
ytest = dataset["ytest"].to_pandas().values.ravel()

# Load model
model = joblib.load("model/GradientBoosting_best_model.pkl")

# Evaluate
pred = model.predict(Xtest)

print("=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)
print(f"Accuracy:  {accuracy_score(ytest, pred):.4f}")
print(f"Precision: {precision_score(ytest, pred, average='weighted'):.4f}")
print(f"Recall:    {recall_score(ytest, pred, average='weighted'):.4f}")
print(f"F1-Score:  {f1_score(ytest, pred, average='weighted'):.4f}")
print("\nClassification Report:")
print(classification_report(ytest, pred))
