
import pandas as pd
import joblib
import os
import json
from sklearn.ensemble import GradientBoostingClassifier
from datasets import load_dataset

# Load data from Hugging Face
dataset = load_dataset(
    "tushar77more/engine_dataset",
    data_files={
        "Xtrain": "data/processed/Xtrain.csv",
        "ytrain": "data/processed/ytrain.csv"
    }
)

Xtrain = dataset["Xtrain"].to_pandas()
ytrain = dataset["ytrain"].to_pandas().values.ravel()

# Use best hyperparameters from tuning
# (These should match the best_params from GridSearchCV)
best_params = {
    "n_estimators": 200,
    "learning_rate": 0.1
}

print(f"Training GradientBoostingClassifier with params: {best_params}")
model = GradientBoostingClassifier(**best_params)
model.fit(Xtrain, ytrain)

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/GradientBoosting_best_model.pkl")

print("Model trained & saved to model/GradientBoosting_best_model.pkl")
