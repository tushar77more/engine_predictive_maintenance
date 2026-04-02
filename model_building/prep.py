
import pandas as pd
from datasets import load_dataset
import os
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# HuggingFace API
api = HfApi(token=os.getenv("MLOps"))

# Load dataset from Hugging Face
dataset = load_dataset(
    "tushar77more/engine_dataset",
    data_files="data/engine_data.csv",
    split="train"
)
df = dataset.to_pandas()
print("Dataset loaded successfully from Hugging Face.")
print("Initial Shape:", df.shape)

# Data Cleaning
df.drop_duplicates(inplace=True)
df.dropna(axis=1, how='all', inplace=True)
print("Shape after cleaning:", df.shape)

# Standardize column names
column_mapping = {
    "Engine rpm": "Engine_RPM",
    "Lub oil pressure": "Lub_Oil_Pressure",
    "Fuel pressure": "Fuel_Pressure",
    "Coolant pressure": "Coolant_Pressure",
    "lub oil temp": "Lub_Oil_Temp",
    "Coolant temp": "Coolant_Temp",
    "Engine Condition": "Engine_Condition"
}
df.rename(columns=column_mapping, inplace=True)
target_col = "Engine_Condition"

if target_col not in df.columns:
    raise ValueError("Target column not found!")

# Outlier Handling
numeric_cols = ["Engine_RPM", "Lub_Oil_Pressure", "Fuel_Pressure",
                "Coolant_Pressure", "Lub_Oil_Temp", "Coolant_Temp"]

for col in numeric_cols:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df[col] = df[col].clip(lower, upper)

print("Outliers capped at 1st/99th percentile")

X = df.drop(columns=[target_col])
y = df[target_col]

# Train-Test Split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train/Test split done")
print("Xtrain:", Xtrain.shape, "Xtest:", Xtest.shape)

# Save Locally
processed_folder_path = "data/processed"
os.makedirs(processed_folder_path, exist_ok=True)

Xtrain.to_csv(os.path.join(processed_folder_path, "Xtrain.csv"), index=False)
Xtest.to_csv(os.path.join(processed_folder_path, "Xtest.csv"), index=False)
ytrain.to_csv(os.path.join(processed_folder_path, "ytrain.csv"), index=False)
ytest.to_csv(os.path.join(processed_folder_path, "ytest.csv"), index=False)

print("Files saved locally")

# Upload to Hugging Face
for fname in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
    fpath = os.path.join(processed_folder_path, fname)
    api.upload_file(
        path_or_fileobj=fpath,
        path_in_repo=f"data/processed/{fname}",
        repo_id="tushar77more/engine_dataset",
        repo_type="dataset",
    )
    print(f"Uploaded: {fpath}")

print("Preprocessing pipeline completed successfully!")
