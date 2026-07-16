"""
Data preparation: loads the raw tourism CSV from the repo, splits into
train/test, and writes the four CSVs to tourism_project/data/splits/.

No cloud storage — everything reads/writes the local filesystem so this
runs identically in Codespaces, GitHub Actions, and local dev.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "tourism_project", "data")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
DATASET_PATH = os.path.join(DATA_DIR, "tourism.csv")

os.makedirs(SPLITS_DIR, exist_ok=True)

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Expected dataset at {DATASET_PATH}. "
        "Drop tourism.csv into tourism_project/data/ before running prep."
    )

tourism_df = pd.read_csv(DATASET_PATH)
print(f"Dataset loaded: {tourism_df.shape[0]} rows, {tourism_df.shape[1]} cols")

# Drop the ID column if present — never useful as a feature
if "CustomerID" in tourism_df.columns:
    tourism_df = tourism_df.drop(columns=["CustomerID"])

target = "ProdTaken"

numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "PitchSatisfactionScore", "NumberOfChildrenVisiting", "MonthlyIncome",
    "Passport", "OwnCar",
]

categorical_features = [
    "TypeofContact", "Occupation", "Gender",
    "ProductPitched", "MaritalStatus", "Designation",
]

X = tourism_df[numeric_features + categorical_features]
y = tourism_df[target]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)

Xtrain.to_csv(os.path.join(SPLITS_DIR, "Xtrain.csv"), index=False)
Xtest.to_csv(os.path.join(SPLITS_DIR, "Xtest.csv"), index=False)
ytrain.to_csv(os.path.join(SPLITS_DIR, "ytrain.csv"), index=False)
ytest.to_csv(os.path.join(SPLITS_DIR, "ytest.csv"), index=False)

print(f"Splits written to {SPLITS_DIR}")
print(f"  Xtrain: {Xtrain.shape}, Xtest: {Xtest.shape}")
