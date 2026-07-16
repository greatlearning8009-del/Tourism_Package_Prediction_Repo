"""
Model training: reads the splits produced by prep.py, does GridSearchCV
on an XGBoost pipeline, logs to a local MLflow sqlite backend, and saves
the fitted pipeline to tourism_project/models/.

The saved model file is what the Streamlit app loads. In CI, this file
gets committed back to the repo by the workflow.
"""

import os
import joblib
import pandas as pd
import xgboost as xgb
import mlflow
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SPLITS_DIR = os.path.join(REPO_ROOT, "tourism_project", "data", "splits")
MODELS_DIR = os.path.join(REPO_ROOT, "tourism_project", "models")
MLRUNS_DIR = os.path.join(REPO_ROOT, "mlruns")
MODEL_FILENAME = "best_tourism_package_model_v1.joblib"
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(MLRUNS_DIR, exist_ok=True)

# Local MLflow with a sqlite backend — no server needed, and file-store
# is deprecated in mlflow >= 3. Artifacts still land under ./mlruns/.
mlflow.set_tracking_uri(f"sqlite:///{os.path.join(MLRUNS_DIR, 'mlflow.db')}")
mlflow.set_experiment("Tourism-Package-Prediction-Experiment")

Xtrain = pd.read_csv(os.path.join(SPLITS_DIR, "Xtrain.csv"))
Xtest = pd.read_csv(os.path.join(SPLITS_DIR, "Xtest.csv"))
ytrain = pd.read_csv(os.path.join(SPLITS_DIR, "ytrain.csv")).squeeze("columns")
ytest = pd.read_csv(os.path.join(SPLITS_DIR, "ytest.csv")).squeeze("columns")

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

# Class-imbalance weight
class_counts = ytrain.value_counts()
class_weight = class_counts[0] / class_counts[1]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric="logloss",
)

# Trimmed grid so CI finishes in minutes. Expand for a full sweep.
param_grid = {
    "xgbclassifier__n_estimators": [75, 125],
    "xgbclassifier__max_depth": [3, 4],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__colsample_bytree": [0.5],
    "xgbclassifier__reg_lambda": [0.5],
}

model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    threshold = 0.45
    y_pred_train = (best_model.predict_proba(Xtrain)[:, 1] >= threshold).astype(int)
    y_pred_test = (best_model.predict_proba(Xtest)[:, 1] >= threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1": test_report["1"]["f1-score"],
    })

    joblib.dump(best_model, MODEL_PATH)
    mlflow.log_artifact(MODEL_PATH, artifact_path="model")

    print(f"Best params: {grid_search.best_params_}")
    print(f"Test F1 (class=1): {test_report['1']['f1-score']:.4f}")
    print(f"Test accuracy:     {test_report['accuracy']:.4f}")
    print(f"Model saved to:    {MODEL_PATH}")
