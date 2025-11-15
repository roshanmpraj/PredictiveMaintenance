import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import mlflow
import os
from huggingface_hub import HfApi

# --- Configuration ---
# HF credentials and model repo are typically set as environment variables in a CI/CD pipeline
HF_USERNAME = os.environ.get("HF_USERNAME", "Roshanmpraj")
HF_MODEL_REPO_ID = f"{HF_USERNAME}/PredictiveMaintenance-XGBoost-Model"
MLFLOW_TRACKING_URI = "sqlite:///mlruns.db" # Local tracking for a self-contained pipeline
MODEL_FILENAME = "xgboost_model.pkl"

# --- Main Training Logic ---
def train_and_register_model():
    """Loads data, trains XGBoost, logs to MLflow, and registers the model to HF."""
    print("Starting model training and tracking...")

    # 1. Load Data (assuming it was downloaded/created by prep.py)
    try:
        X_train = pd.read_csv("X_train.csv")
        X_test = pd.read_csv("X_test.csv")
        y_train = pd.read_csv("y_train.csv").iloc[:, 0]
        y_test = pd.read_csv("y_test.csv").iloc[:, 0]
    except Exception as e:
        print(f"Error loading train/test data: {e}")
        return

    # 2. MLflow Setup
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Predictive Maintenance - XGBoost")

    with mlflow.start_run() as run:
        # 3. Model Definition and Training
        params = {
            'n_estimators': 150,
            'learning_rate': 0.05,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }

        xgb_model = XGBClassifier(**params)
        xgb_model.fit(X_train, y_train)

        # 4. Evaluation
        y_pred = xgb_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # 5. Log Parameters and Metrics
        mlflow.log_params(params)
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

        print(f"Model Metrics: Accuracy={accuracy:.4f}, Recall={recall:.4f}")

        # 6. Save Model Locally
        joblib.dump(xgb_model, MODEL_FILENAME)

        # 7. Register/Upload Model to Hugging Face Model Hub
        try:
            api = HfApi(token=os.environ.get("HF_TOKEN")) # HF_TOKEN must be set in the pipeline

            # Create the model repo if it doesn't exist
            api.create_repo(repo_id=HF_MODEL_REPO_ID, repo_type="model", exist_ok=True)

            # Upload the model file
            api.upload_file(
                path_or_fileobj=MODEL_FILENAME,
                path_in_repo=MODEL_FILENAME,
                repo_id=HF_MODEL_REPO_ID,
                repo_type="model",
            )
            print(f"Best model registered to Hugging Face Model Hub: {HF_MODEL_REPO_ID}")
        except Exception as e:
            print(f"WARNING: Could not upload model to Hugging Face. Ensure HF_TOKEN is set. Error: {e}")

if __name__ == "__main__":
    train_and_register_model()
