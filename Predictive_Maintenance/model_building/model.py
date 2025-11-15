# for model building and evaluation
import pandas as pd
import numpy as np
import os
import joblib # For model saving
# for experimentation tracking and model logging
import mlflow
import mlflow.sklearn
import xgboost as xgb
# for model training and preprocessing
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer # Keep import for robustness, but we won't use it directly for this simple pipeline
# for hugging face space authentication to upload files
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# Define constants for file paths and Hugging Face repositories
api = HfApi(token=os.getenv("HF_TOKEN"))
OUTPUT_DIR = "." # Use current directory where prep.py saved the splits (like the sample)

# NOTE: Set your Hugging Face repo ID for the model artifact
HF_REPO_ID = "Roshanmpraj/engine_predictive_maintenance_data"       # Dataset Repo ID
MODEL_REPO_ID = "Roshanmpraj/PredictiveMaintenance-XGBoost-Model" # Model Repo ID

# Define local file paths (using current directory)
XTRAIN_PATH = os.path.join(OUTPUT_DIR, "Xtrain.csv")
XTEST_PATH = os.path.join(OUTPUT_DIR, "Xtest.csv")
YTRAIN_PATH = os.path.join(OUTPUT_DIR, "ytrain.csv")
YTEST_PATH = os.path.join(OUTPUT_DIR, "ytest.csv")
# Set the model path exactly as you requested
MODEL_PATH = "/content/best_model.pkl" 

# --- MLflow Setup ---
mlflow.set_experiment("Predictive_Maintenance_Experiment")

# =============================
# 1. Load dataset
# =============================
try:
    Xtrain = pd.read_csv(XTRAIN_PATH)
    Xtest = pd.read_csv(XTEST_PATH)
    # Use squeeze() like the sample, assuming target is the first/only column
    ytrain = pd.read_csv(YTRAIN_PATH).squeeze()
    ytest = pd.read_csv(YTEST_PATH).squeeze()
    print("Dataset loaded successfully.")

except Exception as e:
    print(f"FATAL: Failed to load splits from local CSV files. Error: {e}")
    exit()

# =============================
# 2. Imbalance Handling
# =============================
# Handle class imbalance (critical for failure prediction)
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
print(f"Class weight (scale_pos_weight) set to: {class_weight:.2f}")

# Base XGBoost model
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# =============================
# 3. Model Pipeline (FIXED: Simplified Preprocessing)
# =============================
# For numerical data requiring a single full transformation (like scaling),
# place the transformer directly in the make_pipeline.
model_pipeline = make_pipeline(StandardScaler(), xgb_model)


# Hyperparameter grid (renamed keys to match make_pipeline prefix convention)
# Note: The keys must match the steps in the pipeline: 'standardscaler' and 'xgbclassifier'
param_grid = {
    'xgbclassifier__n_estimators': [50, 100],
    'xgbclassifier__max_depth': [3, 5],
    'xgbclassifier__learning_rate': [0.01, 0.1],
    'xgbclassifier__reg_lambda': [0.1, 0.5]
}


# =============================
# 4. Training and MLflow Logging
# =============================
with mlflow.start_run():
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        scoring='f1', 
        cv=3,
        n_jobs=-1
    )
    grid_search.fit(Xtrain, ytrain)

    # Save all results to CSV
    results = pd.DataFrame(grid_search.cv_results_)
    results_csv = os.path.join(OUTPUT_DIR, "gridsearch_results.csv")
    results.to_csv(results_csv, index=False)
    mlflow.log_artifact(results_csv)

    # Log only the best model info
    best_params = grid_search.best_params_
    mlflow.log_params(best_params)

    best_model = grid_search.best_estimator_
    classification_threshold = 0.5 

    # --- Predictions and Reports ---
    # Train predictions
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    # Test predictions
    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    # Reports
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log only final metrics
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_recall": train_report['1']['recall'],
        "train_f1": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_recall": test_report['1']['recall'],
        "test_f1": test_report['1']['f1-score']
    })

    # Save best model locally using the requested path
    joblib.dump(best_model, MODEL_PATH)
    # Log the model artifact to MLflow using the requested command
    mlflow.log_artifact(MODEL_PATH)
    print(f"Best model saved locally: {MODEL_PATH}")

    # Log the model for MLflow Model Registry
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model_artifact",
        registered_model_name="XGBoostPredictiveMaintenance"
    )
    print("Best model logged to MLflow Model Registry.")

    # =============================
    # 5. Upload to Hugging Face Model Hub
    # =============================
    try:
        # Ensure the model repository exists
        api.repo_info(repo_id=MODEL_REPO_ID, repo_type="model")
        print(f"Repo '{MODEL_REPO_ID}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Repo '{MODEL_REPO_ID}' not found. Creating new repo...")
        create_repo(repo_id=MODEL_REPO_ID, repo_type="model", private=False)
        print(f"Repo '{MODEL_REPO_ID}' created.")

    # Upload the serialized model file
    api.upload_file(
        path_or_fileobj=MODEL_PATH,
        path_in_repo=os.path.basename(MODEL_PATH), # Use the filename only
        repo_id=MODEL_REPO_ID,
        repo_type="model"
    )
    print(f"Best model uploaded to Hugging Face Model Hub: {MODEL_REPO_ID}")
