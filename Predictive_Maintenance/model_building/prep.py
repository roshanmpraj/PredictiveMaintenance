# for data manipulation
import pandas as pd
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for feature scaling (Crucial for predictive maintenance sensor data)
from sklearn.preprocessing import StandardScaler
# for hugging face space authentication to upload files
from huggingface_hub import HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
# NOTE: Set your Hugging Face repo ID for upload
HF_REPO_ID = "Roshanmpraj/Predictive_Maintenance" # Corrected repository ID
# Use the correct dataset path from your Hugging Face repository
DATASET_PATH = f"hf://datasets/{HF_REPO_ID}/engine_data.csv"

# =============================
# 1. Load Dataset
# =============================
try:
    df = pd.read_csv(DATASET_PATH)
    # Standardize column names
    df.columns = [
        'Engine_RPM', 'Lub_Oil_Pressure', 'Fuel_Pressure', 'Coolant_Pressure',
        'Lub_Oil_Temperature', 'Coolant_Temperature', 'Engine_Condition'
    ]
    print("Dataset loaded and columns standardized successfully.")
except Exception as e:
    print(f"Failed to load dataset from {DATASET_PATH}. Error: {e}")
    # exit() # Removed exit() to avoid NameError and allow for easier debugging if needed

# =============================
# 2. Data Cleaning & Prepare Target
# =============================

# Ensure target variable is integer type
df['Engine_Condition'] = df['Engine_Condition'].astype(int)
print("Data types ensured for modeling.")

# =============================
# 3. Define target + features and Split
# =============================
target_col = "Engine_Condition"
# All remaining columns are features except the target
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================
# 4. Apply Standard Scaling 📏
# =============================
# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler ONLY on the training data
Xtrain_scaled = scaler.fit_transform(Xtrain)

# Transform both training and testing data
Xtest_scaled = scaler.transform(Xtest)

# Convert the scaled NumPy arrays back to DataFrames, keeping the original column names
Xtrain = pd.DataFrame(Xtrain_scaled, columns=Xtrain.columns)
Xtest = pd.DataFrame(Xtest_scaled, columns=Xtest.columns)

print("Features successfully scaled using StandardScaler.")

# =============================
# 5. Save Locally and Upload to Hugging Face
# =============================

# Define file names and the corresponding dataframes/series for saving
files_to_save = {
    "Xtrain.csv": Xtrain,
    "Xtest.csv": Xtest,
    "ytrain.csv": ytrain,
    "ytest.csv": ytest
}

for filename, data in files_to_save.items():
    # Save locally
    data.to_csv(filename, index=False)

    # Upload to Hugging Face
    api.upload_file(
        path_or_fileobj=filename,
        path_in_repo=filename,  # just the filename
        repo_id=HF_REPO_ID,
        repo_type="dataset",
    )
    print(f"Preprocessed data split '{filename}' uploaded to the Hugging Face dataset repository.")

print("All preprocessed and scaled data uploaded to the Hugging Face dataset repository.")
