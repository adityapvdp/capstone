# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_CAPSTONE_TOKEN"))
DATASET_PATH = "hf://datasets/adityapvdp/Predictive-Maintenance/engine_data.csv"
engine_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Define the target variable
target = "Engine Condition"

# List of numerical predictor features
numeric_features = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp"
]

# Define predictor matrix (X)
X = engine_dataset[numeric_features]

# Define target variable (y)
y = engine_dataset[target]

# Split dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2, # 20% of the data is reserved for testing
    random_state=42,
    stratify=y  # maintain class balance
)

# Save the datasets locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Upload files to Hugging Face dataset repo
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="adityapvdp/Predictive-Maintenance",
        repo_type="dataset",
    )
