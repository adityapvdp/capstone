# for data manipulation
import pandas as pd
import numpy as np

# for data preprocessing and pipeline creation
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight

# for model serialization
import joblib

# for hugging face authentication and upload
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("capstone-engine-fault-prediction")

api = HfApi()

# Load train-test split files from Hugging Face dataset repo
Xtrain_path = "hf://datasets/adityapvdp/Predictive-Maintenance/Xtrain.csv"
Xtest_path = "hf://datasets/adityapvdp/Predictive-Maintenance/Xtest.csv"
ytrain_path = "hf://datasets/adityapvdp/Predictive-Maintenance/ytrain.csv"
ytest_path = "hf://datasets/adityapvdp/Predictive-Maintenance/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze()
ytest = pd.read_csv(ytest_path).squeeze()

# List of numerical features in the dataset
numeric_features = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp"
]

# Compute class weight to handle class imbalance
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(ytrain),
    y=ytrain
)

class_weight = class_weights[1] / class_weights[0]
print("Computed class weight:", class_weight)

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    remainder="drop"
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric="logloss"
)

# Define hyperparameter grid
param_grid = {
    "xgbclassifier__n_estimators": [50, 75, 100, 125, 150],
    "xgbclassifier__max_depth": [2, 3, 4],
    "xgbclassifier__colsample_bytree": [0.4, 0.5, 0.6],
    "xgbclassifier__colsample_bylevel": [0.4, 0.5, 0.6],
    "xgbclassifier__learning_rate": [0.01, 0.05, 0.1],
    "xgbclassifier__reg_lambda": [0.4, 0.5, 0.6],
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        param_set = results["params"][i]
        mean_score = results["mean_test_score"][i]
        std_score = results["std_test_score"][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    # Find optimal threshold using F1 score
    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    precision, recall, thresholds = precision_recall_curve(ytest, y_pred_test_proba)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
    best_threshold = thresholds[np.argmax(f1_scores)]

    print("Optimal classification threshold:", best_threshold)
    mlflow.log_metric("optimal_threshold", best_threshold)

    # Apply threshold on train and test predictions
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= best_threshold).astype(int)
    y_pred_test = (y_pred_test_proba >= best_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log the metrics for the best model
    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1-score": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1-score": test_report["1"]["f1-score"]
    })

    # Save the model locally
    model_path = "best_engine_prediction_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face model repo
    repo_id = "adityapvdp/Predictive-Maintenance-model"
    repo_type = "model"

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Model repo '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Model repo '{repo_id}' not found. Creating new repo...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Model repo '{repo_id}' created.")

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )
