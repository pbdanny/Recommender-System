import os
from pathlib import Path
import pickle
import yaml
import json

import pandas as pd
import mlflow
from mlflow.models import infer_signature
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split, cross_validate

from sklearn.metrics import ndcg_score

current_dir = Path(__file__).parent
config_path = current_dir.parent.parent / 'params.yaml'
data_processed_path = current_dir.parent.parent / 'data' / 'processed'
model_path = current_dir.parent.parent / 'models'

# --- HELPER: Load parameters from YAML ---
def load_params():
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# --- HELPER: Detect Environment ---
def get_redis_host():
    # If running inside a container (env var set), use service name 'redis'
    # If running locally (VSCode), use 'localhost'
    if os.environ.get('AM_I_IN_A_DOCKER_CONTAINER'):
        return 'redis'
    return 'localhost'

def is_running_in_docker_env_var():
    return os.environ.get('AM_I_IN_A_DOCKER_CONTAINER') == 'Yes'

# --- LOAD PARAMETERS ---
params = load_params()
train_params = params['train']
mlflow_params = params['mlflow']

# --- 3. LOAD PROCESSED DATA ---
reader = Reader(line_format="user item rating timestamp", sep="\t")
test_data = Dataset.load_from_file(data_processed_path/'test_data.data', reader=reader)
testset = test_data.build_full_trainset().build_testset()

# --- 2.2 MLflow tracking server
MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(mlflow_params['tracking_uri'] or MLFLOW_TRACKING_URI)
EXPERIMENT_NAME = "RecSys-DVC-Evaluation"
mlflow.set_experiment(mlflow_params['experiment_name'] or EXPERIMENT_NAME)

class ModelPredictor:
    """Model predictor class for inference"""

    def __init__(self, model_name, model_version="latest"):
        # find the latest model in the experiment
        model_uri = f"models:/{model_name}/{model_version}"
        # Load the production model we just logged
        self.model = mlflow.pyfunc.load_model(model_uri)

    def predict(self, predict_df):
        return self.model.predict(predict_df)

def evaluate_model(champion_model=None, challenger_model=None):
    """Evaluate model on test set"""

    # Prepare test dataframe
    test_df = pd.DataFrame(testset, columns=["user", "item", "rating"])

    # Get predictions
    champion_preds = champion_model.predict(test_df)
    challenger_preds = challenger_model.predict(test_df)

    # Calculate RMSE
    champion_rmse = ((test_df['rating'] - champion_preds) ** 2).mean() ** 0.5
    challenger_rmse = ((test_df['rating'] - challenger_preds) ** 2).mean() ** 0.5

    # Create numpy arrays for NDCG calculation with user x item format
    add_pred_df = test_df.copy()
    add_pred_df['champion_pred'] = champion_preds
    add_pred_df['challenger_pred'] = challenger_preds

    y_actual = add_pred_df.pivot(index='user', columns='item', values='rating').fillna(0).to_numpy()
    y_champion = add_pred_df.pivot(index='user', columns='item', values='champion_pred').fillna(0).to_numpy()
    y_challenger = add_pred_df.pivot(index='user', columns='item', values='challenger_pred').fillna(0).to_numpy()


    # Calculate NDCG
    champion_ndcg_all = ndcg_score(y_actual, y_champion)
    challenger_ndcg_all = ndcg_score(y_actual, y_challenger)

   # Calculate NDCG at rank = 30
    champion_ndcg_30 = ndcg_score(y_actual, y_champion, k=30)
    challenger_ndcg_30 = ndcg_score(y_actual, y_challenger, k=30)

    print(f"Champion RMSE: {champion_rmse}")
    print(f"Challenger RMSE: {challenger_rmse}")
    print(f"Champion NDCG: {champion_ndcg_all}")
    print(f"Challenger NDCG: {challenger_ndcg_all}")
    print(f"Champion NDCG@30: {champion_ndcg_30}")
    print(f"Challenger NDCG@30: {challenger_ndcg_30}")

    # Save metrics for DVC
    metrics = {
        "champion_rmse": champion_rmse,
        "challenger_rmse": challenger_rmse,
        "champion_ndcg": champion_ndcg_all,
        "challenger_ndcg": challenger_ndcg_all,
        "champion_ndcg_30": champion_ndcg_30,
        "challenger_ndcg_30": challenger_ndcg_30
    }

    with open(current_dir/'eval_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    # Load champion and challenger models
    champion_model = ModelPredictor(model_name="svd_champion_model")
    challenger_model = ModelPredictor(model_name="svd_candidate_model")
    evaluate_model(champion_model=champion_model, challenger_model=challenger_model)