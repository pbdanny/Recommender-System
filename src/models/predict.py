import pickle
import pandas as pd
import numpy as np
from pathlib import Path

import mlflow
import yaml
import json
import os
import redis

from surprise import Dataset, Reader

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
all_data = Dataset.load_from_file(data_processed_path/'all_data.data', reader=reader)
alldataset = all_data.build_full_trainset()

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

def cache_candidates(candidates):
    redis_host = get_redis_host()
    print(f"Connecting to Redis at {redis_host}...")

    try:
        r = redis.Redis(host=redis_host, port=6379, db=0, decode_responses=True)
        with r.pipeline() as pipe:
            for uid, items in candidates.items():
                pipe.set(f"rec:batch:{uid}", json.dumps(items))
            pipe.execute()
        print("Success: Candidates cached in Redis.")
    except redis.ConnectionError:
        print(f"FATAL: Could not connect to Redis at {redis_host}. Check podman-compose ps.")
    

def batch_predict(data = alldataset):
    """Predict from CSV file"""
    predictor = ModelPredictor(model_name="svd_champion_model")
    
    # Load data
    all_items = [data.to_raw_iid(i) for i in data.all_items()]
    all_users = [data.to_raw_uid(u) for u in data.all_users()]

    candidates = {}
    for uid in all_users:
        # 1. Create prediction DataFrame
        predict_df = pd.DataFrame({'user': [uid]*len(all_items), 'item': all_items})
        
        # 2. Predict (Returns list of floats)
        scores = predictor.predict(predict_df)
        
        # 3. FIX: Assign scores back to DataFrame to sort
        predict_df['score'] = scores
        
        # 4. Sort and take top 500
        top_items = predict_df.sort_values('score', ascending=False).head(500)['item'].tolist()
        candidates[uid] = top_items
    
    cache_candidates(candidates)

if __name__ == "__main__":
    batch_predict()