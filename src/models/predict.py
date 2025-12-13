import pickle
import pandas as pd
import numpy as np
from pathlib import Path

import mlflow
import yaml
import json
import os
import redis

config_path = Path(__file__).parent.parent.parent / 'params.yaml'

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

class ModelPredictor:
    """Model predictor class for inference"""
    
    def __init__(self):
        params = load_params()
        mlflow_params = params['mlflow']
        mlflow.set_tracking_uri(mlflow_params['tracking_uri'])
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(mlflow_params['experiment_name'])
        # find the latest model in the experiment
        models = mlflow.search_logged_models(experiment_ids=[experiment.experiment_id], 
                                            order_by = [{"field_name": "creation_time", "ascending": False}], 
                                            max_results=1,
                                            output_format="list")
        # Check if any runs were found
        if not models:
            raise RuntimeError(f"No runs found for experiment '{mlflow_params['experiment_name']}'.")

        latest_model = models[0]
        model_uri = latest_model.artifact_location

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
    

def batch_predict():
    """Predict from CSV file"""
    predictor = ModelPredictor()
    
    # Load data
    data_processed_path = Path(__file__).parent.parent.parent / 'data' / 'processed'
    train_data = pickle.load(open(data_processed_path/'trainset.pkl', 'rb'))
    all_items = [train_data.to_raw_iid(i) for i in train_data.all_items()]
    all_users = [train_data.to_raw_uid(u) for u in train_data.all_users()]
    
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