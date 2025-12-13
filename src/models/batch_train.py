import os
from pathlib import Path
import pickle
import yaml
import json

import pandas as pd
import mlflow
from mlflow.models import infer_signature
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

import redis

config_path = Path(__file__).parent.parent.parent / 'params.yaml'
data_processed_path = Path(__file__).parent.parent.parent / 'data' / 'processed'
model_path = Path(__file__).parent.parent.parent / 'models'

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

# --- 1. LOAD PROCESSED DATA ---
train_data = pickle.load(open(data_processed_path/'trainset.pkl', 'rb'))
test_data = pickle.load(open(data_processed_path/'testset.pkl', 'rb'))

# --- 2. Log with MLflow ---

# --- 2.1 MLflow tracking local
# MLFLOW_TRACKING_DIR = "../mlruns"
# mlflow_tracking_path = os.path.abspath(MLFLOW_TRACKING_DIR)
# mlflow_tracking_uri = f"file://{os.path.abspath(MLFLOW_TRACKING_DIR)}"
# EXPERIMENT_NAME = "Hybrid_RecSys"
# mlflow.set_tracking_uri(mlflow_tracking_uri) 
# mlflow.set_experiment(EXPERIMENT_NAME)

# --- 2.2 MLflow tracking server
MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(mlflow_params['tracking_uri'] or MLFLOW_TRACKING_URI)
EXPERIMENT_NAME = "RecSys-DVC"
mlflow.set_experiment(mlflow_params['experiment_name'] or EXPERIMENT_NAME)

def train_model():
    # --- 1. Train Model & Log with MLflow ---

    with mlflow.start_run() as run:
        # Create the SVD model
        # params = {"n_factors": 50, "n_epochs": 20, "lr_all": 0.005, "reg_all": 0.02}
        algo = SVD(**train_params)
        # Train
        algo.fit(train_data)

        # Define a custom pyfunc model wrapper
        class SurpriseWrapper(mlflow.pyfunc.PythonModel):
            def load_context(self, context):
                import pickle
                with open(context.artifacts["model_path"], "rb") as f:
                    self.model = pickle.load(f)
                    
            def predict(self, context, model_input, params=None):
                # model_input: DataFrame with columns ['user', 'item']
                preds = [
                    self.model.predict(uid=row["user"], iid=row["item"]).est
                    for _, row in model_input.iterrows()
                ]
                return preds    
        
        print(f"Logging model to MLflow in experiment '{EXPERIMENT_NAME}'...")
    
        # Dump model as pickle before for mlflow to log
        with open(model_path/"model.pkl", "wb") as f:
            pickle.dump(algo, f)
    
        # Input should be a sample DataFrame, not a 'trainset' object
        input_example = pd.DataFrame({"user": ["941"], "item": ["1682"]})
        # Output should be a sample of what your predict() function returns
        output_example = [algo.predict(uid="941", iid="1682").est]
        signature = infer_signature(input_example, output_example)
        
        mlflow.pyfunc.log_model(name="svd_candidate_model",
                                python_model = SurpriseWrapper(), 
                                artifacts={"model_path": mlflow_params['model_path']},
                                signature=signature)
        
        mlflow.log_params(params)    
        mlflow.log_metric("trainset_users", train_data.n_users)
        mlflow.log_metric("trainset_items", train_data.n_items)
        
        # Log evaluation metric
        predictions = algo.test(test_data)
        rmse = accuracy.rmse(predictions)    
        mlflow.log_metric("rmse", rmse)
        
        # Save metrics for DVC
        metrics = {
            'rmse': rmse,
        }
        with open('metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

# --- 3. BATCH Prediction : Load best model and generate candidates for ALL users ---

# --- 3.1 Locate the production model and load ---
def batch_predict():
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    # find the latest model in the experiment
    models = mlflow.search_logged_models(experiment_ids=[experiment.experiment_id], 
                                        order_by = [{"field_name": "creation_time", "ascending": False}], 
                                        max_results=1,
                                        output_format="list")
    # Check if any runs were found
    if not models:
        raise RuntimeError(f"No runs found for experiment '{EXPERIMENT_NAME}'.")

    latest_model = models[0]
    latest_model_id = latest_model.model_id
    model_uri = latest_model.artifact_location

    # --- 3.2. Generate Candidates for ALL users ---
    print("Generating candidates...")

    # Load the production model we just logged
    model = mlflow.pyfunc.load_model(model_uri) 

    all_items = [train_data.to_raw_iid(i) for i in train_data.all_items()]
    all_users = [train_data.to_raw_uid(u) for u in train_data.all_users()]
    # Just do one user for demo speed (User 196)
    demo_users = ['196', '186']

    candidates = {}
    for uid in demo_users:
        # 1. Create prediction DataFrame
        predict_df = pd.DataFrame({'user': [uid]*len(all_items), 'item': all_items})
        
        # 2. Predict (Returns list of floats)
        scores = model.predict(predict_df)
        
        # 3. FIX: Assign scores back to DataFrame to sort
        predict_df['score'] = scores
        
        # 4. Sort and take top 500
        top_items = predict_df.sort_values('score', ascending=False).head(500)['item'].tolist()
        candidates[uid] = top_items
        
    # --- 4. CACHE TO REDIS ---
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
    
if __name__ == "__main__":
    train_model()