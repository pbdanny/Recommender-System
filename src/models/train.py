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
train_data = Dataset.load_from_file(data_processed_path/'train_data.data', reader=reader)
trainset = train_data.build_full_trainset()

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
EXPERIMENT_NAME = "RecSys-DVC-Production"
mlflow.set_experiment(mlflow_params['experiment_name'] or EXPERIMENT_NAME)

def train_model():
    # --- 1. Train Model & Log with MLflow ---

    with mlflow.start_run(run_name="prod_model") as run:
        # Create the SVD model
        # params = {"n_factors": 50, "n_epochs": 20, "lr_all": 0.005, "reg_all": 0.02}
        algo = SVD(**train_params)
        mlflow.log_params(train_params)

        # Train
        # algo.fit(train_data)
        score = cross_validate(algo, train_data, measures=["RMSE", "MAE"], cv=3, verbose=False)

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
            
        # Dump model as pickle before for mlflow to log
        with open(model_path/"model.pkl", "wb") as f:
            pickle.dump(algo, f)
    
        # Input should be a sample DataFrame, not a 'trainset' object
        input_example = pd.DataFrame({"user": ["941"], "item": ["1682"]})
        # Output should be a sample of what your predict() function returns
        output_example = [algo.predict(uid="941", iid="1682").est]
        signature = infer_signature(input_example, output_example)
        
        mlflow.pyfunc.log_model(name="svd_production_model",
                                python_model = SurpriseWrapper(), 
                                artifacts={"model_path": str(model_path/"model.pkl")},
                                signature=signature)
        
        mlflow.log_params(params)    
        mlflow.log_metric("test_rmse", score['test_rmse'].mean())
        mlflow.log_metric("test_mae", score['test_mae'].mean())        
        mlflow.log_metric("trainset_users", train_data.build_full_trainset().n_users)
        mlflow.log_metric("trainset_items", train_data.build_full_trainset().n_items)
                
        # Save metrics for DVC
        metrics = {
            'test_rmse': score['test_rmse'].mean(),
            'test_mae': score['test_mae'].mean()
        }

        with open(current_dir/'train_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

    return None
 
if __name__ == "__main__":
    train_model()