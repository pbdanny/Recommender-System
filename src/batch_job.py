import os
import pickle

import pandas as pd
import mlflow
from mlflow.models import infer_signature

import redis
import json
from collections import defaultdict
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

# --- HELPER: Detect Environment ---
def get_redis_host():
    # If running inside a container (env var set), use service name 'redis'
    # If running locally (VSCode), use 'localhost'
    if os.environ.get('AM_I_IN_A_DOCKER_CONTAINER'):
        return 'redis'
    return 'localhost'

def is_running_in_docker_env_var():
    return os.environ.get('AM_I_IN_A_DOCKER_CONTAINER') == 'Yes'

# --- 1. Train Model & Log with MLflow ---
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file('../data/u.data', reader=reader)

# Split the data into a training and test set
trainset, testset = train_test_split(data, test_size=0.25)

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
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
EXPERIMENT_NAME = "Hybrid_RecSys_v2"
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run() as run:
    # Create the SVD model
    params = {"n_factors": 50, "n_epochs": 20, "lr_all": 0.005, "reg_all": 0.02}
    algo = SVD(**params)
    # Train
    algo.fit(trainset)

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
    model_path = "model_surprise.pkl"
   
    # Save raw model
    with open("model.pkl", "wb") as f:
        pickle.dump(algo, f)
 
    # Input should be a sample DataFrame, not a 'trainset' object
    input_example = pd.DataFrame({"user": ["941"], "item": ["1682"]})
    # Output should be a sample of what your predict() function returns
    output_example = [algo.predict(uid="941", iid="1682").est]
    signature = infer_signature(input_example, output_example)
    
    mlflow.pyfunc.log_model(name="svd_candidate_model",
                            python_model = SurpriseWrapper(), 
                            artifacts={"model_path": "model.pkl"},
                            signature=signature)
    
    mlflow.log_params(params)    
    mlflow.log_metric("trainset_users", trainset.n_users)
    mlflow.log_metric("trainset_items", trainset.n_items)
    
    # Log evaluation metric
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions)    
    mlflow.log_metric("rmse", rmse)
    
    # print(f"Model logged to: runs:/{run.info.run_id}/model")
    # model_uri = f"runs:/{run.info.run_id}/model"
    
# --- 3. BATCH Prediction : Load best model and generate candidates for ALL users ---

# --- 3.1 Locate the production model and load ---
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

all_items = [trainset.to_raw_iid(i) for i in trainset.all_items()]
all_users = [trainset.to_raw_uid(u) for u in trainset.all_users()]
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