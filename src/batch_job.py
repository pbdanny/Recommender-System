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

def is_running_in_docker_env_var():
    return os.environ.get('AM_I_IN_A_DOCKER_CONTAINER') == 'Yes'

# --- 1. Train Model & Log with MLflow ---
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file('../data/u.data', reader=reader)

# Split the data into a training and test set
trainset, testset = train_test_split(data, test_size=0.25)

# --- 2. Log with MLflow ---
# print("Connecting to MLflow at http://localhost:5000")
# mlflow.set_tracking_uri("http://localhost:5000")

# Define MLflow tracking location
MLFLOW_TRACKING_DIR = "../mlruns"
mlflow_tracking_path = os.path.abspath(MLFLOW_TRACKING_DIR)
mlflow_tracking_uri = f"file://{os.path.abspath(MLFLOW_TRACKING_DIR)}"
EXPERIMENT_NAME = "Hybrid_RecSys"
mlflow.set_tracking_uri(mlflow_tracking_uri) 
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run() as run:
    # Create the SVD model
    params = {"n_factors": 50, "n_epochs": 20, "lr_all": 0.005, "reg_all": 0.02}
    algo = SVD(**params)

    # Define a custom pyfunc model wrapper
    class SurpriseRecommenderWrapper(mlflow.pyfunc.PythonModel):
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
    
    # Train
    algo.fit(trainset)
    
    print(f"Logging model to MLflow in experiment '{EXPERIMENT_NAME}'...")

    # Dump model as pickle before for mlflow to log
    model_path = "model_surprise.pkl"
   
    with open(model_path, "wb") as f:
        pickle.dump(algo, f)
 
    # Input should be a sample DataFrame, not a 'trainset' object
    input_example = pd.DataFrame({"user": ["941"], "item": ["1682"]})
    # Output should be a sample of what your predict() function returns
    output_example = [algo.predict(uid="941", iid="1682").est]
    signature = infer_signature(input_example, output_example)
    
    mlflow.pyfunc.log_model(name="svd_candidate_model",
                            python_model = SurpriseRecommenderWrapper(), 
                            artifacts={"model_path": model_path},
                            signature=signature)
    
    mlflow.log_params(params)    
    mlflow.log_metric("trainset_users", trainset.n_users)
    mlflow.log_metric("trainset_items", trainset.n_items)
    
    # Log evaluation metric
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions)    
    mlflow.log_metric("rmse", rmse)
    
# --- 3. Load best model and generate candidates for ALL users ---

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

# Check if running in Docker
if is_running_in_docker_env_var():
    # 1. Get the absolute path to the tracking dir *inside the container*
    # This will resolve to "/app/mlflow_tracking"
    container_tracking_path = os.path.abspath(MLFLOW_TRACKING_DIR)

    # 2. Construct the correct, container-local model URI
    # The required path is: file://<container_path>/<experiment_id>/models/{model_id}/artifacts
    model_uri = (
        f"file://{container_tracking_path}/"
        f"{experiment.experiment_id}/models/{latest_model_id}/artifacts/"
        )
    
else:
    # If not in Docker, use the standard MLflow model URI
    model_uri = (
        f"file://{mlflow_tracking_path}/"
        f"{experiment.experiment_id}/models/{latest_model_id}/artifacts/"
        )

# --- 3.2. Generate Candidates for ALL users ---
print("Generating candidates...")

# Load the production model we just logged
model = mlflow.pyfunc.load_model(model_uri) 

all_items = [trainset.to_raw_iid(i) for i in trainset.all_items()]
all_users = [trainset.to_raw_uid(u) for u in trainset.all_users()]

# In a real system, you'd use a faster batch-prediction method
candidates = defaultdict(list)
for uid in all_users:
    # Create a DataFrame for batch prediction
    predict_df = pd.DataFrame({
        'user': [uid] * len(all_items),
        'item': all_items
    })

    # FIX 1: Assign the prediction results to a new column
    # -----------------------------------------------------------------
    # model.predict() returns a list/Series of scores
    scores = model.predict(predict_df)
    
    # Add scores as a new column
    predict_df['prediction'] = scores

    # Now, sort the DATAFRAME by the 'prediction' column
    top_k = predict_df.sort_values(by='prediction', ascending=False).head(500)

    # And now you can safely access the 'item' column
    candidates[uid] = top_k['item'].tolist()
    # -----------------------------------------------------------------

# --- 3. Populate Redis Cache ---
print("Connecting to Redis cache at http://localhost:6379")
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

print(f"Populating cache with {len(candidates)} users...")
with r.pipeline() as pipe:
    for user_id, item_list in candidates.items():
        # The key is "rec:batch:USER_ID"
        # The value is a JSON list of item IDs
        key = f"rec:batch:{user_id}"
        value = json.dumps(item_list)
        pipe.set(key, value)
    pipe.execute()

print("Batch job complete. Candidates are in Redis.")

r.get("rec:batch:15")  # Example to fetch candidates for user "15"