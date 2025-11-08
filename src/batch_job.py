import pandas as pd
import mlflow
import redis
import json
from collections import defaultdict
from surprise import Dataset, Reader, SVD


# --- 1. Train Model & Log with MLflow ---
print("Connecting to MLflow at http://localhost:5000")
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Hybrid_RecSys")

reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file('data/u.data', reader=reader)
trainset = data.build_full_trainset()

with mlflow.start_run(run_name="Candidate_Model_SVD") as run:
    print("Training SVD model...")
    algo = SVD(n_factors=100, n_epochs=20, random_state=42)
    algo.fit(trainset)

    print("Logging model to MLflow...")
    mlflow.sklearn.log_model(algo, "svd_candidate_model")
    mlflow.log_metric("trainset_users", trainset.n_users)
    mlflow.log_metric("trainset_items", trainset.n_items)
    
    # TODO: log additional metrics as needed
    # This is the "model_v1.0" we will use in production
    model_uri = f"runs:/{run.info.run_id}/svd_candidate_model"
    
# --- 2. Generate Candidates for ALL users ---
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

    # Use the MLflow pyfunc signature
    predictions = model.predict(predict_df)

    # Sort scores and get top 500
    top_k = predictions.sort_values(by='prediction', ascending=False).head(500)
    candidates[uid] = top_k['item'].tolist()

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