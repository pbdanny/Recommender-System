import subprocess
import yaml
import json
from pathlib import Path
import itertools

import mlflow
from mlflow.models import infer_signature
import pandas as pd
import pickle
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split, cross_validate, GridSearchCV

import optuna

current_dir = Path(__file__).parent
config_path = current_dir.parent.parent / 'params.yaml'
data_processed_path = current_dir.parent.parent / 'data' / 'processed'
model_path = current_dir.parent.parent / 'models'

# --- HELPER: Load parameters from YAML ---
def load_params():
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def update_params(params_dict):
    """Update params.yaml with new parameters"""
    with open(config_path, 'w') as f:
        yaml.dump(params_dict, f)

# --- 1. LOAD PARAMETERS ---
params = load_params()
train_params = params['train']
mlflow_params = params['mlflow']

# --- MLflow tracking server
MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(mlflow_params['tracking_uri'] or MLFLOW_TRACKING_URI)
EXPERIMENT_NAME = "RecSys-DVC"
mlflow.set_experiment(mlflow_params['experiment_name'] or EXPERIMENT_NAME)

# --- 3. LOAD PROCESSED DATA ---
reader = Reader(line_format="user item rating timestamp", sep="\t")
train_data = Dataset.load_from_file(data_processed_path/'train_data.data', reader=reader)
trainset = train_data.build_full_trainset()

def objective(trial):
    """Objective function for Optuna hyperparameter optimization."""
    with mlflow.start_run(nested=True, run_name="grid_search") as child_run:
        n_factors = trial.suggest_int('n_factors', 20, 50)
        n_epochs = trial.suggest_int('n_epochs', 10, 20)
        lr_all = trial.suggest_float('lr_all', 0.02, 0.05)
        reg_all = trial.suggest_float('reg_all', 0.4, 0.6)

        params = {
            'n_factors': n_factors,
            'n_epochs': n_epochs,
            'lr_all': lr_all,
            'reg_all': reg_all
        }
        mlflow.log_params(params)

        algo = SVD(**params)
        # Train
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
        
        mlflow.pyfunc.log_model(name="svd_candidate_model",
                                python_model = SurpriseWrapper(), 
                                artifacts={"model_path": str(model_path/"model.pkl")},
                                signature=signature)
        mlflow.log_metric("test_rmse", score['test_rmse'].mean())
        mlflow.log_metric("test_mae", score['test_mae'].mean())
        mlflow.log_metric("trainset_users", train_data.n_users)
        mlflow.log_metric("trainset_items", train_data.n_items)

        return score['test_rmse'].mean()

if __name__ == "__main__":
    # Example: hyperparameter tuning with Optuna and MLflow logging
    with mlflow.start_run(run_name="RecSys-DVC-Optuna") as run:
        # Log the experiment settings
        n_trials = 10
        mlflow.log_param("n_trials", n_trials)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        # Log the best trial and its run ID
        mlflow.log_params(study.best_trial.params)
        mlflow.log_metrics({"best_error": study.best_value})
        if best_run_id := study.best_trial.user_attrs.get("run_id"):
            mlflow.log_param("best_child_run_id", best_run_id)