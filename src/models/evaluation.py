import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import mlflow

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

def evaluate_model():
    """Evaluate model on test set"""
    # Load model
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Load test data
    test_df = pd.read_csv('data/processed/test.csv')
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    metrics = {
        'test_accuracy': float(accuracy_score(y_test, y_pred)),
        'test_precision': float(precision_score(y_test, y_pred, average='weighted')),
        'test_recall': float(recall_score(y_test, y_pred, average='weighted')),
        'test_f1': float(f1_score(y_test, y_pred, average='weighted'))
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_dict = {
        'confusion_matrix': cm.tolist(),
        'labels': sorted(y_test.unique().tolist())
    }
    
    # Save results
    Path('results').mkdir(exist_ok=True)
    
    with open('results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    with open('results/confusion_matrix.json', 'w') as f:
        json.dump(cm_dict, f, indent=2)
    
    # Classification report
    report = classification_report(y_test, y_pred)
    with open('results/classification_report.txt', 'w') as f:
        f.write(report)
    
    print(f"✓ Model evaluation complete")
    print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"  Test F1 Score: {metrics['test_f1']:.4f}")
    print(f"\nClassification Report:")
    print(report)

if __name__ == "__main__":
    evaluate_model()
