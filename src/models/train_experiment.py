import subprocess
import yaml
import json
from pathlib import Path
import itertools

import mlflow
import pickle
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split, cross_validate

current_dir = Path.cwd()
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
train_data = pickle.load(open(data_processed_path/'trainset.pkl', 'rb'))

def run_experiment(experiment_name, param_grid):
    """Run experiments with parameter grid"""
    results = []
    
    # Load base params
    with open(config_path, 'r') as f:
        base_params = yaml.safe_load(f)
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    
    for i, combination in enumerate(itertools.product(*values)):
        print(f"\n{'='*60}")
        print(f"Experiment {i+1}: {experiment_name}")
        print(f"{'='*60}")
        
        # Update parameters
        params = base_params.copy()
        for key, value in zip(keys, combination):
            # Navigate nested dict
            keys_path = key.split('.')
            current = params
            for k in keys_path[:-1]:
                current = current[k]
            current[keys_path[-1]] = value
        
        # Update params file
        update_params(params)
        
        # Run DVC pipeline
        subprocess.run(['dvc', 'repro'], check=True)
        
        # Read results
        with open('metrics.json', 'r') as f:
            metrics = json.load(f)
        
        result = {
            'params': {k: v for k, v in zip(keys, combination)},
            'metrics': metrics
        }
        results.append(result)
        
        print(f"Results: {metrics}")
    
    # Save all results
    Path('experiments').mkdir(exist_ok=True)
    with open(f'experiments/{experiment_name}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Completed {len(results)} experiments")
    print(f"  Results saved to experiments/{experiment_name}_results.json")
    
    return results

if __name__ == "__main__":
    # Example: hyperparameter tuning
    param_grid = {
        'train.n_estimators': [50, 100, 200],
        'train.max_depth': [3, 5, 10],
        'train.learning_rate': [0.01, 0.1]
    }
    
    results = run_experiment('rf_tuning', param_grid)
    
    # Find best result
    best = max(results, key=lambda x: x['metrics']['train_accuracy'])
    print(f"\nBest configuration:")
    print(f"  Params: {best['params']}")
    print(f"  Accuracy: {best['metrics']['train_accuracy']:.4f}")