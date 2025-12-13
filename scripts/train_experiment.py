#!/usr/bin/env python
"""Script to run training experiments with different parameters"""

import subprocess
import yaml
import json
from pathlib import Path
import itertools

def update_params(params_dict):
    """Update params.yaml with new parameters"""
    with open('params.yaml', 'w') as f:
        yaml.dump(params_dict, f)

def run_experiment(experiment_name, param_grid):
    """Run experiments with parameter grid"""
    results = []
    
    # Load base params
    with open('params.yaml', 'r') as f:
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