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
