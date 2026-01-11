# ========== src/api/client.py ==========
"""
Python client for ML Model API
"""
import requests
import pandas as pd
from typing import List, Dict, Optional, Any
import time

class MLAPIClient:
    """Client for interacting with ML Model API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict:
        """Check API health"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_metrics(self) -> Dict:
        """Get API metrics"""
        response = self.session.get(f"{self.base_url}/metrics")
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        response = self.session.get(f"{self.base_url}/model/info")
        response.raise_for_status()
        return response.json()
    
    def predict_batch(
        self,
        features: List[List[Any]],
        feature_names: Optional[List[str]] = None,
        recommend_k: Optional[int] = None,
    ) -> Dict:
        """Make batch predictions or request top-k recommendations.

        - For rating predictions provide rows like `[user, item]`.
        - For recommendations provide rows like `[user]` and set `recommend_k`.
        """
        payload: Dict[str, Any] = {"features": features}
        if feature_names is not None:
            payload["feature_names"] = feature_names
        if recommend_k is not None:
            payload["recommend_k"] = recommend_k
        response = self.session.post(f"{self.base_url}/predict", json=payload)
        response.raise_for_status()
        return response.json()
    
    def predict_single(self, payload: Dict[str]) -> Dict:
        """Make single prediction for a user-item pair.

        Expected payload: {"user": <id>, "item": <id>}
        Returns keys: `prediction`, `prediction_time_ms`, `model_version`.
        """
        response = self.session.post(f"{self.base_url}/predict/single", json=payload)
        response.raise_for_status()
        return response.json()
    
    def predict_from_csv(self, file_path: str) -> Dict:
        """Upload CSV and get predictions"""
        with open(file_path, 'rb') as f:
            files = {'file': (file_path, f, 'text/csv')}
            response = self.session.post(
                f"{self.base_url}/predict/csv",
                files=files
            )
        response.raise_for_status()
        return response.json()
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance"""
        response = self.session.get(f"{self.base_url}/model/feature-importance")
        response.raise_for_status()
        return response.json()
    
    def reload_model(self, model_name: Optional[str] = None) -> Dict:
        """Reload model on the server. Use `model_name` query param if provided."""
        params = {"model_name": model_name} if model_name else {}
        response = self.session.post(f"{self.base_url}/model/reload", params=params)
        response.raise_for_status()
        return response.json()

# ========== Example Usage ==========

def example_usage():
    """Example client usage"""
    
    # Initialize client
    client = MLAPIClient("http://localhost:8000")
    
    # Check health
    print("Health Check:")
    print(client.health_check())
    print()
    
    # Get model info
    print("Model Info:")
    print(client.get_model_info())
    print()
    
    # Single prediction
    print("Single Prediction:")
    result = client.predict_single({"user": "156", "item": "124"})
    print(f"Prediction: {result.get('prediction')}")
    print(f"Time (ms): {result.get('prediction_time_ms')}")
    print(f"Model: {result.get('model_version')}")
    print()
    
    # Batch prediction (user-item pairs)
    print("Batch Prediction (ratings):")
    batch_result = client.predict_batch(
        features=[
            ["155", "123"],
            ["156", "124"],
        ]
    )
    print(f"Predictions: {batch_result.get('predictions')}")
    print(f"Latency: {batch_result.get('prediction_time_ms')}")
    print()

    # Batch recommendations (top-k)
    print("Batch Recommendations:")
    recs = client.predict_batch(
        features=[["155"], ["156"]],
        recommend_k=5
    )
    print(f"Recommendations: {recs.get('predictions')}")
    print()
    
    # Get metrics
    print("API Metrics:")
    metrics = client.get_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")
