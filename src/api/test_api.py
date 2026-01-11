"""
Integration tests for FastAPI endpoints
"""
from fastapi.testclient import TestClient
from models.serve import app
import pytest

client = TestClient(app)

def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "version" in response.json()

def test_health():
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_metrics():
    """Test metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions" in data
    assert "avg_latency_ms" in data

def test_model_info():
    """Test model info endpoint"""
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "model_path" in data

def test_predict_batch():
    """Test batch prediction"""
    payload = {
        "features": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        "feature_names": ["f1", "f2", "f3"]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "probabilities" in data
    assert len(data["predictions"]) == 2

def test_predict_single():
    """Test single prediction"""
    payload = {
        "features": {"f1": 1.0, "f2": 2.0, "f3": 3.0}
    }
    response = client.post("/predict/single", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert 0 <= data["confidence"] <= 1

def test_predict_invalid_input():
    """Test prediction with invalid input"""
    payload = {
        "features": []  # Empty features
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error

def test_feature_importance():
    """Test feature importance endpoint"""
    response = client.get("/model/feature-importance")
    # May return 400 if model doesn't support feature importance
    assert response.status_code in [200, 400]