import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.models.predict import ModelPredictor

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [2, 4, 6, 8, 10],
        'feature3': [3, 6, 9, 12, 15]
    })

@pytest.fixture
def predictor():
    """Initialize model predictor"""
    return ModelPredictor()

def test_model_loads(predictor):
    """Test that model loads successfully"""
    assert predictor.model is not None

def test_predict_shape(predictor, sample_data):
    """Test prediction output shape"""
    predictions = predictor.predict(sample_data)
    assert len(predictions) == len(sample_data)

def test_predict_proba_shape(predictor, sample_data):
    """Test probability output shape"""
    probabilities = predictor.predict_proba(sample_data)
    assert probabilities.shape[0] == len(sample_data)
    assert probabilities.shape[1] > 0

def test_probabilities_sum_to_one(predictor, sample_data):
    """Test that probabilities sum to 1"""
    probabilities = predictor.predict_proba(sample_data)
    sums = probabilities.sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, rtol=1e-5)
