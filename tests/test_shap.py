"""Unit tests for game-theoretic feature interactions using SHAP (interactions/shap.py)."""

import sys
import builtins
import pytest
import numpy as np
from unittest.mock import MagicMock
from xpyrment.interactions.shap import calculate_shap_interactions


def test_calculate_shap_interactions_happy_path(monkeypatch):
    """Asserts calculate_shap_interactions correctly instantiates TreeExplainer and returns interaction values."""
    mock_shap = MagicMock()
    mock_explainer = MagicMock()
    
    # Configure mock hierarchy
    mock_shap.TreeExplainer.return_value = mock_explainer
    dummy_interactions = np.random.default_rng(42).normal(size=(5, 3, 3))
    mock_explainer.shap_interaction_values.return_value = dummy_interactions
    
    # Inject mock shap into sys.modules
    monkeypatch.setitem(sys.modules, "shap", mock_shap)
    
    # Run target function
    dummy_model = MagicMock()
    dummy_data = np.random.default_rng(42).normal(size=(5, 3))
    
    res = calculate_shap_interactions(dummy_model, dummy_data)
    
    # Verify calls and output
    mock_shap.TreeExplainer.assert_called_once_with(dummy_model)
    mock_explainer.shap_interaction_values.assert_called_once_with(dummy_data)
    assert np.allclose(res, dummy_interactions)


def test_calculate_shap_interactions_import_error(monkeypatch):
    """Asserts that calculate_shap_interactions raises an ImportError when the 'shap' library is missing."""
    # Ensure 'shap' is not in sys.modules
    if "shap" in sys.modules:
        monkeypatch.delitem(sys.modules, "shap")
        
    original_import = builtins.__import__
    
    def mock_import(name, *args, **kwargs):
        if name == "shap":
            raise ImportError("mocked import error")
        return original_import(name, *args, **kwargs)
        
    monkeypatch.setattr(builtins, "__import__", mock_import)
    
    dummy_model = MagicMock()
    dummy_data = np.random.default_rng(42).normal(size=(5, 3))
    
    with pytest.raises(ImportError) as exc_info:
        calculate_shap_interactions(dummy_model, dummy_data)
        
    assert "The 'shap' library is required to calculate SHAP interactions" in str(exc_info.value)
    assert "Please install it using: pip install shap" in str(exc_info.value)


def test_calculate_shap_interactions_real_integration():
    """Asserts that calculate_shap_interactions executes cleanly using a real trained tree model and data."""
    from sklearn.tree import DecisionTreeRegressor
    
    # Generate simple training data
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 3))
    y = X[:, 0] * 2.0 + X[:, 1] * 0.5 + rng.normal(scale=0.1, size=20)
    
    model = DecisionTreeRegressor(max_depth=3, random_state=42)
    model.fit(X, y)
    
    # Calculate interactions using real shap
    res = calculate_shap_interactions(model, X)
    
    # Shape of output should be (num_samples, num_features, num_features)
    assert isinstance(res, np.ndarray)
    assert res.shape == (20, 3, 3)

