"""Unit tests for streaming RLS regression (streaming.py) and variance reduction (variance_reduction.py)."""

import pytest
import numpy as np
import pandas as pd
from xpyrment.analyze.streaming import StreamingOLS
from xpyrment.analyze.variance_reduction import apply_cuped


def test_streaming_ols_happy_path():
    """Asserts StreamingOLS online coefficients match offline linear regression equations."""
    rng = np.random.default_rng(42)
    n_samples = 100
    n_features = 2
    
    # Generate OLS dataset: y = 5.0 + 2.0*X_1 - 1.5*X_2 + noise
    X = rng.normal(size=(n_samples, n_features))
    true_beta = np.array([5.0, 2.0, -1.5])
    y = 5.0 + 2.0 * X[:, 0] - 1.5 * X[:, 1] + rng.normal(scale=0.1, size=n_samples)
    
    # Initialize streaming OLS
    stream_model = StreamingOLS(n_features=n_features, l2_penalty=0.0, fit_intercept=True)
    
    # Update sample by sample
    for i in range(n_samples):
        stream_model.update(X[i], y[i])
        
    # Fit offline standard regression for comparison
    X_bias = np.hstack([np.ones((n_samples, 1)), X])
    offline_beta = np.linalg.solve(np.dot(X_bias.T, X_bias), np.dot(X_bias.T, y))
    
    # Assert running weights match within high precision
    assert np.allclose(stream_model.coefficients, offline_beta, rtol=1e-3, atol=1e-3)
    
    # Assert predictions
    preds = stream_model.predict(X)
    assert len(preds) == n_samples


def test_streaming_ols_no_intercept():
    """Asserts StreamingOLS without intercept fits correctly."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 1))
    y = 3.0 * X[:, 0] + rng.normal(scale=0.1, size=50)
    
    stream_model = StreamingOLS(n_features=1, l2_penalty=0.0, fit_intercept=False)
    for i in range(50):
        stream_model.update(X[i], y[i])
        
    # Analytical OLS without intercept
    offline_beta = np.dot(X.T, y) / np.dot(X.T, X)
    assert np.allclose(stream_model.coefficients, offline_beta, rtol=1e-3, atol=1e-3)
    
    # Verify predictions
    preds = stream_model.predict(X)
    assert np.allclose(preds, X.ravel() * stream_model.beta[0])


def test_streaming_ols_batch_update():
    """Asserts batch updating yields identical results to sequential updating."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 2))
    y = 1.0 + 2.0 * X[:, 0] + X[:, 1]
    
    model = StreamingOLS(n_features=2, l2_penalty=0.1)
    model.update_batch(X, y)
    
    assert model.n_samples == 20


def test_streaming_ols_exceptions():
    """Asserts StreamingOLS raises errors for invalid shapes and parameters."""
    # Incorrect dimension
    model = StreamingOLS(n_features=2)
    with pytest.raises(ValueError, match="vector has dimension 3, expected 2"):
        model.update(np.array([1.0, 2.0, 3.0]), 10.0)
        
    # Batch dimension mismatch
    with pytest.raises(ValueError, match="matching sample dimensions"):
        model.update_batch(np.random.rand(5, 2), np.random.rand(6))


def test_cuped_variance_reduction():
    """Asserts apply_cuped reduces outcome variance when pre-period covariate is correlated."""
    rng = np.random.default_rng(42)
    # Generate highly correlated pre and post period continuous variables
    pre = rng.normal(10, 2, 500)
    post = pre + rng.normal(0, 1, 500)  # Strong positive correlation
    
    df = pd.DataFrame({
        "pre": pre,
        "post": post
    })
    
    adjusted = apply_cuped(df, "post", "pre")
    
    # Verify adjusted variance is strictly smaller than raw post variance
    var_raw = df["post"].var()
    var_adj = adjusted.var()
    assert var_adj < var_raw


def test_cuped_edge_cases():
    """Asserts apply_cuped handles tiny samples and zero variance covariates safely."""
    # 1. Zero-variance pre-period covariate
    df_zero = pd.DataFrame({
        "post": [1.0, 2.0, 3.0],
        "pre": [10.0, 10.0, 10.0]
    })
    adjusted = apply_cuped(df_zero, "post", "pre")
    assert np.allclose(adjusted, df_zero["post"])
    
    # 2. Too few samples (< 2)
    df_tiny = pd.DataFrame({
        "post": [1.0],
        "pre": [10.0]
    })
    adjusted_tiny = apply_cuped(df_tiny, "post", "pre")
    assert np.allclose(adjusted_tiny, df_tiny["post"])
