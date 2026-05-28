"""Unit tests for novelty and primacy temporal diagnostics (validate/novelty.py)."""

import pytest
import numpy as np
import pandas as pd
from xpyrment.validate.novelty import check_novelty_effects


def test_check_novelty_effects_stable():
    """Asserts stable treatment effects over time are classified as stable."""
    rng = np.random.default_rng(42)
    # Generate 100 samples
    variant = ["control"] * 50 + ["treatment"] * 50
    time = rng.uniform(0.0, 10.0, 100)
    
    # Stable positive effect of 2.0 without time interaction
    y = np.zeros(100)
    for i, var in enumerate(variant):
        y[i] = (2.0 if var == "treatment" else 0.0) + rng.normal(0.0, 0.5)
        
    df = pd.DataFrame({
        "variant": variant,
        "time": time,
        "revenue": y
    })
    
    res = check_novelty_effects(df, "variant", "revenue", "time")
    assert res["classification"] == "Stable Treatment Effect"
    assert res["treatment"]["coef"] == pytest.approx(2.0, abs=0.3)
    assert res["interaction"]["p_value"] > 0.05


def test_check_novelty_effects_novelty():
    """Asserts novelty effect (high initially, decaying over time) is detected."""
    # Fit: Y = beta_0 + beta_1 * T + beta_2 * t + beta_3 * (T * t)
    # Novelty: beta_1 > 0 and beta_3 < 0, p_3 < 0.05
    n = 200
    variant = ["control"] * 100 + ["treatment"] * 100
    time = np.concatenate([np.linspace(0.0, 5.0, 100), np.linspace(0.0, 5.0, 100)])
    
    y = np.zeros(n)
    for i, var in enumerate(variant):
        t_val = time[i]
        if var == "treatment":
            # Treatment: start at +5.0, decay by -1.5 per unit time
            y[i] = 5.0 - 1.5 * t_val
        else:
            # Control: flat at 0.0
            y[i] = 0.0
            
    df = pd.DataFrame({
        "variant": variant,
        "time": time,
        "revenue": y
    })
    
    res = check_novelty_effects(df, "variant", "revenue", "time")
    assert res["classification"] == "Novelty Effect Detected"
    assert res["treatment"]["coef"] == pytest.approx(5.0, abs=0.1)
    assert res["interaction"]["coef"] == pytest.approx(-1.5, abs=0.1)
    assert res["interaction"]["p_value"] < 0.05


def test_check_novelty_effects_primacy():
    """Asserts primacy effect (negative initially, recovering over time) is detected."""
    # Primacy: beta_1 < 0 and beta_3 > 0, p_3 < 0.05
    n = 200
    variant = ["control"] * 100 + ["treatment"] * 100
    time = np.concatenate([np.linspace(0.0, 5.0, 100), np.linspace(0.0, 5.0, 100)])
    
    y = np.zeros(n)
    for i, var in enumerate(variant):
        t_val = time[i]
        if var == "treatment":
            # Treatment: start at -5.0, recover by +1.5 per unit time
            y[i] = -5.0 + 1.5 * t_val
        else:
            y[i] = 0.0
            
    df = pd.DataFrame({
        "variant": variant,
        "time": time,
        "revenue": y
    })
    
    res = check_novelty_effects(df, "variant", "revenue", "time")
    assert res["classification"] == "Primacy Effect Detected"
    assert res["treatment"]["coef"] == pytest.approx(-5.0, abs=0.1)
    assert res["interaction"]["coef"] == pytest.approx(1.5, abs=0.1)
    assert res["interaction"]["p_value"] < 0.05


def test_check_novelty_effects_datetime():
    """Asserts datetime support parses dates to numeric indices correctly."""
    n = 10
    variant = ["control"] * 5 + ["treatment"] * 5
    dates = pd.date_range("2026-05-01", periods=n, freq="D")
    y = np.arange(1.0, 11.0)
    
    df = pd.DataFrame({
        "variant": variant,
        "time": dates,
        "revenue": y
    })
    
    res = check_novelty_effects(df, "variant", "revenue", "time")
    assert "classification" in res
    assert isinstance(res["time"]["coef"], float)


def test_check_novelty_effects_exceptions():
    """Asserts that check_novelty_effects correctly raises errors for degenerate inputs."""
    # Few samples
    df_few = pd.DataFrame({
        "variant": ["c", "t"],
        "time": [1.0, 2.0],
        "revenue": [10.0, 20.0]
    })
    with pytest.raises(ValueError, match="At least 5 samples are required"):
        check_novelty_effects(df_few, "variant", "revenue", "time")

    # Single group
    df_single = pd.DataFrame({
        "variant": ["control"] * 6,
        "time": range(6),
        "revenue": range(6)
    })
    with pytest.raises(ValueError, match="At least 2 groups are required"):
        check_novelty_effects(df_single, "variant", "revenue", "time")

    # Singular matrix (collinear columns)
    df_singular = pd.DataFrame({
        "variant": ["control", "treatment"] * 5,
        "time": [1.0, 1.0] * 5,
        "revenue": range(10)
    })
    with pytest.raises(ValueError, match="Design matrix is singular"):
        check_novelty_effects(df_singular, "variant", "revenue", "time")
