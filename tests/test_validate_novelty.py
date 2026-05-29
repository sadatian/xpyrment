import pytest
import numpy as np
import pandas as pd
from xpyrment.validate.novelty import check_novelty_effects

def test_check_novelty_effects_stable():
    df = pd.DataFrame({
        "group": ["A", "B", "A", "B", "A", "B"],
        "metric": [10.0, 15.0, 10.0, 15.0, 10.0, 15.0],
        "time": [1, 1, 2, 2, 3, 3]
    })
    res = check_novelty_effects(df, "group", "metric", "time")
    assert res["classification"] == "Stable Treatment Effect"

def test_check_novelty_effects_novelty():
    # beta_1 > 0 and beta_3 < 0
    # Treatment starts high and decays
    df = pd.DataFrame({
        "group": ["A", "B", "A", "B", "A", "B"],
        "metric": [10.0, 20.0, 10.0, 15.0, 10.0, 10.0],
        "time": [1, 1, 2, 2, 3, 3]
    })
    # We will generate more data to ensure significance
    x_t = np.tile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2)
    x_g = np.repeat(["A", "B"], 10)
    y = np.where(x_g == "A", 10.0, 20.0 - 1.0 * x_t)
    df2 = pd.DataFrame({"group": x_g, "metric": y, "time": x_t})
    res2 = check_novelty_effects(df2, "group", "metric", "time")
    assert res2["classification"] == "Novelty Effect Detected"

def test_check_novelty_effects_primacy():
    # beta_1 < 0 and beta_3 > 0
    x_t = np.tile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2)
    x_g = np.repeat(["A", "B"], 10)
    y = np.where(x_g == "A", 10.0, 0.0 + 1.0 * x_t)
    df2 = pd.DataFrame({"group": x_g, "metric": y, "time": x_t})
    res2 = check_novelty_effects(df2, "group", "metric", "time")
    assert res2["classification"] == "Primacy Effect Detected"

def test_check_novelty_effects_datetime():
    df = pd.DataFrame({
        "group": ["A", "B", "A", "B", "A", "B"],
        "metric": [10.0, 15.0, 10.0, 15.0, 10.0, 15.0],
        "time": pd.to_datetime(["2021-01-01", "2021-01-01", "2021-01-02", "2021-01-02", "2021-01-03", "2021-01-03"])
    })
    res = check_novelty_effects(df, "group", "metric", "time")
    assert res["classification"] == "Stable Treatment Effect"

def test_check_novelty_effects_errors():
    df = pd.DataFrame({"g": ["A", "B"], "m": [1, 2], "t": [1, 2]})
    with pytest.raises(ValueError, match="At least 5 samples"):
        check_novelty_effects(df, "g", "m", "t")

    df2 = pd.DataFrame({"g": ["A", "A", "A", "A", "A"], "m": [1, 2, 3, 4, 5], "t": [1, 2, 3, 4, 5]})
    with pytest.raises(ValueError, match="At least 2 groups"):
        check_novelty_effects(df2, "g", "m", "t")

def test_check_novelty_effects_singular():
    df = pd.DataFrame({
        "group": ["A", "B", "A", "B", "A", "B"],
        "metric": [10.0, 15.0, 10.0, 15.0, 10.0, 15.0],
        "time": [1, 1, 1, 1, 1, 1] # singular because time has no variance
    })
    with pytest.raises(ValueError, match="singular"):
        check_novelty_effects(df, "group", "metric", "time")
