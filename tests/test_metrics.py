import numpy as np
import pandas as pd
import pytest
from xpyrment.metrics.taxonomy import MeanMetric, ProportionMetric, RatioMetric


def test_mean_metric_calculation():
    """Tests that continuous mean calculations are robust and accurate."""
    data = {
        "variant": ["control"] * 50 + ["treatment"] * 50,
        "revenue": list(np.random.normal(10, 2, 50)) + list(np.random.normal(12, 2, 50)),
    }
    df = pd.DataFrame(data)

    metric = MeanMetric("Revenue", value_col="revenue")
    res = metric.calculate(df, "variant", "control", "treatment")

    assert res["metric_name"] == "Revenue"
    assert res["metric_type"] == "Mean"
    assert res["control_n"] == 50
    assert res["treatment_n"] == 50
    assert res["control_mean"] > 0
    assert res["treatment_mean"] > 0
    assert "p_value" in res
    assert "ci_lower" in res


def test_proportion_metric_calculation():
    """Tests proportion metric conversions and outcomes."""
    data = {
        "variant": ["control"] * 100 + ["treatment"] * 100,
        "converted": [1] * 10 + [0] * 90 + [1] * 20 + [0] * 80,  # 10% vs 20%
    }
    df = pd.DataFrame(data)

    metric = ProportionMetric("CVR", value_col="converted")
    res = metric.calculate(df, "variant", "control", "treatment")

    assert res["metric_type"] == "Proportion"
    assert pytest.approx(res["control_mean"]) == 0.10
    assert pytest.approx(res["treatment_mean"]) == 0.20
    assert res["p_value"] < 0.10  # Highly significant difference


def test_ratio_metric_delta_method():
    """Tests Taylor-series Delta Method standard error estimations for ratio metrics."""
    # Build stochastically to maintain positive variances
    rng = np.random.default_rng(101)
    n_samples = 200

    data = {
        "variant": ["control"] * n_samples + ["treatment"] * n_samples,
        "clicks": list(rng.binomial(10, 0.1, n_samples)) + list(rng.binomial(10, 0.15, n_samples)),
        "impressions": list(rng.poisson(100, n_samples)) + list(rng.poisson(100, n_samples)),
    }
    df = pd.DataFrame(data)

    metric = RatioMetric("CTR", numerator_col="clicks", denominator_col="impressions")
    res = metric.calculate(df, "variant", "control", "treatment")

    assert res["metric_type"] == "Ratio"
    assert res["control_mean"] > 0
    assert res["treatment_mean"] > 0
    assert res["control_var"] > 0
    assert "p_value" in res


def test_mean_metric_cuped():
    """Tests that CUPED adjustments yield non-negative variance reductions."""
    rng = np.random.default_rng(2022)
    n_samples = 100

    # Strong correlation (0.8) between pre and post revenue
    pre_rev = rng.normal(10, 3, n_samples)
    post_rev = pre_rev + rng.normal(1, 1, n_samples)

    df = pd.DataFrame(
        {
            "variant": ["control"] * 50 + ["treatment"] * 50,
            "pre_revenue": pre_rev,
            "revenue": post_rev,
        }
    )

    metric_no_cuped = MeanMetric("Revenue", value_col="revenue")
    res_no_cuped = metric_no_cuped.calculate(df, "variant", "control", "treatment")

    metric_cuped = MeanMetric("Revenue", value_col="revenue", pre_period_col="pre_revenue")
    res_cuped = metric_cuped.calculate(df, "variant", "control", "treatment")

    assert res_cuped["cuped_applied"] is True
    assert res_cuped["variance_reduction"] > 0.40  # Highly correlated variables reduce variance significantly
    # Confidence Interval under CUPED should be narrower
    ci_range_no_cuped = res_no_cuped["ci_upper"] - res_no_cuped["ci_lower"]
    ci_range_cuped = res_cuped["ci_upper"] - res_cuped["ci_lower"]
    assert ci_range_cuped < ci_range_no_cuped


def test_reproduce_issue_cuped_singular_covariate():
    """Tests that CUPED correctly falls back to unadjusted difference-in-means when covariate variance is near zero."""
    n_samples = 100
    # Create covariate with near-zero variance (~1e-22)
    pre_rev = np.array([10.0] * n_samples, dtype=np.float64)
    pre_rev[0] += 1e-10

    post_rev = np.random.default_rng(42).normal(10, 2, n_samples)

    df = pd.DataFrame(
        {
            "variant": ["control"] * 50 + ["treatment"] * 50,
            "pre_revenue": pre_rev,
            "revenue": post_rev,
        }
    )

    metric_cuped = MeanMetric("Revenue", value_col="revenue", pre_period_col="pre_revenue")
    res_cuped = metric_cuped.calculate(df, "variant", "control", "treatment")

    assert res_cuped["cuped_applied"] is False
    assert res_cuped["variance_reduction"] == 0.0

    # Also test RatioMetric with near-zero pre-period variance
    df_ratio = pd.DataFrame(
        {
            "variant": ["control"] * 50 + ["treatment"] * 50,
            "clicks": post_rev,
            "impressions": [100] * 100,
            "pre_clicks": pre_rev,
            "pre_impressions": [100] * 100,
        }
    )
    ratio_metric = RatioMetric(
        "CTR",
        numerator_col="clicks",
        denominator_col="impressions",
        pre_numerator_col="pre_clicks",
        pre_denominator_col="pre_impressions",
    )
    res_ratio = ratio_metric.calculate(df_ratio, "variant", "control", "treatment")
    assert res_ratio["cuped_applied"] is False
    assert res_ratio["variance_reduction"] == 0.0

