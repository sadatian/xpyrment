"""Unit tests for Experimental Sizing & Power Sizing Planning (Block 44 / plan/power.py)."""

import pytest
import numpy as np
from xpyrment.plan.power import design_experiment, generate_power_curve_data, ExperimentDesignResult


def test_design_experiment_proportion_defaults():
    """Verifies baseline conversions, sample sizes, and summary representation for proportions."""
    # Planning a conversion rate proportion test (10% baseline, relative MDE of 5%)
    result = design_experiment(
        metric_type="proportion",
        baseline_value=0.10,
        mde=0.05,
        mde_type="relative",
        alpha=0.05,
        power=0.80
    )

    # Standard expected proportion sizing
    assert result.details["metric_type"] == "proportion"
    assert result.details["sample_size_per_variant"] == pytest.approx(56512, abs=5)
    assert result.details["total_sample_size"] == pytest.approx(56512 * 2, abs=10)

    # Test summary compiles clean strings
    summary = result.summary()
    assert summary["Parameter"][0] == "Metric Type"
    assert summary["Value"][0] == "Proportion"
    assert "56,512" in summary["Value"][6]

    # repr formatting check
    repr_str = repr(result)
    assert "Experiment Design Summary" in repr_str
    assert "Proportion" in repr_str


def test_design_experiment_mean_and_ratio():
    """Validates t-test sizing computations for continuous means and ratio metrics."""
    # Continuous mean with standard_deviation specified
    result_mean = design_experiment(
        metric_type="mean",
        baseline_value=12.5,
        standard_deviation=4.0,
        mde=1.0,
        mde_type="absolute",
        alpha=0.05,
        power=0.80
    )
    assert result_mean.details["metric_type"] == "mean"
    assert result_mean.details["sample_size_per_variant"] == pytest.approx(251, abs=2)

    # Continuous ratio metrics
    result_ratio = design_experiment(
        metric_type="ratio",
        baseline_value=2.0,
        standard_deviation=1.5,
        mde=0.10,
        mde_type="relative",
        alpha=0.01,
        power=0.90
    )
    assert result_ratio.details["metric_type"] == "ratio"
    assert result_ratio.details["sample_size_per_variant"] > 0


def test_design_experiment_cuped_and_traffic():
    """Asserts CUPED sample size deflation credit and run duration calculations."""
    # Planning with 0.80 pre-post correlation (64% variance reduction, 36% size required)
    result = design_experiment(
        metric_type="mean",
        baseline_value=100.0,
        standard_deviation=15.0,
        mde=0.05,
        mde_type="relative",
        alpha=0.05,
        power=0.80,
        pre_post_correlation=0.80,
        daily_traffic=100
    )

    details = result.details
    assert details["pre_post_correlation"] == 0.80
    assert details["cuped_savings"] == pytest.approx(0.64, abs=0.01)
    
    # Standard size without CUPED is ~141.2 samples per variant
    standard_size = details["sample_size_per_variant"]
    cuped_size = details["cuped_sample_size_per_variant"]
    assert cuped_size == pytest.approx(standard_size * 0.36, abs=0.1)

    # Durations mapping
    assert details["duration_days_standard"] > 0
    assert details["duration_days_cuped"] == pytest.approx(details["duration_days_standard"] * 0.36, abs=0.01)

    summary = result.summary()
    assert "Pre-Post Correlation" in summary["Parameter"]
    assert "Estimated Duration (CUPED)" in summary["Parameter"]


def test_design_experiment_validation_errors():
    """Asserts that invalid metrics, incorrect bounds, and missing SDs raise proper ValueErrors."""
    # 1. Invalid metric type
    with pytest.raises(ValueError, match="metric_type must be one of"):
        design_experiment(metric_type="poisson", baseline_value=10.0)

    # 2. Invalid MDE type
    with pytest.raises(ValueError, match="mde_type must be 'relative' or 'absolute'"):
        design_experiment(metric_type="mean", baseline_value=10.0, mde_type="scaled")

    # 3. Missing standard deviation for continuous mean
    with pytest.raises(ValueError, match="standard_deviation is required for metric type"):
        design_experiment(metric_type="mean", baseline_value=10.0, standard_deviation=None)

    # 4. Proportions out of (0, 1) bounds
    with pytest.raises(ValueError, match="For proportions, baseline_value must be strictly between 0 and 1"):
        design_experiment(metric_type="proportion", baseline_value=1.5)
    with pytest.raises(ValueError, match="For proportions, baseline_value must be strictly between 0 and 1"):
        design_experiment(metric_type="proportion", baseline_value=-0.1)

    # 5. Correlation out of bounds
    with pytest.raises(ValueError, match="pre_post_correlation must be between"):
        design_experiment(metric_type="mean", baseline_value=10.0, standard_deviation=2.0, pre_post_correlation=1.2)
    with pytest.raises(ValueError, match="pre_post_correlation must be between"):
        design_experiment(metric_type="mean", baseline_value=10.0, standard_deviation=2.0, pre_post_correlation=-1.05)

    # 6. Negative daily traffic
    with pytest.raises(ValueError, match="daily_traffic must be positive"):
        design_experiment(metric_type="mean", baseline_value=10.0, standard_deviation=2.0, daily_traffic=-10)


def test_generate_power_curve_data():
    """Verifies generator output array dimensions and contents for plot utilities."""
    # 1. Basic curve generation without correlation
    curve = generate_power_curve_data(
        metric_type="mean",
        baseline_value=5.0,
        standard_deviation=1.0,
        alpha=0.05,
        power=0.80
    )
    assert len(curve["mde_relative"]) == 50
    assert len(curve["sample_size_per_variant"]) == 50
    assert "cuped_sample_size_per_variant" not in curve

    # 2. Curve generation with pre-post correlation
    curve_cuped = generate_power_curve_data(
        metric_type="mean",
        baseline_value=5.0,
        standard_deviation=1.0,
        alpha=0.05,
        power=0.80,
        mde_range=np.array([0.02, 0.05, 0.10]),
        pre_post_correlation=0.50
    )
    assert len(curve_cuped["mde_relative"]) == 3
    assert len(curve_cuped["sample_size_per_variant"]) == 3
    assert len(curve_cuped["cuped_sample_size_per_variant"]) == 3

    # Baseline value of 0 edge cases
    res_mde = design_experiment(
        metric_type="mean",
        baseline_value=0.0,
        standard_deviation=2.0,
        mde=1.0,
        mde_type="absolute"
    )
    assert res_mde.details["mde_relative"] == 0.0
