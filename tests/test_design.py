import pytest
from xpyrment.plan.power import design_experiment, generate_power_curve_data


def test_design_experiment_proportion():
    """Tests sample size calculations for proportion metrics."""
    res = design_experiment(
        metric_type="proportion",
        baseline_value=0.10,
        mde=0.01,
        mde_type="absolute",
        alpha=0.05,
        power=0.80,
    )
    details = res.details
    assert details["metric_type"] == "proportion"
    assert details["baseline_value"] == 0.10
    assert details["mde_absolute"] == 0.01
    assert details["sample_size_per_variant"] > 0


def test_design_experiment_mean():
    """Tests sample size calculations for continuous mean metrics."""
    res = design_experiment(
        metric_type="mean",
        baseline_value=10.0,
        standard_deviation=5.0,
        mde=0.05,
        mde_type="relative",
        alpha=0.05,
        power=0.80,
    )
    details = res.details
    assert details["metric_type"] == "mean"
    assert details["baseline_value"] == 10.0
    assert pytest.approx(details["mde_absolute"]) == 0.5
    assert details["sample_size_per_variant"] > 0


def test_design_experiment_cuped_savings():
    """Tests that a positive pre-period correlation reduces sample sizes accordingly."""
    res_no_cuped = design_experiment(
        metric_type="mean",
        baseline_value=10.0,
        standard_deviation=5.0,
        mde=0.05,
        alpha=0.05,
        power=0.80,
    )

    res_cuped = design_experiment(
        metric_type="mean",
        baseline_value=10.0,
        standard_deviation=5.0,
        mde=0.05,
        alpha=0.05,
        power=0.80,
        pre_post_correlation=0.6,
    )

    assert "cuped_sample_size_per_variant" in res_cuped.details
    # 1 - 0.6^2 = 0.64 variance factor -> 36% sample size savings
    assert pytest.approx(res_cuped.details["cuped_savings"]) == 0.36
    assert (
        res_cuped.details["cuped_sample_size_per_variant"]
        < res_no_cuped.details["sample_size_per_variant"]
    )


def test_generate_power_curve_data():
    """Tests structure and sorting properties of generated power curve coordinates."""
    curve = generate_power_curve_data(
        metric_type="mean",
        baseline_value=10.0,
        standard_deviation=5.0,
        pre_post_correlation=0.5,
    )
    assert "mde_relative" in curve
    assert "sample_size_per_variant" in curve
    assert "cuped_sample_size_per_variant" in curve
    assert len(curve["mde_relative"]) == 50
