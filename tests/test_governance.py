import pytest
from xpyrment.governance.meta_analysis import MetaAnalysis
from xpyrment.governance.p_curve import PCurve


def test_meta_analysis():
    """Validates Fixed-Effects and DerSimonian-Laird Random-Effects pooled estimators and limits."""
    # Mismatched length validations
    with pytest.raises(ValueError):
        MetaAnalysis([1.0], [0.5, 0.5])
    # Single study limit validations
    with pytest.raises(ValueError):
        MetaAnalysis([1.0], [0.5])
    # Negative variance validations
    with pytest.raises(ValueError):
        MetaAnalysis([1.0, 2.0], [0.5, -0.1])

    # Three simulated studies
    estimates = [2.0, 2.5, 1.8]
    variances = [0.5, 0.4, 0.6]

    ma = MetaAnalysis(estimates, variances)

    # 1. Test Fixed Effects
    fe_results = ma.fit_fixed_effects()
    
    # Hand calculations:
    # w1 = 2, w2 = 2.5, w3 = 1.66667 -> sum_w = 6.16667
    # w * est: 4 + 6.25 + 3.0 = 13.25
    # pooled_fe = 13.25 / 6.16667 = 2.1486
    # pooled_se = sqrt(1 / 6.16667) = 0.4027
    assert fe_results["pooled_effect"] == pytest.approx(2.1486, abs=0.001)
    assert fe_results["standard_error"] == pytest.approx(0.4027, abs=0.001)
    assert fe_results["p_value"] < 0.001

    # 2. Test Random Effects
    re_results = ma.fit_random_effects()
    assert "cochrans_q" in re_results
    assert re_results["tau_squared"] >= 0.0
    assert re_results["pooled_effect"] == pytest.approx(2.1486, abs=0.05)


def test_p_curve_analysis():
    """Validates Simonsohn binomial skewness checks for true power versus gaming indicators."""
    # Case A: Too few values
    pc_small = PCurve([0.01, 0.02])
    res_small = pc_small.analyze()
    assert res_small["status"] == "Inconclusive"

    # Case B: Right-skewed distribution (Strong Evidential Value)
    # 10 highly significant p-values below 0.025
    right_skew_p = [0.001, 0.002, 0.005, 0.01, 0.008, 0.012, 0.004, 0.006, 0.02, 0.045]
    pc_right = PCurve(right_skew_p)
    res_right = pc_right.analyze()
    assert res_right["status"] == "Strong Evidential Value"
    assert res_right["n_low_half"] == 9
    assert res_right["n_high_half"] == 1

    # Case C: Left-skewed distribution (Reporting Bias / P-Hacking Detected)
    # Disproportionate clustering just under 0.05
    left_skew_p = [
        0.045, 0.042, 0.048, 0.049, 0.041, 0.047, 0.043, 0.046, 0.044, 0.041,
        0.042, 0.043, 0.01, 0.015, 0.02
    ]
    pc_left = PCurve(left_skew_p)
    res_left = pc_left.analyze()
    assert res_left["status"] == "Reporting Bias / P-Hacking Detected"
    assert res_left["n_low_half"] == 3
    assert res_left["n_high_half"] == 12

def test_p_curve_plot():
    """Asserts that PCurve plot generation returns valid matplotlib canvases and does not fail."""
    import matplotlib.pyplot as plt
    from xpyrment.governance.p_curve import plot_p_curve, PCurve

    # Test with standard distribution
    p_vals = [0.001, 0.002, 0.005, 0.01, 0.008, 0.012, 0.004, 0.006, 0.02, 0.045]

    # Test standalone
    fig, ax = plot_p_curve(p_vals)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)

    # Test via class method
    pc = PCurve(p_vals)
    fig_class, ax_class = pc.plot()
    assert isinstance(fig_class, plt.Figure)
    assert isinstance(ax_class, plt.Axes)
    plt.close(fig_class)

    # Test with empty significant values
    pc_empty = PCurve([0.1, 0.2, 0.3])
    fig_empty, ax_empty = pc_empty.plot()
    assert isinstance(fig_empty, plt.Figure)
    assert isinstance(ax_empty, plt.Axes)
    plt.close(fig_empty)
