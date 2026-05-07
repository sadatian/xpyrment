import pytest
import numpy as np
from xpyrment.quasi.diff_in_diff import DifferenceInDifferences
from xpyrment.quasi.synthetic_control import SyntheticControl


def test_difference_in_differences():
    """Validates DiD treatment effects calculation and parallel trends check accuracy."""
    rng = np.random.default_rng(42)
    n_samples = 300

    # Simulate: 150 control units, 150 treatment units
    treatment = np.repeat([0, 1], n_samples // 2)
    # 150 pre observations, 150 post observations
    post = np.tile(np.repeat([0, 1], n_samples // 4), 2)

    # Base baseline: y = 10.0 + 3.0 * treatment + 2.0 * post + treatment * post * lift + noise
    # Lift of exactly +4.5
    lift = 4.5
    y = 10.0 + 3.0 * treatment + 2.0 * post + (treatment * post) * lift + rng.normal(scale=0.1, size=n_samples)

    did = DifferenceInDifferences()
    did.fit(y, treatment, post)

    # Verify interaction coefficient matches lift and is highly significant
    assert did.treatment_effect == pytest.approx(4.5, abs=0.15)
    assert did.standard_error < 0.1
    assert did.p_value < 0.001

    # --- 2. Test Parallel Trends Validation ---
    # Case A: Truly parallel pre-period trends
    y_pre_parallel = 10.0 + 1.2 * treatment + rng.normal(scale=0.05, size=n_samples)
    assert did.check_parallel_trends(y_pre_parallel, treatment, post) is True

    # Case B: Non-parallel trends (diverging trends in pre-period)
    y_pre_divergent = 10.0 + (treatment * post) * 5.0 + rng.normal(scale=0.05, size=n_samples)
    assert did.check_parallel_trends(y_pre_divergent, treatment, post) is False


def test_synthetic_control_optimization():
    """Validates Abadie convex-weight convergence and treatment path estimation in Synthetic Controls."""
    rng = np.random.default_rng(42)
    T_pre = 50
    T_post = 20

    # Simulate 3 donor pool units with continuous trend histories
    donors_pre = rng.normal(10.0, 1.0, size=(T_pre, 3))
    
    # Pre-period: Treated unit is a convex combination: 0.3 * Donor_0 + 0.7 * Donor_1 + 0.0 * Donor_2
    w_true = np.array([0.3, 0.7, 0.0])
    treated_pre = np.dot(donors_pre, w_true) + rng.normal(scale=0.01, size=T_pre)

    sc = SyntheticControl()
    sc.fit(treated_pre, donors_pre)

    # Asserts weights satisfy convex criteria: sum to 1.0, bounds in [0.0, 1.0]
    assert np.sum(sc.weights) == pytest.approx(1.0)
    assert np.all(sc.weights >= -1e-7)
    assert np.all(sc.weights <= 1.0 + 1e-7)

    # Verify weights align close to true generative combination
    assert sc.weights[0] == pytest.approx(0.3, abs=0.05)
    assert sc.weights[1] == pytest.approx(0.7, abs=0.05)
    assert sc.weights[2] == pytest.approx(0.0, abs=0.05)

    # Post-period simulation: treated unit experiences a massive shift of +8.5
    donors_post = rng.normal(12.0, 1.0, size=(T_post, 3))
    treated_post = np.dot(donors_post, w_true) + 8.5 + rng.normal(scale=0.01, size=T_post)

    effects = sc.estimate_effect(treated_post, donors_post)

    assert effects.shape == (T_post,)
    # Verify estimated treatment effects matches post-treatment shift
    assert np.mean(effects) == pytest.approx(8.5, abs=0.1)
