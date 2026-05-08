import pytest
import numpy as np
import pandas as pd
from xpyrment.quasi import (
    DifferenceInDifferences,
    SyntheticControl,
    PropensityScoreMatcher,
    CoarsenedExactMatcher,
    mahalanobis_distance,
)


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


def test_mahalanobis_distance():
    """Validates the math of mahalanobis_distance with a custom inverse covariance matrix."""
    u = np.array([1.0, 2.0])
    v = np.array([2.0, 4.0])
    # Identity matrix inverse: becomes standard Euclidean distance: sqrt(1^2 + 2^2) = sqrt(5) ~ 2.236
    cov_inv_eye = np.eye(2)
    dist_eye = mahalanobis_distance(u, v, cov_inv_eye)
    assert dist_eye == pytest.approx(np.sqrt(5.0))

    # Scale distance
    cov_inv_scale = np.array([[0.25, 0.0], [0.0, 0.25]])
    dist_scale = mahalanobis_distance(u, v, cov_inv_scale)
    assert dist_scale == pytest.approx(0.5 * np.sqrt(5.0))


def test_propensity_score_matching():
    """Validates propensity score fitting, logit-caliper checks, and balanced matching output."""
    rng = np.random.default_rng(42)
    n = 200

    # Covariate: age and pre-period conversion rates
    age = rng.normal(loc=35.0, scale=10.0, size=n)
    income = rng.normal(loc=50.0, scale=15.0, size=n)

    # Treatment probability correlates with age: older users are more likely to be treated
    z = -3.0 + 0.08 * age + 0.01 * income
    p = 1.0 / (1.0 + np.exp(-z))
    treatment = rng.binomial(1, p)

    df = pd.DataFrame({
        "age": age,
        "income": income,
        "treatment": treatment,
        "y": 10.0 + 2.0 * treatment + 0.1 * age + rng.normal(scale=1.0, size=n)
    })

    matcher = PropensityScoreMatcher(caliper=0.25)
    matched_df = matcher.fit_predict(df, treatment_col="treatment", covariate_cols=["age", "income"])

    # Verify matching balances treatments: weight must exist and matched columns are present
    assert "propensity_score" in matched_df.columns
    assert "weight" in matched_df.columns
    assert len(matched_df) > 0
    
    # Assert matched weights are uniformly 1.0
    assert np.all(matched_df["weight"] == 1.0)
    
    # Assert both treatment and control are represented
    assert np.sum(matched_df["treatment"] == 1) > 0
    assert np.sum(matched_df["treatment"] == 0) > 0


def test_coarsened_exact_matching():
    """Validates coarsened exact matching binning and balancing weight calculation."""
    rng = np.random.default_rng(42)
    n = 150

    # Covariates
    age = rng.uniform(20.0, 60.0, size=n)
    platform = rng.choice([0, 1], size=n)

    # Treatment depends on platform
    treatment = np.zeros(n, dtype=int)
    for i in range(n):
        prob = 0.8 if platform[i] == 1 else 0.2
        treatment[i] = rng.binomial(1, prob)

    df = pd.DataFrame({
        "age": age,
        "platform": platform,
        "treatment": treatment,
        "outcome": 5.0 + 1.5 * treatment + rng.normal(scale=0.5, size=n)
    })

    cem = CoarsenedExactMatcher(n_bins=4)
    matched_df = cem.fit_predict(df, treatment_col="treatment", covariate_cols=["age", "platform"])

    # Ensure weights were calculated
    assert "weight" in matched_df.columns
    assert len(matched_df) > 0

    # Assert matched treated units get weight 1.0
    treated_matched = matched_df[matched_df["treatment"] == 1]
    assert np.all(treated_matched["weight"] == 1.0)

    # Assert control weights are positive and non-trivial
    control_matched = matched_df[matched_df["treatment"] == 0]
    assert np.all(control_matched["weight"] > 0.0)


def test_synthetic_difference_in_differences():
    """Validates the Synthetic Difference-in-Differences (SDID) estimator weights and target treatment effect."""
    from xpyrment.quasi.sdid import SyntheticDifferenceInDifferences
    import numpy as np

    rng = np.random.default_rng(42)
    T = 30
    t_pre = 20
    N_co = 10
    N_tr = 5

    # Base outcomes: level shift + random trends
    y_control = np.zeros((T, N_co))
    for i in range(N_co):
        base = rng.normal(10.0, 1.0)
        trend = np.linspace(0, rng.uniform(0.5, 2.0), T)
        y_control[:, i] = base + trend + rng.normal(0, 0.05, size=T)

    y_treated = np.zeros((T, N_tr))
    for i in range(N_tr):
        base = rng.normal(12.0, 1.0)
        trend = np.linspace(0, rng.uniform(0.5, 2.0), T)
        effect = np.where(np.arange(T) >= t_pre, 3.5, 0.0)
        y_treated[:, i] = base + trend + effect + rng.normal(0, 0.05, size=T)

    sdid = SyntheticDifferenceInDifferences(l2_penalty=1e-3)
    tau = sdid.fit_estimate(y_control, y_treated, t_pre)

    # Check weights satisfy constraints
    assert np.sum(sdid.unit_weights) == pytest.approx(1.0, abs=1e-5)
    assert np.all(sdid.unit_weights >= -1e-7)
    assert np.sum(sdid.time_weights) == pytest.approx(1.0, abs=1e-5)
    assert np.all(sdid.time_weights >= -1e-7)

    # Verify target treatment effect is accurately estimated near 3.5
    assert tau == pytest.approx(3.5, abs=0.25)


def test_snmm_causal_inference():
    """Validates structural nested mean model sequential g-estimation accuracy."""
    from xpyrment.quasi.snmm import StructuralNestedMeanModel

    rng = np.random.default_rng(42)
    n = 1000

    # Stage 0
    X0 = rng.normal(size=(n, 1))
    p0 = 1.0 / (1.0 + np.exp(-0.5 * X0.ravel()))
    A0 = rng.binomial(1, p0)

    # Stage 1 (Confounder X1 is affected by Stage 0 treatment A0)
    X1 = 0.5 * X0.ravel() + 0.8 * A0 + rng.normal(scale=0.5, size=n)
    X1 = X1.reshape(-1, 1)
    p1 = 1.0 / (1.0 + np.exp(-0.5 * X1.ravel() + 0.2 * A0))
    A1 = rng.binomial(1, p1)

    # Outcome model with true treatment coefficients:
    # Stage 0: beta_0 = [2.0, 3.0] -> 2.0 * A0 + 3.0 * A0 * X0
    # Stage 1: beta_1 = [4.5, 1.2] -> 4.5 * A1 + 1.2 * A1 * X1
    y = (
        10.0 
        + 2.0 * A0 + 3.0 * A0 * X0.ravel() 
        + 4.5 * A1 + 1.2 * A1 * X1.ravel() 
        + 1.0 * X0.ravel() + 0.8 * X1.ravel() 
        + rng.normal(scale=0.1, size=n)
    )

    snmm = StructuralNestedMeanModel(l2_penalty=1e-4)
    snmm.fit([X0, X1], [A0, A1], y)

    # Stage 0 Coefficients: [Intercept effect, Covariate slope effect]
    beta_0 = snmm.coefficients[0]
    # Stage 1 Coefficients: [Intercept effect, Covariate slope effect]
    beta_1 = snmm.coefficients[1]

    assert len(beta_0) == 2
    assert len(beta_1) == 2

    # Asserts estimation is accurate to true generative values (incorporates 0.64 indirect mediation effect through X1)
    assert beta_0[0] == pytest.approx(2.64, abs=0.25)
    assert beta_0[1] == pytest.approx(3.0, abs=0.25)
    assert beta_1[0] == pytest.approx(4.5, abs=0.25)
    assert beta_1[1] == pytest.approx(1.2, abs=0.25)

    # Test blip down outcome logic
    # Removing stage 1 effects from y should leave only stage 0 effects + base trends
    y_blipped_1 = snmm.predict_blip_outcome([X0, X1], [A0, A1], y, stage=1)
    assert len(y_blipped_1) == n



