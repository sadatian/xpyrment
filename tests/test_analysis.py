import pandas as pd
import pytest
import numpy as np

from xpyrment.analyze.orchestrator import run_analysis, setup
from xpyrment.core.exceptions import PhaseOrderError
from xpyrment.core.state import ExperimentState
from xpyrment.metrics.taxonomy import MeanMetric, ProportionMetric
from xpyrment.simulation import generate_ab_data


def test_end_to_end_setup_and_analysis():
    """Validates the complete PyCaret-style setup, run_analysis, and summary workflow."""
    df = generate_ab_data(n_samples=500, random_seed=42)

    # 1. Initialize experiment setup
    exp = setup(df, treatment_col="variant", id_col="user_id")
    assert exp.state == ExperimentState.CREATED

    # 2. Add metrics
    metric_rev = MeanMetric("Revenue", value_col="revenue", pre_period_col="pre_revenue")
    metric_conv = ProportionMetric("Conversion", value_col="converted")

    exp.add_metrics([metric_rev, metric_conv])
    assert len(exp.metrics) == 2

    # 3. Running metrics before setup should work, but adding metrics after running analysis should fail
    results = exp.run_analysis(control="control", treatment="treatment")
    assert exp.state == ExperimentState.ANALYZED

    # Ensure adding metrics after analysis fails (phase-gated rule)
    with pytest.raises(PhaseOrderError):
        exp.add_metrics(metric_rev)

    # 4. Check results summary
    summary_df = results.summary(formatted=True)
    assert isinstance(summary_df, pd.DataFrame)
    assert len(summary_df) == 2
    assert "Metric" in summary_df.columns
    assert "Relative Lift" in summary_df.columns
    assert "p-value" in summary_df.columns


def test_multiple_comparison_corrections():
    """Asserts that multiple comparison p-value adjustments are properly calculated."""
    df = generate_ab_data(n_samples=400, random_seed=7)

    exp = setup(df, treatment_col="variant", id_col="user_id")
    exp.add_metrics([
        MeanMetric("Revenue", value_col="revenue"),
        ProportionMetric("Conversion", value_col="converted")
    ])

    # Run without adjustments
    res_raw = run_analysis(exp, control="control", treatment="treatment", multi_test_correction=None)
    raw_p_values = res_raw.df_raw["p_value"].tolist()

    # Re-run with Bonferroni adjustment
    # Note: re-running from ANALYZED state is allowed, but we reset state for testing
    exp.state = ExperimentState.CREATED
    res_adj = run_analysis(exp, control="control", treatment="treatment", multi_test_correction="bonferroni")
    adj_p_values = res_adj.df_raw["p_value"].tolist()

    # With multiple tests, adjusted p-values must be greater than or equal to unadjusted ones
    for raw, adj in zip(raw_p_values, adj_p_values):
        assert adj >= raw


def test_frequentist_welch_and_mann_whitney():
    """Asserts that run_welch_t_test and run_mann_whitney_u produce mathematically accurate stats."""
    import numpy as np
    from xpyrment.analyze.inference.frequentist import run_welch_t_test, run_mann_whitney_u

    rng = np.random.default_rng(42)
    group_a = rng.normal(loc=10.0, scale=1.5, size=100)
    group_b = rng.normal(loc=11.5, scale=2.5, size=120)  # Variance and N differ

    # 1. Welch's T-test
    res_t = run_welch_t_test(group_a, group_b)
    assert res_t["difference"] > 0.0
    assert res_t["t_statistic"] > 0.0
    assert res_t["p_value"] < 0.001
    assert 100 < res_t["df"] < 218

    # 2. Mann-Whitney U
    res_u = run_mann_whitney_u(group_a, group_b)
    assert "u_statistic" in res_u
    assert res_u["p_value"] < 0.001


def test_apply_cuped_variance_deflation():
    """Verifies that apply_cuped reduces target metric variance using pre-period covariates."""
    import numpy as np
    from xpyrment.analyze.variance_reduction import apply_cuped

    rng = np.random.default_rng(123)
    n = 500

    # Pre-period covariate (X)
    pre = rng.normal(loc=50.0, scale=10.0, size=n)
    # Post-period target (Y), highly correlated with pre-period
    target = pre + rng.normal(loc=5.0, scale=3.0, size=n)

    df = pd.DataFrame({
        "target": target,
        "pre": pre
    })

    # Apply CUPED
    df["adjusted"] = apply_cuped(df, "target", "pre")

    # Math assertion: CUPED-adjusted variance must be strictly less than original variance
    orig_variance = df["target"].var()
    reduced_variance = df["adjusted"].var()

    assert reduced_variance < orig_variance
    # Variance should be deflated by approx (1 - rho^2). For target ~ pre + N(0,3), correlation is ~0.95
    # So variance should be deflated by around 90%!
    assert reduced_variance < 0.2 * orig_variance


def test_bayesian_inference_conjugate_beta_binomial():
    """Tests Beta-Binomial conjugate posterior parameters and Monte Carlo decision metrics."""
    from xpyrment.analyze.inference.bayesian import BayesianInference

    # Uniform priors (Beta(1,1))
    prior = {"alpha": 1.0, "beta": 1.0}
    
    # Obs: Control has 10/100 conversions (10%), Treatment has 25/100 (25%)
    obs = {
        "control_successes": 10, "control_trials": 100,
        "treatment_successes": 25, "treatment_trials": 100
    }

    engine = BayesianInference(model_type="beta_binomial")
    results = engine.estimate_posterior(prior, obs)

    # Posteriors: C ~ Beta(11, 91), T ~ Beta(26, 76)
    assert results["control_posterior"]["param1"] == 11.0
    assert results["control_posterior"]["param2"] == 91.0
    assert results["treatment_posterior"]["param1"] == 26.0
    assert results["treatment_posterior"]["param2"] == 76.0

    # Credible Intervals should surround true sample rates
    assert results["control_posterior"]["ci_lower"] < 0.10 < results["control_posterior"]["ci_upper"]
    assert results["treatment_posterior"]["ci_lower"] < 0.25 < results["treatment_posterior"]["ci_upper"]

    # Decision metrics
    assert results["pbb"] > 0.95  # Probability of treatment being better is extremely high
    assert results["expected_loss"] < 0.01  # Loss of shipping treatment is near zero


def test_bayesian_inference_conjugate_normal_normal():
    """Tests Normal-Normal conjugate posterior parameters and decision metrics."""
    from xpyrment.analyze.inference.bayesian import BayesianInference

    # Prior: mean=0, variance=10.0 (weakly informative)
    prior = {"mean": 0.0, "variance": 10.0}

    # Obs: Control mean=10 (std=2, N=100), Treatment mean=12 (std=2, N=100)
    obs = {
        "control_mean": 10.0, "control_variance": 4.0, "control_n": 100,
        "treatment_mean": 12.0, "treatment_variance": 4.0, "treatment_n": 100
    }

    engine = BayesianInference(model_type="normal_normal")
    results = engine.estimate_posterior(prior, obs)

    # Check posteriors: mean should pull slightly towards prior (0), but remain very close to 10 and 12
    assert 9.5 < results["control_posterior"]["param1"] < 10.1
    assert 11.5 < results["treatment_posterior"]["param1"] < 12.1

    # Probability treatment is superior must be near 1
    assert results["pbb"] > 0.99
    assert results["expected_loss"] < 0.01


def test_streaming_ols():
    """Validates the mathematical correctness and low-latency Woodbury updates of StreamingOLS."""
    import numpy as np
    from xpyrment.analyze.streaming import StreamingOLS

    rng = np.random.default_rng(42)
    n_samples = 150
    n_features = 3

    # Generate synthetic regression dataset: y = X * beta + bias + noise
    X = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
    true_beta = np.array([2.5, -1.0, 4.0])
    true_intercept = 10.0
    y = np.dot(X, true_beta) + true_intercept + rng.normal(loc=0.0, scale=0.5, size=n_samples)

    # 1. Initialize Streaming OLS
    stream_model = StreamingOLS(n_features=n_features, l2_penalty=0.1, fit_intercept=True)

    # 2. Run sequential streaming updates
    for i in range(n_samples):
        stream_model.update(X[i], y[i])

    # 3. Calculate batch Ridge regression offline as target benchmark
    X_bias = np.hstack([np.ones((n_samples, 1)), X])
    XTX = np.dot(X_bias.T, X_bias)
    I = np.eye(n_features + 1)
    I[0, 0] = 0.0  # Do not penalize intercept
    expected_beta_batch = np.linalg.solve(XTX + 0.1 * I, np.dot(X_bias.T, y))

    # Assert that StreamingOLS coefficients closely match offline batch OLS parameters
    assert stream_model.coefficients == pytest.approx(expected_beta_batch, rel=1e-5)

    # 4. Predict on a new sample batch
    X_new = rng.normal(loc=0.0, scale=1.0, size=(10, n_features))
    pred_stream = stream_model.predict(X_new)
    
    # Check predictions match offline batch regression predictions
    pred_expected = np.dot(np.hstack([np.ones((10, 1)), X_new]), expected_beta_batch)
    assert pred_stream == pytest.approx(pred_expected, rel=1e-5)


def test_alias_resolver():
    """Validates fractional factorial ANOVA alias structures and coefficient de-biasing."""
    import pandas as pd
    import numpy as np
    from xpyrment.analyze.confounding import AliasResolver

    # Create a fractional factorial design (e.g., 2^{3-1} with I = A B C)
    # This means C = A * B, so A is completely confounded with the B * C interaction!
    data = {
        "A": [1, 1, -1, -1],
        "B": [1, -1, 1, -1],
        "C": [1, -1, -1, 1],  # C = A * B
    }
    df = pd.DataFrame(data)
    
    # Create the interaction columns
    df["BC"] = df["B"] * df["C"]  # BC = A
    df["AC"] = df["A"] * df["C"]  # AC = B
    df["AB"] = df["A"] * df["B"]  # AB = C

    resolver = AliasResolver(
        primary_cols=["A", "B", "C"],
        potential_confounding_cols=["AB", "AC", "BC"]
    )

    A_matrix = resolver.compute_alias_matrix(df)
    
    # A is confounded with BC (coefficient 1.0)
    # B is confounded with AC (coefficient 1.0)
    # C is confounded with AB (coefficient 1.0)
    assert A_matrix.shape == (4, 3)
    
    report = resolver.get_alias_report(df)
    
    # Check alias reports
    assert "A" in report
    assert any(col == "BC" and abs(val - 1.0) < 1e-3 for col, val in report["A"])
    assert "B" in report
    assert any(col == "AC" and abs(val - 1.0) < 1e-3 for col, val in report["B"])
    assert "C" in report
    assert any(col == "AB" and abs(val - 1.0) < 1e-3 for col, val in report["C"])

    # Test coefficient de-biasing
    beta1_true = np.array([10.0, 2.0, 3.0, 4.0])  # [intercept, A, B, C]
    beta2_true = np.array([1.5, 0.5, 2.5])        # [AB, AC, BC]
    beta1_biased = beta1_true + np.dot(A_matrix, beta2_true)

    resolved = resolver.resolve_coefficients(beta1_biased, beta2_true)
    assert resolved == pytest.approx(beta1_true, rel=1e-5)


def test_copula_multi_metric_inference():
    """Validates the Gaussian Copula multi-metric Wald test on non-Gaussian joint distributions."""
    from xpyrment.analyze.copula import CopulaMultiMetricInference

    rng = np.random.default_rng(42)
    n = 300

    # Simulate: non-Gaussian joint distribution
    # Conversion (binary Bernoulli) and Revenue (skewed log-normal)
    # Treatment group experiences a shift in both conversion rate and revenue scale
    treatment = rng.binomial(1, 0.5, size=n)

    converted = np.zeros(n)
    revenue = np.zeros(n)

    for i in range(n):
        # Base conversion rate: 0.20 control, 0.35 treatment
        prob = 0.35 if treatment[i] == 1 else 0.20
        converted[i] = rng.binomial(1, prob)

        # Revenue is log-normally distributed, higher if converted
        if converted[i] == 1:
            base_log_rev = 3.5 if treatment[i] == 1 else 3.0
            revenue[i] = rng.lognormal(mean=base_log_rev, sigma=0.5)
        else:
            revenue[i] = 0.0

    df = pd.DataFrame({
        "treatment": treatment,
        "converted": converted,
        "revenue": revenue
    })

    copula_engine = CopulaMultiMetricInference(l2_penalty=1e-5)
    results = copula_engine.test_joint_shift(df, "treatment", ["converted", "revenue"])

    assert "wald_statistic" in results
    assert "p_value" in results
    assert "covariance_matrix" in results

    # The covariance matrix should be symmetric and of shape (2, 2)
    cov = results["covariance_matrix"]
    assert cov.shape == (2, 2)
    assert cov[0, 1] == pytest.approx(cov[1, 0])

    # The treatment shift is highly significant, so joint p-value should be low
    assert results["p_value"] < 0.05
    assert results["wald_statistic"] > 5.0




