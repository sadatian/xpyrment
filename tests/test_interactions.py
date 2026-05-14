import pytest
import numpy as np
import pandas as pd
from xpyrment.interactions.anova import run_factorial_anova
from xpyrment.interactions.regression import check_treatment_covariate_interaction
from xpyrment.interactions.hstat import compute_friedman_h_statistic
from xpyrment.interactions.detector import InteractionDetector
from xpyrment.analyze.orchestrator import setup

def test_factorial_anova_interaction():
    """Asserts that factorial ANOVA correctly identifies interaction effects in a 2x2 design."""
    # Factor A: [0, 1], Factor B: [0, 1]
    # Y = 10 + 2*A + 2*B + 5*A*B + noise
    df = pd.DataFrame({
        'A': [0, 0, 1, 1] * 25,
        'B': [0, 1, 0, 1] * 25,
    })
    noise = np.random.default_rng(42).normal(0, 0.1, 100)
    df['Y'] = 10.0 + 2.0 * df['A'] + 2.0 * df['B'] + 5.0 * df['A'] * df['B'] + noise
    
    anova_table = run_factorial_anova(df, 'Y ~ A * B')
    
    assert 'A:B' in anova_table.index
    # Interaction p-value should be extremely small
    assert anova_table.loc['A:B', 'PR(>F)'] < 0.001

def test_lrt_interaction_check():
    """Validates Likelihood Ratio Test for treatment-covariate interaction."""
    rng = np.random.default_rng(42)
    n = 200
    treatment = rng.binomial(1, 0.5, size=n)
    covariate = rng.normal(size=n)
    
    # Case 1: Significant interaction
    # Y = 10 + 2*T + 5*T*C + noise
    y_sig = 10.0 + 2.0 * treatment + 5.0 * treatment * covariate + rng.normal(0, 0.1, n)
    df_sig = pd.DataFrame({'T': treatment, 'C': covariate, 'Y': y_sig})
    p_val_sig = check_treatment_covariate_interaction(df_sig, 'T', 'C', 'Y')
    assert p_val_sig < 0.01

    # Case 2: No interaction
    # Y = 10 + 2*T + 2*C + noise
    y_null = 10.0 + 2.0 * treatment + 2.0 * covariate + rng.normal(0, 0.1, n)
    df_null = pd.DataFrame({'T': treatment, 'C': covariate, 'Y': y_null})
    p_val_null = check_treatment_covariate_interaction(df_null, 'T', 'C', 'Y')
    assert p_val_null > 0.05

def test_friedman_h_statistic():
    """Validates H-statistic for linear and non-linear interactions."""
    class DummyModel:
        def predict(self, X):
            # Y = X1 + X2 + 10 * X1 * X2
            return X['X1'] + X['X2'] + 10.0 * X['X1'] * X['X2']
    
    X = pd.DataFrame({
        'X1': np.random.default_rng(42).uniform(0, 1, 50),
        'X2': np.random.default_rng(42).uniform(0, 1, 50)
    })
    
    h_stat = compute_friedman_h_statistic(DummyModel(), X, 'X1', 'X2')
    # Since there is a strong interaction (10 * X1 * X2), h_stat should be non-zero
    assert h_stat > 0.1

def test_interaction_detector_integration():
    """Verifies InteractionDetector identifies HTEs in an Experiment setup."""
    df = pd.DataFrame({
        'variant': [0, 0, 1, 1] * 50,
        'age': np.random.default_rng(42).normal(30, 5, 200),
        'user_id': range(200)
    })
    # Strong interaction with age
    df['revenue'] = 10.0 + 5.0 * df['variant'] * df['age'] + np.random.default_rng(42).normal(0, 1, 200)
    
    exp = setup(df, treatment_col='variant', id_col='user_id', covariates=['age'])
    exp.register_metric('revenue')
    
    detector = InteractionDetector(exp)
    results = detector.detect_all()
    
    assert len(results['heterogeneous_treatment_effects']) > 0
    match = results['heterogeneous_treatment_effects'][0]
    assert match['metric'] == 'revenue'
    assert match['covariate'] == 'age'
    assert match['p_value'] < 0.05
