import pytest
import numpy as np
import pandas as pd
from xpyrment.interpret.decision import generate_launch_recommendation
from xpyrment.interpret.effect_size import compute_cohens_d
from xpyrment.interpret.hte import scan_subgroups_for_hte
from xpyrment.interpret.significance import check_practical_significance

def test_launch_recommendation_logic():
    """Validates the decision engine's ship/no-ship/inconclusive boundaries."""
    # 1. Ship: significant and above cost
    assert "SHIP" in generate_launch_recommendation(p_value=0.01, relative_lift=0.05, cost_threshold=0.02)
    # 2. No-ship: significant but below cost
    assert "NO-SHIP" in generate_launch_recommendation(p_value=0.01, relative_lift=0.01, cost_threshold=0.02)
    # 3. Inconclusive: not significant
    assert "INCONCLUSIVE" in generate_launch_recommendation(p_value=0.10, relative_lift=0.10, cost_threshold=0.02)

def test_cohens_d_calculation():
    """Asserts that standardized effect sizes match analytical expectations."""
    # Groups with shift of 1 standard deviation
    ctrl = np.array([10.0, 11.0, 9.0, 10.0, 10.0]) # Mean 10, low var
    trt = np.array([11.0, 12.0, 10.0, 11.0, 11.0]) # Mean 11, low var
    
    d = compute_cohens_d(ctrl, trt)
    # Mean difference is 1.0. Pooled std is ~0.7. d should be > 1.0
    assert d > 1.0

def test_subgroup_hte_scan():
    """Validates subgroup interaction detection and lift reporting."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        'variant': [0, 1] * 100,
        'platform': ['iOS'] * 100 + ['Android'] * 100,
    })
    
    # iOS has lift of 10.0, Android has lift of 0.0
    y = np.zeros(200)
    # iOS indices: 0 to 99
    y[0:100] = 10.0 + 10.0 * df.loc[0:99, 'variant'] + rng.normal(0, 0.1, 100)
    # Android indices: 100 to 199
    y[100:200] = 10.0 + 0.0 * df.loc[100:199, 'variant'] + rng.normal(0, 0.1, 100)
    df['revenue'] = y
    
    results = scan_subgroups_for_hte(df, 'variant', 'revenue', ['platform'])
    
    assert 'platform' in results
    assert results['platform']['interaction_p_value'] < 0.05
    lifts = results['platform']['subgroup_lifts']
    assert lifts['iOS'] == pytest.approx(1.0, abs=0.1) # (20-10)/10 = 1.0
    assert lifts['Android'] == pytest.approx(0.0, abs=0.1)

def test_practical_significance_check():
    """Simple validation of MVE thresholding."""
    assert check_practical_significance(0.06, 0.05) is True
    assert check_practical_significance(0.04, 0.05) is False
