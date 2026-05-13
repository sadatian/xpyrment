import pandas as pd
from xpyrment.interactions.detector import InteractionDetector
from xpyrment.interactions.anova import run_factorial_anova
from xpyrment.interactions.regression import check_treatment_covariate_interaction
from xpyrment.interactions.shap import calculate_shap_interactions
from xpyrment.interactions.hstat import compute_friedman_h_statistic
from xpyrment.interactions.plots import plot_interaction_heatmap

def test_interactions_stubs():
    detector = InteractionDetector(None)
    assert isinstance(detector.detect_all(), dict)
    
    df = pd.DataFrame({
        'Y': [1, 2, 3, 4, 1.5, 2.5, 3.5, 4.5], 
        'A': [0, 1, 0, 1, 0, 1, 0, 1], 
        'B': [0, 0, 1, 1, 0, 0, 1, 1]
    })
    res_anova = run_factorial_anova(df, 'Y ~ A * B')
    assert isinstance(res_anova, pd.DataFrame)
    
    res_reg = check_treatment_covariate_interaction(df, 'Y', 'A', 'B')
    assert isinstance(res_reg, float)
    
    res_shap = calculate_shap_interactions(None, pd.DataFrame())
    assert isinstance(res_shap, list)
    
    res_hstat = compute_friedman_h_statistic(None, pd.DataFrame(), 'A', 'B')
    assert isinstance(res_hstat, float)
    
    plot_interaction_heatmap(pd.DataFrame())

from xpyrment.interpret.decision import generate_launch_recommendation
from xpyrment.interpret.effect_size import compute_cohens_d
from xpyrment.interpret.hte import scan_subgroups_for_hte
from xpyrment.interpret.significance import check_practical_significance
import numpy as np

def test_interpret_stubs():
    res_dec = generate_launch_recommendation(0.01, 100, 50)
    assert isinstance(res_dec, str)
    
    res_d = compute_cohens_d(np.array([1,2,3]), np.array([4,5,6]))
    assert isinstance(res_d, float)
    
    res_hte = scan_subgroups_for_hte(pd.DataFrame({'T': [1,0], 'M': [1,2], 'S': ['a','b']}), 'T', 'M', ['S'])
    assert isinstance(res_hte, dict)
    
    res_sig = check_practical_significance(0.04, 0.05)
    assert isinstance(res_sig, bool)
