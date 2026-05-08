"""Unit tests for Sample Ratio Mismatch (SRM) Detection & Sequential Guardrails (Block 41)."""

import numpy as np
import pytest
from xpyrment.analyze.srm import SampleRatioMismatchDetector


def test_srm_retrospective_perfect():
    detector = SampleRatioMismatchDetector(target_treatment_ratio=0.5)
    
    # 500 control and 500 treatment
    res = detector.test_retrospective((500, 500))
    assert res["chi_squared_statistic"] == 0.0
    assert res["p_value"] == pytest.approx(1.0)
    assert not res["srm_detected"]


def test_srm_retrospective_mismatch():
    detector = SampleRatioMismatchDetector(target_treatment_ratio=0.5)

    # Big mismatch: 400 control and 600 treatment (instead of 500/500)
    res = detector.test_retrospective((400, 600))
    # Chi-Sq = (400-500)^2/500 + (600-500)^2/500 = 10000/500 + 10000/500 = 20 + 20 = 40
    assert res["chi_squared_statistic"] == pytest.approx(40.0)
    assert res["p_value"] < 1e-9
    assert res["srm_detected"]


def test_srm_sequential_perfect():
    detector = SampleRatioMismatchDetector(target_treatment_ratio=0.5)
    
    # Perfectly balanced sequence of 1000 assignments
    rng = np.random.default_rng(42)
    assignments = rng.binomial(1, 0.5, size=1000)

    res = detector.test_sequential(assignments, delta=0.03, alpha=0.01)
    
    assert len(res["running_likelihood_ratios"]) == 1000
    # No SRM should be detected
    assert not res["srm_detected"]
    assert res["stopped_index"] == -1


def test_srm_sequential_mismatch():
    detector = SampleRatioMismatchDetector(target_treatment_ratio=0.5)

    # Heavily biased sequence: 60% treatment assignment
    rng = np.random.default_rng(42)
    assignments = rng.binomial(1, 0.60, size=2000)

    res = detector.test_sequential(assignments, delta=0.05, alpha=0.01)

    # Bias is large enough to trigger sequential detection
    assert res["srm_detected"]
    assert res["stopped_index"] != -1
    assert 0 < res["stopped_index"] < 2000
