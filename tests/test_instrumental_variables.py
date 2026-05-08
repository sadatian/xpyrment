"""Unit tests for Instrumental Variables with 2-Stage Least Squares (2SLS) (Block 37)."""

import numpy as np
import pytest
from xpyrment.quasi.instrumental_variables import InstrumentalVariables2SLS


def test_instrumental_variables_2sls():
    rng = np.random.default_rng(42)
    N = 300

    # Instrument Z (e.g. encouragement randomly assigned)
    Z = rng.binomial(1, 0.5, size=N)

    # Treatment D (compliance: 70% of those encouraged take treatment, 10% of discouraged take it)
    D = np.zeros(N)
    for i in range(N):
        p_comp = 0.7 if Z[i] == 1 else 0.1
        D[i] = rng.binomial(1, p_comp)

    # Unobserved confounder U (affects both compliance and outcome)
    U = rng.normal(size=N)
    D = np.clip(D + (U > 0.5).astype(float), 0.0, 1.0)

    # Outcome Y: causal effect of treatment D is 3.0, affected by confounder U
    Y = 10.0 + 3.0 * D + 2.0 * U + rng.normal(scale=0.1, size=N)

    iv = InstrumentalVariables2SLS(l2_penalty=1e-5)
    iv.fit(Y, D, Z)

    summary = iv.summary
    assert "complier_average_causal_effect" in summary
    assert "stage1_weak_instrument_f_statistic" in summary

    # Direct OLS of Y on D would suffer from endogeneity/omitted variable bias (inflated coefficient > 3.0)
    # 2SLS recovers the true Complier Average Causal Effect (CACE) near 3.0
    assert iv.cace_ == pytest.approx(3.0, abs=0.5)
    assert iv.se_cace_ > 0.0
    # Instrument should be highly relevant (F-statistic > 10)
    assert iv.stage1_f_stat_ > 10.0
