"""Unit tests for Dynamic Treatment Regimes & Q-Learning (Block 31)."""

import numpy as np
import pytest
from xpyrment.personalize.dtr import DynamicTreatmentRegime, QFactorModel


def test_q_factor_model():
    rng = np.random.default_rng(42)
    N = 100
    H = rng.normal(size=(N, 2))
    A = rng.binomial(1, 0.5, size=N)
    
    # Let Y = 2 + 1.5 * H[:, 0] + 3.0 * A + 2.0 * H[:, 0] * A
    Y = 2.0 + 1.5 * H[:, 0] + 3.0 * A + 2.0 * H[:, 0] * A + rng.normal(scale=0.01, size=N)

    model = QFactorModel(l2_penalty=1e-5)
    model.fit(H, A, Y)

    assert len(model.beta_) == 6  # Intercept, 2 history vars, 1 action, 2 interaction columns
    assert model.beta_[0] == pytest.approx(2.0, abs=0.1)
    assert model.beta_[3] == pytest.approx(3.0, abs=0.1)


def test_dtr_backward_induction():
    rng = np.random.default_rng(42)
    N = 150

    H1 = rng.normal(size=(N, 2))
    A1 = rng.binomial(1, 0.5, size=N)
    
    # Stage 2 history depends on baseline history and Stage 1 action
    H2 = H1 + A1.reshape(-1, 1) + rng.normal(scale=0.1, size=(N, 2))
    A2 = rng.binomial(1, 0.5, size=N)

    # Outcome Y: high reward if treatment complies with high H2[:, 0]
    Y = 5.0 + 2.0 * H2[:, 0] + 4.0 * A2 + 3.0 * H2[:, 0] * A2 + rng.normal(scale=0.05, size=N)

    dtr = DynamicTreatmentRegime(l2_penalty=1e-5)
    dtr.fit(H1, A1, H2, A2, Y)

    rec_1, rec_2 = dtr.recommend(H1, H2)
    assert rec_1.shape == (N,)
    assert rec_2.shape == (N,)
    
    # For positive H2, recommendation should be treated (1.0)
    # Since beta_A2 = 4 and beta_interaction = 3, if H2[i, 0] > -1.33: treatment is superior
    for i in range(N):
        if H2[i, 0] > 0.5:
            assert rec_2[i] == 1.0
        elif H2[i, 0] < -2.0:
            assert rec_2[i] == 0.0
