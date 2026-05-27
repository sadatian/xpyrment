"""Unit tests for Subgroup Heterogeneity Segment Discovery (Block 48)."""

import numpy as np
from xpyrment.personalize.subgroup import SubgroupHeterogeneityDiscoverer


def test_subgroup_heterogeneity_discoverer():
    rng = np.random.default_rng(42)
    N = 200
    P = 2

    X = rng.normal(size=(N, P))
    # Feature 0 determines treatment effect heterogeneity:
    # If feature_0 <= 0, treatment effect is 0
    # If feature_0 > 0, treatment effect is +5.0
    w = rng.binomial(1, 0.5, size=N)
    y = rng.normal(size=N)

    treatment_mask = (w == 1)
    high_effect_mask = (X[:, 0] > 0.0)

    y[treatment_mask & high_effect_mask] += 5.0

    discoverer = SubgroupHeterogeneityDiscoverer(min_sample_size=15, feature_names=["age", "income"])
    discoverer.fit(X, w, y)

    res = discoverer.results
    assert res["heterogeneity_found"] is True
    assert res["best_feature_name"] == "age"
    # The split threshold should be close to 0.0
    assert abs(res["best_threshold"]) < 0.5
    assert res["right_subgroup_cate"] > res["left_subgroup_cate"]
    assert res["cate_gap"] > 3.0
