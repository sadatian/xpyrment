"""Unit tests for Infinite Dirichlet Process Mixture Clustering (Block 39)."""

import numpy as np
import pytest
from xpyrment.personalize.infinite_mixture import InfiniteDirichletClusterer


def test_infinite_dirichlet_clustering():
    rng = np.random.default_rng(42)

    # Generate a clear mixture of two Gaussians: N(-3.0, 0.5) and N(3.0, 0.5)
    size_1 = 30
    size_2 = 30
    g1 = rng.normal(loc=-3.0, scale=0.5, size=size_1)
    g2 = rng.normal(loc=3.0, scale=0.5, size=size_2)
    data = np.concatenate([g1, g2])

    clusterer = InfiniteDirichletClusterer(alpha=0.5, prior_mean=0.0, prior_var=10.0, noise_var=0.25)
    assignments = clusterer.fit_predict(data, n_iters=40, rng=rng)

    assert len(assignments) == 60
    
    # Points in g1 should have similar labels, and points in g2 should have similar labels
    # but different from g1
    labels_g1 = set(assignments[:size_1])
    labels_g2 = set(assignments[size_1:])

    # They should be mostly separated clusters
    assert len(labels_g1.intersection(labels_g2)) == 0
