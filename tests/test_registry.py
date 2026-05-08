"""Unit tests for Metric Registry & Directed Acyclic Graph (DAG) Evaluator (Block 49)."""

import numpy as np
import pytest
from xpyrment.analyze.registry import MetricRegistry


def test_metric_registry_evaluation():
    reg = MetricRegistry()
    reg.add_raw("clicks")
    reg.add_raw("views")
    reg.add_raw("purchases")

    # Derived CTR: clicks / views
    reg.add_derived("ctr", ["clicks", "views"], lambda c, v: c / v)
    # Derived conversion rate: purchases / clicks
    reg.add_derived("cr", ["purchases", "clicks"], lambda p, c: p / c)
    # Compound metric: sum of CTR and CR
    reg.add_derived("compound", ["ctr", "cr"], lambda ctr, cr: ctr + cr)

    raw_data = {
        "clicks": np.array([10.0, 20.0, 30.0]),
        "views": np.array([100.0, 100.0, 100.0]),
        "purchases": np.array([2.0, 4.0, 6.0]),
    }

    results = reg.evaluate(raw_data)

    # Assert CTR calculation
    np.testing.assert_allclose(results["ctr"], np.array([0.1, 0.2, 0.3]))
    # Assert CR calculation
    np.testing.assert_allclose(results["cr"], np.array([0.2, 0.2, 0.2]))
    # Assert Compound calculation
    np.testing.assert_allclose(results["compound"], np.array([0.3, 0.4, 0.5]))


def test_metric_registry_cycles():
    reg = MetricRegistry()
    # Cyclic loop: A depends on B, B depends on A
    reg.add_derived("A", ["B"], lambda b: b)
    reg.add_derived("B", ["A"], lambda a: a)

    raw_data = {"A": np.array([1.0])}
    with pytest.raises(ValueError, match="Cyclic dependency detected"):
        reg.evaluate(raw_data)
