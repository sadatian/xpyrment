"""Unit tests for Multiple Hypothesis Testing (Block 47)."""

import pytest
from xpyrment.analyze.corrections import apply_multiple_testing_correction


def test_multiple_testing_corrections():
    # 5 raw p-values (some highly significant, some completely uniform)
    p_vals = [0.001, 0.015, 0.045, 0.45, 0.89]

    # Test Benjamini-Hochberg (fdr_bh)
    bh_p = apply_multiple_testing_correction(p_vals, method="fdr_bh")
    assert len(bh_p) == 5
    # BH adjustments should preserve order
    assert bh_p[0] <= bh_p[1] <= bh_p[2] <= bh_p[3] <= bh_p[4]

    # Test Benjamini-Yekutieli (fdr_by)
    by_p = apply_multiple_testing_correction(p_vals, method="fdr_by")
    # BY is more conservative than BH under dependency bounds, so by_p should be >= bh_p
    for i in range(len(p_vals)):
        assert by_p[i] >= bh_p[i]

    # Test Hochberg step-up (hochberg)
    hoch_p = apply_multiple_testing_correction(p_vals, method="hochberg")
    assert len(hoch_p) == 5
    assert hoch_p[0] <= hoch_p[1]
