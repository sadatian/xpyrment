from typing import List
from statsmodels.stats.multitest import multipletests


def apply_multiple_testing_correction(
    p_values: List[float], alpha: float = 0.05, method: str = "fdr_bh"
) -> List[float]:
    """Applies multiple testing corrections on p-values using statsmodels.

    Supported methods include: 'bonferroni', 'holm', 'fdr_bh' (Benjamini-Hochberg).
    """
    if not p_values:
        return []

    # Handle NaNs
    import numpy as np
    p_array = np.array(p_values)
    mask = ~np.isnan(p_array)

    if not np.any(mask):
        return p_values

    adjusted_p = p_array.copy()
    _, adj, _, _ = multipletests(p_array[mask], alpha=alpha, method=method)
    adjusted_p[mask] = adj

    return adjusted_p.tolist()
