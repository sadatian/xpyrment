"""Multiple testing correction (MTC) statistical engines.

This module provides correction algorithms for family-wise error rate (FWER) and false discovery rate (FDR).
MTC is critical when testing multiple metrics or variants simultaneously to prevent the dramatic inflation
of false positives (Type I errors).
"""

from typing import List
from statsmodels.stats.multitest import multipletests


def apply_multiple_testing_correction(
    p_values: List[float], alpha: float = 0.05, method: str = "fdr_bh"
) -> List[float]:
    """Applies multiple testing corrections on p-values using statsmodels.

    When performing multiple statistical tests simultaneously, the probability of obtaining at least one
    false positive (rejecting $H_0$ when it is actually true) increases with the number of tests.
    This inflation of Type I error is known as the **Multiple Testing Problem**.

    Mathematical Background of FWER Inflation:
        For $m$ independent tests, each run at nominal significance level $\\alpha$:
        $$\\text{FWER} = P(\\text{at least one false positive}) = 1 - (1 - \\alpha)^m$$
        - If $m = 1$ and $\\alpha = 0.05$, $\\text{FWER} = 0.05$.
        - If $m = 10$ and $\\alpha = 0.05$, $\\text{FWER} = 1 - (0.95)^{10} \\approx 0.40$ ($40\\%$ false positive probability).
        - If $m = 50$ and $\\alpha = 0.05$, $\\text{FWER} \\approx 0.92$ (near-certainty of committing a false positive).

    Supported Correction Methodologies:
        1. **Bonferroni Correction** (`"bonferroni"`):
           Controls the Family-Wise Error Rate (FWER) in the strong sense. It adjusts each p-value by multiplying
           it by the total number of tests $m$:
           $$p^{\\text{adj}}_i = \\min(p_i \\times m, \\ 1.0)$$
           Highly conservative; has low statistical power when $m$ is large or when tests are highly correlated.
        2. **Holm-Bonferroni Procedure** (`"holm"`):
           A step-down FWER control method that is uniformly more powerful than the standard Bonferroni correction.
           It orders the raw p-values: $p_{(1)} \\le p_{(2)} \\le \\dots \\le p_{(m)}$.
           The adjusted p-values are computed sequentially as:
           $$p^{\\text{adj}}_{(i)} = \\max \\left( (m - i + 1) \\times p_{(i)}, \\ p^{\\text{adj}}_{(i-1)} \\right) \\quad \\text{for } i \\ge 1$$
           (with $p^{\\text{adj}}_{(0)} = 0$, bounded above by $1.0$).
        3. **Benjamini-Hochberg (BH) Procedure** (`"fdr_bh"`):
           Controls the **False Discovery Rate (FDR)**, which is the expected proportion of false positives among all
           rejections. This is the preferred method for digital product experimentation (A/B testing with multiple secondary metrics),
           as it provides vastly superior statistical power compared to FWER controllers.
           It orders raw p-values: $p_{(1)} \\le p_{(2)} \\le \\dots \\le p_{(m)}$.
           The adjusted p-values are calculated as:
           $$p^{\\text{adj}}_{(i)} = \\min \\left( \\frac{m}{i} \\times p_{(i)}, \\ p^{\\text{adj}}_{(i+1)} \\right) \\quad \\text{for } i \\le m - 1$$
           (with $p^{\\text{adj}}_{(m)} = p_{(m)}$, bounded above by $1.0$).

    Args:
        p_values (List[float]): List of raw, unadjusted p-values calculated from various metric tests.
        alpha (float): Nominal significance level (e.g., 0.05). Defaults to 0.05.
        method (str): Correction algorithm. Options include `"bonferroni"`, `"holm"`, `"fdr_bh"`.
            Defaults to `"fdr_bh"`.

    Returns:
        List[float]: A list of adjusted p-values, in the same index order as the input.
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
