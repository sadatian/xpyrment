"""Full Factorial classical Design of Experiments (DoE) matrices.

This module provides the `FullFactorialDesign` class, which constructs design matrices containing
every possible combination of factor levels. It is the gold standard for exploratory multi-factor
studies where estimation of high-order interaction terms is required.
"""

from typing import Dict, List
import pandas as pd
from xpyrment.design.doe.base import DesignMatrix


class FullFactorialDesign(DesignMatrix):
    r"""Generates a full factorial experimental design matrix for all factor combinations.

    In a full factorial design, the experiment runs encompass all combinations of levels across all
    factors. This design is highly informative because it enables the orthogonal estimation of:
    - **Main Effects**: The individual impact of changing each factor independently.
    - **Interaction Effects**: How the impact of one factor depends on the level of other factors,
      extending to 2-way, 3-way, ..., and $k$-way interactions.

    Mathematical Context:
        Let $k$ be the number of independent factors, and let $l_i$ be the number of levels for factor $i$
        ($i \in \{1, 2, \dots, k\}$).
        The total number of required experimental runs $N$ is:
        $$N = \prod_{i=1}^{k} l_i$$
        For symmetric $2^k$ designs (where every factor has exactly 2 levels: e.g., low and high):
        $$N = 2^k$$
        While extremely thorough, full factorial designs suffer from the "curse of dimensionality",
        where $N$ grows exponentially as more factors are added, making them economically or temporally
        unfeasible for large numbers of factors (where fractional designs are preferred).

    Examples:
        ??? example "Examples"

            ```python
            >>> # Constructing a 2^2 full factorial design
            >>> factors = {"temp": [100, 200], "pressure": [1.5, 3.0]}
            >>> design = FullFactorialDesign(factors)
            >>> df = design.generate()
            >>> len(df)
            4
            >>> print(df)
            temp  pressure
            0   100       1.5
            1   100       3.0
            2   200       1.5
            3   200       3.0
            ```
    """

    def generate(self) -> pd.DataFrame:
        """Generates the full factorial design matrix.

        Utilizes `itertools.product` to compute the Cartesian product of all factor levels,
        creating an $N \times k$ DataFrame.

        Returns:
            pd.DataFrame: A pandas DataFrame representing the full factorial design matrix.
        """
        import itertools

        keys = list(self.factors.keys())
        values = list(self.factors.values())
        combinations = list(itertools.product(*values))

        df = pd.DataFrame(combinations, columns=keys)
        return df
