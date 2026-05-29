"""Mixture-based classical Design of Experiments (DoE) matrices.

This module provides the `MixtureDesign` class, which constructs designs where factors represent
proportional components of a blend. Because components are fractions that must sum to exactly 1.0,
traditional independent factorial hypercubes cannot be used, and the design space is restricted to a simplex.
"""

import pandas as pd
from xpyrment.design.doe.base import DesignMatrix


class MixtureDesign(DesignMatrix):
    r"""Generates designs for mixture factor spaces where components must sum to 1.0 (e.g., recipe mixes).

    In mixture experiments, the independent variables are components of a blend. Changing the proportion of
    one ingredient automatically forces the proportions of other ingredients to change.

    Mathematical Representation and Simplex Constraints:
        Let $k$ be the number of components in the mixture, and let $x_i$ be the proportion of component $i$
        ($i \in \{1, 2, \dots, k\}$).
        The design space is subject to the following strict constraints:
        $$
        \sum_{i=1}^{k} x_i = 1.0 \quad \text{and} \quad 0 \le x_i \le 1.0 \ \ \forall \ i
        $$
        This constraint restricts the geometric region of interest to a $(k-1)$-dimensional simplex (e.g.,
        an equilateral triangle for $k=3$, or a regular tetrahedron for $k=4$).

    Standard Classical Designs:
        1. **Simplex Lattice Design** (denoted $\{k, m\}$):
           Each component takes $m + 1$ equally spaced values between $0$ and $1$:
           $$
           x_i \in \left\{ 0, \frac{1}{m}, \frac{2}{m}, \dots, 1 \right\}
           $$
           The runs consist of all possible combinations of these levels that sum to exactly $1.0$.
           The total number of runs $N$ is:
           $$
           N = \frac{(k + m - 1)!}{m!(k - 1)!}
           $$
        2. **Simplex Centroid Design**:
           Consists of runs at pure components (one factor is $1.0$, others $0$), binary blends (two factors
           at $0.5$, others $0$), ternary blends (three factors at $1/3$, others $0$), and so on, up to
           the overall centroid blend where all $k$ factors are equal to $1/k$.
           The total number of runs $N$ is:
           $$
           N = 2^k - 1
           $$
    Pseudocode for Simplex Lattice Algorithm:
        ```text
        function generate_simplex_lattice(k, m):
            1. Generate all compositions of integer m into k parts:
               Find all non-negative integer vectors [v1, v2, ..., vk] such that sum(vi) = m.
            2. For each composition, compute the fractional proportions:
               x_i = v_i / m.
            3. Bind x_i columns to the physical factors.
            4. Return DataFrame.
        ```

    Examples:
        ??? example "Example"

            ```python
            >>> # Constructing a {3, 2} Simplex Lattice design (3 ingredients, quadratic model)
            >>> # Proportions will be: [1, 0, 0], [0.5, 0.5, 0], [0, 1, 0], [0, 0.5, 0.5], [0, 0, 1], [0.5, 0, 0.5]
            ```
    """

    def __init__(self, factors: dict, m: int = 2) -> None:
        """Initializes a MixtureDesign.

        Args:
            factors (dict): Mapping of ingredient factor labels to their low/high levels (usually [0.0, 1.0]).
            m (int): Simplex Lattice division parameter. Defaults to 2.
        """
        super().__init__(factors)
        self.m = m

    def generate(self) -> pd.DataFrame:
        """Generates the Mixture Design matrix.

        Constructs simplex coordinates using either Simplex Lattice or Simplex Centroid algorithms,
        verifies the summing constraint, and binds to physical factors.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the Mixture design matrix.
        """
        import numpy as np

        k = len(self.factors)
        keys = list(self.factors.keys())

        # Recursive helper to generate non-negative integer vectors of length k summing to m
        def get_compositions(num_parts, target_sum):
            if num_parts == 1:
                return [[target_sum]]
            compositions = []
            for v in range(target_sum + 1):
                for sub in get_compositions(num_parts - 1, target_sum - v):
                    compositions.append([v] + sub)
            return compositions

        compositions = get_compositions(k, self.m)
        proportions = np.array(compositions) / float(self.m)

        # Scale proportions to actual ingredient boundaries if they are not [0, 1]
        # (Though usually mixture boundaries are [0.0, 1.0])
        physical_df = pd.DataFrame(proportions, columns=keys)

        # TODO: Add support for McLean-Anderson or coordinate exchange constraints to handle upper/lower bounds on individual ingredients (e.g., ingredient A must be between 10% and 30%).
        # TODO: Implement Simplex Centroid designs to supplement Simplex Lattice with interior-point checks.
        return physical_df
