"""Box-Behnken Design (BBD) classical Response Surface Method (RSM) matrices.

This module provides the `BoxBehnkenDesign` class, which constructs Box-Behnken Designs (BBDs).
BBDs are 3-level response surface designs that offer a highly efficient, safe, and cost-effective alternative
to Central Composite Designs (CCDs) for optimizing industrial or digital multi-factor processes.
"""

import pandas as pd
from xpyrment.design.doe.base import DesignMatrix


class BoxBehnkenDesign(DesignMatrix):
    r"""Generates Box-Behnken Designs to estimate non-linear response surfaces in fewer runs.

    A Box-Behnken design is a 3-level response surface design (coded as $-1, 0, +1$) built by combining
    fractional factorials with balanced incomplete block designs (BIBDs). Geometrically, instead of placing
    points at the corners of a multidimensional cube (like CCDs or full factorials), a BBD places points
    specifically at the midpoints of the edges of the process space, combined with replicated center points.

    Safety and Operational Advantages:
        Because BBD does not contain any "corner points" (combinations where all factors are simultaneously
        at their extreme high/low levels, such as $(+1, +1, +1)$ or $(-1, -1, -1)$), it is exceptionally well-suited
        for physical, chemical, or web experiments where extreme combinations are physically dangerous,
        environmentally harmful, or likely to trigger catastrophic system errors or service downtime.

    Mathematical Sizing and Properties:
        The total number of required experimental runs $N$ is:
        $$
        N = 2k(k - 1) + n_c
        $$
        where $k$ is the number of factors and $n_c$ is the number of replicated center points (typically $3$ to $5$).
        For example:
        - $k = 3$: $N = 12 + n_c$ runs (usually $15$ runs with $3$ center points, compared to $15$ for face-centered CCD).
        - $k = 4$: $N = 24 + n_c$ runs (usually $27$ runs, compared to $31$ for CCD).

    Algorithmic Construction:
        The design matrix is built by running standard $2^2$ factorials for pairs of factors while keeping all other
        factors fixed at their center levels ($0$).
        For $k=3$ factors:
        - Block 1: Factors 1 & 2 vary as $2^2$ factorial ($\pm 1$), Factor 3 is fixed at $0$.
        - Block 2: Factors 1 & 3 vary as $2^2$ factorial ($\pm 1$), Factor 2 is fixed at $0$.
        - Block 3: Factors 2 & 3 vary as $2^2$ factorial ($\pm 1$), Factor 1 is fixed at $0$.
        - Center points: All factors are at $0$.

    Pseudocode for the Algorithm:
        ```text
        function generate_bbd(factors, num_center_points):
            1. Determine k = number of factors.
            2. Find all unique pairs (or appropriate balanced blocks) of factors.
            3. Initialize design matrix of size (2k(k-1) + num_center_points) x k with zeros.
            4. For each block/pair of active factors (i, j):
                 Create 4 rows representing the standard 2^2 factorial combinations of (+1, -1) for columns i and j.
                 All other columns in these 4 rows remain 0.
            5. Append num_center_points rows consisting entirely of zeros.
        ```
    """

    def __init__(self, factors: dict, num_center_points: int = 3):
        """Initializes a BoxBehnkenDesign.

        Args:
            factors (dict): Mapping of factor labels to their low/high levels.
            num_center_points (int): Number of center point replicates to append. Defaults to 3.
        """
        super().__init__(factors)
        self.num_center_points = num_center_points

    def generate(self) -> pd.DataFrame:
        """Generates the Box-Behnken Design matrix.

        Constructs the incomplete block structure in coded space, shuffles columns,
        appends center trials, and maps back to physical coordinates.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the BBD matrix.
        """
        import itertools
        import numpy as np

        k = len(self.factors)
        keys = list(self.factors.keys())

        if k < 3:
            raise ValueError("Box-Behnken Design requires at least 3 factors.")

        # 1. Generate incomplete block factorial pairs in coded space [-1.0, 0.0, 1.0]
        coded_rows = []
        pairs = list(itertools.combinations(range(k), 2))
        for i, j in pairs:
            # 2^2 combinations for active factors, keeping other factors at 0.0
            for val_i, val_j in itertools.product([-1.0, 1.0], repeat=2):
                row = [0.0] * k
                row[i] = val_i
                row[j] = val_j
                coded_rows.append(row)

        factorial_df = pd.DataFrame(coded_rows, columns=keys)

        # 2. Append center points (all coded 0.0s)
        center_rows = [[0.0] * k for _ in range(self.num_center_points)]
        center_df = pd.DataFrame(center_rows, columns=keys)

        # 3. Stack blocks
        coded_df = pd.concat([factorial_df, center_df], ignore_index=True)

        # 4. Map coded space to actual physical factor levels
        physical_df = pd.DataFrame()
        for col in keys:
            low, high = self.factors[col]
            mid = (low + high) / 2.0
            half_range = (high - low) / 2.0
            physical_df[col] = mid + coded_df[col] * half_range

        return physical_df
