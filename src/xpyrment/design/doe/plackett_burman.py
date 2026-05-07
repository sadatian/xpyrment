"""Plackett-Burman classical screening Design of Experiments (DoE) matrices.

This module provides the `PlackettBurmanDesign` class, which constructs highly efficient fractional
screening designs. Plackett-Burman designs are optimized to identify main effects using the minimum
number of runs when interaction effects can be assumed to be negligible.
"""

import pandas as pd
from xpyrment.design.doe.base import DesignMatrix


class PlackettBurmanDesign(DesignMatrix):
    r"""Generates Plackett-Burman screening design matrices to identify main effects in few runs.

    Plackett-Burman designs are classical Resolution III fractionals where the number of runs $N$ is a multiple of 4
    (typically $N = 12, 16, 20, 24, 28, 36, 40, \dots$). They allow screening of up to $k = N - 1$ factors
    in $N$ trials.

    Mathematical Context and Confounding:
        Unlike standard $2^{k-p}$ fractional factorials where main effects are completely aliased with specific
        two-way interactions, Plackett-Burman designs exhibit a complex confounding structure where each main effect
        is partially confounded (aliased) with multiple two-way interactions. For example, in an $N=12$ design,
        every main effect is aliased with 45 distinct two-way interactions with fractional coefficient weights of
        $\pm 1/3$.
        This properties makes Plackett-Burman designs highly robust screening tools if the Sparsity of Effects principle
        holds, as interaction effects are diluted and do not fully bias any single main effect estimate.

    Coded Generative Algorithm:
        Plackett-Burman designs are built by taking a specialized "first generator row" of signs ($+$ and $-$),
        cyclically shifting it to form the first $N - 1$ columns, and then appending a final row of all minuses ($-1$).
        Standard generator sequences for different run sizes ($N$) include:
        - $N = 8$: `+ + + - + - -`
        - $N = 12$: `+ + - + + + - - - + -`
        - $N = 16$: `+ + + + - + - + + - - + - - -`
        - $N = 20$: `+ + - - + + - - + + - - + + - - + + -`

    Pseudocode for the Algorithm:
        ```text
        function generate_plackett_burman(factors, N):
            1. Select first generator row of length N - 1 based on run size N.
               (e.g., for N=12: [1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1]).
            2. Construct (N - 1) x (N - 1) matrix by cyclically shifting the generator row.
            3. Append a final row of N - 1 elements, all equal to -1.
               Now we have an N x (N - 1) design matrix in coded [-1, +1] format.
            4. Truncate the columns to match the actual number of factors requested.
            5. Map the coded values back to the actual factor levels.
            6. Return DataFrame.
        ```

    Examples:
        ??? example "Example"

            ```python
            >>> # If we want to screen 10 factors, we can use an N=12 Plackett-Burman design, requiring only 12 runs!
            >>> # Compare this with a full factorial which would require 2^10 = 1024 runs.
            ```
    """

    def generate(self) -> pd.DataFrame:
        """Generates the Plackett-Burman design matrix.

        Builds the cyclic shifts based on standard generator sequences, truncates to match
        the factors dictionary, and scales back to the actual levels.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the design matrix.
        """
        import numpy as np

        k = len(self.factors)
        keys = list(self.factors.keys())

        # Standard Plackett-Burman cyclic generators for N-1 columns
        # N must be a multiple of 4. We find the smallest N >= k + 1.
        target_N = ((k + 1 + 3) // 4) * 4
        if target_N < 8:
            target_N = 8

        generators = {
            8: [1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0],
            12: [1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0],
            16: [1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0],
            20: [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0],
            24: [1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0]
        }

        if target_N not in generators:
            raise ValueError(
                f"No standard Plackett-Burman generator sequence available for run size N={target_N}. "
                "Ensure number of factors k satisfies k <= 23."
            )

        S = generators[target_N]
        n_cols = target_N - 1

        # Build (N-1) x (N-1) matrix by cyclic shifting S
        matrix = np.zeros((n_cols, n_cols))
        for col_idx in range(n_cols):
            # Cyclic shift by col_idx positions
            shifted = S[col_idx:] + S[:col_idx]
            matrix[:, col_idx] = shifted

        # Append final row of all -1.0s to construct complete N x (N-1) design matrix
        final_row = np.array([-1.0] * n_cols).reshape(1, -1)
        coded_matrix = np.vstack([matrix, final_row])

        # Truncate matrix columns to match the actual number of factors requested (k)
        coded_matrix = coded_matrix[:, :k]

        # Convert coded coordinates to physical coordinates and return as DataFrame
        physical_df = pd.DataFrame()
        for idx, col in enumerate(keys):
            low, high = self.factors[col]
            mid = (low + high) / 2.0
            half_range = (high - low) / 2.0
            physical_df[col] = mid + coded_matrix[:, idx] * half_range

        return physical_df
