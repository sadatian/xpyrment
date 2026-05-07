"""Definitive Screening Design (DSD) classical Design of Experiments (DoE) matrices.

This module provides the `DefinitiveScreeningDesign` class, which constructs Definitive Screening Designs (DSDs).
DSDs represent a breakthrough in classical DoE (Jones & Nachtsheim, 2011), enabling the estimation of
main effects, quadratic effects, and two-way interactions simultaneously in a single highly compact 3-level run plan.
"""

import pandas as pd
from xpyrment.design.doe.base import DesignMatrix


class DefinitiveScreeningDesign(DesignMatrix):
    r"""Generates a Definitive Screening Design (DSD) to estimate linear, quadratic, and 2-way terms.

    DSDs are specialized 3-level designs (coded as $-1, 0, +1$). Unlike traditional screening designs
    (which can only estimate linear main effects), a DSD is designed to screen active factors while
    simultaneously modeling curvilinear (quadratic) responses and 2-way interactions without requiring a
    subsequent follow-up Response Surface Method (RSM) design.

    Mathematical Properties and Sizing:
        Let $k$ be the number of factors. The minimum required number of experimental runs $N$ is:
        - For even $k$: $N = 2k + 1$ runs.
        - For odd $k$: $N = 2k + 3$ runs (which represents the next even number of factors plus 3).
        
        The structural properties of the resulting design matrix $X$ include:
        1. **Orthogonality of Main Effects**: All main effects are mutually orthogonal and completely unconfounded
           with other main effects, two-way interactions, and quadratic effects.
        2. **Weak Confounding of Quadratic Terms**: Quadratic terms are not confounded with main effects, and are
           only partially/weakly confounded with two-way interactions.
        3. **No Co-aliased 2-Way Interactions**: No two-way interaction is completely aliased (confounded) with
           any other two-way interaction.

    Conference Matrix Generative Algorithm:
        The core of a DSD is constructed using mathematical **Conference Matrices** ($C$).
        A conference matrix $C_m$ of order $m$ is an $m \times m$ matrix with diagonal entries equal to $0$
        and off-diagonal entries equal to $\pm 1$, satisfying:
        $$C_m^T C_m = (m - 1) I_m$$
        To generate a DSD for $k$ factors:
        1. Generate a conference matrix of appropriate size.
        2. For each row $r_i$ in the conference matrix, create a "folded-over" pair of rows:
           $$r_i \quad \text{and} \quad -r_i$$
           This guarantees that the columns have an average value of exactly zero, centering the design.
        3. Append a final "center point" run consisting entirely of zeros:
           $$[0, 0, \dots, 0]$$
        4. This results in $2k + 1$ (or $2k + 3$) runs in coded $[-1, 0, +1]$ space.
        5. Map these levels back to physical levels (low, medium, high) in `factors`.

    Pseudocode for the Algorithm:
        ```text
        function generate_dsd(factors):
            1. Determine k = number of factors.
            2. Select order m of conference matrix (m = k if k even, m = k + 1 if k odd).
            3. Load/generate conference matrix C of order m.
            4. Create matrix D by stacking [C] and [-C] (size 2m x m).
            5. Append row of all zeros (size (2m + 1) x m).
            6. If k was odd, truncate the last column of the matrix to yield a (2k + 3) x k matrix.
            7. Scale coded values [-1, 0, +1] to the physical factor levels.
            8. Return DataFrame.
        ```

    Examples:
        ??? example "Examples"

            ```python
            >>> # If we have k=4 factors, a full factorial 3-level design requires 3^4 = 81 runs.
            >>> # A DSD requires only 2(4) + 1 = 9 runs! It can still identify active quadratic curvature terms.
            ```
    """

    def generate(self) -> pd.DataFrame:
        """Generates the Definitive Screening Design matrix.

        Looks up or constructs the base conference matrix, performs folding operations,
        appends the center point, and scales the coded levels.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the DSD matrix.
        """
        # TODO: Implement DSD algorithm (using conference matrices)
        return pd.DataFrame()
