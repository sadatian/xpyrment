"""Taguchi robust classical Design of Experiments (DoE) matrices.

This module provides the `TaguchiDesign` class, which selects and constructs Taguchi Orthogonal Arrays (OAs).
Taguchi methods are optimized for quality engineering and robust product design, specifically designed
to minimize performance variance under uncontrollable noise conditions using Signal-to-Noise ($S/N$) ratios.
"""

import pandas as pd
from xpyrment.design.doe.base import DesignMatrix


class TaguchiDesign(DesignMatrix):
    r"""Generates Taguchi orthogonal array designs and S/N ratio mapping specifications.

    Taguchi designs leverage specialized, highly balanced fractional factorials (Orthogonal Arrays, denoted
    as $L_4, L_8, L_9, L_{12}, L_{16}, L_{18}, L_{27}$, etc.) to evaluate factor effects on both the mean and
    variability of responses. Rather than optimizing the mean alone, Taguchi methods prioritize "robustness"
    — making the system insensitive to uncontrollable "noise factors".

    ### Mathematical Specifications for Signal-to-Noise ($S/N$, denoted $\eta$) Ratios

    Taguchi methods convert multiple experimental response replicates $y_1, y_2, \dots, y_n$ at each run
    into an $S/N$ ratio ($\eta$ in decibels) depending on the optimization objective:

    1. **Smaller-The-Better** (e.g., latency, defects, material wear):
       $$\eta = -10 \log_{10} \left( \frac{1}{n} \sum_{i=1}^{n} y_i^2 \right)$$
    2. **Larger-The-Better** (e.g., conversion rate, user engagement, revenue):
       $$\eta = -10 \log_{10} \left( \frac{1}{n} \sum_{i=1}^{n} \frac{1}{y_i^2} \right)$$
    3. **Nominal-The-Best** (e.g., precise target dimensions, exact fluid viscosity):
       $$\eta = 10 \log_{10} \left( \frac{\bar{y}^2}{s^2} \right)$$
       where $\bar{y}$ is the sample mean and $s^2$ is the sample variance across the replicates.

    ### Orthogonal Array Database Selection Algorithm

    * The user requests a specific array (e.g., $L_9$, which supports up to 4 factors with 3 levels each).
    * The algorithm maps the requested physical factors to columns in the standard predefined $L_9$ template:

    | Run | Col 1 (Coded) | Col 2 (Coded) | Col 3 (Coded) | Col 4 (Coded) |
    | :--- | :--- | :--- | :--- | :--- |
    | **1** | 1 | 1 | 1 | 1 |
    | **2** | 1 | 2 | 2 | 2 |
    | **3** | 1 | 3 | 3 | 3 |
    | **4** | 2 | 1 | 2 | 3 |
    | **5** | 2 | 2 | 3 | 1 |
    | **6** | 2 | 3 | 1 | 2 |
    | **7** | 3 | 1 | 3 | 2 |
    | **8** | 3 | 2 | 1 | 3 |
    | **9** | 3 | 3 | 2 | 1 |

    These levels are mapped back to the physical factor levels provided in `factors`.

    Attributes:
        array_name (str): The name of the target Taguchi Orthogonal Array (e.g., `"L9"`, `"L18"`, `"L27"`).

    Examples:
        ??? example "Example"

            ```python
            >>> # Selecting an L9 array (supports up to 4 factors at 3 levels)
            >>> factors = {
            ...     "temperature": [100, 150, 200],
            ...     "pressure": [1.0, 1.5, 2.0],
            ...     "catalyst": [0.01, 0.05, 0.10]
            ... }
            >>> design = TaguchiDesign(factors, array_name="L9")
            >>> # Generated matrix will contain exactly 9 balanced runs.
            ```
    """

    def __init__(self, factors: dict, array_name: str):
        """Initializes a TaguchiDesign.

        Args:
            factors (dict): Mapping of factor labels to their lists of levels.
            array_name (str): The name of the target orthogonal array, e.g. `"L9"`.
        """
        super().__init__(factors)
        self.array_name = array_name

    def generate(self) -> pd.DataFrame:
        """Generates the Taguchi Orthogonal Array design matrix.

        Looks up the standard template corresponding to `array_name`, binds the active
        factors to template columns, and maps them to physical levels.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the Taguchi design matrix.
        """
        import numpy as np

        if self.array_name.upper() != "L9":
            raise ValueError(
                f"Taguchi array '{self.array_name}' is not currently implemented. Only 'L9' is supported."
            )

        # Standard L9 template (coded levels 1, 2, 3)
        l9_matrix = np.array([
            [1, 1, 1, 1],
            [1, 2, 2, 2],
            [1, 3, 3, 3],
            [2, 1, 2, 3],
            [2, 2, 3, 1],
            [2, 3, 1, 2],
            [3, 1, 3, 2],
            [3, 2, 1, 3],
            [3, 3, 2, 1]
        ])

        k = len(self.factors)
        if k > 4:
            raise ValueError("L9 Orthogonal Array supports at most 4 factors.")

        keys = list(self.factors.keys())
        physical_df = pd.DataFrame()

        for idx, col in enumerate(keys):
            levels = self.factors[col]
            if len(levels) != 3:
                raise ValueError(
                    f"Each factor in Taguchi L9 design must have exactly 3 levels. Factor '{col}' has {len(levels)} levels."
                )

            coded_col = l9_matrix[:, idx]
            physical_df[col] = [levels[c - 1] for c in coded_col]

        # TODO: Add automatic lookup support for L12, L16, and L18 mixed-level orthogonal arrays.
        # TODO: Integrate signal-to-noise ratio (SNR) loss analysis plots for parameter robust design.
        return physical_df
