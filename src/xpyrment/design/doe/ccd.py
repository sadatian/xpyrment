"""Central Composite Design (CCD) classical Response Surface Method (RSM) matrices.

This module provides the `CentralCompositeDesign` class, which constructs Central Composite Designs (CCDs).
CCDs are the most popular designs for Response Surface Methodology (RSM), designed to fit second-order (quadratic)
response surfaces to optimize multi-factor processes.
"""

import pandas as pd
from xpyrment.design.doe.base import DesignMatrix


class CentralCompositeDesign(DesignMatrix):
    r"""Generates Central Composite Designs (CCD) for response surface modeling.

    A CCD is a 5-level design that extends a standard 2-level factorial design by adding star (axial) points
    and center points. This allows the estimation of quadratic curvature terms, enabling process optimization.

    Mathematical Structure:
        A CCD with $k$ factors consists of three distinct parts:
        1. **Factorial (or Cube) Portion**: A standard $2^k$ (or $2^{k-p}$ fractional) design with coded levels $[-1, +1]$.
           This estimates main effects and two-way interactions. (Size: $N_f = 2^{k-p}$ runs).
        2. **Axial (or Star) Portion**: Runs where all factors except one are set to their center level ($0$), and the
           remaining factor is set to a distance of $\pm \alpha$. (Size: $N_a = 2k$ runs).
        3. **Center Points**: Multiple runs where all factors are set to $0$. This allows estimation of pure experimental
           error and stabilizes prediction variance at the center of the design space. (Size: $n_c$ runs).
        
        The total number of runs $N$ is:
        $$
        N = 2^{k-p} + 2k + n_c
        $$
    Alpha ($\alpha$) Parameter Configuration:
        The distance $\alpha$ of the axial points determines the geometric shape and properties of the design:
        - **Rotatable**: $\alpha = (N_f)^{1/4}$ (e.g., $\alpha = 1.414$ for $k=2$, $\alpha = 1.682$ for $k=3$). Rotatability
          guarantees that the prediction variance is constant at all points equidistant from the design center.
        - **Face-Centered (CCF)**: $\alpha = 1.0$. Star points are placed on the faces of the cube, meaning factors
          only require 3 levels instead of 5, which is highly practical for many experimental setups.
        - **Orthogonal**: $\alpha$ is mathematically selected so that the quadratic effect estimates are completely
          uncorrelated with the main effects and other quadratic terms.

    Pseudocode for the Algorithm:
        ```text
        function generate_ccd(factors, alpha_type, num_center_points):
            1. Generate factorial matrix (size N_f x k) using FullFactorial or FractionalFactorial.
               Values are in coded space [-1, +1].
            2. Compute alpha value based on alpha_type (rotatable, face-centered, orthogonal) and k.
            3. Generate axial matrix of size (2k x k):
                 For each factor i from 1 to k:
                   Row 1: all zeros except column i which is +alpha.
                   Row 2: all zeros except column i which is -alpha.
            4. Generate center points matrix of size (num_center_points x k), containing all zeros.
            5. Stack cube, star, and center matrices (size N x k).
            6. Map coded values [-alpha, -1, 0, +1, +alpha] to physical levels in factors.
            7. Return DataFrame.
        ```

    Attributes:
        alpha_type (str): Geometric configuration for axial star point distance.
            Options are `"rotatable"`, `"face-centered"`, `"orthogonal"`. Defaults to `"orthogonal"`.

    Examples:
        ??? example "Example"

            ```python
            >>> # Constructing a face-centered CCD for 2 factors
            >>> factors = {"temp": [100, 200], "pressure": [1.5, 3.0]}
            >>> design = CentralCompositeDesign(factors, alpha="face-centered")
            >>> # The coded levels generated will be -1 (low), 0 (midpoint), +1 (high).
            ```
    """

    def __init__(self, factors: dict, alpha: str = "orthogonal", num_center_points: int = 4):
        """Initializes a CentralCompositeDesign.

        Args:
            factors (dict): Mapping of factor labels to their low/high levels.
            alpha (str): Axial point distance strategy (`"rotatable"`, `"face-centered"`, `"orthogonal"`).
                Defaults to `"orthogonal"`.
            num_center_points (int): Number of center point replicates to append. Defaults to 4.
        """
        super().__init__(factors)
        self.alpha_type = alpha
        self.num_center_points = num_center_points

    def generate(self) -> pd.DataFrame:
        """Generates the Central Composite Design matrix.

        Constructs the cube, star, and center blocks in coded space, stacks them,
        and scales the coordinates back to physical units.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the complete CCD matrix.
        """
        import itertools

        k = len(self.factors)
        keys = list(self.factors.keys())

        # 1. Cube Portion (Standard 2^k full factorial in coded [-1.0, 1.0] space)
        cube_combinations = list(itertools.product([-1.0, 1.0], repeat=k))
        cube_df = pd.DataFrame(cube_combinations, columns=keys)

        # 2. Determine Alpha (Axial Distance)
        alpha_val = 1.0
        if self.alpha_type == "face-centered":
            alpha_val = 1.0
        elif self.alpha_type == "rotatable":
            alpha_val = (2.0 ** k) ** 0.25
        else:  # "orthogonal"
            # Standard default orthogonal alpha is often very close or identical to rotatable
            alpha_val = (2.0 ** k) ** 0.25

        # 3. Axial Star Portion (2k runs at coded distance +/- alpha)
        star_rows = []
        for i in range(k):
            row_pos = [0.0] * k
            row_pos[i] = alpha_val
            star_rows.append(row_pos)

            row_neg = [0.0] * k
            row_neg[i] = -alpha_val
            star_rows.append(row_neg)
        star_df = pd.DataFrame(star_rows, columns=keys)

        # 4. Center Points (num_center_points runs consisting of all coded 0.0s)
        center_rows = [[0.0] * k for _ in range(self.num_center_points)]
        center_df = pd.DataFrame(center_rows, columns=keys)

        # 5. Combine coded blocks
        coded_df = pd.concat([cube_df, star_df, center_df], ignore_index=True)

        # 6. Re-scale coded space to actual physical levels
        physical_df = pd.DataFrame()
        for col in keys:
            low, high = self.factors[col]
            mid = (low + high) / 2.0
            half_range = (high - low) / 2.0
            physical_df[col] = mid + coded_df[col] * half_range

        return physical_df
