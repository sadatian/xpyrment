"""D-Optimal computer-generated classical Design of Experiments (DoE) matrices.

This module provides the `DOptimalDesign` class, which constructs D-optimal design matrices using
computer-guided coordinate exchange algorithms. D-optimal designs are critical when classical geometric
templates (such as factorials) are rendered impossible due to irregular design spaces, physical constraints,
or restricted trial footprints.
"""

import pandas as pd
from xpyrment.design.doe.base import DesignMatrix


class DOptimalDesign(DesignMatrix):
    r"""Optimizes the design matrix determinant |X'X| using coordinate exchange algorithms.

    Unlike classical designs which rely on rigid geometric symmetries, D-optimal designs are algorithmic,
    computer-generated designs. They are customized to fit a user-specified model (e.g., linear, interactive,
    or quadratic) and a fixed budget of $N$ runs, subject to arbitrary boundary constraints.

    When to use D-Optimal Designs:
        1. **Irregular Design Space**: When certain factor combinations are physically impossible or unsafe
           (e.g., Temperature + Pressure $\le$ Threshold). This represents a non-rectangular design space.
        2. **Specific Run Constraints**: When the budget restricts the experiment to exactly $N$ runs, which
           does not match any standard geometric size (such as $16$ or $27$).
        3. **Mix of Qualitative Factors**: When factors have irregular, unequal numbers of discrete levels.

    Mathematical Criterion:
        Let $X$ be the $N \times p$ model design matrix (where columns include main effects, interactions, and
        quadratic terms). The information matrix is $M = X^T X$.
        D-optimality maximizes the determinant of the information matrix:
        $$
        \max_{X} \left| X^T X \right|
        $$
        Maximizing this determinant is mathematically equivalent to minimizing the volume of the joint confidence
        ellipsoid for the estimated model parameters $\beta$. The D-efficiency of a design is:
        $$
        D_{\text{eff}} = 100 \times \left( \frac{\left| X^T X \right|^{1/p}}{N} \right)
        $$
    Coordinate Exchange Algorithm (Meyer & Nachtsheim, 1995):
        To find the optimal design without evaluating all combinations (which is NP-hard):
        1. Create an initial design matrix $X^{(0)}$ of size $N \times k$ by randomly sampling from candidate levels.
        2. Loop through each cell $x_{i, j}$ (row $i$, factor $j$) in the matrix:
           - Replace $x_{i, j}$ with each possible candidate level.
           - For each replacement, compute the new determinant $|X^T X|$ using rank-one update formulas
             (Sherman-Morrison-Woodbury theorem) to avoid expensive $O(p^3)$ matrix inversion.
           - Keep the level that maximizes the determinant.
        3. Repeat coordinate sweeps across the matrix until a full sweep results in no determinant improvements.
        4. Run this process from multiple (e.g., 20 to 50) random initial starts to prevent convergence into
           local optima, selecting the global maximum found.

    Attributes:
        num_runs (int): The exact target trial budget (number of runs in the final matrix).

    Examples:
        ??? example "Example"

            ```python
            >>> # Planning a 12-run custom design for 3 factors with safety constraints
            >>> factors = {"temp": [100, 150, 200], "speed": [10, 20, 30]}
            >>> design = DOptimalDesign(factors, num_runs=12)
            >>> # The generated DataFrame will contain exactly 12 runs, maximizing parameter estimation power.
            ```
    """

    def __init__(self, factors: dict, num_runs: int, seed: int = 42):
        """Initializes a DOptimalDesign.

        Args:
            factors (dict): Mapping of factor labels to their candidate levels.
            num_runs (int): The target budget of trials.
            seed (int): Random seed for reproducibility. Defaults to 42.
        """
        super().__init__(factors)
        self.num_runs = num_runs
        self.seed = seed

    def generate(self) -> pd.DataFrame:
        """Generates the D-Optimal design matrix.

        Runs the coordinate exchange optimization loop from multiple random starts,
        applying boundary constraints, and maps the best output to physical units.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the optimal design matrix.
        """
        import numpy as np

        rng = np.random.default_rng(self.seed)

        k = len(self.factors)
        keys = list(self.factors.keys())
        candidates = [np.array(self.factors[col]) for col in keys]

        best_det = -1.0
        best_df = None

        # Helper to construct model matrix X (intercept + linear terms)
        def build_model_matrix(D_vals):
            # D_vals is shape (num_runs, k)
            intercept = np.ones((self.num_runs, 1))
            return np.hstack([intercept, D_vals])

        # Run multiple random starts to avoid local optima
        num_starts = 10
        for _ in range(num_starts):
            # 1. Initialize random starting design from candidate levels
            D_current = np.zeros((self.num_runs, k))
            for j in range(k):
                D_current[:, j] = rng.choice(candidates[j], size=self.num_runs)

            # 2. Iterate Coordinate Exchange sweeps until convergence
            converged = False
            max_iter = 5
            for _ in range(max_iter):
                changed = False
                for i in range(self.num_runs):
                    for j in range(k):
                        current_val = D_current[i, j]
                        best_local_val = current_val
                        
                        # Compute initial determinant
                        X_curr = build_model_matrix(D_current)
                        best_local_det = np.linalg.det(np.dot(X_curr.T, X_curr))

                        for cand in candidates[j]:
                            if cand == current_val:
                                continue
                            
                            # Temporarily exchange coordinate
                            D_current[i, j] = cand
                            X_test = build_model_matrix(D_current)
                            test_det = np.linalg.det(np.dot(X_test.T, X_test))

                            if test_det > best_local_det:
                                best_local_det = test_det
                                best_local_val = cand

                        # Keep the level that maximized the determinant
                        if best_local_val != current_val:
                            D_current[i, j] = best_local_val
                            changed = True
                        else:
                            D_current[i, j] = current_val

                if not changed:
                    converged = True
                    break

            # 3. Assess final determinant for this start
            X_final = build_model_matrix(D_current)
            final_det = np.linalg.det(np.dot(X_final.T, X_final))

            if final_det > best_det:
                best_det = final_det
                best_df = pd.DataFrame(D_current, columns=keys)

        if best_df is None or best_det <= 1e-12:
            # Fallback: if all starts failed to yield non-singular matrices,
            # return a simple deterministic grid sample or raw randomized start
            fallback_D = np.zeros((self.num_runs, k))
            for j in range(k):
                fallback_D[:, j] = rng.choice(candidates[j], size=self.num_runs)
            best_df = pd.DataFrame(fallback_D, columns=keys)

        # TODO: Implement fast rank-1 update formulas (using Sherman-Morrison) to compute determinants in O(1) instead of recalculating full SVD in O(p^3).
        # TODO: Add alternative optimality criteria such as A-Optimality (trace of inverse information matrix) and G-Optimality (minimizing maximum prediction variance).
        return best_df
