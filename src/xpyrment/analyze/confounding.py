"""Multi-Factor Fractional ANOVA Confounding Resolvers for sparse experimental designs.

Computes the alias matrix to algebraically isolate main-effect projections from
joint 2-way and 3-way factor confounding manifolds.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


class AliasResolver:
    """Computes and resolves the alias structure in fractional factorial designs.

    # TODO: Implement sequential D-optimal design updates that minimize the trace of the alias matrix projection error
    # covariance, Cov(beta_1_true) = sigma^2 (X_1^T X_1)^{-1} + A Cov(beta_2) A^T.
    """

    def __init__(self, primary_cols: List[str], potential_confounding_cols: List[str]) -> None:
        """Initializes the AliasResolver.

        Args:
            primary_cols (List[str]): Column names representing primary effects of interest (e.g., main effects).
            potential_confounding_cols (List[str]): Column names representing potential higher-order confounding
                effects (e.g., 2-way or 3-way interactions).
        """
        self.primary_cols = primary_cols
        self.potential_confounding_cols = potential_confounding_cols
        self.alias_matrix_ = None

    def compute_alias_matrix(self, df: pd.DataFrame, l2_penalty: float = 1e-6) -> np.ndarray:
        """Computes the alias matrix: A = (X1^T X1 + lambda * I)^{-1} X1^T X2.

        Args:
            df (pd.DataFrame): Input design or experimental data containing all specified columns.
            l2_penalty (float): Regularization parameter for the inverse computation to ensure numerical stability.
                Defaults to 1e-6.

        Returns:
            np.ndarray: The alias matrix A of shape (len(primary_cols) + 1, len(potential_confounding_cols)).
                The first row corresponds to the unpenalized intercept/bias term.
        """
        N = len(df)
        
        # Build X1 with bias term (intercept)
        X1 = np.hstack([np.ones((N, 1)), df[self.primary_cols].to_numpy()])
        X2 = df[self.potential_confounding_cols].to_numpy()

        # Regularized normal equations solver: (X1^T X1 + lambda * I) A = X1^T X2
        XTX = np.dot(X1.T, X1)
        XTX_reg = XTX + l2_penalty * np.eye(X1.shape[1])
        # Do not regularize the intercept term
        XTX_reg[0, 0] = XTX[0, 0]

        X1TX2 = np.dot(X1.T, X2)
        self.alias_matrix_ = np.linalg.solve(XTX_reg, X1TX2)
        return self.alias_matrix_

    def get_alias_report(self, df: pd.DataFrame, threshold: float = 1e-3) -> Dict[str, List[Tuple[str, float]]]:
        """Generates a structured report detailing which primary effects are aliased with which confounding effects.

        Args:
            df (pd.DataFrame): Input design or experimental data.
            threshold (float): Only reports alias coefficients whose absolute value exceeds this threshold.
                Defaults to 1e-3.

        Returns:
            Dict[str, List[Tuple[str, float]]]: A dictionary mapping primary effect names (and 'intercept')
                to lists of (confounding_col_name, alias_coefficient) tuples.
        """
        A = self.compute_alias_matrix(df)
        report = {}

        labels = ["intercept"] + self.primary_cols

        for i, primary_label in enumerate(labels):
            row_aliases = []
            for j, confounding_col in enumerate(self.potential_confounding_cols):
                coef = float(A[i, j])
                if abs(coef) > threshold:
                    row_aliases.append((confounding_col, coef))
            if row_aliases:
                report[primary_label] = row_aliases

        return report

    def resolve_coefficients(self, beta1_biased: np.ndarray, beta2_prior: np.ndarray) -> np.ndarray:
        """Resolves/decouples true primary coefficients given biased estimates and prior confounding parameters.

        Uses the relationship: beta1_true = beta1_biased - A * beta2_prior

        Args:
            beta1_biased (np.ndarray): Biased primary coefficients (including intercept) of shape (len(primary_cols) + 1,).
            beta2_prior (np.ndarray): Prior known or estimated higher-order confounding coefficients
                of shape (len(potential_confounding_cols),).

        Returns:
            np.ndarray: Cleaned/debiased primary coefficients of shape (len(primary_cols) + 1,).
        """
        if self.alias_matrix_ is None:
            raise ValueError("Alias matrix must be computed using compute_alias_matrix before resolving coefficients.")
        
        return beta1_biased - np.dot(self.alias_matrix_, beta2_prior)
