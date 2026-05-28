"""Covariate Balance and Multi-Dimensional Unit Matching algorithms for observational studies.

Provides Coarsened Exact Matching (CEM), Propensity Score Matching (PSM), and
Mahalanobis Distance Matching (MDM) to optimize covariate balance.

# TODO: Support high-performance KD-Tree indexing to accelerate nearest-neighbor caliper searches on multi-million row observational datasets.
"""

from typing import List
import numpy as np
import pandas as pd


class LogisticRegression:
    """Simple Logistic Regression classifier solved via Gradient Descent for Propensity Score Estimation."""

    def __init__(self, lr: float = 0.1, max_iter: int = 1000, tol: float = 1e-5) -> None:
        """Initializes the logistic regression model.

        Args:
            lr (float): Learning rate. Defaults to 0.1.
            max_iter (int): Maximum training iterations. Defaults to 1000.
            tol (float): Tolerance threshold for convergence. Defaults to 1e-5.
        """
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Computes the element-wise logistic sigmoid function."""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """Fits the model coefficients on the training set using SciPy optimization solvers.

        Args:
            X (np.ndarray): Design matrix of shape (N, P).
            y (np.ndarray): Target labels of shape (N,).
        """
        from scipy.optimize import minimize

        N, P = X.shape
        X_bias = np.hstack([np.ones((N, 1)), X])
        
        def loss(w):
            z = np.dot(X_bias, w)
            preds = self._sigmoid(z)
            # Binary Cross Entropy
            epsilon = 1e-15
            preds = np.clip(preds, epsilon, 1.0 - epsilon)
            return -np.sum(y * np.log(preds) + (1 - y) * np.log(1 - preds)) / N
            
        def grad(w):
            z = np.dot(X_bias, w)
            preds = self._sigmoid(z)
            return np.dot(X_bias.T, preds - y) / N

        initial_weights = np.zeros(P + 1)
        res = minimize(loss, initial_weights, jac=grad, method="L-BFGS-B", options={"maxiter": self.max_iter, "gtol": self.tol})
        self.weights = res.x
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts continuous event probabilities of shape (N,)."""
        N = X.shape[0]
        X_bias = np.hstack([np.ones((N, 1)), X])
        return self._sigmoid(np.dot(X_bias, self.weights))


def mahalanobis_distance(u: np.ndarray, v: np.ndarray, cov_inv: np.ndarray) -> float:
    """Computes the Mahalanobis distance between two multi-dimensional unit vectors.

    d(u, v) = sqrt((u - v)^T * Sigma^{-1} * (u - v))

    Args:
        u (np.ndarray): First point of shape (D,).
        v (np.ndarray): Second point of shape (D,).
        cov_inv (np.ndarray): Precomputed inverse covariance matrix of shape (D, D).

    Returns:
        float: Calculated Mahalanobis distance.
    """
    diff = u - v
    return float(np.sqrt(np.dot(diff, np.dot(cov_inv, diff.T))))


class PropensityScoreMatcher:
    """Implements Propensity Score Matching (PSM) with calipers."""

    def __init__(self, caliper: float = 0.2) -> None:
        """Initializes the PropensityScoreMatcher.

        Args:
            caliper (float): Caliper width threshold in standard deviations of propensity score logits.
                Defaults to 0.2.
        """
        self.caliper = caliper
        self.model = LogisticRegression()

    def fit_predict(
        self, df: pd.DataFrame, treatment_col: str, covariate_cols: List[str]
    ) -> pd.DataFrame:
        """Matches treated and control units using Propensity Scores.

        Args:
            df (pd.DataFrame): Input dataset.
            treatment_col (str): Binary indicator column for treatment assignment.
            covariate_cols (List[str]): List of column names representing matching covariates.

        Returns:
            pd.DataFrame: A filtered matched dataset containing a 'weight' column representing
                matched units weights (1.0 for each matched unit).
        """
        X = df[covariate_cols].to_numpy()
        y = df[treatment_col].to_numpy()

        self.model.fit(X, y)
        prop_scores = self.model.predict_proba(X)

        result_df = df.copy()
        result_df["propensity_score"] = prop_scores
        result_df["weight"] = 0.0

        treated_indices = np.where(y == 1)[0]
        control_indices = np.where(y == 0)[0]

        # Logit transformation of propensity score prevents bound skewing near 0 and 1
        logits = np.log(prop_scores / (1.0 - prop_scores + 1e-12))
        caliper_val = self.caliper * np.std(logits)

        matched_controls = set()

        from scipy.spatial.distance import cdist

        matched_controls = set()
        
        logits_t = logits[treated_indices].reshape(-1, 1)
        logits_c = logits[control_indices].reshape(-1, 1)
        
        # Vectorized absolute distance computation
        dist_matrix = cdist(logits_t, logits_c, metric='cityblock')
        
        weight_col_idx = result_df.columns.get_loc("weight")

        for i, idx in enumerate(treated_indices):
            # Sort controls by distance
            sorted_c_indices_local = np.argsort(dist_matrix[i])
            
            for local_c_idx in sorted_c_indices_local:
                dist = dist_matrix[i, local_c_idx]
                if dist > caliper_val:
                    break  # Caliper violation, no more matches for this treated unit
                    
                c_idx = control_indices[local_c_idx]
                if c_idx not in matched_controls:
                    # Assign matches weights of 1.0
                    result_df.iloc[idx, weight_col_idx] = 1.0
                    result_df.iloc[c_idx, weight_col_idx] = 1.0
                    matched_controls.add(c_idx)
                    break

        # Filter out unmatched rows
        return result_df[result_df["weight"] > 0].copy()


class CoarsenedExactMatcher:
    """Implements Coarsened Exact Matching (CEM)."""

    def __init__(self, n_bins: int = 5) -> None:
        """Initializes the CoarsenedExactMatcher.

        Args:
            n_bins (int): Number of bins to coarsen continuous continuous features into.
                Defaults to 5.
        """
        self.n_bins = n_bins

    def fit_predict(
        self, df: pd.DataFrame, treatment_col: str, covariate_cols: List[str]
    ) -> pd.DataFrame:
        """Executes CEM by binning covariates and matching identical strata.

        Calculates appropriate balancing weight adjustments for control units in matching strata:
            w_i = (C_total / T_total) * (T_s / C_s)

        Args:
            df (pd.DataFrame): Input dataset.
            treatment_col (str): Binary indicator column for treatment assignment.
            covariate_cols (List[str]): List of column names representing matching covariates.

        Returns:
            pd.DataFrame: A filtered matched dataset containing a 'weight' column representing
                appropriate matching weights (1.0 for treated, CEM-weighted for control units).
        """
        df_coarse = df.copy()

        # 1. Coarsen continuous variables
        for col in covariate_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Create uniform bin bounds
                df_coarse[col] = pd.cut(df[col], bins=self.n_bins, labels=False)

        # 2. Group into strata using the coarsened covariates
        strata = df_coarse.groupby(covariate_cols)

        result_df = df.copy()
        result_df["weight"] = 0.0

        matched_treated_total = 0
        matched_control_total = 0

        strata_info = []

        # First pass: find matched strata and compute counts
        for _, group_indices in strata.groups.items():
            sub_df = df_coarse.loc[group_indices]
            n_treat = np.sum(sub_df[treatment_col] == 1)
            n_control = np.sum(sub_df[treatment_col] == 0)

            if n_treat > 0 and n_control > 0:
                matched_treated_total += n_treat
                matched_control_total += n_control
                strata_info.append((group_indices, n_treat, n_control))

        if matched_treated_total == 0 or matched_control_total == 0:
            # No matches found: return empty dataframe with weights
            return result_df[result_df["weight"] > 0].copy()

        # Second pass: assign weights
        for indices, n_t, n_c in strata_info:
            sub_indices = df.loc[indices].index
            for idx in sub_indices:
                t_val = df.loc[idx, treatment_col]
                if t_val == 1:
                    result_df.loc[idx, "weight"] = 1.0
                else:
                    # CEM weight formula: (C_total / T_total) * (T_s / C_s)
                    weight = (matched_control_total / matched_treated_total) * (n_t / n_c)
                    result_df.loc[idx, "weight"] = weight

        return result_df[result_df["weight"] > 0].copy()
