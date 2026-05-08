"""Double Machine Learning (DML) with K-Fold Cross-Fitting (Chernozhukov et al., 2018).

Estimates unbiased causal treatment effects under high-dimensional nuisance parameters
using Robinson's residual-on-residual OLS regression.
"""

from typing import Type, Any, Tuple
import numpy as np
from scipy.stats import norm
from xpyrment.personalize.meta_learners import RidgeRegressor


class DoubleMachineLearning:
    """Double Machine Learning estimator with K-Fold cross-fitting for treatment effect estimation.

    # TODO: Support estimating Heterogeneous Treatment Effects (CATE) via local polynomial residual-on-residual regression
    # or kernel-weighted Robinson's OLS, tau(x) = ArgMin_tau sum_i K((X_i - x)/h) * (tilde{Y}_i - tilde{T}_i * tau)^2.
    """

    def __init__(
        self,
        model_y_class: Type[Any] = RidgeRegressor,
        model_t_class: Type[Any] = RidgeRegressor,
        n_folds: int = 5,
        model_y_kwargs: dict = None,
        model_t_kwargs: dict = None,
    ) -> None:
        """Initializes the DoubleMachineLearning estimator.

        Args:
            model_y_class (Type): Class of regression model used for outcome Y prediction.
                Defaults to RidgeRegressor.
            model_t_class (Type): Class of regression model used for treatment T prediction (propensity model).
                Defaults to RidgeRegressor.
            n_folds (int): Number of folds for cross-fitting. Defaults to 5.
            model_y_kwargs (dict): Keyword arguments for outcome model instantiation.
            model_t_kwargs (dict): Keyword arguments for treatment model instantiation.
        """
        self.model_y_class = model_y_class
        self.model_t_class = model_t_class
        self.n_folds = n_folds
        self.model_y_kwargs = model_y_kwargs or {"alpha": 1.0}
        self.model_t_kwargs = model_t_kwargs or {"alpha": 1.0}

        self.treatment_effect_ = 0.0
        self.standard_error_ = 0.0
        self.p_value_ = 1.0

    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray, seed: int = 42) -> "DoubleMachineLearning":
        """Executes Robinson's residual-on-residual regression with cross-fitting.

        Args:
            X (np.ndarray): Covariate feature matrix of shape (N, P).
            treatment (np.ndarray): Binary treatment assignment vector of shape (N,).
            y (np.ndarray): Numeric outcome vector of shape (N,).
            seed (int): Random seed for deterministic K-Fold partitioning. Defaults to 42.
        """
        N = X.shape[0]
        treatment = treatment.ravel()
        y = y.ravel()

        # 1. Deterministic K-Fold split
        indices = np.arange(N)
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        folds = np.array_split(indices, self.n_folds)

        y_res = np.zeros(N)
        t_res = np.zeros(N)

        # 2. Sequential Out-of-Fold (OOF) residual computation
        for k in range(self.n_folds):
            test_idx = folds[k]
            train_idx = np.concatenate([folds[i] for i in range(self.n_folds) if i != k])

            # Instantiate fold-specific models
            model_y = self.model_y_class(**self.model_y_kwargs)
            model_t = self.model_t_class(**self.model_t_kwargs)

            # Fit outcome model: Y ~ m(X) on training set
            model_y.fit(X[train_idx], y[train_idx])
            # Fit treatment model: T ~ g(X) on training set
            model_t.fit(X[train_idx], treatment[train_idx])

            # Predict out-of-fold residuals
            y_res[test_idx] = y[test_idx] - model_y.predict(X[test_idx])
            t_res[test_idx] = treatment[test_idx] - model_t.predict(X[test_idx])

        # 3. Robinson's Residual-on-Residual regression: y_res = tau * t_res + epsilon
        # Closed-form OLS: tau = (t_res^T t_res)^-1 * t_res^T y_res
        t_res_sq = np.dot(t_res, t_res)
        if t_res_sq < 1e-12:
            raise ValueError("Treatment residuals have zero variance; treatment cannot be estimated.")

        self.treatment_effect_ = float(np.dot(t_res, y_res) / t_res_sq)

        # Compute standard error: se = sqrt( sigma^2 / (t_res^T t_res) )
        residuals = y_res - self.treatment_effect_ * t_res
        sigma_sq = np.dot(residuals, residuals) / (N - 1)
        self.standard_error_ = float(np.sqrt(sigma_sq / t_res_sq))

        # Compute two-sided p-value under asymptotic normality
        z_stat = self.treatment_effect_ / max(self.standard_error_, 1e-12)
        self.p_value_ = float(2.0 * (1.0 - norm.cdf(abs(z_stat))))

        return self

    @property
    def treatment_effect(self) -> float:
        """Returns the estimated unbiased treatment effect (tau)."""
        return self.treatment_effect_

    @property
    def standard_error(self) -> float:
        """Returns the asymptotic standard error of the treatment effect estimate."""
        return self.standard_error_

    @property
    def p_value(self) -> float:
        """Returns the two-sided p-value under asymptotic normality."""
        return self.p_value_
