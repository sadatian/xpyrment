"""Meta-Regression with Knapp-Hartung and Newey-West HAC Standard Errors (Block 38).

Aggregates study-level treatment effect estimates across various trials and cohorts,
supporting DerSimonian-Laird between-study variance, Knapp-Hartung standard error corrections,
and Newey-West Heteroskedasticity and Autocorrelation Consistent (HAC) covariance structures.
"""

from typing import Dict, Optional, Union
import numpy as np


class MetaRegressor:
    r"""Solves random-effects meta-regression models with Knapp-Hartung or Newey-West HAC standard errors.

    ??? mathbox "Mathematical Specifications of Meta-Regression"
        Meta-regression aggregates study-level treatment effect estimates across various trials and cohorts.
        
        Let $y_j$ be the estimated treatment effect of study $j \in \{1, \dots, J\}$, let $v_j$ be the within-study
        variance, and let $X_j$ be a $P$-dimensional vector of study-level covariates. The random-effects meta-regression
        model is:
        $$
        y_j = X_j \beta + u_j + \epsilon_j
        $$
        where $u_j \sim N(0, \tau^2)$ is the between-study random effect, and $\epsilon_j \sim N(0, v_j)$ is the within-study error.
        
        1. **DerSimonian-Laird Heterogeneity Variance Estimation**:
           The between-study variance parameter $\tau^2$ is estimated using a closed-form method of moments:
           $$
           \tau^2 = \max\left(0, \frac{Q - (J - P)}{\sum_{j=1}^J w_{FE, j} - \text{tr}((X^T W_{FE} X)^{-1} X^T W_{FE}^2 X)}\right)
           $$
           where $Q = \sum_{j=1}^J (y_j - X_j \hat{\beta}_{FE})^2 / v_j$ is Cochran's Q statistic, and $W_{FE} = \text{diag}(1/v_j)$ represents fixed-effect weights.
           
        2. **Coefficients WLS Solver**:
           With the estimated $\tau^2$, the final random-effects weights are $w_j = 1 / (v_j + \tau^2)$. The coefficients are solved via:
           $$
           \hat{\beta} = (X^T W X)^{-1} X^T W y
           $$
           where $W = \text{diag}(w_j)$.

        3. **Knapp-Hartung (KH) Covariance Adjustment**:
           Standard random-effects models can underestimate standard errors when $J$ is small. KH adjusts the variance-covariance matrix:
           $$
           \text{Cov}(\hat{\beta})_{KH} = q \cdot (X^T W X)^{-1}
           $$
           where $q$ is a scaling multiplier based on weighted residual sum of squares:
           $$
           q = \max\left(1.0, \frac{\sum_{j=1}^J w_j (y_j - X_j \hat{\beta})^2}{J - P}\right)
           $$

        4. **Newey-West HAC Covariance Correction**:
           If the studies are ordered chronologically and exhibit serial correlation, Newey-West HAC standard errors can be computed.
           The scores (estimating function contributions) for each study $j$ are:
           $$
           g_j = w_j e_j X_j^T
           $$
           where $e_j = y_j - X_j \hat{\beta}$ is the study-level residual.
           The Heteroskedasticity and Autocorrelation Consistent (HAC) long-run covariance of these scores is estimated with the Bartlett kernel:
           $$
           \Omega_{HAC} = \Gamma_0 + \sum_{l=1}^L \left(1 - \frac{l}{L+1}\right) (\Gamma_l + \Gamma_l^T)
           $$
           where $\Gamma_l = \sum_{j=l+1}^J g_j g_{j-l}^T$.
           The final sandwich covariance estimator is:
           $$
           \text{Cov}(\hat{\beta})_{HAC} = (X^T W X)^{-1} \Omega_{HAC} (X^T W X)^{-1}
           $$
    """

    def __init__(self, l2_penalty: float = 1e-5) -> None:
        """Initializes the meta-regressor.

        Args:
            l2_penalty (float): Ridge penalty for numerical stability. Defaults to 1e-5.
        """
        self.l2_penalty = l2_penalty
        self.beta_: np.ndarray = np.array([])
        self.se_: np.ndarray = np.array([])
        self.p_values_: np.ndarray = np.array([])
        self.tau_sq_: float = 0.0  # Between-study variance parameter
        self.cov_type_: str = "classic"

    def fit(
        self,
        outcomes: np.ndarray,
        variances: np.ndarray,
        covariates: np.ndarray,
        cov_type: str = "classic",
        hac_lag: Optional[int] = None,
    ) -> "MetaRegressor":
        """Fits the random-effects meta-regression model.

        Args:
            outcomes (np.ndarray): Study-level estimated treatment effects of shape (J,).
            variances (np.ndarray): Study-level within-study variances (SE^2) of shape (J,).
            covariates (np.ndarray): Covariate matrix of shape (J, P).
            cov_type (str): Covariance/standard error adjustment type (`"classic"` for Knapp-Hartung, `"hac"` for Newey-West). Defaults to `"classic"`.
            hac_lag (Optional[int]): Spectral lag bandwidth parameter ($L$). If None, computed automatically. Defaults to None.
        """
        J = len(outcomes)
        P = covariates.shape[1] if len(covariates.shape) > 1 else 1

        if J == 0:
            raise ValueError("Cannot fit meta-regression on empty dataset.")

        y = outcomes.astype(float)
        v = variances.astype(float)
        X = covariates.reshape(J, P).astype(float)

        if np.any(v <= 0.0):
            raise ValueError("Study within-study variances must be positive.")

        self.cov_type_ = cov_type

        # 1. Estimate initial fixed-effects coefficients to get residuals
        W_fe = np.diag(1.0 / v)
        XTWX_fe = np.dot(np.dot(X.T, W_fe), X) + self.l2_penalty * np.eye(P)
        beta_fe = np.linalg.solve(XTWX_fe, np.dot(np.dot(X.T, W_fe), y))

        # Heterogeneity statistic Q
        residuals_fe = y - np.dot(X, beta_fe)
        Q = np.sum((residuals_fe ** 2) / v)

        # 2. Compute between-study variance tau^2 using DerSimonian-Laird (DL) formula
        # Trace parameter: trace( W - W * X * (X^T * W * X)^-1 * X^T * W )
        inv_XTWX_fe = np.linalg.pinv(XTWX_fe)
        term1 = np.sum(1.0 / v)
        
        W_X = X / v.reshape(-1, 1)
        term2 = np.trace(np.dot(np.dot(W_X.T, W_X), inv_XTWX_fe))
        
        denom = term1 - term2
        if denom <= 0.0:
            denom = 1e-9

        self.tau_sq_ = max(0.0, (Q - (J - P)) / denom)

        # 3. Compute final random-effects coefficients using weights w_j = 1 / (v_j + tau^2)
        w_re = 1.0 / (v + self.tau_sq_)
        W_re = np.diag(w_re)
        
        XTWX_re = np.dot(np.dot(X.T, W_re), X) + self.l2_penalty * np.eye(P)
        self.beta_ = np.linalg.solve(XTWX_re, np.dot(np.dot(X.T, W_re), y))

        # 4. Compute standard error adjustments
        inv_XTWX_re = np.linalg.pinv(XTWX_re)
        residuals_re = y - np.dot(X, self.beta_)
        df = J - P

        if cov_type == "classic":
            # Knapp-Hartung (KH) scale factor q
            if df > 0:
                q = np.sum(w_re * (residuals_re ** 2)) / df
            else:
                q = 1.0

            # Enforce positive scale factor
            if q < 1.0:
                q = 1.0

            cov_matrix = q * inv_XTWX_re
        elif cov_type == "hac":
            # Newey-West HAC covariance calculations
            # Incorporate weights into scores g_j = w_j * e_j * X_j
            G = (w_re * residuals_re)[:, np.newaxis] * X  # shape (J, P)

            # Determine lag bandwidth L
            if hac_lag is not None:
                L = int(hac_lag)
            else:
                L = int(np.floor(4.0 * (J / 100.0) ** (2.0 / 9.0)))

            if L >= J:
                L = J - 1
            if L < 0:
                L = 0

            # Gamma_0 (standard White heteroskedasticity matrix)
            Gamma_0 = np.dot(G.T, G)

            # Bartlett kernel auto-covariance lagged sum
            Gamma_lag_sum = np.zeros((P, P))
            for l in range(1, L + 1):
                w_l = 1.0 - (l / (L + 1.0))
                Gamma_l = np.dot(G[l:].T, G[:-l])
                Gamma_lag_sum += w_l * (Gamma_l + Gamma_l.T)

            Omega_HAC = Gamma_0 + Gamma_lag_sum

            # Sandwich estimator: (X^T W X)^-1 * Omega_HAC * (X^T W X)^-1
            cov_matrix = np.dot(np.dot(inv_XTWX_re, Omega_HAC), inv_XTWX_re)
            # Small sample degrees-of-freedom correction
            df_corr = J / (J - P) if J > P else 1.0
            cov_matrix = cov_matrix * df_corr
        else:
            raise ValueError(f"Unknown cov_type: {cov_type}. Choose 'classic' or 'hac'.")

        self.se_ = np.sqrt(np.maximum(0.0, np.diag(cov_matrix)))

        # 5. Two-tailed p-values using Student's t-distribution with (J - P) df
        from scipy.stats import t as t_dist
        self.p_values_ = np.zeros(P)
        for i in range(P):
            se = self.se_[i]
            if se > 0 and df > 0:
                t_stat = self.beta_[i] / se
                self.p_values_[i] = float(2 * (1 - t_dist.cdf(abs(t_stat), df)))
            else:
                self.p_values_[i] = 1.0

        return self

    @property
    def results(self) -> Dict[str, Union[float, np.ndarray, str]]:
        """Returns fitted parameters, tau^2, and coefficient stats."""
        res = {
            "between_study_variance_tau_sq": self.tau_sq_,
            "coefficients": self.beta_,
            "standard_errors": self.se_,
            "p_values": self.p_values_,
            "cov_type": self.cov_type_,
        }
        # Keep old/alternative keys for backwards compatibility
        if self.cov_type_ == "classic":
            res["knapp_hartung_standard_errors"] = self.se_
        else:
            res["hac_standard_errors"] = self.se_
        return res

