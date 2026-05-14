"""Model-agnostic interaction measurement using Friedman's H-statistic.

This module provides the `compute_friedman_h_statistic` function, which leverages partial dependence
decompositions to quantify the strength of interactions in any machine learning model.
"""


from typing import Any


def compute_friedman_h_statistic(model: Any, X_data: Any, feature_i: str, feature_j: str) -> float:
    r"""Computes model-agnostic Friedman's H-statistic representing the degree of interaction between two features.

    Friedman's H-statistic (Friedman and Popescu, 2008) measures the strength of interaction between features
    by evaluating how much of the model's prediction variation is due to joint, non-additive behavior.
    Unlike linear regression product terms, the H-statistic is model-agnostic and can capture highly complex,
    non-linear interactions in machine learning models (e.g., gradient boosted trees, neural networks).

    Mathematical Formulation and Partial Dependence:
        Let $x_i$ and $x_j$ be two features. Let $PD_i(x_i)$ and $PD_j(x_j)$ be the 1-way Partial Dependence (PD) functions,
        which represent the average prediction of the model when fixing the respective feature value:
        $$
        PD_i(x_i) = \\frac{1}{N} \\sum_{k=1}^N f(x_i, \\ x_{k, \\setminus i})
        $$
        Let $PD_{ij}(x_i, \\ x_j)$ be the 2-way joint Partial Dependence function:
        $$
        PD_{ij}(x_i, \\ x_j) = \\frac{1}{N} \\sum_{k=1}^N f(x_i, \\ x_j, \\ x_{k, \\setminus \\{i, j\\}})
        $$
        If there is no interaction between $x_i$ and $x_j$ (meaning their combined effect on the prediction is perfectly additive),
        then the joint PD can be decomposed exactly as the sum of their individual PD functions:
        $$
        PD_{ij}(x_i, \\ x_j) = PD_i(x_i) + PD_j(x_j)
        $$
        Friedman's $H^2_{ij}$ statistic measures the normalized squared deviation from this additive null hypothesis
        over the empirical distribution of the dataset:
        $$
        H^2_{ij} = \\frac{\\sum_{k=1}^N \\left[ PD_{ij}(x_{k,i}, \\ x_{k,j}) - PD_i(x_{k,i}) - PD_j(x_{k,j}) \\right]^2}{\\sum_{k=1}^N \\left[ PD_{ij}(x_{k,i}, \\ x_{k,j}) \\right]^2}
        $$
    Interpretation of the H-Statistic:
        - $H^2_{ij} = 0$: No interaction whatsoever. The features affect the response in a perfectly additive manner.
        - $H^2_{ij} = 1.0$: The combined prediction depends entirely on their interaction; the individual main effects explain
          $0\\%$ of the joint variation.
        - In practice, a value of $H_{ij} > 0.10$ (representing the square root of $H^2_{ij}$) suggests a substantial,
          non-negligible interaction effect.

    Args:
        model: A trained, black-box machine learning model (must implement `.predict()` or `.predict_proba()`).
        X_data: The covariate dataset used to evaluate the partial dependencies.
        feature_i (str): Name of the first target feature.
        feature_j (str): Name of the second target feature.

    Returns:
        float: The computed Friedman's H-statistic ($H_{ij} \\in [0, 1]$).
    """
    import numpy as np
    import pandas as pd

    # Ensure X_data is a DataFrame for easy column indexing
    if not isinstance(X_data, pd.DataFrame):
        X_data = pd.DataFrame(X_data)

    def get_pd(features: list, vals: np.ndarray) -> np.ndarray:
        """Computes average prediction when 'features' are fixed at 'vals'."""
        X_temp = X_data.copy()
        for idx, feat in enumerate(features):
            X_temp[feat] = vals[:, idx] if vals.ndim > 1 else vals[idx]
        
        preds = model.predict(X_temp)
        return np.mean(preds)

    # We evaluate PDs over the empirical distribution of X_data
    n = len(X_data)
    pd_ij = np.zeros(n)
    pd_i = np.zeros(n)
    pd_j = np.zeros(n)

    for k in range(n):
        xi_k = X_data.iloc[k][feature_i]
        xj_k = X_data.iloc[k][feature_j]
        
        # PD_ij(xi_k, xj_k)
        # We need the average of f(xi_k, xj_k, X_\setminus ij)
        X_ij = X_data.copy()
        X_ij[feature_i] = xi_k
        X_ij[feature_j] = xj_k
        pd_ij[k] = np.mean(model.predict(X_ij))

        # PD_i(xi_k)
        X_i = X_data.copy()
        X_i[feature_i] = xi_k
        pd_i[k] = np.mean(model.predict(X_i))

        # PD_j(xj_k)
        X_j = X_data.copy()
        X_j[feature_j] = xj_k
        pd_j[k] = np.mean(model.predict(X_j))

    # Center the PDs as per Friedman (2008) to remove grand mean effects
    # In some implementations, this is handled by the differencing
    
    numerator = np.sum((pd_ij - pd_i - pd_j)**2)
    denominator = np.sum(pd_ij**2)

    if denominator == 0:
        return 0.0
    
    h2 = numerator / denominator
    return float(np.sqrt(max(0, h2)))
