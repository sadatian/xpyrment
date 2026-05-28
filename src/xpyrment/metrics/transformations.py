"""Mathematical transformations and normalization utilities for experimental telemetry.

This module provides preprocessing transformations designed to stabilize variance, normalize
skewed distributions (common in commercial monetization data), or prepare metric distributions
for downstream frequentist and Bayesian hypothesis testing.
"""

import numpy as np
import pandas as pd


def log_transform(df: pd.DataFrame, col: str) -> pd.Series:
    r"""Transforms continuous metrics using a shifted natural log transformation.

    Highly skewed distributions, such as revenue per user or session durations, often violate
    the normality assumptions of classical parametric tests (e.g., Student's or Welch's t-test).
    Applying a natural log transformation normalizes the distribution and stabilizes variance
    (homoscedasticity). The addition of 1 ensures that zero values remain mapped to zero.

    ??? mathbox "Mathematical Representation"

        The transformation is defined as:
        $$
        y_{\text{transformed}} = \ln(y + 1)
        $$
        This is mathematically equivalent to:
        $$
        \log1p(y)
        $$
        which maintains numerical precision for extremely small values of $y \approx 0$.

    Args:
        df (pd.DataFrame): The source DataFrame containing the column to transform.
        col (str): The name of the target column in `df` representing the skewed metric.

    Returns:
        pd.Series: A new pandas Series containing the log-transformed values.

    Examples:
        ??? example "Example"

            ```python
            >>> import pandas as pd
            >>> df = pd.DataFrame({"revenue": [0.0, 10.0, 150.5]})
            >>> log_transform(df, "revenue")
            0    0.000000
            1    2.397895
            2    5.020586
            Name: revenue, dtype: float64
            ```
    """
    return np.log1p(df[col])


def delta_normalization(df: pd.DataFrame, col: str) -> pd.Series:
    r"""Normalizes metrics using the Delta Method expansion (Stub/Scaffolding).

    The Delta Method is a general technique for approximating the variance of a function of random
    variables. For non-linear transformations or aggregate metrics (e.g., Click-Through-Rate where the
    denominator is not fixed), direct variance calculations are biased. Delta normalization computes a
    Taylor series expansion of the target function around its expected value to derive an asymptotically
    normal approximation.

    ??? mathbox "Mathematical Context"

        Let $g(X)$ be a differentiable function of a random variable $X$ with mean $\mu$ and variance $\sigma^2$.
        The first-order Taylor expansion of $g(X)$ about $\mu$ is:
        $$
        g(X) \approx g(\mu) + g'(\mu)(X - \mu)
        $$
        Taking the variance of this linear approximation yields:
        $$
        \text{Var}(g(X)) \approx [g'(\mu)]^2 \sigma^2
        $$
        For multidimensional vectors, such as ratio estimates of the form $g(X, Y) = X / Y$, the Taylor expansion
        incorporates the covariance between numerator and denominator:
        $$
        \text{Var}\left(\frac{X}{Y}\right) \approx \frac{1}{\mu_Y^2} \text{Var}(X) + \frac{\mu_X^2}{\mu_Y^4} \text{Var}(Y) - 2 \frac{\mu_X}{\mu_Y^3} \text{Cov}(X, Y)
        $$
    Args:
        df (pd.DataFrame): The source DataFrame containing the metric columns.
        col (str): The name of the column representing the metric to normalize.

    Returns:
        pd.Series: A pandas Series of normalized values.
    """
    series = df[col]
    mean_val = series.mean()
    std_val = series.std(ddof=1)

    if pd.isna(std_val) or std_val == 0.0:
        return pd.Series(0.0, index=series.index, name=col)

    return (series - mean_val) / std_val
