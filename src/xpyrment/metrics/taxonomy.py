r"""Standardized metrics taxonomy, calculation engines, and variance reduction routines.

This module houses the core representation of experimental measurements in `xpyrment`.
It implements a modular metrics hierarchy to support continuous (`MeanMetric`), binary
(`ProportionMetric`), and aggregate ratio (`RatioMetric`) metrics. All metrics support Welch's
t-test for hypothesis testing, and the continuous and ratio engines support integrated
CUPED (Controlled-comparison Using Pre-Existing Data) for variance reduction.

Mathematical Background:
    1. **Welch's t-test**:
       Unlike Student's t-test, Welch's t-test does not assume equal variances between control and
       treatment groups. The test statistic is:
       $$t = \frac{\bar{Y}_T - \bar{Y}_C}{\sqrt{\frac{s_C^2}{N_C} + \frac{s_T^2}{N_T}}}$$
       And degrees of freedom $\nu$ are approximated using the Welch-Satterthwaite equation:
       $$\nu \approx \frac{\left(\frac{s_C^2}{N_C} + \frac{s_T^2}{N_T}\right)^2}{\frac{\left(s_C^2 / N_C\right)^2}{N_C - 1} + \frac{\left(s_T^2 / N_T\right)^2}{N_T - 1}}$$

    2. **CUPED (Controlled-comparison Using Pre-Existing Data)**:
       CUPED utilizes pre-experiment covariate data ($X$) to explain away pre-existing variance in
       the experiment period metric ($Y$), thereby increasing statistical power.
       $$Y_{\text{CUPED}} = Y - \theta (X - E[X])$$
       where $\theta$ is the optimal scaling factor computed as:
       $$\theta = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}$$
       The variance of the CUPED-adjusted metric is:
       $$\text{Var}(Y_{\text{CUPED}}) = \text{Var}(Y)(1 - \rho^2)$$
       where $\rho$ is the Pearson correlation coefficient between $Y$ and $X$.

    3. **Delta Method for Ratios**:
       For a ratio metric $R = U / V$ (e.g., clicks/impressions), the sample variance is approximated
       using a first-order Taylor expansion:
       $$\text{Var}(R) \approx \frac{1}{\mu_V^2} \text{Var}(U) + \frac{\mu_U^2}{\mu_V^4} \text{Var}(V) - 2 \frac{\mu_U}{\mu_V^3} \text{Cov}(U, V)$$
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from scipy import stats


class BaseMetric(ABC):
    """Abstract base class representing a statistical metric in the experiment taxonomy.

    All custom metrics must inherit from `BaseMetric` and implement the abstract `.calculate()`
    method to return a standardized `MetricResult` dictionary.

    Attributes:
        name (str): The unique descriptive name of the metric.
    """

    def __init__(self, name: str):
        """Initializes a BaseMetric.

        Args:
            name (str): Unique descriptive name of the metric.
        """
        self.name = name

    @abstractmethod
    def calculate(
        self, df: pd.DataFrame, treatment_col: str, control: str, treatment: str
    ) -> Dict[str, Any]:
        """Abstract method to compute statistics for control and treatment groups.

        Args:
            df (pd.DataFrame): The experimental dataset.
            treatment_col (str): Column name identifying experimental groups/arms.
            control (str): The value in `treatment_col` representing the control group.
            treatment (str): The value in `treatment_col` representing the treatment group.

        Returns:
            Dict[str, Any]: A compliant `MetricResult` dictionary containing mean, variance,
                p-value, confidence intervals, power, etc.
        """
        pass

    def _calculate_stats(
        self,
        mean_c: float,
        mean_t: float,
        var_c: float,
        var_t: float,
        n_c: int,
        n_t: int,
        alpha: float = 0.05,
    ) -> Dict[str, float]:
        r"""Computes statistical p-value, confidence intervals, and power using Welch's t-test.

        This method executes Welch's t-test to compare control and treatment means when group
        variances are unequal. It also computes statistical power using a non-central t-distribution.

        Mathematical Representation:
            Degrees of freedom ($\nu$) approximation:
            $$\nu = \frac{\left(\frac{\sigma_C^2}{N_C} + \frac{\sigma_T^2}{N_T}\right)^2}{\frac{\left(\sigma_C^2 / N_C\right)^2}{N_C - 1} + \frac{\left(\sigma_T^2 / N_T\right)^2}{N_T - 1}}$$
            Standard error of difference:
            $$\text{SE}_{\text{diff}} = \sqrt{\frac{\sigma_C^2}{N_C} + \frac{\sigma_T^2}{N_T}}$$
            Confidence Interval:
            $$\text{CI} = (\bar{Y}_T - \bar{Y}_C) \pm t_{\text{crit}, 1-\alpha/2, \nu} \times \text{SE}_{\text{diff}}$$
            Power ($1-\beta$) is calculated using the Non-Central Parameter (NCP):
            $$\text{NCP} = \frac{|\bar{Y}_T - \bar{Y}_C|}{\text{SE}_{\text{diff}}}$$

        Args:
            mean_c (float): Control group mean ($\bar{Y}_C$).
            mean_t (float): Treatment group mean ($\bar{Y}_T$).
            var_c (float): Control group variance ($s^2_C$).
            var_t (float): Treatment group variance ($s^2_T$).
            n_c (int): Number of control units ($N_C$).
            n_t (int): Number of treatment units ($N_T$).
            alpha (float): Significance level (Type I error probability). Defaults to 0.05.

        Returns:
            Dict[str, float]: Standardized dict containing `p_value`, `ci_lower`, `ci_upper`,
                `rel_ci_lower`, `rel_ci_upper`, and `power`.
        """
        num = (var_c / n_c + var_t / n_t) ** 2
        den = ((var_c / n_c) ** 2) / (n_c - 1) + ((var_t / n_t) ** 2) / (n_t - 1)
        df = num / den if den > 0 else (n_c + n_t - 2)

        se_diff = np.sqrt(var_c / n_c + var_t / n_t)
        diff = mean_t - mean_c

        if se_diff > 0:
            t_stat = diff / se_diff
            p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=df))

            t_crit = stats.t.ppf(1 - alpha / 2, df=df)
            ci_lower = diff - t_crit * se_diff
            ci_upper = diff + t_crit * se_diff

            ncp = np.abs(diff) / se_diff
            t_crit_alpha = stats.t.ppf(1 - alpha / 2, df=df)
            power = 1 - stats.t.cdf(t_crit_alpha, df=df, loc=ncp) + stats.t.cdf(-t_crit_alpha, df=df, loc=ncp)
        else:
            p_val = 1.0
            ci_lower = diff
            ci_upper = diff
            power = 0.0

        relative_lift = diff / mean_c if mean_c != 0 else 0.0
        rel_ci_lower = ci_lower / mean_c if mean_c != 0 else 0.0
        rel_ci_upper = ci_upper / mean_c if mean_c != 0 else 0.0

        return {
            "p_value": p_val,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "rel_ci_lower": rel_ci_lower,
            "rel_ci_upper": rel_ci_upper,
            "power": power,
        }


class MeanMetric(BaseMetric):
    """A metric representing a continuous or numeric value (e.g., average revenue, sessions).

    Supports optional pre-period CUPED (Controlled-comparison Using Pre-Existing Data) adjustment
    to explain away pre-existing variance and dramatically lower required sample sizes or runtimes.

    Attributes:
        value_col (str): The column in the DataFrame containing active experiment period values.
        pre_period_col (Optional[str]): The column containing pre-experiment baseline values for CUPED.
    """

    def __init__(
        self,
        name: str,
        value_col: str,
        pre_period_col: Optional[str] = None,
    ):
        """Initializes a MeanMetric.

        Args:
            name (str): Unique descriptive name of the metric.
            value_col (str): Column name containing experiment period values.
            pre_period_col (Optional[str]): Column name containing pre-experiment baseline values for CUPED.
                Defaults to None (no CUPED applied).
        """
        super().__init__(name)
        self.value_col = value_col
        self.pre_period_col = pre_period_col

    def calculate(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        control: str,
        treatment: str,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Calculates descriptive and Welch's t-test statistics for the mean metric.

        Drops missing values on the value column. If `pre_period_col` is provided,
        performs joint missing drop and executes a standard linear CUPED adjustment:
        $$Y_i^{\text{CUPED}} = Y_i - \theta (X_i - \bar{X})$$

        Args:
            df (pd.DataFrame): The experimental dataset.
            treatment_col (str): Column identifying treatment assignments.
            control (str): Control arm identifier value in `treatment_col`.
            treatment (str): Treatment arm identifier value in `treatment_col`.
            alpha (float): Significance level for Welch's confidence intervals. Defaults to 0.05.

        Returns:
            Dict[str, Any]: A completed `MetricResult` dictionary.

        Raises:
            ValueError: If either control or treatment group becomes empty after filtering.
        """
        df_clean = df.dropna(subset=[self.value_col]).copy()

        c_mask = df_clean[treatment_col] == control
        t_mask = df_clean[treatment_col] == treatment

        n_c = int(np.sum(c_mask))
        n_t = int(np.sum(t_mask))

        if n_c == 0 or n_t == 0:
            raise ValueError(f"Control or treatment group is empty for metric {self.name}.")

        y = df_clean[self.value_col].to_numpy()

        cuped_achieved = False
        variance_reduction = 0.0

        if self.pre_period_col and self.pre_period_col in df_clean.columns:
            df_clean = df_clean.dropna(subset=[self.pre_period_col])
            c_mask = df_clean[treatment_col] == control
            t_mask = df_clean[treatment_col] == treatment
            n_c = int(np.sum(c_mask))
            n_t = int(np.sum(t_mask))

            y = df_clean[self.value_col].to_numpy()
            x = df_clean[self.pre_period_col].to_numpy()

            var_x = np.var(x, ddof=1)
            if var_x > 0:
                cov_yx = np.cov(y, x, ddof=1)[0, 1]
                theta = cov_yx / var_x
                mean_x_global = np.mean(x)

                y_cuped = y - theta * (x - mean_x_global)

                y_c = y_cuped[c_mask]
                y_t = y_cuped[t_mask]

                mean_c = float(np.mean(y_c))
                mean_t = float(np.mean(y_t))

                var_c = float(np.var(y_c, ddof=1))
                var_t = float(np.var(y_t, ddof=1))

                cuped_achieved = True

                orig_var = np.var(y, ddof=1)
                adjusted_var = np.var(y_cuped, ddof=1)
                if orig_var > 0:
                    variance_reduction = max(0.0, (orig_var - adjusted_var) / orig_var)
            else:
                mean_c = float(np.mean(y[c_mask]))
                mean_t = float(np.mean(y[t_mask]))
                var_c = float(np.var(y[c_mask], ddof=1))
                var_t = float(np.var(y[t_mask], ddof=1))
        else:
            mean_c = float(np.mean(y[c_mask]))
            mean_t = float(np.mean(y[t_mask]))
            var_c = float(np.var(y[c_mask], ddof=1))
            var_t = float(np.var(y[t_mask], ddof=1))

        stats_dict = self._calculate_p_and_ci(
            mean_c=mean_c,
            mean_t=mean_t,
            var_c=var_c,
            var_t=var_t,
            n_c=n_c,
            n_t=n_t,
            alpha=alpha,
        )

        diff = mean_t - mean_c
        relative_lift = diff / mean_c if mean_c != 0 else 0.0

        results = {
            "metric_name": self.name,
            "metric_type": "Mean",
            "control_mean": mean_c,
            "treatment_mean": treatment_mean if (treatment_mean := mean_t) is not None else mean_t,
            "control_var": var_c,
            "treatment_var": var_t,
            "control_n": n_c,
            "treatment_n": n_t,
            "absolute_difference": diff,
            "relative_lift": relative_lift,
            "cuped_applied": cuped_achieved,
            "variance_reduction": variance_reduction,
            **stats_dict,
        }

        return results


class ProportionMetric(MeanMetric):
    """A metric representing a binary/proportion rate (e.g., conversion rate, success rate).

    Inherits continuous logic from `MeanMetric`, as proportions can be modelled asymptotically
    using normal approximations (Z-test/t-test) under the Central Limit Theorem.
    """

    def calculate(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        control: str,
        treatment: str,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Calculates proportion conversion rates, differences, and statistical significance.

        Drops missing values, calculates means and variances of binary inputs, and delegates
        to `MeanMetric.calculate` while overriding the metric type string to "Proportion".

        Args:
            df (pd.DataFrame): The experimental dataset.
            treatment_col (str): Column identifying treatment assignments.
            control (str): Control arm identifier value.
            treatment (str): Treatment arm identifier value.
            alpha (float): Significance level. Defaults to 0.05.

        Returns:
            Dict[str, Any]: Standardized results dict with "metric_type" set to "Proportion".
        """
        res = super().calculate(df, treatment_col, control, treatment, alpha)
        res["metric_type"] = "Proportion"
        return res


class RatioMetric(BaseMetric):
    """A metric calculated as the ratio: sum(numerator) / sum(denominator) (e.g., Click-Through-Rate).

    Employs the Delta Method to approximate ratio-level variances and supports double-covariate
    ratio-level CUPED adjustments to independently reduce variance in numerator and denominator.

    Attributes:
        numerator_col (str): The column containing active period numerator values.
        denominator_col (str): The column containing active period denominator values (must be $>0$).
        pre_numerator_col (Optional[str]): Column containing pre-experiment baseline numerator values.
        pre_denominator_col (Optional[str]): Column containing pre-experiment baseline denominator values.
    """

    def __init__(
        self,
        name: str,
        numerator_col: str,
        denominator_col: str,
        pre_numerator_col: Optional[str] = None,
        pre_denominator_col: Optional[str] = None,
    ):
        """Initializes a RatioMetric.

        Args:
            name (str): Unique descriptive name.
            numerator_col (str): Active numerator column name.
            denominator_col (str): Active denominator column name.
            pre_numerator_col (Optional[str]): Pre-experiment numerator column. Defaults to None.
            pre_denominator_col (Optional[str]): Pre-experiment denominator column. Defaults to None.
        """
        super().__init__(name)
        self.numerator_col = numerator_col
        self.denominator_col = denominator_col
        self.pre_numerator_col = pre_numerator_col
        self.pre_denominator_col = pre_denominator_col

    def calculate(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        control: str,
        treatment: str,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        r"""Calculates ratio values, Delta-method variances, and statistical significance.

        Cleans missing values and non-positive denominators. If double-covariates are present,
        separately fits linear CUPED adjustments to the numerator and denominator series:
        $$U_i^{\text{CUPED}} = U_i - \theta_U (U_{i,\text{pre}} - \bar{U}_{\text{pre}})$$
        $$V_i^{\text{CUPED}} = V_i - \theta_V (V_{i,\text{pre}} - \bar{V}_{\text{pre}})$$
        The ratio variance is then estimated using the Delta Method formulation:
        $$\text{Var}\left(\frac{U}{V}\right) \approx \frac{1}{\bar{V}^2} \left[ \text{Var}(U) + R^2 \text{Var}(V) - 2 R \text{Cov}(U, V) \right]$$

        Args:
            df (pd.DataFrame): The experimental dataset.
            treatment_col (str): Column identifying treatment assignments.
            control (str): Control arm identifier value.
            treatment (str): Treatment arm identifier value.
            alpha (float): Significance level. Defaults to 0.05.

        Returns:
            Dict[str, Any]: Completed `MetricResult` dictionary.

        Raises:
            ValueError: If either control or treatment group becomes empty after filtering.
        """
        df_clean = df.dropna(subset=[self.numerator_col, self.denominator_col]).copy()
        df_clean = df_clean[df_clean[self.denominator_col] > 0]

        c_mask = df_clean[treatment_col] == control
        t_mask = df_clean[treatment_col] == treatment

        n_c = int(np.sum(c_mask))
        n_t = int(np.sum(t_mask))

        if n_c == 0 or n_t == 0:
            raise ValueError(f"Control or treatment group is empty for ratio metric {self.name}.")

        cuped_achieved = False
        variance_reduction = 0.0

        num = df_clean[self.numerator_col].to_numpy()
        den = df_clean[self.denominator_col].to_numpy()

        if (
            self.pre_numerator_col
            and self.pre_denominator_col
            and self.pre_numerator_col in df_clean.columns
            and self.pre_denominator_col in df_clean.columns
        ):
            df_clean = df_clean.dropna(subset=[self.pre_numerator_col, self.pre_denominator_col])
            df_clean = df_clean[df_clean[self.pre_denominator_col] > 0]

            c_mask = df_clean[treatment_col] == control
            t_mask = df_clean[treatment_col] == treatment
            n_c = int(np.sum(c_mask))
            n_t = int(np.sum(t_mask))

            num = df_clean[self.numerator_col].to_numpy()
            den = df_clean[self.denominator_col].to_numpy()
            pre_num = df_clean[self.pre_numerator_col].to_numpy()
            pre_den = df_clean[self.pre_denominator_col].to_numpy()

            var_pre_num = np.var(pre_num, ddof=1)
            var_pre_den = np.var(pre_den, ddof=1)

            if var_pre_num > 0 and var_pre_den > 0:
                cov_num = np.cov(num, pre_num, ddof=1)[0, 1]
                theta_num = cov_num / var_pre_num
                mean_pre_num_global = np.mean(pre_num)
                num_cuped = num - theta_num * (pre_num - mean_pre_num_global)

                cov_den = np.cov(den, pre_den, ddof=1)[0, 1]
                theta_den = cov_den / var_pre_den
                mean_pre_den_global = np.mean(pre_den)
                den_cuped = den - theta_den * (pre_den - mean_pre_den_global)

                num_c, num_t = num_cuped[c_mask], num_cuped[t_mask]
                den_c, den_t = den_cuped[c_mask], den_cuped[t_mask]

                mean_num_c, mean_num_t = np.mean(num_c), np.mean(num_t)
                mean_den_c, mean_den_t = np.mean(den_c), np.mean(den_t)

                ratio_c = mean_num_c / mean_den_c
                ratio_t = mean_num_t / mean_den_t

                var_num_c = np.var(num_c, ddof=1)
                var_den_c = np.var(den_c, ddof=1)
                cov_num_den_c = np.cov(num_c, den_c, ddof=1)[0, 1]

                var_ratio_c = (1 / (mean_den_c**2)) * (
                    var_num_c + (ratio_c**2) * var_den_c - 2 * ratio_c * cov_num_den_c
                )

                var_num_t = np.var(num_t, ddof=1)
                var_den_t = np.var(den_t, ddof=1)
                cov_num_den_t = np.cov(num_t, den_t, ddof=1)[0, 1]

                var_ratio_t = (1 / (mean_den_t**2)) * (
                    var_num_t + (ratio_t**2) * var_den_t - 2 * ratio_t * cov_num_den_t
                )

                cuped_achieved = True

                orig_ratio_global = np.mean(num) / np.mean(den)
                orig_var_ratio = (1 / (np.mean(den) ** 2)) * (
                    np.var(num, ddof=1)
                    + (orig_ratio_global**2) * np.var(den, ddof=1)
                    - 2 * orig_ratio_global * np.cov(num, den, ddof=1)[0, 1]
                )

                adj_ratio_global = np.mean(num_cuped) / np.mean(den_cuped)
                adj_var_ratio = (1 / (np.mean(den_cuped) ** 2)) * (
                    np.var(num_cuped, ddof=1)
                    + (adj_ratio_global**2) * np.var(den_cuped, ddof=1)
                    - 2 * adj_ratio_global * np.cov(num_cuped, den_cuped, ddof=1)[0, 1]
                )

                if orig_var_ratio > 0:
                    variance_reduction = max(0.0, (orig_var_ratio - adj_var_ratio) / orig_var_ratio)
            else:
                mean_num_c, mean_num_t = np.mean(num[c_mask]), np.mean(num[t_mask])
                mean_den_c, mean_den_t = np.mean(den[c_mask]), np.mean(den[t_mask])
                ratio_c = mean_num_c / mean_den_c
                ratio_t = mean_num_t / mean_den_t

                var_num_c = np.var(num[c_mask], ddof=1)
                var_den_c = np.var(den[c_mask], ddof=1)
                cov_num_den_c = np.cov(num[c_mask], den[c_mask], ddof=1)[0, 1]
                var_ratio_c = (1 / (mean_den_c**2)) * (
                    var_num_c + (ratio_c**2) * var_den_c - 2 * ratio_c * cov_num_den_c
                )

                var_num_t = np.var(num[t_mask], ddof=1)
                var_den_t = np.var(den[t_mask], ddof=1)
                cov_num_den_t = np.cov(num[t_mask], den[t_mask], ddof=1)[0, 1]
                var_ratio_t = (1 / (mean_den_t**2)) * (
                    var_num_t + (ratio_t**2) * var_den_t - 2 * ratio_t * cov_num_den_t
                )
        else:
            mean_num_c, mean_num_t = np.mean(num[c_mask]), np.mean(num[t_mask])
            mean_den_c, mean_den_t = np.mean(den[c_mask]), np.mean(den[t_mask])
            ratio_c = mean_num_c / mean_den_c
            ratio_t = mean_num_t / mean_den_t

            var_num_c = np.var(num[c_mask], ddof=1)
            var_den_c = np.var(den[c_mask], ddof=1)
            cov_num_den_c = np.cov(num[c_mask], den[c_mask], ddof=1)[0, 1]
            var_ratio_c = (1 / (mean_den_c**2)) * (
                var_num_c + (ratio_c**2) * var_den_c - 2 * ratio_c * cov_num_den_c
            )

            var_num_t = np.var(num[t_mask], ddof=1)
            var_den_t = np.var(den[t_mask], ddof=1)
            cov_num_den_t = np.cov(num[t_mask], den[t_mask], ddof=1)[0, 1]
            var_ratio_t = (1 / (mean_den_t**2)) * (
                var_num_t + (ratio_t**2) * var_den_t - 2 * ratio_t * cov_num_den_t
            )

        stats_dict = self._calculate_p_and_ci(
            mean_c=ratio_c,
            mean_t=ratio_t,
            var_c=var_ratio_c,
            var_t=var_ratio_t,
            n_c=n_c,
            n_t=n_t,
            alpha=alpha,
        )

        diff = ratio_t - ratio_c
        relative_lift = diff / ratio_c if ratio_c != 0 else 0.0

        results = {
            "metric_name": self.name,
            "metric_type": "Ratio",
            "control_mean": ratio_c,
            "treatment_mean": ratio_t,
            "control_var": var_ratio_c,
            "treatment_var": var_ratio_t,
            "control_n": n_c,
            "treatment_n": n_t,
            "absolute_difference": diff,
            "relative_lift": relative_lift,
            "cuped_applied": cuped_achieved,
            "variance_reduction": variance_reduction,
            **stats_dict,
        }

        return results
