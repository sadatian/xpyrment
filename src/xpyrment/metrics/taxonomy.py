from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from scipy import stats


class BaseMetric(ABC):
    """Abstract base class for all experiment metrics."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def calculate(
        self, df: pd.DataFrame, treatment_col: str, control: str, treatment: str
    ) -> Dict[str, Any]:
        """Calculates statistics for control and treatment groups."""
        pass

    def _calculate_p_and_ci(
        self,
        mean_c: float,
        mean_t: float,
        var_c: float,
        var_t: float,
        n_c: int,
        n_t: int,
        alpha: float = 0.05,
    ) -> Dict[str, float]:
        """Computes statistical p-value, confidence intervals, and power using Welch's t-test."""
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

    Supports CUPED adjustment if pre-period data is provided.
    """

    def __init__(
        self,
        name: str,
        value_col: str,
        pre_period_col: Optional[str] = None,
    ):
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
            "treatment_mean": mean_t,
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
    """A metric representing a binary/proportion rate (e.g., conversion rate)."""

    def calculate(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        control: str,
        treatment: str,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        res = super().calculate(df, treatment_col, control, treatment, alpha)
        res["metric_type"] = "Proportion"
        return res


class RatioMetric(BaseMetric):
    """A metric calculated as the ratio: sum(numerator) / sum(denominator) (e.g., CTR).

    Uses Delta Method for variance estimation. Supports Ratio-level CUPED.
    """

    def __init__(
        self,
        name: str,
        numerator_col: str,
        denominator_col: str,
        pre_numerator_col: Optional[str] = None,
        pre_denominator_col: Optional[str] = None,
    ):
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
