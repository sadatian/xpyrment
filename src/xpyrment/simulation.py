"""Synthetic data generators for validation, profiling, and unit testing.

This module provides the `generate_ab_data` utility, which generates realistic experimental datasets containing continuous,
binary, and ratio variables across pre-period and post-period windows with stochastic correlation structures.
"""

import numpy as np
import pandas as pd


def generate_ab_data(
    n_samples: int = 10000,
    treatment_fraction: float = 0.5,
    baseline_revenue: float = 10.0,
    treatment_effect_revenue: float = 0.5,
    baseline_conversion: float = 0.15,
    treatment_effect_conversion: float = 0.02,
    baseline_clicks_mean: float = 5.0,
    baseline_impressions_mean: float = 100.0,
    treatment_effect_clicks: float = 0.3,
    pre_period_correlation: float = 0.70,
    random_seed: int = 42,
) -> pd.DataFrame:
    r"""Generates synthetic A/B test data simulating continuous, binary, and ratio metrics.

    Constructs a high-fidelity synthetic evaluation dataset. This generator is crucial for testing the validity of the
    statistical engines, validating Type I / Type II error rates, and profiling variance reduction (CUPED) performance.
    It generates both pre-period and post-period metrics to support covariate-adjustment modeling.

    ??? mathbox "Mathematical and Generative Specifications"

        1. **Continuous Metric (Revenue) with Pre/Post Covariance**:
           Revenue is modeled using a bivariate normal distribution to inject a pre-defined correlation ($\rho$)
           between pre-period (covariate) and post-period (outcome) performance.
           - Let $Y_i = [Y_{i, \text{pre}}, Y_{i, \text{post}}]^T$ be the revenue vector for unit $i$.
           - Under the Control variant, the mean vector is $\boldsymbol{\mu}_C = [\mu_{\text{baseline}}, \mu_{\text{baseline}}]^T$.
           - Under the Treatment variant, the mean vector is $\boldsymbol{\mu}_T = [\mu_{\text{baseline}}, \mu_{\text{baseline}} + \delta_{\text{rev}}]^T$.
           - The covariance matrix $\boldsymbol{\Sigma}$ is configured using standard deviation $\sigma$ and target correlation $\rho$:
             $$
             \boldsymbol{\Sigma} = \begin{bmatrix} \sigma^2 & \rho \sigma^2 \\ \rho \sigma^2 & \sigma^2 \end{bmatrix}
             $$
           - We sample $Y_i \sim \mathcal{N}_2(\boldsymbol{\mu}_k, \boldsymbol{\Sigma})$ and apply a non-negative floor:
             $$
             Y_{i} \leftarrow \max(Y_i, 0)
             $$
        2. **Binary Rate Metric (Conversions)**:
           Conversions are modeled as independent Bernoulli trials:
           - For Control: $Converted_i \sim \text{Bernoulli}(p_C)$ where $p_C = p_{\text{baseline}}$.
           - For Treatment: $Converted_i \sim \text{Bernoulli}(p_T)$ where $p_T = \min(\max(p_{\text{baseline}} + \delta_{\text{conv}}, 0), 1)$.

        3. **Ratio Metric (Clicks and Impressions for Click-Through Rate)**:
           Simulates CTR stochastically, introducing user-level heterogeneity and a positive skew:
           - Post-period impressions follow a Poisson distribution:
             $$
             Impressions_i \sim \text{Poisson}(\lambda_{\text{baseline\_impressions}})
             $$
             with a minimum threshold of $1$ to prevent divisions by zero.
           - Click probabilities for each user follow a Beta distribution to model user variance (Beta-Binomial stochastics):
             $$
             p_{i, \text{CTR}} \sim \text{Beta}(a_k, b_k)
             $$
             where the shape parameters $a_k, b_k$ are derived to match the expected CTR of the respective group:
             $$
             a_k = \text{CTR}_k \times 10, \quad b_k = (1 - \text{CTR}_k) \times 10
             $$
           - Finally, individual clicks are simulated using Binomial trials:
             $$
             Clicks_i \sim \text{Binomial}(Impressions_i, \ p_{i, \text{CTR}})
             $$
    Args:
        n_samples (int): The total number of experimental units (users) to simulate. Defaults to 10000.
        treatment_fraction (float): The probability of assignment to the Treatment group. Defaults to 0.5.
        baseline_revenue (float): The baseline mean revenue ($\mu_{\text{baseline}}$). Defaults to 10.0.
        treatment_effect_revenue (float): The absolute revenue lift in Treatment ($\delta_{\text{rev}}$). Defaults to 0.5.
        baseline_conversion (float): The baseline conversion rate ($p_{\text{baseline}}$). Defaults to 0.15.
        treatment_effect_conversion (float): The absolute conversion lift in Treatment ($\delta_{\text{conv}}$). Defaults to 0.02.
        baseline_clicks_mean (float): Baseline expected clicks. Defaults to 5.0.
        baseline_impressions_mean (float): Baseline expected impressions ($\lambda$). Defaults to 100.0.
        treatment_effect_clicks (float): Incremental clicks in Treatment. Defaults to 0.3.
        pre_period_correlation (float): The target correlation coefficient ($\rho$) between pre and post continuous revenue. Defaults to 0.70.
        random_seed (int): Pseudo-random seed to guarantee reproducibility. Defaults to 42.

    Returns:
        pd.DataFrame: A DataFrame containing simulated user IDs, variant assignments, and pre/post metrics:
            - `"user_id"` (str): Unique identifier formatted as `USER_######`.
            - `"variant"` (str): Assignment labels (`"control"` or `"treatment"`).
            - `"pre_revenue"` (float): Pre-period continuous covariate metric.
            - `"revenue"` (float): Post-period continuous outcome metric.
            - `"converted"` (int): Binary conversion indicator ($0$ or $1$).
            - `"pre_impressions"` (int): Pre-period ratio denominator.
            - `"pre_clicks"` (int): Pre-period ratio numerator.
            - `"impressions"` (int): Post-period ratio denominator.
            - `"clicks"` (int): Post-period ratio numerator.
    """
    rng = np.random.default_rng(random_seed)

    # 1. Assign User IDs and Treatment groups
    user_ids = [f"USER_{i:06d}" for i in range(1, n_samples + 1)]
    variants = ["control", "treatment"]
    treatment_probs = [1.0 - treatment_fraction, treatment_fraction]
    variant_assignments = rng.choice(variants, size=n_samples, p=treatment_probs)

    # 2. Continuous Metrics (Revenue) with Pre-Period Correlation
    # Simulating using a Bivariate Normal Distribution transformed to be positive
    mean_pre = baseline_revenue
    mean_post_c = baseline_revenue
    mean_post_t = baseline_revenue + treatment_effect_revenue

    std_val = 5.0

    cov_matrix = [
        [std_val**2, pre_period_correlation * (std_val**2)],
        [pre_period_correlation * (std_val**2), std_val**2],
    ]

    # Generate for Control
    n_c = int(np.sum(variant_assignments == "control"))
    bivariate_c = rng.multivariate_normal([mean_pre, mean_post_c], cov_matrix, size=n_c)

    # Generate for Treatment
    n_t = n_samples - n_c
    bivariate_t = rng.multivariate_normal([mean_pre, mean_post_t], cov_matrix, size=n_t)

    # Recombine results preserving index alignments
    pre_revenue = np.zeros(n_samples)
    revenue = np.zeros(n_samples)

    c_idx = 0
    t_idx = 0
    for i, var in enumerate(variant_assignments):
        if var == "control":
            pre_revenue[i] = bivariate_c[c_idx, 0]
            revenue[i] = bivariate_c[c_idx, 1]
            c_idx += 1
        else:
            pre_revenue[i] = bivariate_t[t_idx, 0]
            revenue[i] = bivariate_t[t_idx, 1]
            t_idx += 1

    # Keep numbers positive
    pre_revenue = np.clip(pre_revenue, 0, None)
    revenue = np.clip(revenue, 0, None)

    # 3. Binary Metrics (Conversions)
    converted = np.zeros(n_samples, dtype=int)
    for i, var in enumerate(variant_assignments):
        prob = baseline_conversion if var == "control" else (baseline_conversion + treatment_effect_conversion)
        prob = np.clip(prob, 0.0, 1.0)
        converted[i] = rng.binomial(n=1, p=prob)

    # 4. Ratio Metrics (Clicks and Impressions for CTR)
    # Both clicks and impressions are correlated stochastically, with positive variance
    pre_impressions = rng.poisson(lam=baseline_impressions_mean, size=n_samples)
    pre_impressions = np.clip(pre_impressions, 1, None)

    impressions = rng.poisson(lam=baseline_impressions_mean, size=n_samples)
    impressions = np.clip(impressions, 1, None)

    # Click proportions
    base_ctr = baseline_clicks_mean / baseline_impressions_mean
    treatment_ctr = (baseline_clicks_mean + treatment_effect_clicks) / baseline_impressions_mean

    pre_clicks = np.zeros(n_samples, dtype=int)
    clicks = np.zeros(n_samples, dtype=int)

    for i, var in enumerate(variant_assignments):
        pre_ctr_user = rng.beta(base_ctr * 10, (1.0 - base_ctr) * 10)
        pre_clicks[i] = rng.binomial(n=int(pre_impressions[i]), p=pre_ctr_user)

        user_ctr_prob = base_ctr if var == "control" else treatment_ctr
        ctr_user = rng.beta(user_ctr_prob * 10, (1.0 - user_ctr_prob) * 10)
        clicks[i] = rng.binomial(n=int(impressions[i]), p=ctr_user)

    # Compile dataset
    df = pd.DataFrame(
        {
            "user_id": user_ids,
            "variant": variant_assignments,
            "pre_revenue": pre_revenue,
            "revenue": revenue,
            "converted": converted,
            "pre_impressions": pre_impressions,
            "pre_clicks": pre_clicks,
            "impressions": impressions,
            "clicks": clicks,
        }
    )

    return df


class ExperimentSimulator:
    """Runs extensive Monte Carlo simulations to validate experimental designs and algorithms.

    TODO: Add synthetic panel non-compliance treatment estimation corrections using Instrumental Variables (LATE/CACE).
    TODO: Support synthetic network topology configurations to simulate clustered graph-based interference spillovers.
    """

    def __init__(self, random_seed: int = 42) -> None:
        """Initializes the experiment simulator with a pseudo-random seed."""
        self.rng = np.random.default_rng(random_seed)

    def generate_synthetic_panel(
        self,
        n_samples: int = 1000,
        baseline_mean: float = 10.0,
        treatment_effect: float = 1.0,
        non_compliance_rate: float = 0.0,
        spillover_effect: float = 0.0,
    ) -> pd.DataFrame:
        """Generates a synthetic panel with potential non-compliance and spillover.

        Args:
            n_samples (int): Number of experimental units.
            baseline_mean (float): Baseline outcome intercept.
            treatment_effect (float): Treatment causal impact (CATE).
            non_compliance_rate (float): Probability of failing to comply with assignment.
            spillover_effect (float): Causal impact spilled over onto control units from treatment.
        """
        # Baseline covariates
        age = self.rng.normal(loc=35, scale=10, size=n_samples)
        tenure = self.rng.exponential(scale=3, size=n_samples)

        # Assigned variant (Intent to Treat)
        assigned_treatment = self.rng.binomial(n=1, p=0.5, size=n_samples)

        # Actual compliance decision
        actual_treatment = assigned_treatment.copy()
        if non_compliance_rate > 0.0:
            # Randomly flip compliance based on compliance rate
            flip_mask = self.rng.random(size=n_samples) < non_compliance_rate
            actual_treatment[flip_mask] = 1 - actual_treatment[flip_mask]

        # Outcomes under potential outcomes framework:
        # Y_i = baseline_mean + 0.1 * age_i + 0.5 * tenure_i + treatment_effect * actual_treatment_i
        #       + spillover_effect * (1 - actual_treatment_i) * (proportion of treated)
        # Note: We simulate a simplified spillover model
        prop_treated = float(np.mean(actual_treatment))
        spillover_term = spillover_effect * prop_treated

        y_base = baseline_mean + 0.1 * age + 0.5 * tenure + self.rng.normal(loc=0, scale=2.0, size=n_samples)
        y = y_base + (treatment_effect * actual_treatment) + (spillover_term * (1 - actual_treatment))

        return pd.DataFrame({
            "unit_id": np.arange(1, n_samples + 1),
            "covariate_age": age,
            "covariate_tenure": tenure,
            "assigned_treatment": assigned_treatment,
            "actual_treatment": actual_treatment,
            "outcome": y,
        })

    def run_monte_carlo(
        self,
        n_simulations: int = 50,
        n_samples: int = 500,
        baseline_mean: float = 10.0,
        treatment_effect: float = 1.0,
        non_compliance_rate: float = 0.0,
        spillover_effect: float = 0.0,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Runs repeated Monte Carlo trials to compute empirical power, bias, and MSE of a standard T-test.

        Returns:
            Dict[str, Any]: Calculated statistical performance metrics.
        """
        from scipy.stats import ttest_ind

        rejections = 0
        estimates = []

        for _ in range(n_simulations):
            df = self.generate_synthetic_panel(
                n_samples=n_samples,
                baseline_mean=baseline_mean,
                treatment_effect=treatment_effect,
                non_compliance_rate=non_compliance_rate,
                spillover_effect=spillover_effect,
            )

            # Analyze using actual compliance (As-Treated estimation)
            y_trt = df.loc[df["actual_treatment"] == 1, "outcome"].values
            y_ctrl = df.loc[df["actual_treatment"] == 0, "outcome"].values

            if len(y_trt) < 2 or len(y_ctrl) < 2:
                continue

            stat, p_val = ttest_ind(y_trt, y_ctrl, equal_var=False)
            
            if p_val < alpha:
                rejections += 1

            est_effect = float(np.mean(y_trt) - np.mean(y_ctrl))
            estimates.append(est_effect)

        estimates = np.array(estimates)
        avg_estimate = float(np.mean(estimates)) if len(estimates) > 0 else 0.0
        
        # Bias: E[beta_hat] - beta
        bias = avg_estimate - treatment_effect
        # MSE: E[(beta_hat - beta)^2]
        mse = float(np.mean((estimates - treatment_effect) ** 2)) if len(estimates) > 0 else 0.0
        # Empirical Power / Type I error (rejection rate)
        rejection_rate = float(rejections / n_simulations) if n_simulations > 0 else 0.0

        return {
            "num_simulations_run": n_simulations,
            "empirical_mean_estimate": avg_estimate,
            "empirical_bias": bias,
            "empirical_mean_squared_error": mse,
            "empirical_rejection_rate": rejection_rate,
        }

