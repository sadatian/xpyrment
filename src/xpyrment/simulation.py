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
    """Generates synthetic A/B test data simulating continuous, binary, and ratio metrics.

    Includes correlated pre-period values for CUPED analysis.
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
