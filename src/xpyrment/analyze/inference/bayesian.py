"""Bayesian conjugate models and decision-making parameters.

This module provides the `BayesianInference` class, which estimates conjugate posterior distributions
(such as Beta-Binomial and Normal-Normal), computing decision metrics including the Probability of Being Best (PBB),
Expected Loss, and Region of Practical Equivalence (ROPE).
"""


class BayesianInference:
    r"""Computes Bayesian posterior parameters, probability of being best, expected loss, and ROPE.

    Bayesian inference offers a direct, probabilistic interpretation of treatment effects, avoiding the complex
    and frequently misunderstood reasoning of frequentist p-values. It provides answers to intuitive questions like:
    "What is the probability that Treatment B is superior to Control A?" or "What is the expected loss if I ship Treatment B?"

    Mathematical Formulation of Conjugate Models:
        Conjugate models allow the analytical calculation of posterior distributions without requiring expensive
        Markov Chain Monte Carlo (MCMC) sampling:

        1. **Beta-Binomial Model** (for binary conversion rates, $p \in [0, 1]$):
           - Prior: $p \sim \text{Beta}(\alpha_0, \beta_0)$ (e.g., $\text{Beta}(1, 1)$ for a flat, uniform prior).
           - Likelihood: Binomial ($k$ conversions out of $n$ trials).
           - Posterior:
             $$p|k, n \sim \text{Beta}(\alpha_0 + k, \ \beta_0 + n - k)$$
        2. **Normal-Normal Model** (for continuous averages, $\mu \in \mathbb{R}$, with known variance $\sigma^2$):
           - Prior: $\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)$.
           - Likelihood: Normal ($N$ observations with sample mean $\bar{Y}$ and variance $\sigma^2$).
           - Posterior:
             $$\mu| \bar{Y} \sim \mathcal{N}(\mu_N, \sigma_N^2)$$
             where the posterior precision ($1/\sigma_N^2$) and posterior mean ($\mu_N$) are calculated as:
             $$\frac{1}{\sigma_N^2} = \frac{1}{\sigma_0^2} + \frac{N}{\sigma^2} \quad \text{and} \quad \mu_N = \sigma_N^2 \left( \frac{\mu_0}{\sigma_0^2} + \frac{N\bar{Y}}{\sigma^2} \right)$$

    Decision-Making Criteria and Analytics:
        - **Probability of Being Best (PBB)**: The probability that the treatment parameter $\theta_T$ is strictly greater
          than the control parameter $\theta_C$:
          $$\text{PBB} = P(\theta_T > \theta_C) = \int_{-\infty}^{\infty} \int_{\theta_C}^{\infty} f_T(\theta_T) f_C(\theta_C) \, d\theta_T \, d\theta_C$$
          (Typically estimated via Monte Carlo sampling: drawing 100k random samples from each posterior and calculating the fraction where sample $t > c$).
        - **Expected Loss ($L$)**: The expected metric drop if the treatment is shipped but is actually inferior:
          $$L(T) = \mathbb{E}[\max(\theta_C - \theta_T, 0)]$$
          If the expected loss is below a certain threshold $\epsilon$ (the "acceptable risk level"), the treatment can be
          safely deployed.
        - **Region of Practical Equivalence (ROPE)**: Establishes a range $[-\delta, \delta]$ representing differences so small
          they are practically equivalent to zero. If the posterior distribution of the difference ($\theta_T - \theta_C$) lies
          entirely inside the ROPE, we conclude that the two variants are equivalent.

    Attributes:
        model_type (str): Conjugate model pairing label. Options: `"beta_binomial"`, `"normal_normal"`.
            Defaults to `"beta_binomial"`.
    """

    def __init__(self, model_type: str = "beta_binomial"):
        """Initializes a BayesianInference.

        Args:
            model_type (str): Conjugate model to use (`"beta_binomial"` or `"normal_normal"`).
                Defaults to `"beta_binomial"`.
        """
        self.model_type = model_type

    def estimate_posterior(self, prior_params: dict, observed_data: dict) -> dict:
        """Estimates conjugate posterior distributions based on prior settings and raw observations.

        Performs the analytical conjugate update formulas, then computes PBB, Expected Loss, and
        credible intervals.

        Args:
            prior_params (dict): Prior parameters (e.g., `{"alpha": 1, "beta": 1}` or `{"mean": 0, "variance": 1}`).
            observed_data (dict): Observed outcomes (conversions and counts, or means and variances).

        Returns:
            dict: Posterior distribution parameters, credible intervals, and decision metrics.
        """
        import numpy as np
        from scipy import stats

        if self.model_type == "beta_binomial":
            # Prior parameters
            a0 = prior_params.get("alpha", 1.0)
            b0 = prior_params.get("beta", 1.0)

            # Observed data
            k_c = observed_data.get("control_successes", observed_data.get("k_c", 0))
            n_c = observed_data.get("control_trials", observed_data.get("n_c", 1))
            k_t = observed_data.get("treatment_successes", observed_data.get("k_t", 0))
            n_t = observed_data.get("treatment_trials", observed_data.get("n_t", 1))

            # Posterior parameters
            alpha_c_post = a0 + k_c
            beta_c_post = b0 + n_c - k_c

            alpha_t_post = a0 + k_t
            beta_t_post = b0 + n_t - k_t

            # Generate samples for Monte Carlo simulation of PBB and Expected Loss
            samples_c = stats.beta.rvs(alpha_c_post, beta_c_post, size=20000, random_state=42)
            samples_t = stats.beta.rvs(alpha_t_post, beta_t_post, size=20000, random_state=42)

            # Credible intervals (95%)
            ci_c_lower, ci_c_upper = stats.beta.ppf([0.025, 0.975], alpha_c_post, beta_c_post)
            ci_t_lower, ci_t_upper = stats.beta.ppf([0.025, 0.975], alpha_t_post, beta_t_post)

            # Probability of being best (treatment > control)
            pbb = float(np.mean(samples_t > samples_c))
            # Expected loss of treatment
            expected_loss = float(np.mean(np.maximum(samples_c - samples_t, 0.0)))

            return {
                "control_posterior": {
                    "param1": float(alpha_c_post),
                    "param2": float(beta_c_post),
                    "ci_lower": float(ci_c_lower),
                    "ci_upper": float(ci_c_upper)
                },
                "treatment_posterior": {
                    "param1": float(alpha_t_post),
                    "param2": float(beta_t_post),
                    "ci_lower": float(ci_t_lower),
                    "ci_upper": float(ci_t_upper)
                },
                "pbb": pbb,
                "expected_loss": expected_loss
            }

        elif self.model_type == "normal_normal":
            # Prior parameters
            mu_0 = prior_params.get("mean", prior_params.get("mu_0", 0.0))
            var_0 = prior_params.get("variance", prior_params.get("sigma_0_sq", 1.0))

            # Observed data
            mean_c = observed_data.get("control_mean", observed_data.get("mean_c", 0.0))
            var_c = observed_data.get("control_variance", observed_data.get("var_c", 1.0))
            n_c = observed_data.get("control_n", observed_data.get("n_c", 1))

            mean_t = observed_data.get("treatment_mean", observed_data.get("mean_t", 0.0))
            var_t = observed_data.get("treatment_variance", observed_data.get("var_t", 1.0))
            n_t = observed_data.get("treatment_n", observed_data.get("n_t", 1))

            # Posterior variance: 1 / var_post = 1 / var_0 + n / var_sample
            prec_0 = 1.0 / var_0
            
            # Control posterior
            prec_c = prec_0 + n_c / var_c
            var_c_post = 1.0 / prec_c
            mu_c_post = var_c_post * (mu_0 * prec_0 + n_c * mean_c / var_c)

            # Treatment posterior
            prec_t = prec_0 + n_t / var_t
            var_t_post = 1.0 / prec_t
            mu_t_post = var_t_post * (mu_0 * prec_0 + n_t * mean_t / var_t)

            # Generate samples
            samples_c = stats.norm.rvs(mu_c_post, np.sqrt(var_c_post), size=20000, random_state=42)
            samples_t = stats.norm.rvs(mu_t_post, np.sqrt(var_t_post), size=20000, random_state=42)

            # Credible intervals (95%)
            ci_c_lower, ci_c_upper = stats.norm.ppf([0.025, 0.975], mu_c_post, np.sqrt(var_c_post))
            ci_t_lower, ci_t_upper = stats.norm.ppf([0.025, 0.975], mu_t_post, np.sqrt(var_t_post))

            pbb = float(np.mean(samples_t > samples_c))
            expected_loss = float(np.mean(np.maximum(samples_c - samples_t, 0.0)))

            return {
                "control_posterior": {
                    "param1": float(mu_c_post),
                    "param2": float(var_c_post),
                    "ci_lower": float(ci_c_lower),
                    "ci_upper": float(ci_c_upper)
                },
                "treatment_posterior": {
                    "param1": float(mu_t_post),
                    "param2": float(var_t_post),
                    "ci_lower": float(ci_t_lower),
                    "ci_upper": float(ci_t_upper)
                },
                "pbb": pbb,
                "expected_loss": expected_loss
            }
        else:
            raise ValueError(f"Unknown Bayesian model type: {self.model_type}")

        # TODO: Implement conjugate Gamma-Poisson model pairing for discrete count metrics (such as page views or clicks).
        # TODO: Add numerical integration solvers to compute PBB and Expected Loss exactly without relying on Monte Carlo simulations.
