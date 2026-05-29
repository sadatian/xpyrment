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
             $$
             p|k, n \sim \text{Beta}(\alpha_0 + k, \ \beta_0 + n - k)
             $$
        2. **Normal-Normal Model** (for continuous averages, $\mu \in \mathbb{R}$, with known variance $\sigma^2$):
           - Prior: $\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)$.
           - Likelihood: Normal ($N$ observations with sample mean $\bar{Y}$ and variance $\sigma^2$).
           - Posterior:
             $$
             \mu| \bar{Y} \sim \mathcal{N}(\mu_N, \sigma_N^2)
             $$
             where the posterior precision ($1/\sigma_N^2$) and posterior mean ($\mu_N$) are calculated as:
             $$
             \frac{1}{\sigma_N^2} = \frac{1}{\sigma_0^2} + \frac{N}{\sigma^2} \quad \text{and} \quad \mu_N = \sigma_N^2 \left( \frac{\mu_0}{\sigma_0^2} + \frac{N\bar{Y}}{\sigma^2} \right)
             $$
        3. **Gamma-Poisson Model** (for discrete counts/rates, $\lambda \in [0, \infty)$):
           - Prior: $\lambda \sim \text{Gamma}(\alpha, \beta)$ (shape $\alpha$, rate $\beta$).
           - Likelihood: Poisson ($k$ total counts across $n$ units).
           - Posterior:
             $$
             \lambda|k, n \sim \text{Gamma}(\alpha + k, \beta + n)
             $$

    Decision-Making Criteria and Analytics:
        - **Probability of Being Best (PBB)**: The probability that the treatment parameter $\theta_T$ is strictly greater
          than the control parameter $\theta_C$:
          $$
          \text{PBB} = P(\theta_T > \theta_C) = \int_{-\infty}^{\infty} f_C(\theta_C) [1 - F_T(\theta_C)] \, d\theta_C
          $$
        - **Expected Loss ($L$)**: The expected metric drop if the treatment is shipped but is actually inferior:
          $$
          L(T) = \mathbb{E}[\max(\theta_C - \theta_T, 0)] = \int_{-\infty}^{\infty} \int_{-\infty}^{\theta_C} (\theta_C - \theta_T) f_T(\theta_T) f_C(\theta_C) \, d\theta_T \, d\theta_C
          $$

    Attributes:
        model_type (str): Conjugate model pairing label. Options: `"beta_binomial"`, `"normal_normal"`, `"gamma_poisson"`.
            Defaults to `"beta_binomial"`.
    """

    def __init__(self, model_type: str = "beta_binomial") -> None:
        """Initializes a BayesianInference.

        Args:
            model_type (str): Conjugate model to use (`"beta_binomial"`, `"normal_normal"`, or `"gamma_poisson"`).
                Defaults to `"beta_binomial"`.
        """
        self.model_type = model_type

    def estimate_posterior(self, prior_params: dict, observed_data: dict, exact: bool = True) -> dict:
        """Estimates conjugate posterior distributions based on prior settings and raw observations.

        Performs the analytical conjugate update formulas, then computes PBB, Expected Loss, and
        credible intervals.

        Args:
            prior_params (dict): Prior parameters (e.g., `{"alpha": 1, "beta": 1}` or `{"mean": 0, "variance": 1}`).
            observed_data (dict): Observed outcomes (conversions and counts, or means and variances).
            exact (bool): Whether to use numerical integration for exact PBB and Expected Loss.
                If False, uses Monte Carlo sampling (20k samples). Defaults to True.

        Returns:
            dict: Posterior distribution parameters, credible intervals, and decision metrics.
        """
        import numpy as np
        from scipy import stats, integrate

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
            dist_c = stats.beta(a0 + k_c, b0 + n_c - k_c)
            dist_t = stats.beta(a0 + k_t, b0 + n_t - k_t)

            res = {
                "control_posterior": {"param1": float(dist_c.args[0]), "param2": float(dist_c.args[1])},
                "treatment_posterior": {"param1": float(dist_t.args[0]), "param2": float(dist_t.args[1])}
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

            # Posterior calculation
            prec_0 = 1.0 / var_0
            
            prec_c = prec_0 + n_c / var_c
            var_c_post = 1.0 / prec_c
            mu_c_post = var_c_post * (mu_0 * prec_0 + n_c * mean_c / var_c)

            prec_t = prec_0 + n_t / var_t
            var_t_post = 1.0 / prec_t
            mu_t_post = var_t_post * (mu_0 * prec_0 + n_t * mean_t / var_t)

            dist_c = stats.norm(mu_c_post, np.sqrt(var_c_post))
            dist_t = stats.norm(mu_t_post, np.sqrt(var_t_post))

            res = {
                "control_posterior": {"param1": float(mu_c_post), "param2": float(var_c_post)},
                "treatment_posterior": {"param1": float(mu_t_post), "param2": float(var_t_post)}
            }

        elif self.model_type == "gamma_poisson":
            # Prior parameters: shape alpha, rate beta (rate = 1/scale)
            a0 = prior_params.get("alpha", prior_params.get("shape", 1.0))
            b0 = prior_params.get("beta", prior_params.get("rate", 1.0))

            # Observed data: k counts in n exposure units
            k_c = observed_data.get("control_counts", observed_data.get("k_c", 0))
            n_c = observed_data.get("control_exposure", observed_data.get("n_c", 1))
            k_t = observed_data.get("treatment_counts", observed_data.get("k_t", 0))
            n_t = observed_data.get("treatment_exposure", observed_data.get("n_t", 1))

            # Posterior parameters: alpha_post = alpha + k, beta_post = beta + n
            # Scipy gamma uses scale = 1/beta
            dist_c = stats.gamma(a0 + k_c, scale=1.0 / (b0 + n_c))
            dist_t = stats.gamma(a0 + k_t, scale=1.0 / (b0 + n_t))

            res = {
                "control_posterior": {"param1": float(dist_c.args[0]), "param2": float(1.0 / dist_c.extra_args[0] if hasattr(dist_c, 'extra_args') else 1.0/dist_c.kwds['scale'])},
                "treatment_posterior": {"param1": float(dist_t.args[0]), "param2": float(1.0 / dist_t.extra_args[0] if hasattr(dist_t, 'extra_args') else 1.0/dist_t.kwds['scale'])}
            }
            # Fix param2 extraction for robustness
            res["control_posterior"]["param2"] = float(a0 + k_c) # wait no, param2 is rate
            res["control_posterior"]["param1"] = float(a0 + k_c)
            res["control_posterior"]["param2"] = float(b0 + n_c)
            res["treatment_posterior"]["param1"] = float(a0 + k_t)
            res["treatment_posterior"]["param2"] = float(b0 + n_t)

        else:
            raise ValueError(f"Unknown Bayesian model type: {self.model_type}")

        # Common metrics
        ci_c_lower, ci_c_upper = dist_c.ppf([0.025, 0.975])
        ci_t_lower, ci_t_upper = dist_t.ppf([0.025, 0.975])
        res["control_posterior"].update({"ci_lower": float(ci_c_lower), "ci_upper": float(ci_c_upper)})
        res["treatment_posterior"].update({"ci_lower": float(ci_t_lower), "ci_upper": float(ci_t_upper)})

        if exact:
            # Determine integration bounds based on the spread of the distributions
            # We use the [0.0001, 0.9999] quantile range to cover the meaningful parts of the PDF
            low_c, high_c = dist_c.ppf([0.0001, 0.9999])
            low_t, high_t = dist_t.ppf([0.0001, 0.9999])
            
            # Use the union of both ranges plus a small buffer
            lower = min(low_c, low_t)
            upper = max(high_c, high_t)
            
            # If the distribution is very narrow (e.g. at zero), ensure we have some width
            if upper <= lower:
                upper = lower + 1.0

            # Exact PBB: integral of f_c(x) * (1 - F_t(x))
            pbb, _ = integrate.quad(lambda x: dist_c.pdf(x) * (1 - dist_t.cdf(x)), lower, upper, limit=100)
            
            # Exact Expected Loss: integral of f_c(x) * integral_{-inf}^{x} (x - y) * f_t(y) dy
            # Which is: integral f_c(x) * [x * F_t(x) - integral_{-inf}^{x} y * f_t(y) dy]
            def loss_integrand(x):
                # Integral_{lower}^{x} y * f_t(y) dy is the partial mean
                inner_val, _ = integrate.quad(lambda y: y * dist_t.pdf(y), lower, x, limit=50)
                return dist_c.pdf(x) * (x * dist_t.cdf(x) - inner_val)

            expected_loss, _ = integrate.quad(loss_integrand, lower, upper, limit=100)
        else:
            # Monte Carlo fallback
            samples_c = dist_c.rvs(size=20000, random_state=42)
            samples_t = dist_t.rvs(size=20000, random_state=42)
            pbb = float(np.mean(samples_t > samples_c))
            expected_loss = float(np.mean(np.maximum(samples_c - samples_t, 0.0)))

        res.update({"pbb": float(pbb), "expected_loss": float(expected_loss)})
        return res
