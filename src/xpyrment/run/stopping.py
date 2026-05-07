"""Mixture Sequential Probability Ratio Test (mSPRT) sequential stopping rules.

This module provides the `StoppingRules` class, which implements sequential early-stopping criteria
using mixture Sequential Probability Ratio Tests (mSPRTs) to allow continuous monitoring of live
A/B tests while strictly controlling the Type I error rate.
"""


class StoppingRules:
    r"""Implements early-stopping rules like mixture sequential probability ratio tests (mSPRT).

    In traditional fixed-sample A/B testing, peeking at the results before the planned sample size is reached
    and stopping the test early if the p-value is significant severely inflates the Type I error rate (the peeking problem).
    mSPRT solves this by constructing a sequence of likelihood ratios that form a non-negative martingale.

    Mathematical Theory and Martingale Boundaries:
        Under the null hypothesis $H_0$ (no effect), the mixture likelihood ratio sequence $\Lambda_n$ is a martingale
        with expected value $E[\Lambda_n] = 1$. By **Doob's Martingale Inequality**, the probability that the likelihood
        ratio ever exceeds a critical threshold $1/\alpha$ at *any* point in the infinite sequence is strictly bounded by $\alpha$:
        $$P\left( \sup_{n \ge 1} \Lambda_n \ge \frac{1}{\alpha} \right) \le \alpha$$
        This properties allows experimenters to continuously monitor ("peek at") the data in real-time. If the likelihood
        ratio crosses the boundary, they can stop the experiment immediately with a mathematically guaranteed Type I error
        rate controlled at $\alpha$.

    Likelihood Ratio Formulation ($\Lambda_n$):
        For a continuous metric with cumulative sample variance $\sigma^2$ and a normal mixing distribution
        $H(\theta) = \mathcal{N}(0, \tau^2)$ over the expected effect size $\theta$, the mixture likelihood ratio
        at accumulated sample size $n$ is calculated as:
        $$\Lambda_n = \sqrt{\frac{\sigma^2}{\sigma^2 + n\tau^2}} \exp \left( \frac{n^2 \bar{Y}_n^2 \tau^2}{2\sigma^2(\sigma^2 + n\tau^2)} \right)$$
        where:
        - $\bar{Y}_n$: The observed sample mean difference between treatment and control at step $n$.
        - $\sigma^2$: The estimated daily/unit baseline variance of the metric.
        - $\tau^2$: The mixing parameter (tuning parameter) representing the prior variance of the expected effect size
          (usually selected to match the Minimum Detectable Effect, MDE).

    Attributes:
        alpha (float): The target Type I error rate (e.g., 0.05).

    Example:
        >>> # Monitoring a live test with alpha = 0.05
        >>> rules = StoppingRules(alpha=0.05)
        >>> # If the calculated likelihood ratio lambda_value crosses 1/alpha = 20.0, we stop early.
        >>> rules.check_msprt_stop(lambda_value=25.4)
        True
    """

    def __init__(self, alpha: float = 0.05):
        """Initializes StoppingRules.

        Args:
            alpha (float): The desired false-positive probability bound (Type I error rate). Defaults to 0.05.
        """
        self.alpha = alpha

    def check_msprt_stop(self, lambda_value: float) -> bool:
        r"""Determines if the mSPRT likelihood ratio exceeds the safety stopping bounds.

        Args:
            lambda_value (float): The calculated mixture likelihood ratio ($\Lambda_n$) for the active metric.

        Returns:
            bool: True if the likelihood ratio exceeds the martingale boundary ($1/\alpha$), indicating
                that the null hypothesis can be rejected and the experiment can be stopped immediately.
        """
        # Boundaries typically equal 1 / alpha
        boundary = 1.0 / self.alpha
        return lambda_value > boundary

