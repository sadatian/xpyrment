"""Sequential inference, continuous peeking, and always-valid confidence intervals (AVCI).

This module provides the `SequentialInference` class, which computes always-valid confidence intervals (AVCIs) and
sequential monitoring boundaries to allow safe, continuous visual exploration of experimental metrics over time.
"""


class SequentialInference:
    """Computes sequential monitoring bounds and always-valid confidence intervals (AVCIs).

    In traditional experimentation, looking at confidence intervals before the test finishes (peeking) is statistically
    hazardous. Always-Valid Confidence Intervals (AVCIs) solve this by providing a sequence of intervals that cover the
    true parameter $\\theta$ at *all* steps $n \\ge 1$ simultaneously with a probability of at least $1 - \\alpha$.

    Mathematical Definition and Boundary Formulas:
        Let $C_n$ be the confidence interval calculated at sample size $n$. For any nominal error rate $\\alpha \\in (0, 1)$:
        $$\\mathbb{P} \\left( \\forall n \\ge 1, \\ \\theta \\in C_n \\right) \\ge 1 - \\alpha$$
        This is an incredibly powerful property: it allows the experimenter to continuously plot the confidence interval
        over time, and if the interval does not contain zero at any point, the experiment can be stopped immediately
        with a guaranteed Type I error rate controlled at $\\alpha$.

        AVCI Derivation from mSPRT:
        By inverting the mixture Sequential Probability Ratio Test (mSPRT) statistic for a normally distributed metric with
        unit baseline variance $\\sigma^2$ and mixing tuning parameter $\\tau^2$ (representing prior effect variance), the
        always-valid confidence interval at step $n$ is:
        $$C_n = \\left[ \\bar{Y}_n - W_n, \\ \\bar{Y}_n + W_n \\right]$$
        where $\\bar{Y}_n$ is the observed sample mean difference, and the sequential margin of error $W_n$ is:
        $$W_n = \\sqrt{\\frac{2\\sigma^2(\\sigma^2 + n\\tau^2)}{n^2\\tau^2} \\ln\\left( \\frac{1}{\\alpha} \\sqrt{\\frac{\\sigma^2 + n\\tau^2}{\\sigma^2}} \\right)}$$
        
        Properties of $W_n$:
        - For very small $n$, $W_n$ is wider than the traditional fixed-sample Wald margin of error ($z_{1-\\alpha/2} \\sigma / \\sqrt{n}$),
          which mathematically penalizes and compensates for the continuous peeking.
        - As $n \\to \\infty$, the sequential margin $W_n$ asymptotically matches the rate of the traditional interval,
          ensuring zero long-term loss in statistical efficiency.
    """

    def calculate_always_valid_ci(self, sample_size: int, alpha: float) -> tuple:
        """Computes continuous monitoring boundaries to prevent alpha inflation from peeking.

        Calculates the exact sequential margin of error ($W_n$) at a given sample size, yielding
        always-valid confidence bounds around the observed effect size.

        Args:
            sample_size (int): The current accumulated number of units/trials ($n$).
            alpha (float): The target false-positive rate ($\\alpha$).

        Returns:
            tuple: A tuple of floats `(lower_bound, upper_bound)` representing the always-valid interval bounds
                relative to the mean difference.
        """
        # TODO: Implement sequential boundary functions (mSPRT or Pocock)
        return (-1.0, 1.0)
