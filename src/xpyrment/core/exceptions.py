"""Custom exception classes for the xpyrment core system.

This module defines the specific exception taxonomy used to enforce state-machine rules,
statistical validity checks, and experimental design constraints across the entire
lifecycle of digital and classical experiments.
"""


class PhaseOrderError(Exception):
    """Raised when an operation is performed in an invalid state/phase.

    This exception is a core mechanism of the phase-gated execution flow. It prevents:
    - Downstream actions (e.g., calling `.analyze()` or `.report()`) from being executed
      before upstream requirements (e.g., `.design()` or `.validate()`) are complete.
    - Upstream state reversals (e.g., transitioning back to `CREATED` or `PLANNED`
      once an experiment is already `RUNNING` or `ANALYZED`), which could lead to
      post-hoc configuration tampering or invalid statistical analysis.

    Mathematical & Operational Context:
        The experimental lifecycle is governed by a strict directed, non-cyclic graph
        of transitions:
        CREATED -> PLANNED -> DESIGNED -> RUNNING -> ANALYZED -> REPORTED.
        Any transition where index(target_state) < index(current_state) violates this
        unidirectional flow and triggers this error (with the exception of re-running
        the analysis on the same or new frozen data, which remains in the ANALYZED state).

    Attributes:
        message (str): Explains the invalid state transition attempt and the active state.

    Examples:
        ??? example "Example"

            ```python
            >>> from xpyrment.core.state import ExperimentState
            >>> from xpyrment.core.exceptions import PhaseOrderError
            >>> raise PhaseOrderError("Cannot transition backwards from RUNNING to PLANNED.")
            ```
    """
    pass


class SRMError(Exception):
    r"""Raised when a Sample Ratio Mismatch (SRM) is detected during validation.

    SRM occurs when the observed ratio of sample counts assigned to treatment arms
    significantly deviates from the pre-specified expected ratio (e.g., 50/50 split).
    An SRM is a critical indicator of data quality issues, selection bias, or bugs
    in the randomization/assignment mechanism.

    Mathematical Background:
        A Pearson Chi-Square Goodness-of-Fit test is performed to evaluate the discrepancy:
        $$
        \chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}
        $$
        where $O_i$ is the observed count in arm $i$ and $E_i$ is the expected count under
        the planned split. The degrees of freedom is $k - 1$.
        This exception is raised if the resulting p-value is extremely small (typically
        p < 0.001), indicating that the deviation is highly unlikely to have occurred
        by chance alone.

    Remediation Procedures:
        1. Halt the experiment or analysis immediately.
        2. Audit the randomization and unit assignment pipeline for bugs.
        3. Check for data loss, telemetry issues, or delay in event ingestion pipelines.
        4. Validate that the assignment tracking log captures all users on first touch.

    Attributes:
        message (str): Explains the observed vs. expected sample counts and the p-value.
    """
    pass


class AliasError(Exception):
    r"""Raised when fractional factorial alias confounding is violated or misconfigured.

    In fractional factorial classical Design of Experiments (DoE), only a fraction
    of all possible factor combinations is run. This results in confounding (aliasing),
    where the estimate of a specific main effect is mathematically indistinguishable
    from a multi-factor interaction term.

    This exception is raised when:
    - A user tries to estimate an effect that is completely confounded with another
      active effect of equal or lower order (violating the specified design Resolution).
    - The defined alias structure does not match the actual combinations present
      in the design matrix.
    - The design resolution (III, IV, or V) is insufficient to support the hypothesis or
      interaction analysis requested.

    Mathematical Context:
        Let $X$ be the design matrix and $C = (X^T X)^{-1} X^T Y$ be the parameter estimates.
        If the design is fractional, some columns of $X$ are linear combinations of others,
        leading to a rank-deficient matrix where unique solutions for all factors and
        interactions do not exist. The alias relation matrix $A$ defines which terms are
        confounded:
        $$
        E[\hat{\beta}_1] = \beta_1 + A \beta_2
        $$
        An AliasError prevents the system from proceeding with invalid or unresolvable
        confounding structures.

    Attributes:
        message (str): Details the confounded factors or resolution constraint violated.
    """
    pass
