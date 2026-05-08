---
trigger: model_decision
description: Whenever debugging numerical or statistical failures, modifying mathematical routines under `src/`, or adding new math solvers.
---

# 🧮 Rule: Algorithmic Correctness & Numerical Maintenance (v1 Era)

## ⚡ Model Decision Activation
This rule is **activated** whenever an AI agent makes a decision to debug, modify, or optimize any statistical, mathematical, or algorithmic routines in `src/xpyrment/`.

When activated, the agent MUST document its mathematical reasoning, debugging hypothesis, or numerical changes in a **"Model Decision"** block in its output before editing any files.

---

## 1. Mathematical Regression Prevention
Any bug fix or enhancement to the numerical engine must guarantee mathematical equivalence to the documented specifications:
* **Outcome Verification**: Verify that bug fixes do not shift existing test statistical results (such as posterior means, p-values, or degrees of freedom) unless correcting an explicitly diagnosed mathematical error.
* **Double-Precision Standards**: All intermediate statistical computations (sum of squares, covariance pools, matrix operations) must be executed in 64-bit float precision (`np.float64`) to prevent accumulative precision loss.
* **Degenerate Edge Cases**: Any algorithmic maintenance must protect against:
  - Singular covariance matrices in regressions.
  - Zero-variance arrays in frequentist and Bayesian models.
  - Sample sizes of $N \le 1$ in treatment arms.
  These must raise descriptive domain-specific errors (e.g. `ValueError`, `ZeroDivisionError`, `SingularMatrixError`) with helpful logging rather than unhandled library crashes.

---

## 2. Advanced Debugging & Intermediate Logging
When diagnosing a mathematical anomaly or statistical error:
* **Log Intermediate States**: Log key statistics at intermediate solver stages (e.g. the SVD singular values, standard errors, matrix determinants, or coordinate exchange iteration deltas) using our telemetry engine.
* **Tolerance Assertion Checks**: Use robust analytical comparison assertions (`np.allclose(a, b, rtol=1e-5, atol=1e-8)`) when verifying outputs against offline standard solutions.
* **Avoid Hard-coded Tolerances**: Ensure tolerances scale adaptively with data dimensions and variance magnitude.

---

## 3. Safe Numerical & Solvers Extensions
When implementing new mathematical solvers (e.g., for block bootstrapping, HAC standard errors, or non-linear decay estimation):
* **Vectorized Safeguards**: Write highly optimized vectorized operations using NumPy and SciPy; avoid slow Python loops over datasets.
* **Chunking Limits**: Calibrate memory-heavy calculations (such as large-scale bootstraps) to enforce active memory chunk limits (e.g., 10,000,000 elements per batch) to prevent memory spikes in resource-constrained container environments.
* **Log-Space Likelihoods**: Keep sequential SPRT likelihood calculations, probabilities of being best, and Bayesian expected loss integrations in log-space where mathematically possible to mitigate underflow/overflow anomalies.