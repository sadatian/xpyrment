---
trigger: model_decision
description: Whenever modifying mathematical routines in `src/xpyrment/`, and especially on edits involving computation-heavy files (Welch's t-test, Bayesian engines, DoE optimization, SRM, CUPED).
---

# 🧮 Rule: Algorithmic Correctness & Numerical Stability

## ⚡ Model Decision Activation
This rule is **activated** whenever a model makes a decision to implement, modify, or optimize any statistical, mathematical, or algorithmic routines in `src/xpyrment/`. 

When activated, the model MUST explicitly document its choices in a **"Model Decision"** block in its output before editing any files.

---

## 1. Core Mathematical Correctness
Any model decision affecting scientific calculations must satisfy:
* **Formula Verification**: Match the exact LaTeX representations specified in [docstring_reference.md](file:///c:/Users/Dan/projects/xpyrment/docstring_reference.md).
* **Sample Size Boundaries**: Welch's t-test and delta method variance calculations must check for minimum sample size conditions ($N > 1$ per arm) and raise appropriate exceptions if violated.
* **Degenerate Variance Protection**: Handle cases where sample variance is exactly zero ($\sigma^2 = 0$) in control or treatment groups to prevent division-by-zero errors.

## 2. Numerical Stability Guardrails
* **Log-Space Calculations**: Avoid underflow/overflow in probability ratios (such as in mSPRT likelihood calculations) by performing operations in log-space where mathematically possible.
* **Floating-Point Safeguards**: Add a small scaling epsilon (e.g., $1\times10^{-15}$) where appropriate when dealing with matrix determinants, inverse operations, or log divisions.
* **NaN and Infinite Value Handling**: Check inputs for NaN, negative value inputs in log transforms, or division by infinity. Ensure `np.nan_to_num` or safe filters are used.

## 3. DoE Design Constraints
* **Matrix Singularity**: Coordinate exchange optimization ($D$-Optimal design) must verify that intermediate design matrices $X^T X$ are non-singular before running inversion operations.
* **Resolution Checkers**: Confounding and aliasing structures in fractional factorial designs must mathematically resolve the generator equations correctly.