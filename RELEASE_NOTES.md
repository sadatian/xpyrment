# Release Notes - v1.2.0.0

## 🚀 New Features & High-Performance Optimizations
- **Profiling Suite**: Added `xpyrment.profiler` submodule for lightweight memory and execution profiling.
- **High-Performance NumPy Vectorization**: Rewrote mock A/A split validation and multi-armed bandit arm selection (Thompson Sampling & UCB1) to use vectorized NumPy broadcasting, drastically reducing simulation runtimes.
- **SciPy Optimization for Quasi-Experiments**: Replaced custom Python gradient descent with `scipy.optimize.minimize` (L-BFGS-B) for Propensity Score Matching. Completely vectorized the nearest-neighbor matching algorithm using `scipy.spatial.distance.cdist`.
- **FDR Control**: Integrated Benjamini-Hochberg False Discovery Rate control diagnostics into A/A testing validation.
- **Tamper-Proof Audit Trails**: Added local SQLite-backed replication and prepared digital signature payloads for `AuditTrail`.
- **Factorial ANOVA Integration**: `run_factorial_anova` now leverages `statsmodels` OLS and `anova_lm` functions.

## 🐛 Bug Fixes & Algorithmic Hardening
- **SRM Diagnostic Stability**: Guarded Sample Ratio Mismatch (SRM) chi-square goodness-of-fit routines against zero-counts and low-expected-count boundary conditions.
- **Degenerate Bootstrap Handlers**: Added intermediary guards to the BCa non-parametric inference engine to safely catch and collapse completely degenerate (zero-variance) bootstrap resample distributions, preventing math domain failures on sparse binary vectors.
- **Design of Experiments Strict Validation**: Embedded strict configuration guards inside `FractionalFactorialDesign` (requiring exactly 2 levels) and `DefinitiveScreeningDesign` (requiring exactly 3 levels) to reject invalid factorial setups before matrix mapping.
- **CUPED Singular Covariate Protection**: Guarded CUPED variance reduction adjustments against near-zero covariate variance configurations (`Var(X) ~ 0`). Both continuous and ratio metrics now safely fall back to unadjusted Difference-in-Means / unadjusted Delta Method calculations rather than suffering from numerical floating-point instability.

## 📦 Installation
```bash
pip install xpyrment==1.2.0.0
```
