# Changelog

## [1.3.0.0] - 2026-05-14

### Added
- **Multivariate Covariate Balance Diagnostics** (`validate/balance.py`): Integrated **Mahalanobis Distance** for joint multi-feature balance verification and **Kolmogorov-Smirnov (KS)** distance tests for full distribution shape alignment beyond standardized mean differences.
- **Gamma-Poisson Bayesian Model** (`analyze/inference/bayesian.py`): Implemented conjugate Gamma-Poisson posterior estimation for discrete count-based metrics (e.g., sessions, clicks).
- **Exact Numerical Integration for Bayesian Inference** (`analyze/inference/bayesian.py`): Replaced Monte Carlo approximations with high-precision numerical quadrature (via `scipy.integrate.quad`) for calculating **Probability of Being Best (PBB)** and **Expected Loss**.
- **Multi-Lag Carryover Modeling** (`design/doe/carryover.py`): Extended `CarryoverDecomposition` to support multiple historical lags ($T_{t-2}, T_{t-3}$, etc.) with distinct decay vectors.
- **Profile Likelihood Confidence Intervals** (`design/doe/carryover.py`): Added a robust profile likelihood fallback solver for joint asymptotic confidence intervals of decay parameters.

### Changed
- Promoted package version to `1.3.0.0`.
- Optimized Bayesian integration stability by using dynamic finite limits based on posterior distribution spread, preventing numerical failures in narrow distributions.
- Consolidated temporary Block T test suites into core test files (`test_balance.py`, `test_analysis.py`, `test_design.py`).

### Fixed
- Resolved attribute naming conflicts in `CarryoverDecomposition` to support vectorized multi-lag results.
- Fixed numerical precision issues in Bayesian PBB calculations for highly concentrated normal distributions.
