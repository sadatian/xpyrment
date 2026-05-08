# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-05-08

### Added
- **Centralized Telemetry & Profiling Engine** (`core/telemetry.py`): Structured JSON logging formats and a robust context manager / decorator execution profiler using high-resolution performance counters (`perf_counter`) and trackmalloc.
- **Robust Outlier & Input Validation Guard** (`validate/clean.py`): Advanced `clean_array` and collinearity verification checks leveraging Singular Value Decomposition (SVD) matrix rank computations.
- **Standalone Interactive Report Generator** (`report/generator.py`): Beautiful, responsive HTML dashboard generation utilizing premium CSS design structures (harmonic HSL colors, modern typography, grid alignment) along with detailed GitHub-compatible Markdown summary tables.
- **Polished Command Line Interface (CLI)** (`cli.py`): Multi-command console application supporting analytical power analysis, CSV covariate balance evaluations (standardized mean differences), and analytical OLS regressions.
- **High-Performance Resampling Bootstrap** (`analyze/inference/bootstrap.py`): Highly vectorized percentile and Bias-Corrected and Accelerated (BCa) bootstrap interval estimations backed by robust safety checks and 10,000,000 element batch memory chunk caps.
- **Unified Fluent Orchestrator** (`core/experiment.py`, `analyze/orchestrator.py`): Registered multi-metric and baseline covariate setups via topological metric registries, enabling automated covariate balance checking (`res.love_plot()`) and covariate-adjusted CUPED routing.
- **Runnable Showcase Demo Script** (`examples/run_experiment_lifecycle.py`): A complete, end-to-end simulated digital experiment lifecycle showing sample sizing, balancing checks, and CUPED variance reduction.

### Changed
- Promoted package version to stable release `1.0.0` in both `_version.py` and `pyproject.toml`.
- Upgraded and structured our `mkdocs.yml` configurations to utilize theme slate-dark / light toggles and code block features.

### Verified
- Executed continuous verification audits — all 138/138 unit tests are passing completely green.
- Verified docstring math formatting and compiled our clean mkdocs site with zero errors.

---

## [0.1.0] - 2026-05-06

### Added
- **Randomization & Stratification Engine** (`design/splits.py`, `design/stratification.py`): Hashing assignments based on MurmurHash3, cluster randomizations, and stratified partitions.
- **Classical DoE Submodules** (`design/doe/`): Robust factorial design configurations, Definitive Screening Designs (DSD), Taguchi Orthogonal Arrays, Central Composite Designs (CCD), and coordinate exchange D-Optimal algorithms.
- **Statistical Inference Engines** (`analyze/inference/`): Welch's t-test, Mann-Whitney U, Delta method ratio variance estimations, and conjugate Bayesian models (Beta-Binomial, Normal-Normal, Gamma-Poisson).
- **Diagnostics & Validate Checks** (`validate/`): Retrospective Pearson Chi-Square Sample Ratio Mismatch (SRM) checks, Standardized Mean Difference (SMD) covariate metrics, and time-series novelty/primacy effect models.
- **Adaptive Multi-Armed Bandits** (`bandit/`): Thompson Sampling, UCB1, and Epsilon-greedy bandits with radial basis function Gaussian Process Bayesian optimizers.
- **Quasi-Experimental Classifiers** (`quasi/`): Double Machine Learning (DML), Difference-in-Differences (DiD), Synthetic Difference-in-Differences (SDID), and Synthetic Controls.
- **Symmetric Encryption & Security** (`network/federated.py`, `network/privacy.py`): Homomorphic Paillier encryption, SMPC federated variance pools, and differential privacy noise addition filters.
- **Immutable Audit Trails** (`report/audit.py`): Cryptographically chained, state-gating ledger tracking transitions from `CREATED` to `REPORTED`.
