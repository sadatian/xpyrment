# Changelog

## [1.6.1.0] - 2026-05-28

### Added
- **5 New High-Fidelity Test Suites**: Created dedicated unit/integration test suites for `shap.py`, `validate_novelty.py`, `network_identity.py`, `streaming_extreme.py`, and `frequentist.py` achieving near 100% code coverage.
- **Dependency Clean-Rebuild (Poetry Integration)**: Configured and migrated the local virtual environment to use Poetry as the single source of truth based on `pyproject.toml`, resolving dynamic dependency resolving conflicts. Added clean-rebuild guidelines to `.agents/rules/implenmentation-guide.md` to prevent future python dependency issues.

### Fixed
- **SPA Dashboard Server Bugs**: Fully resolved three key runtime bugs in `src/xpyrment/run/hub.py`:
  1. Fixed `AttributeError` in `/api/module/personalize/train` route by invoking `.estimate_effect(X)` instead of `.predict(X)` on `TLearner`.
  2. Fixed `GraphPartitioner` `ImportError` in `/api/module/network/cluster` by importing and correctly invoking `EntropyBalancedGraphPartitioner` over simulated adjacency lists.
  3. Resolved `InteractionDetector` constructor misalignment in `/api/module/interactions/anova` route by constructing a valid `Experiment` via the orchestrator `setup(df)` API and calling `.detect_all()`.
- **CLI & Statistical Edge Cases**: Fixed boundary/exception tests in `test_cli.py` (mocked app server startup), GPD tails in `test_extreme.py` (constant tail and MOM bounds safeguards), Winsorization bounds in `test_outliers.py`, and spent alpha calculations in `test_sequential.py`.

## [1.6.0.0] - 2026-05-27
### Added
- Created the primary `XpyrmentHubServer` dashboard.
- Introduced `app` subcommand to CLI to launch the interactive UI hub.
- Dashboard features fully-functional backends for Continuous Monitoring, Experimental Design, Quasi-Experiments, Personalization, Governance, and Interactions over a shared, live datastore.

## [1.5.2.0] - 2026-05-23

### Added
- **Live Causal Inference & Lift Analysis (Phase 2)**: Developed and integrated real-time Welch's t-test causal estimations and relative lift percentage computations over registered dashboard metrics.
- **Correlated Causal Background Data Simulator**: Redesigned the background simulation thread to generate continuous `metric_value` and `pre_value` (pre-period baseline covariate) logs in real-time ($X_i \sim \mathcal{N}(100.0, 15.0^2)$).
- **CUPED Variance Reduction REST Endpoint**: Added `/api/cuped/toggle` POST endpoint to control variance reduction dynamically and thread-safely.
- **Glassmorphic UI Causal Panel & Visual CI Contracting Bar**: Integrated horizontal CI visual boundary bars displaying real-time contraction when CUPED is activated, along with glowing relative lift percentages, statistical power, and significance badges.

### Changed
- Promoted package version to `1.5.2.0`.
- Added comprehensive unit and integration tests under `tests/test_webui.py` achieving 100% green test passes and 99% coverage on `webui.py`.

## [1.5.0.0] - 2026-05-22

### Added
- **Dynamic SRM Shutoff Webhooks & Alert System (Block 61)**: Implemented pluggable HTTP webhook listener hooks and alert triggers inside `LiveMonitor` (`src/xpyrment/run/monitor.py`) supporting Slack (rich Markdown notifications), Email (notification body format), and generic custom JSON POST webhooks, featuring complete fault isolation between registered handlers.
- **High-Performance Parquet & DuckDB Streaming Ingestion (Block 62)**: Engineered out-of-core streaming statistics computation in `DuckDBIngester` (`src/xpyrment/run/ingestion.py`) to process multi-gigabyte Parquet datasets. Implemented out-of-core Standardized Mean Difference (SMD), Pearson Chi-Square, and Welch's t-test stats completely bypassing RAM load limits.
- **Deep Learning CATE Meta-Learners (Block 63)**: Developed `DragonNet` (`src/xpyrment/personalize/dragonnet.py`), a mathematically rigorous 3-headed joint representation neural network in pure NumPy. Enforced joint optimization with L2 regularization, Tanh shared layers, and propensity score clipping boundaries ($10^{-7}$) to guard against confounding selection bias.
- **Autoregressive & Block-Bootstrap Covariance Structures (Block 64)**: Integrated Moving Block Bootstrap (MBB) and Circular Block Bootstrap (CBB) resampling engines in `src/xpyrment/analyze/inference/bootstrap.py` to preserve temporal dependencies, along with Knapp-Hartung and Newey-West HAC standard error covariance estimators in `MetaRegressor` (`src/xpyrment/analyze/meta_regression.py`) to correct for serial auto-correlations.
- **Interactive Live-Streaming Dashboard Web-UI (Block 65)**: Created `ExperimentDashboardServer` (`src/xpyrment/run/webui.py`) serving a thread-safe glassmorphic dark-mode live-monitoring dashboard with polling interfaces, live Chart.js/SVG fallback trends, sequential SPRT martingale pathing, and active anomaly alerts.

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
