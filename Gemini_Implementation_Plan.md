# Implementation Plan: xpyrment Python Package (v1.0.0.1)

This is the active implementation plan for **`xpyrment`**—a highly modular, phase-gated library designed to support the entire lifecycle of industrial-scale digital experimentation and classical Design of Experiments (DoE).

* For the full, un-concentrated detail of blocks 1 to 60, see the [v0.1.0.0 Archive Implementation Plan](file:///c:/Users/Dan/projects/xpyrment/Gemini_Implementation_Plan_v0.1.0.0.md).

---

## 🏷️ Stable Era Versioning Protocol

This repository adheres to a strict 4-digit versioning system (`Major.Minor.Patch.Revision`) to reflect development scopes:
* **Extremely small changes and recommits**: Increment revision by `+0.0.0.1` (e.g., `1.0.0.0` $\rightarrow$ `1.0.0.1`).
* **Bug fixes**: Increment patch by `+0.0.1.0` and zero out any downstream digits (e.g., `1.0.0.1` $\rightarrow$ `1.0.1.0`).
* **Features added**: Increment minor version by `+0.1.0.0` and zero out any downstream digits (e.g., `1.0.1.5` $\rightarrow$ `1.1.0.0`).
* **Major releases**: Increment major version by `+1.0.0.0` and zero out any downstream digits (e.g., `1.1.2.3` $\rightarrow$ `2.0.0.0`).
* **Rule of Reset**: Incrementing a higher digit automatically resets (zeroes out) all digits downstream of it.

---

## 📋 Concentrated Phase 1 (Completed Foundation Blocks 1-60)

The foundational engine (comprising the initial 60 milestone blocks) is **100% completed, verified with 138/138 green unit tests, and packaged for stable v1.0.0 release**. Below is a concentrated mapping of our architecture:

### 1. Core State Machine, Registry & Infrastructure
* **State Machine & Phase Gating** (`core/state.py`, `core/experiment.py`): Restricts method calls according to valid phase boundaries (`CREATED` $\rightarrow$ `PLANNED` $\rightarrow$ `DESIGNED` $\rightarrow$ `RUNNING` $\rightarrow$ `ANALYZED` $\rightarrow$ `REPORTED`), throwing `PhaseOrderError` on violations.
* **Tamper-Evident Ledger** (`report/audit.py`): Cryptographically chains state updates via SHA-256 signatures: $h_k = H(t_k \parallel a_k \parallel d_k \parallel h_{k-1})$.
* **Cross-Device Resolution** (`network/identity.py`): DSU-based `IdentityRegistry` resolving session-stitching leaks in $O(\alpha(N))$ time.
* **JSON Serialization** (`core/serialization.py`): Unified recursive serialization (`to_dict` and `to_json`) for estimators and analysis results.
* **Profiling & Telemetry** (`core/telemetry.py`): Features JSON logging formatters and an dual-use `ExecutionProfiler` capturing peaks in `tracemalloc` memory and high-resolution durations.

### 2. Design of Experiments (DoE) & Multi-Armed Bandits
* **Randomization splits** (`design/splits.py`, `design/stratification.py`): Clean deterministic MurmurHash3 hashing, stratified partitions, and cluster assignments.
* **Classical DoE Schemes** (`design/doe/`): Factorial (full/fractional), definitive screening (DSD), Taguchi Orthogonal Arrays, EVOP step scheduling, central composite (CCD), D-optimal coordinate exchange, and Latin Hypercube Sampling.
* **Multi-Armed Bandits** (`bandit/`): Multi-arm adaptive exploration (Epsilon-Greedy, UCB1, Thompson Sampling) with GP surrogate Bayesian optimization tuning.
* **OPE & Non-Stationarity** (`bandit/ope.py`, `bandit/non_stationary.py`): Off-Policy evaluation (IPS, SN-IPS, Doubly Robust) and Sliding-Window / Discounted Thompson Sampling for drifting baselines.

### 3. Advanced Causal Inference & Quasi-Experiments
* **Synthetic Controls & Panels** (`quasi/`): Standard and Synthetic Difference-in-Differences (SDID), Abadie SLSQP-optimized synthetic control, and Nuclear-Norm panel matrix completion via Singular Value Thresholding (SVT).
* **Meta-Learners & DML** (`personalize/`): Closed-form S-Learner, T-Learner, Propensity-weighted X-Learner, and Double Machine Learning (DML) with multi-variable Ridge/ElasticNet and K-Fold cross-fitting.
* **Dynamic Treatment Regimes** (`personalize/dtr.py`): Two-stage backward induction Q-learning models for personalized sequential decisions.
* **Instrumental Variables** (`quasi/instrumental_variables.py`): Complier Average Causal Effects (CACE) modeled via analytical Two-Stage Least Squares (2SLS).

### 4. Large-Scale Analytics & Robustness Safeguards
* **Variance Reduction (CUPED)** (`analyze/variance_reduction.py`, `metrics/taxonomy.py`): Automated CUPED linear adjustments mapped to pre-period covariates, achieving up to $88\%+$ reduction in variance.
* **Statistical Inference Engines** (`analyze/inference/`): Welch's t-test, Wilcoxon, Delta-method ratio variance, and nonparametric vectorized BCa bootstrapping with memory chunk limits.
* **Sequential Peeking** (`analyze/sequential.py`, `analyze/srm.py`): Wald's sequential mSPRT always-valid confidence intervals and Lan-DeMets alpha spending functions.
* **Input Diagnostics & Cleaning** (`validate/clean.py`): SVD rank checking for collinearities, NaN/inf cleaning, and sample size warnings.

---

## 🚀 Phase 2 (v1.0.0.0 & Beyond: Advanced Enterprise Enhancements)

With the foundation locked down, the roadmap shifts to high-scale performance, real-time monitoring, and automatic integration hooks:

### 📌 Block 61: Dynamic SRM Shutoff Webhooks & Alert System
* **Goal**: Provide automated alert dispatch and dynamic shutoff triggers when sequential SRM or guardrail metric breaches are detected.
* **Technical Spec**: Implement pluggable webhook listener hooks inside the live monitor that dispatch payloads to external endpoints (e.g., Slack, Datadog, or PagerDuty) if Chi-square SRM $p < 0.001$.

### 📌 Block 62: High-Performance Parquet & DuckDB Streaming Ingestion
* **Goal**: Bypass in-memory Pandas dataframe bottlenecks for enterprise-scale multi-gigabyte datasets.
* **Technical Spec**: Integrate `duckdb` backend connections inside `run/ingestion.py` to stream-calculate covariate balance and Welch's standard errors directly from parquet folders or relational databases.

### 📌 Block 63: Deep Learning CATE Meta-Learners (Dragonnet / CausalML)
* **Goal**: Support non-linear high-dimensional heterogeneous treatment effect modeling.
* **Technical Spec**: Create a neural network estimator block using standard mathematical matrices to support multi-layer joint representations of treatment propensity and outcome surfaces.

### 📌 Block 64: Autoregressive & Block-Bootstrap Covariance Structures
* **Goal**: Correct confidence interval coverage for time-series experiments exhibiting high autocorrelation (such as continuous switchback designs).
* **Technical Spec**: Implement block-bootstrap resampling engines and Newey-West HAC standard errors on our meta-regression models to adjust for serial dependencies.

### 📌 Block 65: Interactive Live-Streaming Dashboard Web-UI
* **Goal**: Provide an interactive visual web interface to configure, track, and monitor running experiments in real-time.
* **Technical Spec**: Build a lightweight, standalone UI that displays live traffic distributions, running mSPRT p-values, and metric charts.

---

## 🛠️ Developer Workflow & Guardrails

To maintain high development quality, future implementations of Phase 2 blocks must strictly adhere to the following rules:

### 🔄 Rule A: Interactive Progress-Gating Update
* Update this file (`Gemini_Implementation_Plan.md`) immediately after the completion of every block with:
  1. What was accomplished.
  2. What must be done next.

### 🧪 Rule B: Continuous Validation Verification
* Proactively execute the test suite (`pytest`) and compile documentation (`mkdocs build`) after any code additions.
* Ensure zero compilation warnings or test failures before proceeding.

---

## 🔍 v1.0.0 Release Readiness Audit (Completed)

We conducted a comprehensive, production-level audit of the repository to identify and implement any missing requirements for a stable, professional v1.0.0 open-source release.

### a) What was accomplished:
1. **Added MIT LICENSE File**: Created a standard `LICENSE` file in the root matching the MIT License specified in the package metadata (`pyproject.toml`) and `README.md`.
2. **Added CHANGELOG.md**: Created a polished, standard-compliant changelog documenting our version history and complete progression from initial beta releases up to v1.0.0.
3. **Packaging & Wheel Validation**: Installed the standard Python `build` package and executed compilation; successfully generated `.tar.gz` and `.whl` files with zero packaging anomalies or metadata errors.
4. **Imports & Test Verification**: Verified silent importing of `xpyrment` namespace and complete coverage (all 138/138 tests passing 100% green).
5. **Adjusted Agent Rules for Maintenance Era**: Re-authored all `.agents/rules/` files to shift the focus from greenfield construction to systematic debugging sequences, backwards compatibility guarantees, and non-breaking, additive Phase 2 increments.

## 🎨 Logo & Icon Asset Integration (Completed)

We integrated the custom brand logos and multi-size favicons throughout the documentation and compiled HTML report generators.

### a) What was accomplished:
1. **Copied Assets**: Copied the brand icon assets (`favicon_16.png`, `favicon_32.png`, `favicon_64.png`, `xpyment_logo_wbg.png`, `xpyment_logo_wbg.svg`) from the build output directory to the source folders (`docs/assets/icons/` and `src/xpyrment/assets/icons/`).
2. **Packaged Package Data**: Configured `pyproject.toml` to declare `assets/icons/*` package-data so they are distributed inside python `.whl`/`.tar.gz` distributions.
3. **MkDocs Integration**: Integrated logo and favicon settings in `mkdocs.yml` and created `docs/overrides/main.html` to inject all favicon sizes (16x16, 32x32, 64x64, any vector) dynamically based on standard-compliant sizes.
4. **HTML Report Generator Branding**: Updated `ExperimentReportGenerator` inside `src/xpyrment/report/generator.py` to automatically load and inline the brand assets:
   - Smaller favicons (16x16, 32x32, 64x64) are base64-encoded as data URIs and injected into the HTML `<head>`.
   - The primary vector logo (SVG) is embedded raw in the `<header>` block for larger than 64px crisp layout.
5. **Testing Verification**: Added complete pytest coverage assertions inside `tests/test_report.py` ensuring successful asset extraction and injection. Built with zero compilation warnings and 100% test success.
6. **Verified Default Light Mode Theme**: Verified that `scheme: default` (light mode) is defined as the first palette block in `mkdocs.yml`, making it the default theme for all clean/new documentation visits.

## 🌟 Light Mode Standardization & Dense Status Badge Crowding (Completed)

We consolidated the documentation platform's styling to support Light Mode exclusively, linked repository metadata, added a dense, feature-rich array of status badges, and bumped the package minor revision.

### a) What was accomplished:
1. **Removed Dark Mode Theme Switcher**: Pruned dark palette configurations (`scheme: slate`) and toggles inside `mkdocs.yml` to lock the documentation portal strictly into Light Mode (`scheme: default`).
2. **Added Repository Metadata Links**: Registered `repo_name: sadatian/xpyrment` and `repo_url` in the top level of `mkdocs.yml`. This automatically activates a premium top-navigation GitHub button containing the repository icon and live info.
3. **Engineered Dense Status Badges**: Injected a comprehensive, professional, multi-row layout of Shields/Badges directly under the title in `docs/index.md` covering PyPI status, python versions, automated testing counts, coverage, licensing, build status, commit metrics, code-size, issues, release stability, cryptographic chaining stamps, and engine math protocols. This presents an extremely detailed and active overview.
4. **Enforced Revision Incrementing**: Bumped the release version up by `0.0.0.1` (from `1.0.0` to `1.0.0.1`) across both `pyproject.toml` and `src/xpyrment/_version.py` files.
5. **Testing**: Verified build compilation with `mkdocs build` and verified the complete 138/138 test suite runs cleanly.

## 🏷️ Establishment of Strict 4-Digit Versioning Protocol (Completed)

We established and formalized the custom 4-digit versioning protocol across all rules and planning guidelines.

### a) What was accomplished:
1. **Added to Implementation Plan**: Documented the strict 4-digit versioning rules (`Major.Minor.Patch.Revision`) in `Gemini_Implementation_Plan.md`.
2. **Added to Agent Rules Guide**: Injected the custom 4-digit rules under the "Strict Four-Digit Versioning & Compliance Rule" section in the `.agents/rules/implenmentation-guide.md` guidelines.
3. **Formalized Downstream Reset Rule**: Explicitly detailed the rule of reset where higher-order increments zero out downstream digits.

## 🚀 Pip Upgrade & PyPI Publishing Master Guide (Completed)

We upgraded pip in the active virtual environment and compiled a comprehensive, step-by-step master guide for publishing the package to PyPI.

### a) What was accomplished:
1. **Upgraded pip**: Ran python pip upgrade to install the latest pip version (`26.1.1`) in the active `.venv` environment.
2. **Created PyPI Master Guide**: Authored a detailed, secure, and modern guide in [add_to_pypi_guide.md](file:///C:/Users/Dan/.gemini/antigravity/brain/af0bb444-d791-4a8c-aab5-fc2c62b1052e/add_to_pypi_guide.md) detailing build compilation, TWINE checking, API Token setup, TestPyPI staging, and official live publishing.

### b) What must be done next:
1. Proceed with Phase 2 blocks, beginning with **Block 61** (Dynamic SRM Shutoff Webhooks & Alert System).

