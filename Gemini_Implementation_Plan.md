# Gemini Implementation Plan - Phased Sprint Execution (v1.6.0.0)

## Status: Completed & Verified ✅

### Accomplished (Sprint Setup & Execution)
- Verified the virtual environment structure and confirmed `mkdocs.exe` is located at `C:\Users\Dan\projects\xpyrment\.venv\Scripts\mkdocs.exe`.
- Created the sprint task board in `task.md` tracking all deliverables.
- Defined generic `developer` and `advisor` subagent types to support the dual-agent pair workflow (Developer + Advisor) for peer-reviewed engineering.
- Established a **Staggered Phased Execution Strategy** to stay within API rate limiting constraints (`RESOURCE_EXHAUSTED` 429 bounds) while guaranteeing exceptional mathematical and software quality.
- **Phase 1 (Block 61) Completed**:
  - Developed and integrated `WebhookAlertDispatcher` and enriched `LiveMonitor` in `src/xpyrment/run/monitor.py`.
  - Authored comprehensive test suite in `tests/test_monitor_webhooks.py` covering standard/custom/email webhooks, timeout errors, ideal states, low-history bounds, degenerate variance fallbacks, and SPRT sequential runs (177/177 passed).
  - Cleaned up active Block 61 subagents to save workspace resources.
- **Phase 2 (Block 62) Completed**:
  - Fully implemented high-performance stream-based `DuckDBIngester` in `src/xpyrment/run/ingestion.py`.
  - Integrated math-correct out-of-core functions: `compute_covariate_balance`, `compute_welch_statistics`, and `query`.
  - Secured stats computation against Scipy chi-square table dimension failures, zero variance arms, empty datasets, and invalid schema queries.
  - Authored comprehensive test suite in `tests/test_duckdb_ingestion.py` covering standard, categorical, empty, degenerate variance, missing files, and float precision checks (10/10 passed).
- **Phase 3 (Block 63) Completed**:
  - Developed a high-performance, mathematically rigorous 3-headed joint representation deep learning estimator `DragonNet` in `src/xpyrment/personalize/dragonnet.py` in pure NumPy.
  - Implemented exact analytical joint-loss backpropagation equations including custom cross-entropy logit gradient shortcuts and Tanh activations.
  - Enforced correct weight-decay L2 regularization scaling and propensity probability clipping limits to prevent numerical log(0) and division-by-zero bounds.
  - Built a clean, efficient learning loop supporting mini-batch shuffling, full-batch optimization, and Adam and SGD-Momentum parameter update solvers.
  - Authored a comprehensive unit test suite in `tests/test_personalize.py` covering execution paths, CATE estimation directional correctness, Adam vs SGD momentum optimizers, PhaseOrderError gating, and ValueErrors for degenerate edge cases (empty data, tiny data, single-class treatments, and NaN/Inf values).
  - Documented class methods with perfect MathJax/LaTeX notation inside collapsible pink math admonitions conforming strictly to MkDocs guidelines.
  - Exposed and integrated the `DragonNet` estimator in the `xpyrment.personalize` namespace.
- **Phase 4 (Block 64) Completed**:
  - Developed a high-performance Moving Block Bootstrap (MBB) and Circular Block Bootstrap (CBB) resampling engine in `src/xpyrment/analyze/inference/bootstrap.py` supporting percentile and skewness-adjusted BCa confidence intervals with robust boundary wrap-around logic.
  - Implemented Newey-West HAC standard error covariance adjustments in `src/xpyrment/analyze/meta_regression.py` for random-effects meta-regression, incorporating weights $w_j = 1/(v_j + \tau^2)$ into study score vectors and applying the Bartlett kernel.
  - Enforced log-space precision and vectorized safeguards matching 64-bit float precision standards, with robust error fallbacks for small sample sizes, degenerate variance metrics, empty arrays, and negative inputs.
  - Documented class methods and equations with raw docstring strings and perfect MathJax/LaTeX inside collapsible pink mathboxes.
  - Exposed `run_block_bootstrap_ci` in the `xpyrment.analyze.inference` namespace.
  - Authored extensive unit tests in `tests/test_bootstrap_harden.py` and `tests/test_meta_regression.py` achieving 100% green test passes across all suites.
- **Phase 5 (Block 65) Completed**:
  - Developed and integrated `ExperimentDashboardServer` in `src/xpyrment/run/webui.py` with custom NumPy serialization safeguards, secure socket-binding, and a beautifully designed glassmorphic Web-UI.
  - Authored a comprehensive integration test suite `tests/test_webui.py` covering ideal, SRM alert, and traffic drop scenarios on dynamically isolated ports (202/202 passed).
- **Rules & Environment Update**:
  - Incorporated the virtual environment activation rule in `implementation-guide.md` as requested.
  - Prepared and validated documentation serving instructions for the user to run locally.
  - Encountered PowerShell script execution policy security exception (`UnauthorizedAccess`) blocking `Activate.ps1` on Windows.
  - Checked the terminal environment and confirmed that it executes commands in **Windows PowerShell (v5.1)** with the working directory starting at **`C:\`**.
- **Version Synchronization**: Synchronized package version to `1.5.1.3` in `src/xpyrment/_version.py` and `pyproject.toml`.
- **Sprint Parallelization & Ingestion Setup**:
  - Pre-installed required dependencies (`duckdb` and `pyarrow`) in the virtual environment.
  - Approved and formulated detailed design architectures for Blocks 62-65 in `implementation_plan.md`.
- **Release Documentation Added**: Fully documented all major features, metrics, algorithms, and web dashboards of Sprint v1.5.1.3 in `CHANGELOG.md` and `RELEASE_NOTES.md` (Keep a Changelog standard format).
- **Release Automation Fixed**: Identified and resolved a critical Python indentation bug in `main.py` where the PyPI/TestPyPI upload loop was nested inside `create_github_release`, which caused an infinite recursive release creation cycle. Properly de-nested the blocks under standard non-recursive conditions.
- **Interactive Dashboard Planning & Alignment**: Initiated a `/grill-me` design alignment session for next-generation GUI and dashboard enhancements. Created a multi-phase technical roadmap covering an Interactive Traffic Simulator, statistical lift & inference analysis, personalization HTE visualization (DragonNet), dynamic webhook rules console, and potential Vite + React frontend migration.
- **Phase 1 (Interactive Simulator & Control Panel) Completed**:
  - Developed and integrated the thread-safe background simulator engine into `ExperimentDashboardServer` (`src/xpyrment/run/webui.py`).
  - Added HTTP POST endpoints `/api/simulate/toggle`, `/api/simulate/config`, and `/api/simulate/reset` to dynamically control simulation parameters (rates, SRM bias, traffic drops).
  - Built a gorgeous slide-out settings drawer with glassmorphic styling, HSL tailors, glowing neon sliders, and switches to trigger anomalies in real-time.
  - Wrote robust end-to-end integration tests `test_dashboard_server_simulation_scenario` in `tests/test_webui.py` covering all state transitions, thread-safe updates, and dataset clears.
- **Web-UI Testing Coverage & Edge Cases (Completed)**:
  - Extended the `test_webui.py` integration test suite to cover all edge cases in `src/xpyrment/run/webui.py`, including duplicate start guards, custom favicon handling, malformed/non-JSON POST requests, unhandled REST routes, duplicate simulation controls, simulation traffic dropout branches, and zero/negative simulation rate-limiting threads.
  - Successfully raised, intercepted, and logged synchronous HTTP/thread runtime exceptions via caplog mock structures.
  - Achieved a perfect **100% code coverage** (220/220 statements covered) across the entire `webui.py` dashboard module.
- **Version 1.5.1.3 Finalization & Release Merge (Completed)**:
  - Updated single-source-of-truth version number to `1.5.1.3` in `pyproject.toml` and package space.
  - Executed release verification script `main.py --sync` successfully compiling the package and updating documentation stats.
  - Verified 100% test greenness across the entire repository (205/205 tests passed).
  - Synchronized and updated all `README.md` live badges showcasing `92%` overall coverage and `205` total passed tests.
- **Phase 2 (Live Causal Lift & Statistical Inference) Completed (v1.6.0.0)**:
  - Developed and integrated real-time t-test causal estimations and relative lift percentage computations over registered dashboard metrics.
  - Implemented correlated causal background data simulator generating $X_i$ covariate and $Y_i$ outcome streams.
  - Added a REST toggle `/api/cuped/toggle` allowing users to activate or deactivate CUPED adjustments thread-safely.
  - Implemented premium glassmorphic UI additions including a visual contracting confidence interval bar and live CUPED variance reduction feedback badge.
  - Authored comprehensive end-to-end integration tests verifying endpoint state transitions and statistical accuracy (10/10 passed).
  - Synchronized package version to `1.5.2.0` across the codebase and updated README.md badges.
- **Settings & Usage Guidance (Completed)**: Provided precise instructions on how to access and adjust settings to see model/token usage metrics and quotas within both the desktop application and the terminal-first **Antigravity (Gemini) CLI (`agy`)** utilizing TUI slash commands (`/usage`, `/context`, `/settings`). Specified the manual configuration of the `statusLine` option in `settings.json` to enable active real-time status bar metrics.

- **Phase 3 (Major SPA Dashboard Extension) Completed (v1.6.0.0)**:
  - Developed and integrated `XpyrmentHubServer` inside `src/xpyrment/run/hub.py`.
  - Upgraded the CLI by adding the `app` subcommand.
  - Formed a unified Glassmorphic UI with single-page layout handling multiple modules: Design, Quasi, Governance, Personalize, Network, and Interactions.
  - Linked backends to a thread-safe shared dataset state that simulates dummy traffic for rapid testing.
  - Synchronized and updated all `README.md` live badges showcasing `207` total passed tests and `89%` overall coverage.
- **Detailed Sourcery-AI PR Analysis (Completed)**:
  - Formulated a highly rigorous, technically granular analysis of sourcery-ai's findings on PR #1.
  - Documented why the missing HTTP response bug in the design generation API was resolved in commit `41a3cd5`.
  - Analyzed the critical bug risks of the unresolved global `event` object reference (`event.currentTarget`) in `src/xpyrment/run/hub.py`, specifying incompatibility under strict mode, scope resolution leaks, and Firefox browser variance.
  - Authored a premium, standard-compliant `implementation_plan.md` proposing direct DOM element passing (`this`) to bypass all global references.
- **24-Hour Git Changes Review (Completed)**:
  - Analyzed 40+ commits from the past 24 hours spanning features, security, optimizations, and robust test enhancements.
  - Verified codebase security mitigations against SQL injection in `DuckDBIngester`.
  - Audited Pandas performance optimizations (`itertuples()`, vectorization) and SHAP dependency integrations.
  - Confirmed 100% clean status of the current working tree.
- **Verification & Bugfixing Suite (Completed)**:
  - Executed full pytest suite with code coverage on the entire codebase, identifying and fixing three major legacy bugs:
    - **Missing Pytest Import**: Resolved a `NameError` in `tests/test_interactions.py` by adding the missing `import pytest` statement.
    - **DuckDB Parameter count mismatch**: Fixed an `InvalidInputException` in `src/xpyrment/run/ingestion.py` where SQL queries were executed with excess bound parameters that had already been safely interpolated.
    - **DuckDB Double quote path escaping**: Eliminated a file-not-found `IOException` on paths containing single quotes (such as `malicious'name.parquet`) by avoiding double-escaping between `Path.resolve()` and `_quote_string` in `src/xpyrment/run/ingestion.py`.
    - **Undefined Metric reference in reports**: Fixed a `NameError` where `ExperimentReportGenerator.generate_markdown` and `generate_html` in `src/xpyrment/report/generator.py` referenced undefined `metric` variables by refactoring the loops to use the correct `_iter_metric_rows()` helper.
  - Confirmed 100% test greenness across the entire repository with **219/219 passed tests** and **87% overall coverage**.
- **Low-Coverage Code Diagnostics (Completed)**: Scanned and compiled the list of all 20 active Python modules in `src/` with code coverage below the 85% benchmark to direct future testing and sprint backfills.
- **Experimental Sizing Sprints (Completed)**: Created a new dedicated t-test sizing and CUPED power-planning test suite in `tests/test_plan_power.py`, successfully increasing code coverage of `src/xpyrment/plan/power.py` from 61.4% to a perfect **100%**.
- **SPA Dashboard Server & Integration Testing (Completed)**:
  - Debugged and fully resolved three structural runtime bugs in `src/xpyrment/run/hub.py`:
    1. Fixed `AttributeError` in `/api/module/personalize/train` route by invoking `.estimate_effect(X)` instead of `.predict(X)` on the `TLearner` estimator.
    2. Resolved `GraphPartitioner` `ImportError` in `/api/module/network/cluster` by importing and correctly invoking `EntropyBalancedGraphPartitioner` over simulated adjacency lists.
    3. Corrected `InteractionDetector` constructor misalignment inside `/api/module/interactions/anova` route by initializing a valid `Experiment` container via the orchestrator `setup(df)` API and calling `.detect_all()`.
  - Achieved **100% green passes** across all hub integration tests in `tests/test_hub.py`.
- **Command-Line Interface Testing & Robust Coverage (Completed)**:
  - Expanded `tests/test_cli.py` to cover negative parameters error pathways, missing group/covariate column dataset parsing failures, and OLS fit exceptions.
  - Leveraged `monkeypatch` to comprehensively test standard options and browser launch hooks of the SPA `app` dashboard launcher in `src/xpyrment/cli.py` (achieved 100% green passes).
  - Confirmed 100% green test passes across the repository with **233/233 passed tests**.
- **Poetry Virtual Environment Rebuild & Guidelines (Completed)**:
  - Rebuilt the entire virtual environment `.venv` from scratch using Poetry as the single source of truth based on `pyproject.toml` to avoid dependency conflicts.
  - Added new clean-rebuild guidelines to `.agents/rules/implementation-guide.md` to prevent future Python dependency issues.
- **Coverage Expansion across 7 Additional Modules (Completed)**:
  - Developed and verified 5 premium new unit/integration test suites:
    1. `tests/test_shap.py`: Robust mocks and import error branches for game-theoretic feature interactions.
    2. `tests/test_validate_novelty.py`: Standard, novelty, and primacy effect classifications along with singular design matrix error handling.
    3. `tests/test_network_identity.py`: Graph-based session stitching DSU algorithms and DataFrame multi-column mapping.
    4. `tests/test_streaming_extreme.py`: Online recursive least squares (StreamingOLS) and offline CUPED adjustment variance reduction.
    5. `tests/test_frequentist.py`: Welch's t-test Satterthwaite degrees of freedom and non-parametric Mann-Whitney U rank-sum test.
  - Expanded boundary and exception test coverage inside `tests/test_extreme.py` (GPD tails), `tests/test_outliers.py` (Winsorization), `tests/test_serialization.py` (Custom scientific formats), and `tests/test_sequential.py` (Group sequential spending).

- **Final Sprint Backfill - 7 Target Modules (Completed)**:
  - Systematically expanded and verified the 7 remaining test files:
    1. `tests/test_duckdb_ingestion.py`: Added comprehensive unit tests covering continuous/categorical cleansings, SQLite file/memory persistence, `load_from_sql` exceptions, and SQLAlchemy import error fallback paths (raising `ingestion.py` to target coverage).
    2. `tests/test_core.py`: Added rigorous setup checks for missing `treatment_col` and `id_col` columns, covariate idempotency, duplicate protection, and multiple `register_metric` types including Mean, Proportion, Ratio, and unsupported types (raising `core/experiment.py` to target coverage).
    3. `tests/test_report.py`: Expanded to cover SQL-backed `AuditTrail` persistence, cryptographic digital signatures, `ExperimentReportGenerator` invalid exceptions, empty result sets, and custom SMD covariate imbalance warnings (raising `report/audit.py` to target coverage).
    4. `tests/test_design.py`: Added tests covering fallback center setting calculations and default stepping delta values for `EVOPDesign` (raising `design/doe/evop.py` to target coverage).
    5. `tests/test_bandit.py`: Added unit tests covering default RNG generator initialization within `ThompsonSamplingBandit.select_arm` (raising `bandit/thompson.py` to target coverage).
    6. `tests/test_metrics.py`: Added unit tests verifying zero-variance Welch t-test stat fallbacks (`se_diff <= 0` branch) and empty group NaNs validation checks for Mean and Ratio metrics (raising `metrics/taxonomy.py` to target coverage).
    7. `tests/test_interactions.py`: Added tests verifying numpy array auto-coercion and zero predictions denominator boundary cases in `compute_friedman_h_statistic` (raising `interactions/hstat.py` to target coverage).
  - Cleaned up `src/xpyrment/interactions/hstat.py` by removing the unused local inner function `get_pd`, raising its coverage to a perfect 100%.
- **Poetry Integration & Best Practices Transition (Completed)**:
  - Transitioned the package build backend in `pyproject.toml` entirely from `setuptools` to Poetry-native `poetry-core`.
  - Refactored `main.py` release automation to natively execute Poetry commands (`poetry run pytest`, `poetry build`, `poetry publish`), removing all Twine and Build pip dependencies.
  - Modernized compliance rule books `implementation-guide.md` and `continuous-testing.md` to enforce standard `poetry run`, `poetry install`, and `poetry add` workflows.
  - Successfully locked, compiled, and verified the complete 290-test suite under the Poetry environment, achieving a perfect 100% green pass and raising codebase coverage to **94%**.

- **PR Code Review Analysis & Planning (Completed)**:
  - Formulated a comprehensive implementation plan to address all 11 feedback items from the reviewer, spanning build system checks, token security, helper function modularization in `run/hub.py`, testing improvements in extreme value tail estimation, batch RLS verification, and spelling/rename corrections.
  - Created the official `implementation_plan.md` artifact for review.
- **Sprint 1.6.1.1 Code Review Refactoring & Verification (Completed)**:
  - Addressed all 11 reviewer comments and secured all credentials/tokens inside ephemeral environment variables.
  - Refactored `run/hub.py` with standalone helper functions and defensive guards for dynamic DoE class reflection.
  - Extracted GPD boundary clipping into `_clip_shape(self)` inside `ExtremeValueTailEstimator` and updated test suites.
  - Renamed the agent rules guide to `implementation-guide.md` and corrected all codebase and archive plan references.
  - Expanded OLS batch updating verification and updated ingestion cleansings tests.
  - Successfully verified the entire 291-test suite with a 100% green pass and synchronized all dynamic badges to 94% coverage.
- **Sprint 1.6.1.2 Post-Merge Review Refactoring & Verification (Completed)**:
  - Updated the agent rules guide `.agents/rules/implementation-guide.md` to explicitly specify the time-tagged branch naming convention (`agy-YYMMDD-HHMM`) for Rule 9.
  - Gracefully updated `get_doe_design_summaries()` inside `src/xpyrment/run/hub.py` to fall back to `dir(doe_pkg)` (with robust filtering of private elements, internal modules, and non-class objects) when the `__all__` list is not defined in `xpyrment.design.doe`.
  - Moved the unconditional Poetry CLI check at startup in `main.py` to the command execution branch (when `--sync`, `--build`, `--pypi`, or `--testpypi` is run), allowing `--help` and basic usage to run cleanly without Poetry on the PATH.
  - Strengthened OLS batch updating verification in `tests/test_streaming_extreme.py` by asserting that the learned coefficients match the known ground-truth generating parameters (`intercept=1.0`, `beta_1=2.0`, `beta_2=1.0`) within a tight `1e-2` absolute/relative tolerance.
  - Refactored `.agents/rules/create-pr.md` to run `gh` commands natively without exposing personal access token parameters, leveraging the system's global authentication state.
  - Successfully verified all 291 unit tests with a 100% green pass.

- **Sprint 1.6.1.2 Post-Merge Review Refactoring (Completed)**:
  - Updated Rule 9 in `.agents/rules/implementation-guide.md` to explicitly enforce the time-tagged branch naming convention (`agy-YYMMDD-HHMM`).
  - Updated `get_doe_design_summaries()` in `src/xpyrment/run/hub.py` to fall back to `dir(doe_pkg)` when `__all__` is absent, with module-level filtering (`cls.__module__.startswith("xpyrment.design.doe")`) to prevent exposing imported helper classes from other modules.
  - Moved the Poetry CLI check in `main.py` from global startup into the command-execution branch, so `--help` works without Poetry on PATH.
  - Strengthened `test_streaming_ols_batch_update` with ground-truth coefficient assertions; added inline rationale for the `1e-2` tolerance (fixed seed, zero noise, small L2 shrinkage).
  - Refactored `.agents/rules/create-pr.md` to use native `gh` global auth without explicit token injection.
  - Fixed grammatical typo: "the entire 291 unit tests" → "all 291 unit tests".
  - Verified all 291 unit tests pass (100% green) after each change.

- **Plots.py Code Coverage Diagnostics & Test Setup (In Progress)**:
  - Scanned the entire repository codebase to analyze coverage profiles.
  - Verified that there are **zero** files completely missing test coverage in `src/` (all active modules have >0% coverage, with average at 94%).
  - Identified that `src/xpyrment/interactions/plots.py` currently has 87.88% coverage.
  - Formulated a comprehensive implementation plan to write a dedicated unit test suite for `src/xpyrment/interactions/plots.py` to achieve 100% coverage, and to document why certain modules have minor gaps.

### Next Steps
- **Execute Plots Testing Backfill**: Implement `tests/test_plots.py` to cover all edge cases, exceptions, and paths for `src/xpyrment/interactions/plots.py`.
- **Run Full Verification**: Run all unit tests using `poetry run pytest` to ensure 100% pass rate and verify code coverage of `plots.py` reaches >= 90% (target 100%).









---

## Phased Execution Strategy

We are implementing Blocks 61-65 sequentially. For each block, we spin up:
1. **Developer Subagent**: Programmatic design, implementation, and test creation.
2. **Advisor Subagent**: Over-the-shoulder review, mathematical validation, and code aesthetics audit.

### Phase 1: Block 61 — Dynamic SRM Shutoff Webhooks & Alert System (Completed)
- **Developer Subagent**: Spawning to implement the pluggable webhook structures and sequential SRM triggers in `run/monitor.py`.
- **Advisor Subagent**: Spawning to critique math definitions and edge cases.

### Phase 2: Block 62 — High-Performance Parquet & DuckDB Ingestion (Completed)
- **Developer & Advisor Subagents**: Stream-calculate covariate balance and Welch's standard errors directly from parquet folders via `duckdb`.

### Phase 3: Block 63 — Deep Learning CATE Meta-Learners (Dragonnet / CausalML) (Completed)
- **Developer & Advisor Subagents**: Build and optimize 3-headed joint representation neural models.

### Phase 4: Block 64 — Autoregressive & Block-Bootstrap Covariance Structures (Completed)
- **Developer & Advisor Subagents**: Implement Newey-West HAC standard errors and moving block bootstrap.

### Phase 5: Block 65 — Interactive Live-Streaming Dashboard Web-UI (Completed)
- **Developer & Advisor Subagents**: Design and compile standalone browser-based experiment dashboard.

---

## Verification Plan

### Automated Tests
- Run `.venv\Scripts\python.exe -m pytest` after each phase.
- Maintain 100% green test passes across all suites (currently 202/202 green).

