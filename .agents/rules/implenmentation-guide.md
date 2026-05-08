---
trigger: model_decision
description: When debugging existing issues, adding new backward-compatible features, or updating tests and functionality within the stable v1 project.
---

# 🧭 xpyrment Implementation & AI Agent Rules Guide (v1 Stable Era)

This document defines the strict rules, workflows, and guidelines for maintaining, debugging, and incrementally extending the stable **`xpyrment`** Python package. Any AI agent modifying this repository must strictly adhere to these guidelines to preserve the integrity of our v1.0.0 release.

---

## 1. Core Principles of the v1 Era

With the release of v1.0.0, our development philosophy shifts from rapid greenfield execution to **production stability, predictability, and safety**.

1. **Absolute Backward Compatibility**: The public API surface (specifically `setup`, `run_analysis`, `Experiment`, `MeanMetric`, `ProportionMetric`, `RatioMetric`, `design_experiment`, `check_srm`, and CLI commands) is locked. Do NOT introduce breaking changes to signatures, expected types, or public return structures without explicit approval.
2. **Safe Feature Increments**: All new capabilities (such as those scheduled in Phase 2) must be designed as additive or opt-in components (e.g., via configuration flags, additive keyword arguments, or new submodules) to avoid disrupting existing workflows.
3. **Structured Debugging over Hot-patching**: Do not randomly mutate code to fix issues. Follow the systematic **Differential Diagnosis Sequence** detailed in Section 5.

---

## 2. Architectural & Dependency Rules

To prevent circular dependencies and spaghetti architecture, imports must follow our strict, one-way phase hierarchy. Downstream phases may import from upstream dependencies, but upstream components must remain completely ignorant of downstream implementations.

**Chain of Hierarchy**:
`metrics/` (leaf) $\rightarrow$ `core/` $\rightarrow$ `plan/` $\rightarrow$ `design/` $\rightarrow$ `validate/` $\rightarrow$ `run/` $\rightarrow$ `analyze/` $\rightarrow$ `interactions/` $\rightarrow$ `interpret/` $\rightarrow$ `report/` (terminal)

### Direct Import Permissions & Prohibitions
| Submodule | Supported Direct Imports | Prohibited Imports | Rationale / Constraints |
| :--- | :--- | :--- | :--- |
| **`metrics/`** | None (Leaf module) | `core/`, `plan/`, `design/`, ... | Global metric taxonomy. Must remain free of execution contexts. |
| **`core/`** | `metrics/` | `plan/`, `design/`, `analyze/`, ... | State machine, registry, exceptions. Free of math execution logic. |
| **`plan/`** | `core/`, `metrics/` | `design/`, `run/`, `analyze/`, ... | Hypotheses & sample-size calculators before setup. |
| **`design/`** | `core/`, `metrics/` | `run/`, `analyze/`, `interactions/` | Randomization structures, split patterns, DoE. |
| **`validate/`**| `core/`, `metrics/` | `run/`, `analyze/`, `interpret/` | Balance checks, AA tests, and SRM before analyzing. |
| **`run/`** | `core/`, `design/`, `validate/` | `analyze/`, `interactions/` | Assignment tracking, ingestion, stopping logic. |
| **`analyze/`** | `core/`, `metrics/`, `run/` | `interactions/`, `interpret/`, ... | Statistical inference, CUPED, corrections. |
| **`interactions/`**| `analyze/`, `design/` | `interpret/`, `report/` | Factorial interaction terms, ANOVA alias, SHAP. |
| **`interpret/`**| `analyze/`, `interactions/`, ... | `report/` | Subgroup scanners, economic decisions. |
| **`report/`** | **All phases** | None (Terminal consumer) | Consumes everything to compile exportable summaries. |

---

## 3. Experiment State Machine & Gating Rules

The central orchestrator `Experiment` (`core/experiment.py`) enforces strict state transitions via `ExperimentState` (`core/state.py`):
`CREATED` $\rightarrow$ `PLANNED` $\rightarrow$ `DESIGNED` $\rightarrow$ `RUNNING` $\rightarrow$ `ANALYZED` $\rightarrow$ `REPORTED`

### Phase Gating Requirements:
1. **Transition Flow**: State transitions can only advance sequentially. Regression to a previous state is prohibited.
2. **Phase Verification**: Before running operations of phase $X$, verify the current state is exactly the predecessor state (or already advanced to $X$). If violated, raise `PhaseOrderError` (defined in `core/exceptions.py`).
3. **Cryptographic Signatures**: The `ExperimentRegistry` (`core/registry.py`) computes SHA-256 signatures of specifications to prevent post-hoc changes (pre-registration validation).

---

## 4. Codebase Map & Functionality Index

Refer to `docstring_reference.md` for complete API signatures, parameters, and return types.

- **`core/`** (State Machine, Registry & Serialization)
  - `experiment.py` $\rightarrow$ `Experiment` class (the user-facing entrypoint and orchestrator).
  - `state.py` $\rightarrow$ `ExperimentState` (Enum: CREATED, PLANNED, DESIGNED, RUNNING, ANALYZED, REPORTED).
  - `exceptions.py` $\rightarrow$ `PhaseOrderError`, `SRMError`, `AliasError` (custom domain exceptions).
  - `registry.py` $\rightarrow$ `ExperimentRegistry` (SHA-256 config checksum hashing).
  - `serialization.py` $\rightarrow$ `to_dict` and `to_json` serialization wrappers.
  - `telemetry.py` $\rightarrow$ `ExecutionProfiler` and JSON telemetry logging.
- **`metrics/`** (Taxonomy & Normalization)
  - `taxonomy.py` $\rightarrow$ `BaseMetric`, `MeanMetric`, `ProportionMetric`, `RatioMetric` (implements CUPED logic inside `.calculate()`).
- **`plan/`** (Hypothesis & Power)
  - `power.py` $\rightarrow$ `PowerAnalysis` (computes MDE, power curves, required sample sizes).
- **`design/`** (Randomization & Design of Experiments)
  - `doe/` (DoE sub-package): full/fractional, definative (DSD), Taguchi, response surface, D-optimal exchange.
- **`validate/`** (Pre-analysis Diagnostics & Input Cleaning)
  - `clean.py` $\rightarrow$ Array validators, collinearity SVD checks, and size checks.
  - `srm.py` $\rightarrow$ Chi-square Sample Ratio Mismatch validation.
- **`analyze/`** (Statistical Inference Engine)
  - `orchestrator.py` $\rightarrow$ `AnalysisResult` (compiles summaries, generates forest plots).
- **`report/`** $\rightarrow$ `generator.py` (standalone HTML/Markdown reports), `audit.py` (cryptographic trails).

---

## 5. Strict AI Agent Debugging & Maintenance Rules

1. **Systematic Bug Diagnosis Sequence**:
   When debugging a regression or mathematical failure, do NOT edit production files immediately. Follow these exact steps:
   - **Step A: Capture State**: Collect and inspect input values, dimensions, variances, and state variables using our profiling logs.
   - **Step B: Isolate in Test**: Write a minimal failing unit test under `tests/` reproducing the exact bug (e.g., passing singular arrays or extreme values). Confirm that the test fails.
   - **Step C: Safely Correct**: Apply the correction in the target `src/` file.
   - **Step D: Regression Run**: Run the entire test suite to guarantee the fix did not break downstream dependencies.
2. **Additive-First Feature Integration**:
   When adding a new feature (e.g., from Phase 2):
   - Do not replace existing methods or change signatures.
   - Inject new features cleanly via opt-in parameters, separate utility functions, or isolated submodules.
3. **Preserve LaTeX & docstrings**:
   - Maintain all LaTeX mathematical notations in docstrings.
   - Ensure code edits do not introduce docstring format violations that break MkDocs builds.
4. **Strict Four-Digit Versioning & Compliance Rule**:
   All source code and visual changes must strictly adhere to the following 4-digit versioning schema (`Major.Minor.Patch.Revision`):
   - **Extremely small changes and recommits**: Increment revision by `+0.0.0.1` (e.g., `1.0.0.0` $\rightarrow$ `1.0.0.1`).
   - **Bug fixes**: Increment patch by `+0.0.1.0` and zero out any downstream digits (e.g., `1.0.0.1` $\rightarrow$ `1.0.1.0`).
   - **Features added**: Increment minor version by `+0.1.0.0` and zero out any downstream digits (e.g., `1.0.1.5` $\rightarrow$ `1.1.0.0`).
   - **Major releases**: Increment major version by `+1.0.0.0` and zero out any downstream digits (e.g., `1.1.2.3` $\rightarrow$ `2.0.0.0`).
   - *Rule of Reset*: Any increment of a higher-order digit **must** reset (zero out) all digits downstream of it.
5. **No Broken Skeletons**:
   - All newly added files, helpers, or hooks must be fully implemented, documented, and covered with unit tests before declaring the task finished.
6. **No Automated Documentation Serving**:
   - Do NOT run the `mkdocs serve` command directly on behalf of the user.
   - If a live-served documentation preview is required, only compile the documentation locally via `mkdocs build` to check for compilation issues, and ask the user to run `mkdocs serve` separately in their own terminal.