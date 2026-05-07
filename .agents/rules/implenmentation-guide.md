---
trigger: model_decision
description: When working on files within `src` and `tests` folders, specifically with implementing algorithms and functionality within the primary project.
---

# 🧭 xpyrment Implementation & AI Agent Rules Guide

This document defines the strict architectural rules, module structures, and mathematical anchors for the **`xpyrment`** Python package. Any agent modifying this repository must strictly adhere to these guidelines to maintain design integrity and prevent regression.

---

## 1. Architectural & Dependency Rules
To prevent circular dependencies and spaghetti architecture, imports must follow a strict, one-way phase hierarchy. Downstream phases may import from upstream dependencies, but upstream components must remain completely ignorant of downstream implementations.

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

## 2. Experiment State Machine & Gating Rules
The central orchestrator `Experiment` (`core/experiment.py`) enforces strict state transitions via `ExperimentState` (`core/state.py`):
`CREATED` $\rightarrow$ `PLANNED` $\rightarrow$ `DESIGNED` $\rightarrow$ `RUNNING` $\rightarrow$ `ANALYZED` $\rightarrow$ `REPORTED`

### Phase Gating Requirements:
1. **Transition Flow**: State transitions can only advance sequentially. Regression to a previous state is prohibited.
2. **Phase Verification**: Before running operations of phase $X$, verify the current state is exactly the predecessor state (or already advanced to $X$). If violated, raise `PhaseOrderError` (defined in `core/exceptions.py`).
3. **Cryptographic Signatures**: The `ExperimentRegistry` (`core/registry.py`) computes SHA-256 signatures of specifications to prevent post-hoc changes (pre-registration validation).

---

## 3. Codebase Map & Functionality Index
Refer to `docstring_reference.md` for complete API signatures, parameters, and return types.

- **`core/`** (State Machine & Registry)
  - `experiment.py` $\rightarrow$ `Experiment` class (the user-facing entrypoint and orchestrator).
  - `state.py` $\rightarrow$ `ExperimentState` (Enum: CREATED, PLANNED, DESIGNED, RUNNING, ANALYZED, REPORTED).
  - `exceptions.py` $\rightarrow$ `PhaseOrderError`, `SRMError`, `AliasError` (custom domain exceptions).
  - `registry.py` $\rightarrow$ `ExperimentRegistry` (SHA-256 config checksum hashing).
  - `types.py` $\rightarrow$ Shared TypedDicts and structural dataclasses.
- **`metrics/`** (Taxonomy & Normalization)
  - `taxonomy.py` $\rightarrow$ `BaseMetric`, `MeanMetric`, `ProportionMetric`, `RatioMetric` (implements CUPED logic inside `.calculate()`).
  - `guardrails.py` $\rightarrow$ `GuardrailMetric` (monitors boundary violations and handles abort rules).
  - `transformations.py` $\rightarrow$ log-normalizations and delta approximations.
- **`plan/`** (Hypothesis & Power)
  - `hypothesis.py` $\rightarrow$ `HypothesisSpec` (directional hypotheses, alpha/beta levels, metric bindings).
  - `power.py` $\rightarrow$ `PowerAnalysis` (computes MDE, power curves, required sample sizes).
  - `duration.py` $\rightarrow$ Estimators for traffic requirements and run times.
  - `preregistration.py` $\rightarrow$ Serializes plans into immutable registration cards.
- **`design/`** (Randomization & Design of Experiments)
  - `doe/` (DoE sub-package):
    - `base.py` $\rightarrow$ Abstract `DesignMatrix` class.
    - `full_factorial.py` / `fractional_factorial.py` $\rightarrow$ Matrix generation & resolution checks.
    - `taguchi.py` / `dsd.py` $\rightarrow$ Orthogonal arrays & Definitive Screening.
    - `ccd.py` / `box_behnken.py` $\rightarrow$ Response Surface Methodology.
    - `d_optimal.py` $\rightarrow$ Coordinate exchange determinant optimization.
    - `lhs.py` / `mixture.py` / `switchback.py` / `evop.py` $\rightarrow$ Specialized designs.
  - `splits.py` $\rightarrow$ `SplitDefinition` (handles traffic allocations, ramp-ups, and holdouts).
  - `randomization.py` $\rightarrow$ Cryptographic salt/hash assignment mapping.
  - `stratification.py` $\rightarrow$ Stratified and cluster randomized partitioning.
- **`validate/`** (Pre-analysis Diagnostics)
  - `srm.py` $\rightarrow$ Chi-square Sample Ratio Mismatch validation.
  - `aa_test.py` $\rightarrow$ Validates null hypothesis distributions.
  - `balance.py` $\rightarrow$ Covariate baseline symmetry checks.
  - `novelty.py` $\rightarrow$ Detects and flags novelty/primacy effects.
- **`run/`** (Data Collection & Stopping Boundaries)
  - `ingestion.py` $\rightarrow$ Data adapters (SQL, streaming, CSV).
  - `assignment.py` $\rightarrow$ Assignment deduplication and logging.
  - `monitor.py` $\rightarrow$ Real-time peek boundaries.
  - `stopping.py` $\rightarrow$ Implements sequential stopping checks.
- **`analyze/`** (Statistical Inference Engine)
  - `orchestrator.py` $\rightarrow$ `AnalysisResult` (compiles summaries, generates forest plots).
  - `variance_reduction.py` $\rightarrow$ CUPED/CUPAC regression adjustment utilities.
  - `corrections.py` $\rightarrow$ `apply_multiple_testing_correction` (Bonferroni, Holm, False Discovery Rate Benjamini-Hochberg).
  - `inference/router.py` $\rightarrow$ Maps metric types and design selections to computational engines.
  - `inference/frequentist.py` $\rightarrow$ Welch's t-test and Mann-Whitney U tests.
  - `inference/bayesian.py` $\rightarrow$ `BayesianInference` (Beta-Binomial, Normal-Normal posteriors, Expected Loss, ROPE).
  - `inference/sequential.py` $\rightarrow$ `SequentialInference` (mSPRT martingale limits, Always-Valid CI boundaries).
  - `inference/bootstrap.py` $\rightarrow$ Bias-Corrected and Accelerated (BCa) bootstrap.
- **`interactions/`** $\rightarrow$ `detector.py`, `anova.py`, `regression.py`, `shap.py` (models treatment-covariate interactions).
- **`interpret/`** $\rightarrow$ `effect_size.py`, `hte.py` (subgroup scanners), `decision.py` (economic advice).
- **`report/`** $\rightarrow$ `card.py` (`ExperimentCard`), `audit.py` (cryptographic logging), `export.py` (serialization).

---

## 4. Mathematical & Algorithmic Anchors
To verify math or implement enhancements, refer to the LaTeX formulas in the corresponding sections of `docstring_reference.md`:

1. **Sample Ratio Mismatch (SRM) Chi-Square**: Implemented in `xpyrment.validate.srm`. Ref: [srm.py](file:///c:/Users/Dan/projects/xpyrment/src/xpyrment/validate/srm.py).
2. **Welch's T-Test & Satterthwaite Degrees of Freedom**: Implemented in `xpyrment.analyze.inference.frequentist`. Ref: [frequentist.py](file:///c:/Users/Dan/projects/xpyrment/src/xpyrment/analyze/inference/frequentist.py).
3. **CUPED Variance Reduction**:
   - $Y_i^{\text{CUPED}} = Y_i - \theta(X_i - \bar{X})$, where $\theta = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}$
   - Implemented in `xpyrment.metrics.taxonomy` (inside `MeanMetric.calculate` and `RatioMetric.calculate`).
4. **Bayesian Conjugate Updating & Expected Loss**:
   - Beta-Binomial: $\text{Beta}(\alpha_0 + k, \beta_0 + n - k)$
   - Normal-Normal: Mean/variance updates.
   - Expected Loss: $L(T) = \mathbb{E}[\max(\theta_C - \theta_T, 0)]$ via Monte Carlo posteriors.
   - Implemented in `xpyrment.analyze.inference.bayesian`.
5. **mSPRT & Always-Valid Confidence Intervals (AVCI)**:
   - Margin $W_n = \sqrt{\frac{2\sigma^2(\sigma^2 + n\tau^2)}{n^2\tau^2} \ln\left( \frac{1}{\alpha} \sqrt{\frac{\sigma^2 + n\tau^2}{\sigma^2}} \right)}$
   - Implemented in `xpyrment.analyze.inference.sequential`.
6. **Multiple Comparison Corrections**:
   - Holm-Bonferroni step-down, FDR Benjamini-Hochberg step-up.
   - Implemented in `xpyrment.analyze.corrections`.
7. **BCa Bootstrap Resampling**:
   - Corrects percentile bounds for median bias $z_0$ and skewness acceleration $a$.
   - Implemented in `xpyrment.analyze.inference.bootstrap`.

---

## 5. Strict AI Agent Coding Rules
1. **Never Introduce Circular Imports**: Every import statement must strictly conform to the hierarchy in Section 1. Never add imports from a downstream package into an upstream file.
2. **Maintain State Invariant Checkers**: Always invoke state assertion helpers at key execution gates to enforce transition discipline.
3. **Preserve LaTeX in Docstrings**: Keep all standard mathematical notations (standard LaTeX formatted in `$$` or `$`) in docstrings. Do not alter existing docstrings unless correcting math bugs.
4. **No Placeholders**: Never write TODOs or pass blocks in production modules; implement complete, production-ready logic (unless implicitly prompted or you have confirmed its necessity).
5. **Strict Test-Driven Development (TDD)**: Test files should be created first before implementing algorithms within production files. Always sketch out expected behaviors and edge cases in the corresponding test suite before coding.
6. **Verify Project Health Regularly**:
   - Run the test suite: `python -m pytest` or `& .venv\Scripts\python.exe -m pytest`.
   - Ensure zero errors or warnings from the MkDocs build system when making changes that alter docstrings.