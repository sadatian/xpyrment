# Implementation Plan: xpyrment Python Package (v1 Stable Maintenance Era)

This is the active, concentrated implementation plan for **`xpyrment`**—a highly modular, phase-gated library for digital experimentation and classical Design of Experiments (DoE).

Following the successful completion of Phase 1 (Core Foundation Blocks 1-60) and subsequent stable revisions, this plan has been concentrated to shift focus toward:
1. **Solidifying Project Scope**: Ensuring existing modules remain robust, predictable, and compliant with v1 requirements.
2. **Redefining & Verifying Project Requirements**: Verifying that API behavior matches mathematical specifications, handles edge cases, and remains fully backward-compatible.
3. **Necessary Incremental Improvements & Bug Fixes**: Executing deliberate, non-breaking, and thoroughly tested enhancements under our strict 4-digit versioning protocol.

## 🗄️ Historical Archives

For full historical detail of past milestones and initial construction phases, please consult the archives:
* 📦 **[v0.1.0.0 Archive Implementation Plan](file:///c:/Users/Dan/projects/xpyrment/Gemini_Implementation_Plan_v0.1.0.0.md)**: Greenfield development, initial core structures, and design submodules.
* 📦 **[v1.1.2.3 Archive Implementation Plan](file:///c:/Users/Dan/projects/xpyrment/Gemini_Implementation_Plan_v1.1.2.3.md)**: Completing Phase 1 milestones, establishing versioning automation, styling overhauls, and documentation improvements.

---

## 🏷️ Strict 4-Digit Versioning Protocol

All changes made during this maintenance era must strictly adhere to the `Major.Minor.Patch.Revision` versioning format:
* **Extremely small polishments / styling adjustments**: Increment revision by `+0.0.0.1` (4th digit, `0.0.0.x`) (e.g., `1.1.2.3` $\rightarrow$ `1.1.2.4`).
* **Bug fixes / error corrections**: Increment patch by `+0.0.1.0` (3rd digit, `0.0.x.0`) and zero downstream digits (e.g., `1.1.2.4` $\rightarrow$ `1.1.3.0`).
* **Features added / new tools / automation integrations**: Increment minor version by `+0.1.0.0` (2nd digit, `0.x.0.0`) and zero downstream digits (e.g., `1.1.3.0` $\rightarrow$ `1.2.0.0`).
* **Major releases / complete structural overhauls**: Increment major version by `+1.0.0.0` (1st digit, `x.0.0.0`) and zero downstream digits (e.g., `1.2.0.4` $\rightarrow$ `2.0.0.0`).
* *Rule of Reset*: Any increment of a higher-order digit **must** reset (zero out) all digits downstream of it.

---

## 📐 Current Core Project Scope & Requirements

The project scope is strictly bounded to the following main pillars. Any changes or fixes must maintain compatibility and correctness within these domains:

### 1. Robust Lifecycle & Phase-Gated Orchestration (`core/`)
* **Requirement**: Enforce sequential, immutable state-transitions through `CREATED` $\rightarrow$ `PLANNED` $\rightarrow$ `DESIGNED` $\rightarrow$ `RUNNING` $\rightarrow$ `ANALYZED` $\rightarrow$ `REPORTED` via `Experiment` orchestrator.
* **Verification**: `PhaseOrderError` must be raised on invalid transitions or actions taken out of order.

### 2. Comprehensive Metric Taxonomy (`metrics/`)
* **Requirement**: Support `proportion`, `mean`, `ratio`, and `revenue` metric calculations with automated CUPED variance reduction.
* **Verification**: Verify variance reduction factors and check for zero-variance or extreme-skew failures.

### 3. Rigorous Classical & Modern DoE Schemes (`design/doe/`)
* **Requirement**: Provide D-optimal, definitive screening, Taguchi, central composite, and fractional factorial design engines with clear alias confounding checks.
* **Verification**: Ensure generated matrices meet mathematical constraints (orthogonality, D-efficiency maximization) and handle level limits.

### 4. Input Diagnostics, Data Verification & Cleaning (`validate/`)
* **Requirement**: Perform sample ratio mismatch (SRM) chi-square checking, A/A null validity permutations, and multi-dimensional covariate balance evaluations.
* **Verification**: Automatically detect assignment biases and raise warnings/errors before analyzing outcomes.

### 5. Multi-Armed Bandits & Adaptive Exploration (`bandit/`)
* **Requirement**: Support multi-arm adaptive allocations (Epsilon-Greedy, UCB1, Thompson Sampling) with custom Gaussian Process Bayesian optimization parameter-tuning.
* **Verification**: Confirm cumulative regret trajectories are bounded and exploration-exploitation thresholds adapt correctly under drift.

---

## 🎯 Active Maintenance & Verification Backlog

Below is the active work backlog aimed at solidifying the package scope and verifying its correctness:

### 📋 Block A: Systematic Requirements Audit & Edge Case Verification
* **Goal**: Systematically verify all core submodules against extreme input bounds, singular matrix designs, and empty dataframe edge cases.
* **Technical Spec**: Author comprehensive test cases under `tests/` specifically targeted at boundary condition behavior (e.g., singular covariate matrix during OLS/CUPED, zero allocations in SRM, high skew in BCa bootstrap).
* **Priority**: High (Prevents production downtime under unexpected real-world data).

### 📋 Block B: Documentation Validation & Warnings Scrub
* **Goal**: Ensure the entire documentation suite builds with 100% clean output and zero formatting or macro errors.
* **Technical Spec**: Audit and verify that LaTeX MathJax rendering is correct across all generated API pages, and that no unescaped markdown syntax is present in docstrings.
* **Priority**: Medium.

---

## 🚀 Primary Future Goals & Performance Optimization

### 1. Multithreading & Parallelization
* **Objective**: Leverage concurrent execution models to speed up computationally expensive tasks.
* **Focus Areas**: Parallelize long-running simulation loops, bootstrap resampling routines (e.g., BCa bootstrap), and Monte Carlo simulations in design engines and bandits.

### 2. High-Performance Pre-built Methods (NumPy/SciPy First)
* **Objective**: Replace slow, native Python loops and operations with highly optimized vector and matrix operations.
* **Focus Areas**: Prioritize **NumPy** array broadcasting, vectorized linear algebra solvers, and optimized math library implementations over standard Python iterators and comprehension lists.

### 3. Computation Profiling and Bottleneck Identification
* **Objective**: Conduct systematic profiling of memory and CPU utilization.
* **Focus Areas**: Identify hotspots in CUPED variance reduction, Gaussian Process tuning, and multi-dimensional covariate balancing routines.

---

## 🛠️ Developer Workflow Guardrails

All maintenance work must follow these strict guardrails:
1. **Interactive Progress Updates**: Update this file (`Gemini_Implementation_Plan.md`) immediately after the completion of any activity, detailing accomplished work and subsequent steps.
2. **Continuous Testing (Continuous Validation Verification)**: Proactively run the test suite (`pytest`) and compile documentation locally (`mkdocs build`) after any code edits to ensure zero regressions.
3. **No Automated Documentation Serving**: Do NOT run the `mkdocs serve` command directly. Prompt the user to run it if they need to check a live preview.

---

## 📈 Activity & Progress Log

### 1. Implementation Plan Archive & Concentration (v1.1.2.3 $\rightarrow$ v1.1.2.4)
* **a) What was accomplished**:
  - Saved full historical milestones and blocks 1-60 details to [Gemini_Implementation_Plan_v1.1.2.3.md](file:///c:/Users/Dan/projects/xpyrment/Gemini_Implementation_Plan_v1.1.2.3.md).
  - Consolidated and concentrated the active `Gemini_Implementation_Plan.md` to establish the Maintenance Era scope, core package requirements, versioning protocol, and the active verification backlog.
  - Bumped the package revision to `1.1.2.4` in `pyproject.toml` and verified dynamic compilation.
* **b) What must be done next**:
  - Proceed with systematically auditing and verifying project requirements (Block A), starting with identifying any potential requirement gaps in our statistical algorithms or CLI tools.

### 2. Integration of Performance Optimization & Parallelization Future Goals (v1.1.2.4 $\rightarrow$ v1.1.2.5)
* **a) What was accomplished**:
  - Added "Primary Future Goals & Performance Optimization" section to the active implementation plan.
  - Formulated goals for integrating multithreading, parallelization, and high-performance NumPy/SciPy operations across key computational bottlenecks.
  - Bumped the package revision to `1.1.2.5` in `pyproject.toml` to track plan updates.
* **b) What must be done next**:
  - Proceed with systematically auditing and verifying project requirements (Block A), keeping performance optimization paths (such as vectorization and NumPy array broadcasting) in mind when refactoring statistical computations.

### 3. GPLv3 Mashup & Humorous LICENSE Revision
* **a) What was accomplished**:
  - Rewrote and mashed up the [LICENSE](file:///c:/Users/Dan/projects/xpyrment/LICENSE) with GNU GPLv3 terms to create the "GNU General Public Slop License v3.1-SLOP".
  - Kept humorous copyleft clauses and the word "slop", but elevated the tone to extremely professional/polite legal language (e.g., replacing unprofessional terms like "dumbification" with formal concepts like "Intellectual Acuity Protection" and "reduction in the Licensee's cognitive faculties").
  - Included the "Mutual Career Obsolescence" clause detailing the economic consequences of over-reliance on AI for software development.
* **b) What must be done next**:
  - Continue executing active maintenance backlog tasks under Block A (Systematic Requirements Audit & Edge Case Verification).


