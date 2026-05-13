# Developer Guide & Versioning Protocol: xpyrment Python Package

This document serves as the high-level roadmap, core architecture scope, versioning guidelines, and active developer guidelines for **`xpyrment`**—a highly modular, phase-gated library for digital experimentation and classical Design of Experiments (DoE).

> [!TIP]
> The active, checklist-based development task board has been separated into **[TASKS.md](file:///c:/Users/Dan/projects/xpyrment/TASKS.md)** to enable dynamic GitHub-Flavored Markdown task lists and native GitHub Issue integrations.

---

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
* *Rule of Reset*: Any increment of a higher-order digit **must** reset (zero out) all downstream digits.

---

## 📐 Core Project Scope & Requirements

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

### 4. Dynamic Badge Synchronization & Remote PyPI Fetching (v1.1.2.5 $\rightarrow$ v1.1.2.6)
* **a) What was accomplished**:
  - Enhanced the version synchronization script in [main.py](file:///c:/Users/Dan/projects/xpyrment/main.py) to run the test suite and coverage calculation dynamically during doc building and deployment phases.
  - Implemented dynamic API fetching of the latest published package version from actual PyPI (`pypi.org`), falling back automatically to TestPyPI (`test.pypi.org`) if not yet released on main PyPI.
  - Formulated robust regular expressions to dynamically extract and sync the actual total passed test counts, coverage percentages, and remote PyPI version directly into [README.md](file:///c:/Users/Dan/projects/xpyrment/README.md).
  - Successfully built documentation using `mkdocs build`, verifying that all badges (including the PyPI badge updating automatically to `1.1.2.3` from TestPyPI) were synchronized without any manual edits.
* **b) What must be done next**:
  - Continue executing active maintenance backlog tasks under Block A (Systematic Requirements Audit & Edge Case Verification).

### 5. Build and Publish Automation CLI (v1.1.2.6 $\rightarrow$ v1.1.2.7)
* **a) What was accomplished**:
  - Created a CLI entry point directly in [main.py](file:///c:/Users/Dan/projects/xpyrment/main.py) utilizing standard library `argparse` to automate key development release tasks.
  - Added support for `--sync` to update version identifiers and shields on demand, `--build` to clean past residues and compile robust `.whl` and `.tar.gz` packages using Python `build`, and `--testpypi`/`--pypi` to upload artifacts using `twine`.
  - Configured lazy, self-healing installations for `build` and `twine` to guarantee executable availability without pre-requisite configurations.
* **b) What must be done next**:
  - Continue executing active maintenance backlog tasks under Block A (Systematic Requirements Audit & Edge Case Verification).

### 6. Backlog Splitting & Professional GitHub Integration (v1.1.2.7 $\rightarrow$ v1.1.2.8)
* **a) What was accomplished**:
  - Splitted the centralized task lists out of the implementation plan into an independent, premium **[TASKS.md](file:///c:/Users/Dan/projects/xpyrment/TASKS.md)** board using GitHub-Flavored Markdown checklist syntax.
  - Established native GitHub issue-tracking integration by designing standard-compliant issue templates in `.github/ISSUE_TEMPLATE/` (for bugs, feature requests, maintenance tasks, and repository configurations).
  - Fully cleaned and verified local workspace tracking branches.
* **b) What must be done next**:
  - Continue executing active maintenance backlog tasks specified in **[TASKS.md](file:///c:/Users/Dan/projects/xpyrment/TASKS.md)** (Block A: Systematic Requirements Audit & Edge Case Verification).

### 7. Unified PyPI & GitHub Release Automation (v1.1.2.8 $\rightarrow$ v1.1.2.9)
* **a) What was accomplished**:
  - Engineered direct integration between the PyPI publishing pipeline and the GitHub release ecosystem.
  - Added support for reading and parsing release descriptions from a central **[RELEASE_NOTES.md](file:///c:/Users/Dan/projects/xpyrment/RELEASE_NOTES.md)** file at the root directory.
  - Implemented automated Git tagging (`v{version}`), origin tag pushing, and formal **GitHub Release** creation via the GitHub REST API (including automatic binary wheel & sdist asset uploads).
  - Bumped the package version to `1.1.2.9` across all files, ran the sync tools to align documentation and badges, and ran the complete 138-test suite with a 100% pass rate.
* **b) What must be done next**:
  - Perform the official release build and publish sequence to launch v1.1.2.9 live!
  - Continue executing active maintenance backlog tasks specified in **[TASKS.md](file:///c:/Users/Dan/projects/xpyrment/TASKS.md)** (Block A: Systematic Requirements Audit & Edge Case Verification).


