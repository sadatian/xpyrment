# Active Development Backlog & Task Board

This is the central task board for the **`xpyrment`** project maintenance. It is designed to be fully integrated with **GitHub Issues**, **Pull Requests**, and **GitHub Project Boards**.

> [!NOTE]
> You can convert any checkbox task below directly into a tracked GitHub Issue or link it to a PR using the GitHub UI, or reference the task checklist when closing issues!

---

## 📋 Block A: Systematic Requirements Audit & Edge Case Verification
* **Objective**: Systematically audit all analytical models against mathematical boundary conditions, singular matrix configurations, and empty arrays to prevent production crashes.
* **Status**: 🔄 In Progress

- [ ] **A.1 CUPED Singular Covariate Matrices**
  - [ ] Implement protective condition when the covariate variance is zero or collinear.
  - [ ] Ensure proper fallback to standard Difference-in-Means when regression adjustments are unsolvable.
  - [ ] Add regression tests under `tests/test_metrics.py`.
- [ ] **A.2 Sample Ratio Mismatch (SRM) Extreme Low Sample Sizes**
  - [ ] Guard SRM chi-square tests when cell counts are zero or extremely low (e.g., < 5).
  - [ ] Log helpful diagnostics warnings instead of crashing with division-by-zero or math domain errors.
- [ ] **A.3 BCa Bootstrap Extreme Resampling Bounds**
  - [ ] Handle scenarios where the bootstrap distributions are perfectly degenerate or have zero variance.
  - [ ] Add test cases simulating zero variance metrics in `tests/test_bootstrap_harden.py`.
- [ ] **A.4 Design of Experiments Level Limits & Edge Configurations**
  - [ ] Add validators to raise descriptive exceptions when level counts do not align with fractional factorial or definitive screening generator matrices.

---

## 📋 Block B: Documentation Validation & Warnings Scrub
* **Objective**: Guarantee a 100% clean documentation compilation with correct mathematical symbols and zero formatting alerts.
* **Status**: 📅 Planned

- [ ] **B.1 MathJax Character Rendering Sweep**
  - [ ] Audit LaTeX formulas in both code docstrings and `.md` pages to ensure perfect compatibility with Material for MkDocs.
- [ ] **B.2 Unescaped Code Snippets in Docstrings**
  - [ ] Double-check docstrings of newly added modules (`quasi`, `plan`) to ensure all variables and parameters are fenced correctly.
- [ ] **B.3 Dynamic Macros Verification**
  - [ ] Verify that all automated macros (`list_doe_designs`, `list_metrics`, `cli_help`) evaluate cleanly on fresh builds.

---

## 🚀 Primary Future Goals & Performance Optimization
* **Objective**: Accelerate statistical computations and multi-armed bandit simulation runs through vectorization and parallelism.
* **Status**: 📅 Long-term Roadmap

- [ ] **P.1 Multithreading & Parallelization**
  - [ ] Parallelize the long-running simulation loops inside the Thompson Sampling and UCB bandit agents.
  - [ ] Vectorize Monte Carlo resampling steps within the bootstrap modules.
- [ ] **P.2 High-Performance Pre-built Methods (NumPy First)**
  - [ ] Audit performance hotspots and replace standard native loops with vectorized NumPy array broadcasting.
  - [ ] Optimize covariate balancing algorithms through SciPy optimization solvers.
- [ ] **P.3 Execution Profiling Suite**
  - [ ] Create a lightweight memory and execution profiler script to monitor analysis runtimes over large datasets.
