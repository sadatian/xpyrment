---
trigger: model_decision
description: Activated on any bug debugging, regression investigation, testing update, or safe feature addition.
---

# 🧪 Rule: Continuous Verification & Regression Debugging (v1 Era)

## ⚡ Model Decision Activation
This rule is **activated** whenever an AI agent makes a decision to debug a bug, write a regression test, or implement a safe new feature.

When activated, the agent MUST explicitly outline its test architecture and reproduction hypothesis in a **"Model Decision"** block in its output.

---

## 1. Deferred Regression Debugging Flow
When resolving a bug, regression, or calculation issue, do NOT edit production code and run tests immediately. Follow this strict verification loop:
1. **Implement the Fix**: Modify the target file in `src/` to resolve the bug or implement the new feature.
2. **Batch Test Runs**: Keep pytests deferred until after a complete set of major debugging or new feature implementation is done. It is useless to run `pytest` before everything.
3. **Write Regression Tests**: After implementing major fixes/features, write unit tests under `tests/` ensuring exhaustive boundary coverage.
4. **Full Suite Regression Check**: Execute the complete test suite (`pytest`) in batch at the end to confirm zero existing tests or downstream modules are broken.

---

## 2. Test Architecture for New Features
When adding safe, backward-compatible new features:
* **Exhaustive Input Coverage**: Test suites must assert:
  - **Ideal Data**: Standard expected data matrices.
  - **Boundary/Extreme Values**: Constant columns, null series, infs/NaNs, collinear inputs, single-element arrays, extreme scales.
  - **Structural Mismatches**: Unaligned series indexes, shape mismatches, invalid types.
* **Random Seed Isolation**:
  - All stochastic operations (bootstrap, simulations, randomized partitions, bandits) must use local, isolated `np.random.Generator` objects initialized by a configurable integer seed.
  - Never call global `np.random.seed()` as it creates state leakage across test modules.
* **Prohibition of Dummy Tests ("Coverups")**:
  - Do NOT create "dummy tests" or "coverups" that merely call code wrapped in `try/except: pass` without meaningful assertions just to artificially inflate coverage.
  - Every test MUST assert functional correctness, correctly setup necessary dependencies, and validate outputs.
  - Test files like `test_coverage_backfill.py` containing zero-assertion logic are strictly forbidden.

---

## 3. Deployment & Release Readiness Checks
Before completing any task, execute:
* **Unit Verification**: Run `poetry run pytest` to verify 100% test success across all 290+ test cases.
* **Documentation Health**: Run `poetry run mkdocs build` to confirm that any docstring changes do not trigger mkdocstrings errors.