---
trigger: model_decision
description: On any feature addition, regression debugging, or testing updates.
---

# 🧪 Rule: Continuous Rigorous Testing

## ⚡ Model Decision Activation
This rule is **activated** whenever a model makes a decision to edit existing tests, write new features, or implement statistical code under `src/`.

When activated, the model MUST explicitly document its test architecture in a **"Model Decision"** block, explaining how the implementation is verified.

---

## 1. Test-First Paradigm (TDD)
* **Test Creation Sequence**: Write the corresponding test files under `tests/` *before* editing the target files in `src/`.
* **Behavior Assertions**: Test suites must assert:
  1. **Perfect Inputs**: Expected behavior with realistic data.
  2. **Boundary Inputs**: Empty series, single-element arrays, extreme outliers, zero-variance arrays.
  3. **Mismatched Inputs**: Array length mismatches, incompatible types, incorrect indexes.

## 2. Random Seed Deterministism
* **Reproducibility**: All simulation, bootstrap, and random assignment routines (such as Latin Hypercube Sampling, bootstrapping, or synthetic data generators) must allow passing a fixed integer seed to ensure perfect test reproducibility.
* **Seed Isolation**: Tests must use isolated `np.random.Generator` objects instead of global `np.random.seed()` to prevent side-effects on other tests.

## 3. Strict Verification & Regression Gate
* **Local Test Execution**: Proactively run `pytest` via `.venv\Scripts\python.exe -m pytest` to verify the modified code path and any downstream modules.
* **Documentation Compiles**: Verify that new docstrings do not introduce mkdocstrings parsing failures.