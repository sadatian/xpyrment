---
trigger: model_decision
description: Activated on any bug debugging, regression investigation, testing update, or safe feature addition.
---

# 🧪 Rule: Continuous Verification & Regression Debugging (v1 Era)

## ⚡ Model Decision Activation
This rule is **activated** whenever an AI agent makes a decision to debug a bug, write a regression test, or implement a safe new feature.

When activated, the agent MUST explicitly outline its test architecture and reproduction hypothesis in a **"Model Decision"** block in its output.

---

## 1. Gold-Standard Regression Debugging Flow
When resolving a bug, regression, or calculation issue, do NOT edit production code immediately. Follow this strict verification loop:
1. **Write Reproduction Test First**: Before applying any fix under `src/`, write a reproducing unit test in the corresponding test suite (e.g., prefixing with `test_reproduce_issue_...`).
2. **Verify Failure**: Execute the test runner on the new test only to confirm that it fails under the current implementation.
3. **Implement the Fix**: Modify the target file in `src/` to resolve the bug.
4. **Assert Success**: Execute the reproduction test to confirm that it now passes successfully.
5. **Full Suite Regression Check**: Execute the complete test suite (`pytest`) to confirm that zero existing tests or downstream modules are broken.

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

---

## 3. Deployment & Release Readiness Checks
Before completing any task, execute:
* **Unit Verification**: Run `.venv\Scripts\python.exe -m pytest` to verify 100% test success across all 140+ test cases.
* **Documentation Health**: Run `.venv\Scripts\python.exe -m mkdocs build` to confirm that any docstring changes do not trigger mkdocstrings errors.