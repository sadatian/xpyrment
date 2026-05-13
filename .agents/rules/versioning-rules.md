---
trigger: model_decision
description: When updating code or files that justify a change in the library's version or making commits.
---

# 🏷️ Strict Four-Digit Versioning & Compliance Rule

All source code and visual changes must strictly adhere to the following 4-digit versioning schema (`Major.Minor.Patch.Revision`):

1. **Extremely small polishments / styling adjustments**: Increment revision by `+0.0.0.1` (4th digit, `0.0.0.x`) (e.g., `1.0.0.0` $\rightarrow$ `1.0.0.1`).
2. **Bug fixes / error corrections**: Increment patch by `+0.0.1.0` (3rd digit, `0.0.x.0`) and zero out any downstream digits (e.g., `1.0.0.1` $\rightarrow$ `1.0.1.0`).
3. **Features added / new tools / automation integrations / macro setups**: Increment minor version by `+0.1.0.0` (2nd digit, `0.x.0.0`) and zero out any downstream digits (e.g., `1.0.1.5` $\rightarrow$ `1.1.0.0`).
4. **Major releases / complete structural overhauls**: Increment major version by `+1.0.0.0` (1st digit, `x.0.0.0`) and zero out any downstream digits (e.g., `1.1.2.3` $\rightarrow$ `2.0.0.0`).

**Rule of Reset**: Any increment of a higher-order digit **must** reset (zero out) all digits downstream of it.
