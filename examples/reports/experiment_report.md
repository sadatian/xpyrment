# 📊 Q2 Premium Revenue Growth Test (CUPED)

## 📌 Executive Summary
- **Nominal Significance Level (Alpha)**: `0.05`
- **Total Samples**: `2,010` (Control: `1,013`, Treatment: `997`)
- **Sample Ratio Mismatch (SRM)**: 🟢 **PASSED** (p-value: `0.721182`)
- **Covariate Balance**: 🟢 **ALL COVARIATES BALANCED** (SMD <= 0.1)

## 📈 Metric Performance
| Metric | Type | Control Mean | Treatment Mean | Relative Lift | P-Value | Significance | CUPED |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **revenue** | `Mean` | 9.7043 | 10.2558 | **+5.68%** | `0.00002` | 🌟 **Significant** | ✅ |

## ⚖️ Covariate Balance Love Plot
```text
=================== COVARIATE BALANCE LOVE PLOT ===================
Covariate Name            | SMD Balance                     | Value 
-------------------------------------------------------------------
pre_revenue               | [----------X----------] | +0.0223
================---------------------------------================
Legend: [Left: Treatment < Control] | [Center '|': Balanced] | [Right: Treatment > Control]
```