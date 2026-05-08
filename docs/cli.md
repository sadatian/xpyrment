# 💻 Command Line Interface (CLI) Reference

The `xpyrment` Python package includes a command-line tool designed for rapid analytical queries, statistical checks, and experiment planning. This reference guide dynamically showcases all available command sub-interfaces directly from the active module.

---

## 🛠️ Global CLI Interface

To run the CLI tool, call the package directly or use the `xpyrment` binary command:

{{ cli_help('--help') }}

---

## ⚡ Power Analysis and Sample Size Calculator (`xpyrment power`)

The `power` command computes statistical power curves, minimum detectable effects (MDE), and the required sample size per variant for frequentist evaluation.

{{ cli_help('power --help') }}

---

## 📊 Covariate Balance Checker (`xpyrment balance`)

Before running or analyzing an experiment, use the `balance` command to inspect Standardized Mean Differences (SMD) on pre-period covariates across your allocation groups to detect pre-assignment bias.

{{ cli_help('balance --help') }}

---

## 📈 Tabular Ordinary Least Squares (`xpyrment regress`)

The `regress` command executes an ordinary least squares linear regression solver on a local CSV table to check for linear coefficients, standard errors, t-statistics, and p-values.

{{ cli_help('regress --help') }}
