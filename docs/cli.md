# 💻 Command Line Interface (CLI) Reference

The `xpyrment` Python package includes an enterprise-ready command-line interface (CLI) tool designed for rapid analytical queries, statistical balance auditing, and experiment planning directly from your terminal. 

By utilizing the CLI, engineers, data scientists, and analysts can calculate power requirements, audit data quality, and fit regressions without having to launch interactive Jupyter Notebooks or write Python script boilerplate.

---

## 🛠️ Global CLI Interface

The CLI is registered as a global entry point binary called `xpyrment`. You can invoke it globally or call the module using Python directly (`python -m xpyrment.cli`):

```bash
xpyrment --help
# Or alternatively:
python -m xpyrment.cli --help
```

### Dynamic CLI Help Screen
The active commands, options, and global flags are compiled dynamically from the executable source code:

{{ cli_help('--help') }}

---

## ⚡ Power Analysis and Sample Size Calculator (`xpyrment power`)

The `power` command computes standard statistical power curves, minimum detectable effects (MDE), and the required sample size per variant for frequentist evaluations.

### Theoretical Context & Parameters
When planning an experiment, determining the correct sample size is essential to guarantee that your test has sufficient statistical power to detect real product changes without inflating false-negative rates. 

The CLI implements the two-sample t-test sample sizing algorithm. It computes requirements using five core parameters:
- **Minimum Detectable Effect (`--mde`)**: The relative or absolute percentage lift you wish to detect.
- **Baseline Standard Deviation (`--std`)**: The natural variance of your key metric in the historical population.
- **Statistical Power (`--power`)**: The target probability of rejecting the null hypothesis when a real treatment effect exists ($1 - \beta$, usually set to `0.80`).
- **Significance Level (`--alpha`)**: The maximum acceptable false-positive rate ($\alpha$, usually set to `0.05`).
- **Allocation Ratio (`--ratio`)**: The ratio of the sample size in the Treatment group to the Control group ($n_T / n_C$, defaults to `1.0` for equal $50/50$ splits).

### Interactive Sizing Examples

!!! example "Equal Allocation (50/50 Split)"
    To size an experiment looking to detect a **$2.5\%$ relative lift** on a metric with a baseline mean of **100.0** and standard deviation of **15.0**:
    ```bash
    xpyrment power --mde 2.5 --std 15.0 --power 0.80 --alpha 0.05 --ratio 1.0
    ```

!!! example "Unequal Allocation (90/10 Rollout)"
    During low-exposure rollouts, you might want to allocate only 10% of traffic to treatment (`--ratio 0.1111`):
    ```bash
    xpyrment power --mde 5.0 --std 25.0 --power 0.80 --alpha 0.05 --ratio 0.111
    ```

### Command Options Reference
The active options for the `power` command are dynamically generated below:

{{ cli_help('power --help') }}

---

## 📊 Covariate Balance Checker (`xpyrment balance`)

Before running or analyzing an experiment, use the `balance` command to inspect Standardized Mean Differences (SMD) on pre-period covariates across your allocation groups to detect pre-assignment selection bias.

### Standardized Mean Difference (SMD)
In random online experiments, any pre-existing differences between control and treatment groups (e.g., historical revenue, user sign-up ages, or device breakdowns) can pollute your post-period analysis and invalidate causal inference. 

To audit the quality of random split assignments, we calculate the Standardized Mean Difference (SMD) for each covariate:
$$
SMD = \frac{\bar{X}_T - \bar{X}_C}{\sqrt{\frac{s_T^2 + s_C^2}{2}}}
$$
Where $\bar{X}_T$ and $\bar{X}_C$ are the sample means of the covariate in the treatment and control groups, and $s_T^2, s_C^2$ are their respective sample variances.

!!! mathbox "The Golden SMD Rule"
    An $|SMD| < 0.1$ is the standard industrial threshold for a perfectly balanced covariate. Any $|SMD| \ge 0.1$ indicates a statistically significant difference between the groups, flagging a severe assignment imbalance or tracking leak.

### Balance Checking Examples

!!! example "Checking Covariates from CSV"
    Assuming you have a local dataset named `user_data.csv` containing columns for the assignment group (`variant`) and pre-period metrics (`pre_revenue`, `user_age`, `tenure_days`):
    ```bash
    xpyrment balance --csv user_data.csv --group-col variant --covariates pre_revenue,user_age,tenure_days
    ```

### Command Options Reference
The active options for the `balance` command are dynamically generated below:

{{ cli_help('balance --help') }}

---

## 📈 Tabular Ordinary Least Squares (`xpyrment regress`)

The `regress` command executes a high-performance ordinary least squares (OLS) linear regression solver on a local CSV table.

### Theoretical Context
Regression is a powerful tool for causal inference. By fitting a linear model, you can estimate the treatment effect while simultaneously adjusting for continuous pre-period covariates or user characteristics to significantly reduce residual variance and boost statistical power:
$$
Y_i = \beta_0 + \beta_1 T_i + \sum_{j=1}^{p} \gamma_j X_{ij} + \varepsilon_i
$$
Where:
- $Y_i$: The dependent variable (e.g., post-period revenue).
- $T_i$: The treatment assignment dummy variable ($0$ for Control, $1$ for Treatment).
- $X_{ij}$: Covariates (e.g., pre-period revenue, user age).
- $\beta_1$: The estimated causal treatment effect.
- $\varepsilon_i$: The random error term.

### Regression Modeling Examples

!!! example "Basic Treatment Estimation"
    Fit a simple model estimating the effect of the treatment variant on revenue:
    ```bash
    xpyrment regress --csv experimental_data.csv --y-col revenue --x-cols variant
    ```

!!! example "Covariate-Adjusted (CUPED-equivalent) Estimation"
    Boost statistical power by including pre-period baseline metrics and covariates:
    ```bash
    xpyrment regress --csv experimental_data.csv --y-col revenue --x-cols variant,pre_revenue,user_age
    ```

### Command Options Reference
The active options for the `regress` command are dynamically generated below:

{{ cli_help('regress --help') }}

---

## 🚀 Advanced CI/CD Integration

Because the `xpyrment` CLI outputs structured text and exits with a standard system code (`0` for success, non-zero for failures), it can be easily embedded in automated data pipelines, ETL testing, and continuous integration flows.

### Pre-Experiment Balance Gate Script
Here is an example of an automated Bash script you can use inside a Jenkins, GitHub Actions, or GitLab CI/CD runner to block biased experiment launches before exposure:

```bash
#!/bin/bash
echo "=== Running Pre-Experiment Randomization Balance Audit ==="

# Execute balance check and capture the output
balance_output=$(xpyrment balance --csv daily_exposures.csv --group-col variant --covariates pre_period_clicks,pre_period_revenue)
exit_status=$?

if [ $exit_status -ne 0 ]; then
    echo "❌ Error executing covariate balance check!"
    exit 1
fi

# Parse output for severe SMD imbalances (>= 0.10)
if echo "$balance_output" | grep -q "SMD >= 0.1" || echo "$balance_output" | grep -q "SMD <= -0.1"; then
    echo "⚠️ CRITICAL: Covariate balance check failed! Significant pre-period imbalance detected."
    echo "$balance_output"
    exit 2
else
    echo "✅ Covariate balance check passed! Group assignments are statistically balanced."
    echo "$balance_output"
    exit 0
fi
```
