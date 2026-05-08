# xpyrment 🧪

<p align="left">
  <img src="https://img.shields.io/pypi/v/xpyrment?color=blue&style=flat-square&logo=pypi&logoColor=white" alt="PyPI version" />
  <img src="https://img.shields.io/pypi/pyversions/xpyrment?color=teal&style=flat-square&logo=python&logoColor=white" alt="Python Support" />
  <img src="https://img.shields.io/badge/tests-138%20passed-success?style=flat-square&logo=pytest" alt="Tests" />
  <img src="https://img.shields.io/badge/coverage-100%25-brightgreen?style=flat-square" alt="Coverage" />
  <img src="https://img.shields.io/github/license/sadatian/xpyrment?color=orange&style=flat-square" alt="License" />
  <img src="https://img.shields.io/badge/docs-mkdocs--material-blueviolet?style=flat-square&logo=materialformkdocs" alt="Documentation Status" />
</p>
<p align="left">
  <img src="https://img.shields.io/github/actions/workflow/status/sadatian/xpyrment/ci.yml?branch=main&style=flat-square&logo=github&label=build" alt="Build Status" />
  <img src="https://img.shields.io/github/last-commit/sadatian/xpyrment?style=flat-square" alt="Last Commit" />
  <img src="https://img.shields.io/github/commit-activity/m/sadatian/xpyrment?style=flat-square&color=magenta" alt="Commit Activity" />
  <img src="https://img.shields.io/github/languages/code-size/sadatian/xpyrment?style=flat-square" alt="Code Size" />
  <img src="https://img.shields.io/github/issues/sadatian/xpyrment?style=flat-square&color=red" alt="Open Issues" />
</p>
<p align="left">
  <img src="https://img.shields.io/badge/release-v{{ version }}%20stable-emerald?style=flat-square" alt="Release" />
  <img src="https://img.shields.io/badge/ledger-SHA--256%20chained-yellowgreen?style=flat-square" alt="Audit Ledger" />
  <img src="https://img.shields.io/badge/stats-Welch%20%7C%20mSPRT%20%7C%20CUPED-blue?style=flat-square" alt="Statistical Engine" />
  <img src="https://img.shields.io/badge/DoE-Full%2FFractional%2FTaguchi%2FDSD-crimson?style=flat-square" alt="Industrial DoE" />
  <img src="https://img.shields.io/badge/maintainer-Dan%20Sadatian-darkgreen?style=flat-square" alt="Maintainer" />
</p>

`xpyrment` is an enterprise-grade, low-code Python library designed for **experiment design, classical Design of Experiments (DoE), and statistical causal inference**. 

It provides an elegant, object-oriented fluent API to orchestrate the entire lifecycle of digital experimentation (A/B testing) alongside the rigorous mathematical techniques of modern, enterprise-scale platforms. It features native support for **CUPED (variance reduction)**, **ratio metrics via the Delta method**, **multiple comparison corrections**, **Sample Ratio Mismatch (SRM) diagnostics**, **mixture SPRT continuous monitoring (mSPRT)**, **Bayesian inference**, and classical **industrial DoE design matrices**.

---

## 🌟 Key Features

* **Unified Fluent Orchestrator API**: Initialize experiments, define metric structures, run statistical evaluations, and compile publication-ready summaries or plots in a clean, state-gated object-oriented pipeline.
* **Rigorous Variance Reduction (CUPED)**: Built-in support for standard CUPED (continuous metrics) and **Ratio CUPED** (numerator and denominator adjustment). Reduces variance and sample size requirements by up to 88%+.
* **Ratio Metric Precision**: Precise variance estimation of ratio metrics (e.g., CTR, revenue per click) where both numerator and denominator are stochastic, using first-order Taylor expansion (**Delta method**).
* **Classical Design of Experiments (DoE)**: Full and Fractional Factorial, Plackett-Burman, Taguchi Orthogonal Arrays, Definitive Screening Designs (DSD), Response Surface Methodologies (CCD & Box-Behnken), and D-Optimal coordinate exchange.
* **Continuous Monitoring & Early Stopping**: Always-valid confidence intervals and sequential monitoring boundaries via **mixture SPRT (mSPRT)** and Pocock/O'Brien-Fleming alpha-spending functions.
* **Experimental Diagnostics**: Built-in automated Chi-square tests to detect **Sample Ratio Mismatch (SRM)**, pre-experiment covariate balance validation with Standardized Mean Differences (SMD), and time-series novelty/primacy effect detectors.
* **Multi-Testing Correction**: Guard against Type I error inflation by automatically adjusting p-values for multiple metrics using Holm-Bonferroni, Bonferroni, or Benjamini-Hochberg (FDR).
* **Multi-Armed Bandits & Adaptive Traffic**: Dynamically allocate traffic using Beta-Binomial / Normal-Normal **Thompson Sampling**, standard/decaying **$\varepsilon$-Greedy**, and classical **UCB1** optimistic exploration. Supports sliding-window and discounted Thompson Sampling for drifting baselines.
* **Heterogeneous Treatment Effects (HTE)**: Personalize variant targeting using CATE estimators (**S-Learner**, **T-Learner**, and propensity-weighted **X-Learner**) alongside custom bootstrapped **Causal Forests**.
* **Synthetic Controls & Quasi-Experiments**: Analyze unrandomized policy deployments using Abadie SLSQP-constrained **Synthetic Controls**, multi-variable **Difference-in-Differences (DiD)** regressions, and Synthetic DiD (SDID).
* **Premium Standalone Reports**: Instantly export summaries into beautiful, portable, responsive CSS-styled HTML dashboards and GitHub-compatible Markdown summary tables.
* **Audit Trail Security**: Cryptographically chain and sign state updates via a SHA-256 tamper-evident ledger, ensuring experiment metadata and configuration parameters remain auditable.

---

## ⚙️ Installation

To install the stable release of `xpyrment` from PyPI, simply run:

```bash
pip install xpyrment
```

For development and contributor setups (including `pytest`, `black`, and `mypy`), clone the repository and install in editable mode:

```bash
git clone https://github.com/sadatian/xpyrment.git
cd xpyrment
pip install -e .[dev]
```

---

## 🚀 Quickstart Tutorial

This quickstart guides you through the entire A/B testing lifecycle: designing, simulating, configuring, and analyzing.

### 1. Experiment Design (Power Analysis)

Before launching your test, calculate the sample size required to detect a $5\%$ relative lift in a key continuous metric (e.g., Average Order Value = \$100, standard deviation = \$35).

```python
import xpyrment as xp

# Calculate sample size for a standard t-test
design = xp.design_experiment(
    metric_type="mean",
    baseline_value=100.0,
    standard_deviation=35.0,
    mde=0.05,                  # 5% relative lift
    mde_type="relative",
    alpha=0.05,                # Significance level (Type I error)
    power=0.80,                # Target power (1 - Type II error)
    pre_post_correlation=0.75, # Optional: Pre-Post correlation to calculate CUPED savings!
    daily_traffic=5000         # Optional: Daily user traffic to calculate duration
)

print(design)
```

**Output:**
```text
=========================================
       Experiment Design Summary        
=========================================
Metric Type                   : Mean
Baseline Value                : 100.0000
Target MDE (Absolute)         : 5.0000
Target MDE (Relative)         : 5.00%
Significance Level (Alpha)    : 5.00%
Statistical Power (1-Beta)    : 80.00%
Sample Size Per Variant       : 1,537
Total Sample Size Required    : 3,074
Pre-Post Correlation          : 0.75
CUPED Sample Size Per Variant : 672
CUPED Total Sample Size       : 1,344
CUPED Sample Size Savings     : 43.8%
Daily Traffic                 : 5,000/day
Estimated Duration (Standard) : 0.6 days
Estimated Duration (CUPED)    : 0.3 days
=========================================
```

#### Visualizing Power Curves
Generate coordinates and plot required sample sizes against a range of MDEs to see the impact of CUPED:

```python
# Generate power curve coordinates
curve_data = xp.generate_power_curve_data(
    metric_type="mean",
    baseline_value=100.0,
    standard_deviation=35.0,
    pre_post_correlation=0.75
)

# Plot standard vs. CUPED required sample sizes
xp.plot_power_curve(curve_data)
```

---

### 2. Generate Synthetic A/B Test Data

Let's generate simulated experimental data of 10,000 users split 50/50, complete with pre-period covariates so we can demonstrate CUPED and ratio metric evaluations:

```python
df = xp.generate_ab_data(
    n_samples=10000,
    treatment_effect_revenue=2.5,        # +$2.50 absolute lift
    treatment_effect_conversion=0.015,    # +1.5% absolute lift
    treatment_effect_clicks=0.06,         # +6% relative lift in click ratios
    pre_period_correlation=0.82,          # Correlation between pre- and post- period
    random_seed=42
)

print(df.head())
```

| user_id | variant | pre_revenue | revenue | converted | pre_clicks | pre_impressions | clicks | impressions |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| USER_000001 | control | 4.47 | 4.42 | 0 | 4 | 93 | 5 | 96 |
| USER_000002 | treatment | 56.78 | 61.22 | 1 | 6 | 112 | 8 | 108 |
| USER_000003 | control | 51.12 | 48.91 | 0 | 5 | 105 | 3 | 99 |
| USER_000004 | treatment | 32.54 | 36.90 | 0 | 3 | 82 | 4 | 88 |

---

### 3. Setup and Run Analysis

Initialize the experiment environment using the `setup` function, define your metrics (with pre-period specifications for automatic CUPED), and run your analysis!

```python
# 1. Initialize experiment setup
exp = xp.setup(
    data=df, 
    treatment_col="variant", 
    id_col="user_id"
)

# 2. Define your metrics
# Continuous metric (Average revenue) with automatic CUPED!
revenue = xp.MeanMetric(
    name="Average Revenue per User", 
    value_col="revenue", 
    pre_period_col="pre_revenue"
)

# Proportion metric (Conversion rate)
conversion = xp.ProportionMetric(
    name="Purchase Conversion Rate", 
    value_col="converted"
)

# Ratio metric (Click-Through-Rate = sum(clicks)/sum(impressions)) with ratio CUPED!
ctr = xp.RatioMetric(
    name="Click-Through-Rate (CTR)", 
    numerator_col="clicks", 
    denominator_col="impressions",
    pre_numerator_col="pre_clicks",
    pre_denominator_col="pre_impressions"
)

# 3. Add metrics to the experiment container
exp.add_metrics([revenue, conversion, ctr])

# 4. Run Analysis (optionally apply multi-test corrections like 'fdr_bh')
results = exp.run_analysis(
    control="control", 
    treatment="treatment",
    multi_test_correction="fdr_bh"
)
```

---

### 4. Review and Visualize Results

#### Standard Summary DataFrame
Call `.summary()` to get a polished, publication-ready pandas DataFrame with automatic statistical significance annotations (`*` for $p < 0.05$, `**` for $p < 0.01$, `***` for $p < 0.001$).

```python
summary_df = results.summary()
print(summary_df)
```

| Metric | Type | Control Mean | Treatment Mean | Relative Lift | 95% CI (Rel) | p-value | Post-hoc Power | CUPED | Var Reduction |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Average Revenue per User | Mean | 49.9542 | 52.4712 | +5.04% | [+3.78%, +6.30%] | 0.0000*** | 100.0% | Yes | 68.3% |
| Purchase Conversion Rate | Proportion | 0.0990 | 0.1172 | +18.42% | [+4.12%, +32.72%] | 0.0112* | 73.1% | No | - |
| Click-Through-Rate (CTR) | Ratio | 0.0498 | 0.0528 | +5.95% | [+4.11%, +7.78%] | 0.0000*** | 100.0% | Yes | 71.2% |

!!! tip "Variance Reduction Advantage"
    CUPED was automatically applied to both **Average Revenue** and **Click-Through-Rate**, achieving over **68%** and **71%** variance reduction respectively! This dramatically narrowed our confidence intervals and amplified our statistical power.

#### Forest Plot Visualization
Call `.plot()` to render a gorgeous forest plot representing confidence intervals. Statistically significant lifts are automatically rendered in vibrant teal, while others are shown in subtle gray.

```python
# Render the forest plot
results.plot()
```

#### Covariate Balance Verification (Love Plot)
```python
# Print an ASCII love plot directly in the console
print(results.love_plot())
```

---

### 5. Generate Standalone HTML Reports

With the v1 release, you can export beautiful standalone HTML dashboards or Markdown cards representing your experimental results, complete with embedded modern styling, KPI metrics, and covariate balance logs.

```python
from xpyrment.report.generator import ExperimentReportGenerator

# Initialize the report generator with the analysis results
reporter = ExperimentReportGenerator(results, experiment_name="Mobile Landing Page Redesign")

# Save a premium responsive HTML dashboard (fully styled, self-contained)
reporter.save_html("reports/ab_experiment_dashboard.html")

# Save a GitHub-compatible Markdown summary card
reporter.save_markdown("reports/ab_experiment_summary.md")
```

---

## 🔬 Subpackage Taxonomy & Dependency Flow

To support industrial-scale digital tests and classical DoE, the package has been structured under `src/xpyrment` following a one-way dependency gating layout to avoid circular references:

```text
metrics/     ← Houses core metric taxonomy and guardrail thresholds.
core/        ← Powers the phase gating lifecycle & spec registries.
plan/        ← Computes pre-registration power/durations.
design/      ← Handles randomizations, splits & DoE matrices.
validate/    ← Houses SRM checks and covariate balance tests.
run/         ← Handles ingestion & mSPRT monitors.
analyze/     ← Orchestrates frequentist/Bayesian engines.
interactions/← Decomposes multi-factor ANOVA interaction terms.
interpret/   ← Infers ship/no-ship decisions.
report/      ← Terminal consumer of all phases. Compiles audit trails & exportable reports.
```

---

## 📖 Mathematical Framework

### Welch's t-test
For continuous metrics without a pre-period covariate, the standard error of the mean difference is:
$$SE = \sqrt{\frac{s_C^2}{n_C} + \frac{s_T^2}{n_T}}$$
Degrees of freedom are computed via the Welch-Satterthwaite equation to handle unequal sample sizes and variances.

### Delta Method (Ratio Metrics)
Because click-through-rates or revenue ratios are calculated as:
$$R = \frac{\sum_i X_i}{\sum_i Y_i} = \frac{\bar{X}}{\bar{Y}}$$
the variance of the ratio cannot be computed using standard methods because the denominator $Y$ is a random variable. We employ a first-order Taylor expansion (Delta method) to estimate variance:
$$Var(R) \approx \frac{1}{\mu_Y^2} Var(X) + \frac{\mu_X^2}{\mu_Y^4} Var(Y) - 2\frac{\mu_X}{\mu_Y^3} Cov(X, Y)$$

### CUPED (Controlled-experiments Using Pre-Experiment Data)
CUPED adjusts post-period metrics by subtracting the portion of variance explained by pre-period performance:
$$Y_i^* = Y_i - \theta (X_i - \mu_{X, global})$$
where $\theta = \frac{Cov(Y, X)}{Var(X)}$ is computed across the pooled data.
The variance of the CUPED-adjusted metric is reduced by a factor of $1 - \rho^2$ (where $\rho$ is the correlation coefficient):
$$Var(Y^*) = Var(Y) (1 - \rho^2)$$

For ratio metrics, `xpyrment` applies CUPED adjustment separately to the numerator and denominator before applying the Delta method on adjusted vectors—a technique pioneered by Netflix and Uber.

### Sample Ratio Mismatch (SRM) Goodness-of-Fit
A Pearson Chi-square test is calculated on the observed sample counts against the expected design weights to flag assignment bugs early:
$$\chi^2 = \sum_i \frac{(O_i - E_i)^2}{E_i}$$
If the test p-value $< 0.001$, an `SRMError` is raised.

---

## 🛠️ Local Development & Testing

We use `pytest` for unit testing. To set up your local environment:

1. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On macOS/Linux
   ```

2. Install the package in editable mode with development dependencies:
   ```bash
   pip install -e .[dev]
   ```

3. Run the unit test suite:
   ```bash
   pytest
   ```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
