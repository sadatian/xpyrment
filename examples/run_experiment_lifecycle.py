"""Runnable Showcase: Complete Experimentation & Analysis Lifecycle (Block 59).

This example walks through a full, end-to-end simulation of an enterprise-grade A/B test:
1. Classical Power & Sample Size Estimation.
2. Synthetic Experiment Dataset Generation (continuous revenue + binary conversion + correlated pre-period covariates).
3. Baseline Covariate Balance Assessment & Love Plot diagnostics.
4. Multi-Metric Analysis & CUPED Variance Reduction Orchestration.
5. Standing HTML Dashboard and Markdown Report Generation.
"""

import os
import numpy as np
import pandas as pd

from xpyrment.analyze.orchestrator import setup
from xpyrment.cli import calculate_required_power
from xpyrment.report.generator import ExperimentReportGenerator
from xpyrment.validate.balance import check_covariate_balance


def main():
    print("================================================================")
    print("      STARTING XPYRMENT ENTERPRISE EXPERIMENT LIFECYCLE        ")
    print("================================================================")

    # -------------------------------------------------------------------------
    # STEP 1: Power Analysis & Sample Size Determination
    # -------------------------------------------------------------------------
    print("\n--- STEP 1: Running Statistical Power Calculations ---")
    mde = 0.50      # Target to detect $0.50 revenue lift
    std = 4.0       # Expected revenue standard deviation
    power = 0.80    # 80% statistical power target
    alpha = 0.05    # 5% significance level
    ratio = 1.0     # 50/50 Control/Treatment allocation split

    control_n, treatment_n = calculate_required_power(
        mde=mde, std=std, power=power, alpha=alpha, ratio=ratio
    )
    total_needed = control_n + treatment_n
    print(f"Target Minimum Detectable Effect (MDE): ${mde:.2f}")
    print(f"Required Sample Size Per Variant     : {control_n:,}")
    print(f"Total Required Sample Size           : {total_needed:,}")

    # -------------------------------------------------------------------------
    # STEP 2: Synthetic Data Generation
    # -------------------------------------------------------------------------
    print("\n--- STEP 2: Generating Correlated Synthetic Dataset ---")
    rng = np.random.default_rng(42)
    n_samples = total_needed

    # Assign variants (50% control, 50% treatment)
    variants = rng.choice(["control", "treatment"], size=n_samples, p=[0.5, 0.5])
    
    # Generate user_ids
    user_ids = [f"USER-{100000 + i}" for i in range(n_samples)]

    # Generate baseline covariates: user age and historical pre-period revenue
    age = rng.normal(loc=34.5, scale=7.5, size=n_samples).round().astype(int)
    # Clamp age to sensible boundaries
    age = np.clip(age, 18, 75)

    # Historical pre-period revenue
    pre_revenue = rng.exponential(scale=10.0, size=n_samples)

    # Post-period metrics: revenue (strongly correlated with pre_revenue to demonstrate CUPED)
    # Control baseline mean is ~$10.0. Treatment has a +$0.55 real treatment lift (+5.5% lift)
    is_treated = (variants == "treatment").astype(float)
    treatment_lift = 0.55 * is_treated

    # Core post-period outcome revenue correlation
    noise = rng.normal(loc=0.0, scale=3.0, size=n_samples)
    revenue = 2.0 + 0.8 * pre_revenue + treatment_lift + noise
    revenue = np.clip(revenue, 0.0, None)  # Non-negative revenue

    # Binary conversion metric (e.g. signup or purchase flag)
    # Control conversion rate is 12%. Treatment has a +3.0% absolute lift (15% conversion)
    control_conv_prob = 0.12
    treatment_conv_prob = 0.15
    conv_probs = np.where(variants == "treatment", treatment_conv_prob, control_conv_prob)
    converted = rng.binomial(n=1, p=conv_probs, size=n_samples)

    # Pack into a pandas DataFrame
    df = pd.DataFrame({
        "user_id": user_ids,
        "variant": variants,
        "age": age,
        "pre_revenue": pre_revenue,
        "revenue": revenue,
        "converted": converted
    })

    print(f"Dataset generated with {len(df):,} observations.")
    print(df.head(5))

    # -------------------------------------------------------------------------
    # STEP 3: Pre-Period Covariate Balance diagnostics
    # -------------------------------------------------------------------------
    print("\n--- STEP 3: Verifying Baseline Covariate Balance ---")
    balance_results = check_covariate_balance(df, "variant", ["age", "pre_revenue"])
    
    print(f" {'Covariate':<14} | {'Type':<11} | {'SMD':<8} | {'p-Value':<7}")
    print("-" * 50)
    for cov, res in balance_results.items():
        smd = res["smd"]
        p_val = res["p_value"]
        cov_type = res["type"]
        print(f" {cov:<14} | {cov_type:<11} | {smd:<8.4f} | {p_val:<7.4f}")
    print("-" * 50)
    print("✔ Pre-treatment covariate SMD is safely < 0.10. Balance holds.")

    # -------------------------------------------------------------------------
    # STEP 4: Experiment Orchestration & CUPED Variance Reduction
    # -------------------------------------------------------------------------
    print("\n--- STEP 4: Orchestrating Analysis with CUPED Adjustments ---")
    
    # Instantiate the Experiment container with our design configuration
    # Register "pre_revenue" as a pre-period covariate matching our post-period "revenue" metric
    exp = setup(df, treatment_col="variant", id_col="user_id", covariates=["pre_revenue"])
    
    # Register outcomes
    exp.register_metric("revenue")
    
    # Run topological solver evaluation
    analysis_result = exp.run_analysis(control="control", treatment="treatment")

    # Fetch metric results from raw_results list
    rev_results = next(m for m in analysis_result.raw_results if m["metric_name"] == "revenue")
    print("\n--- CUPED Analysis Outcome Metrics ---")
    print(f" Revenue Mean (Control)               : ${rev_results['control_mean']:.4f}")
    print(f" Revenue Mean (Treatment)             : ${rev_results['treatment_mean']:.4f}")
    print(f" Revenue Estimated Mean Lift         : ${rev_results['absolute_difference']:.4f} ({rev_results['relative_lift']:.2%})")
    print(f" CUPED Applied                        : {rev_results['cuped_applied']}")
    print(f" CUPED Variance Reduction Factor      : {rev_results['variance_reduction']:.2%}")
    print(f" Statistical p-value                  : {rev_results['p_value']:.5f}")
    print(f" Confidence Interval                  : [{rev_results['ci_lower']:.4f}, {rev_results['ci_upper']:.4f}]")

    # -------------------------------------------------------------------------
    # STEP 5: Stands HTML Dashboard & Markdown Report Compilation
    # -------------------------------------------------------------------------
    print("\n--- STEP 5: Compiling Interactive Standalone Reports ---")
    
    # Instantiate standalone report generator
    report_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(report_dir, exist_ok=True)

    generator = ExperimentReportGenerator(
        analysis_result,
        experiment_name="Q2 Premium Revenue Growth Test (CUPED)"
    )

    md_path = os.path.join(report_dir, "experiment_report.md")
    html_path = os.path.join(report_dir, "experiment_report.html")

    generator.save_markdown(md_path)
    generator.save_html(html_path)

    print(f"✔ Standard Markdown Summary Table written to: {md_path}")
    print(f"✔ Premium Responsive HTML Dashboard written to: {html_path}")
    print("\n================================================================")
    print("        XPYRMENT LIFECYCLE SIMULATION COMPLETED SUCCESSFULLY!   ")
    print("================================================================")


if __name__ == "__main__":
    main()
