"""Command-Line Interface (CLI) & Schema Validator (Block 58).

Provides a premium command-line wrapper to perform sample size/power calculations, verify
covariate balance profiles from custom CSV datasets, and run rapid ordinary least squares regressions.
"""

import argparse
import math
import os
import sys
from typing import List, Optional
import numpy as np
import pandas as pd
from scipy.stats import norm

from xpyrment.quasi.diff_in_diff import fit_ols
from xpyrment.validate.balance import check_covariate_balance


def calculate_required_power(
    mde: float,
    std: float,
    power: float = 0.80,
    alpha: float = 0.05,
    ratio: float = 1.0
) -> tuple:
    """Computes sample size bounds for standard two-sample Z-test with arbitrary allocation ratio.

    Args:
        mde (float): Minimum Detectable Effect size.
        std (float): Standard deviation of target metric.
        power (float): Target statistical power (1 - beta). Defaults to 0.80.
        alpha (float): Significance level (Type I error). Defaults to 0.05.
        ratio (float): Allocation ratio (N_T / N_C). Defaults to 1.0.

    Returns:
        tuple: Required (control_size, treatment_size) integers.
    """
    if mde <= 0 or std <= 0 or ratio <= 0:
        raise ValueError("MDE, standard deviation, and allocation ratio must be strictly positive.")
    if not (0.0 < power < 1.0) or not (0.0 < alpha < 1.0):
        raise ValueError("Power and alpha values must lie strictly between 0 and 1.")

    z_alpha = norm.ppf(1.0 - alpha / 2.0)
    z_beta = norm.ppf(power)

    # N_C = ((z_a + z_b)^2 * (1 + 1/r) * std^2) / MDE^2
    nc = ((z_alpha + z_beta) ** 2 * (1.0 + 1.0 / ratio) * (std ** 2)) / (mde ** 2)
    control_size = math.ceil(nc)
    treatment_size = math.ceil(control_size * ratio)

    return (control_size, treatment_size)


def handle_power(args: argparse.Namespace) -> None:
    """Handles parsing and console printing for the power calculation subcommand."""
    try:
        nc, nt = calculate_required_power(
            mde=args.mde,
            std=args.std,
            power=args.power,
            alpha=args.alpha,
            ratio=args.ratio
        )

        print("==================================================")
        print("         POWER & REQUIRED SAMPLE SIZE            ")
        print("==================================================")
        print(f" Minimum Detectable Effect (MDE) : {args.mde:.4f}")
        print(f" Metric Standard Deviation       : {args.std:.4f}")
        print(f" Target Statistical Power        : {args.power:.2f}")
        print(f" Significance Level (Alpha)      : {args.alpha:.3f}")
        print(f" Allocation Ratio (N_T / N_C)    : {args.ratio:.2f}")
        print("--------------------------------------------------")
        print(f" Required Control Group Size     : {nc:,}")
        print(f" Required Treatment Group Size   : {nt:,}")
        print(f" Total Required Sample Size      : {nc + nt:,}")
        print("==================================================")

    except ValueError as e:
        print(f"Power Calculation Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


def handle_balance(args: argparse.Namespace) -> None:
    """Handles reading CSV files, running standardized balance checks, and printing ASCII summary matrices."""
    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found at '{args.csv}'", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}", file=sys.stderr)
        sys.exit(1)

    # Ensure targeted columns are contained
    if args.group_col not in df.columns:
        print(f"Error: Group column '{args.group_col}' not found in CSV columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    covariates = [col.strip() for col in args.covariates.split(",") if col.strip()]
    for cov in covariates:
        if cov not in df.columns:
            print(f"Error: Covariate column '{cov}' not found in CSV columns.", file=sys.stderr)
            sys.exit(1)

    try:
        results = check_covariate_balance(df, args.group_col, covariates)

        print("==================================================")
        print("          COVARIATE BALANCE REPORT (SMD)          ")
        print("==================================================")
        print(f" Dataset Path    : {args.csv}")
        print(f" Group Column    : {args.group_col}")
        print("--------------------------------------------------")
        print(f" {'Covariate':<14} | {'Type':<11} | {'SMD':<8} | {'p-Value':<7}")
        print("--------------------------------------------------")

        imbalanced_detected = False
        for cov, res in results.items():
            smd = res.get("smd", 0.0)
            p_val = res.get("p_value", 1.0)
            cov_type = res.get("type", "numeric")
            
            # Highlight SMD warnings above standard threshold 0.1
            smd_flag = "⚠️ " if abs(smd) > 0.1 else "  "
            
            print(f"{smd_flag}{cov:<12} | {cov_type:<11} | {smd:<8.4f} | {p_val:<7.4f}")
            if abs(smd) > 0.1:
                imbalanced_detected = True

        print("--------------------------------------------------")
        if imbalanced_detected:
            print(" ⚠️  WARNING: Some covariates are significantly imbalanced (SMD > 0.1).")
        else:
            print(" ✔  All targeted covariates are well balanced (SMD <= 0.1).")
        print("==================================================")

    except Exception as e:
        print(f"Balance Verification Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


def handle_regress(args: argparse.Namespace) -> None:
    """Handles reading CSV datasets, parsing regression vectors, fitting OLS solvers, and printing reports."""
    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found at '{args.csv}'", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}", file=sys.stderr)
        sys.exit(1)

    if args.y_col not in df.columns:
        print(f"Error: Dependent variable '{args.y_col}' not found in CSV columns.", file=sys.stderr)
        sys.exit(1)

    x_cols = [col.strip() for col in args.x_cols.split(",") if col.strip()]
    for col in x_cols:
        if col not in df.columns:
            print(f"Error: Predictor variable '{col}' not found in CSV columns.", file=sys.stderr)
            sys.exit(1)

    try:
        y = df[args.y_col].to_numpy()
        X = df[x_cols].to_numpy()

        results = fit_ols(X, y)

        beta = results["beta"]
        se = results["standard_errors"]
        t_stats = results["t_stats"]
        p_values = results["p_values"]
        df_error = results["df"]

        print("=============================================================")
        print("                 OLS REGRESSION ANALYSIS REPORT              ")
        print("=============================================================")
        print(f" Dataset Path    : {args.csv}")
        print(f" Dependent Var   : {args.y_col}")
        print("-------------------------------------------------------------")
        print(f" {'Parameter':<14} | {'Coefficient':<11} | {'Std Error':<9} | {'t-Stat':<8} | {'p-Value':<7}")
        print("-------------------------------------------------------------")

        # Intercept is the first beta entry
        p_val_str = f"{p_values[0]:<7.4f}" if p_values[0] >= 0.001 else "<0.001 "
        print(f" {'Intercept':<14} | {beta[0]:<11.4f} | {se[0]:<9.4f} | {t_stats[0]:<8.3f} | {p_val_str}")

        # Individual covariates follow
        for idx, col in enumerate(x_cols):
            coef_val = beta[idx + 1]
            se_val = se[idx + 1]
            t_val = t_stats[idx + 1]
            p_val = p_values[idx + 1]
            p_val_s = f"{p_val:<7.4f}" if p_val >= 0.001 else "<0.001 "

            print(f" {col:<14} | {coef_val:<11.4f} | {se_val:<9.4f} | {t_val:<8.3f} | {p_val_s}")

        print("-------------------------------------------------------------")
        print(f" Residual Degrees of Freedom: {df_error}")
        print("=============================================================")

    except Exception as e:
        print(f"Regression Execution Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    """Builds and returns the master command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="xpyrment",
        description="Enterprise-grade experiment design, classical DoE, and statistical analysis CLI wrapper."
    )

    subparsers = parser.add_subparsers(dest="subcommand", required=True, help="Subcommands to execute.")

    # 1. Power calculator
    power_parser = subparsers.add_parser("power", help="Run Z-test sample size / power calculations.")
    power_parser.add_argument("--mde", type=float, required=True, help="Minimum Detectable Effect.")
    power_parser.add_argument("--std", type=float, required=True, help="Standard deviation of outcome metric.")
    power_parser.add_argument("--power", type=float, default=0.80, help="Target power level (0 to 1). Defaults to 0.80.")
    power_parser.add_argument("--alpha", type=float, default=0.05, help="Significance alpha level (0 to 1). Defaults to 0.05.")
    power_parser.add_argument("--ratio", type=float, default=1.0, help="Treatment-to-Control allocation ratio. Defaults to 1.0.")

    # 2. Covariate Balance diagnostic
    balance_parser = subparsers.add_parser("balance", help="Evaluate baseline covariate balance from a CSV dataset.")
    balance_parser.add_argument("--csv", type=str, required=True, help="Path to input CSV dataset file.")
    balance_parser.add_argument("--group-col", type=str, required=True, help="CSV column name containing assignment group/variants.")
    balance_parser.add_argument("--covariates", type=str, required=True, help="Comma-separated list of covariate column names.")

    # 3. Quick OLS regress solver
    regress_parser = subparsers.add_parser("regress", help="Fit an OLS regression solver to arbitrary CSV outcomes.")
    regress_parser.add_argument("--csv", type=str, required=True, help="Path to input CSV dataset file.")
    regress_parser.add_argument("--y-col", type=str, required=True, help="Outcome target variable name.")
    regress_parser.add_argument("--x-cols", type=str, required=True, help="Comma-separated predictor column names.")

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Central entry point routing args to targeted handler functions."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.subcommand == "power":
        handle_power(args)
    elif args.subcommand == "balance":
        handle_balance(args)
    elif args.subcommand == "regress":
        handle_regress(args)


if __name__ == "__main__":
    main()
