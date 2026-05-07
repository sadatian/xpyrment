"""xpyrment: A unified, mathematically rigorous platform for experimental design and analysis.

The `xpyrment` library is a complete framework for online A/B testing and physical Design of Experiments (DoE).
It unifies the entire experiment lifecycle into a clean, fluent python API:
- **Design & Plan**: Analytical power curves, sample size calculators, classical fractional/definitive designs.
- **Validate**: Real-time Sample Ratio Mismatch (SRM) checks, covariate balance tests, and novelty effects.
- **Run & Monitor**: Chronological binning, continuous monitoring margins, first-touch attribution guards.
- **Analyze**: High-powered continuous and ratio CUPED variance reduction, Holm/BH multiple testing corrections.
- **Inference**: Flexible Frequentist (Welch's, Mann-Whitney U), Bayesian (conjugate models), Sequential (mSPRT, AVCI), and non-parametric Bootstrap engines.
- **Interpret**: Cohen's d effect sizes, subgroup HTE screening, and minimum valuable economic decisions.
- **Report**: Immutable cryptographic compliance trails, standard-compliant Experiment Cards, and publication-ready plots.

Fluent API Example Workflow:
    ```python
    import xpyrment as xp

    # 1. Generate synthetic A/B test data
    data = xp.generate_ab_data(n_samples=5000, pre_period_correlation=0.75)

    # 2. Setup the experiment container and register metrics/covariates
    exp = xp.setup(data, variant_col="variant", unit_col="user_id")
    exp.register_metric("revenue", metric_type="mean", covariate="pre_revenue")
    exp.register_metric("converted", metric_type="proportion")

    # 3. Perform pre-analysis SRM validation
    xp.check_srm(exp, expected_ratio=0.50)

    # 4. Execute the complete analysis workflow (applies CUPED and corrections)
    results = exp.run_analysis()

    # 5. Review results or output a standard Experiment Card
    card = results.to_experiment_card()
    print(card.to_json())
    ```
"""

from xpyrment._version import __version__
from xpyrment.core.experiment import Experiment
from xpyrment.metrics.taxonomy import BaseMetric, MeanMetric, ProportionMetric, RatioMetric
from xpyrment.plan.power import design_experiment, generate_power_curve_data
from xpyrment.validate.srm import check_srm
from xpyrment.analyze.orchestrator import setup, run_analysis
from xpyrment.report.export import plot_forest, plot_power_curve
from xpyrment.simulation import generate_ab_data

__all__ = [
    "__version__",
    "Experiment",
    "BaseMetric",
    "MeanMetric",
    "ProportionMetric",
    "RatioMetric",
    "design_experiment",
    "generate_power_curve_data",
    "check_srm",
    "setup",
    "run_analysis",
    "plot_forest",
    "plot_power_curve",
    "generate_ab_data",
]
