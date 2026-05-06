import pandas as pd


def check_novelty_effects(df: pd.DataFrame, treatment_col: str, metric_col: str, time_col: str) -> dict:
    """Detects novelty or primacy effects by tracking treatment effect size evolution over time."""
    # TODO: Implement time-series slope/interaction checks
    return {}
