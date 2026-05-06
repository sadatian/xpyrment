import numpy as np
import pandas as pd
from xpyrment.simulation import generate_ab_data


def test_generate_ab_data():
    """Tests shape and boundary values of synthetic A/B datasets generated for xpyrment."""
    n_samples = 1000
    df = generate_ab_data(n_samples=n_samples, treatment_fraction=0.5, random_seed=42)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == n_samples

    expected_cols = {
        "user_id",
        "variant",
        "pre_revenue",
        "revenue",
        "converted",
        "pre_impressions",
        "pre_clicks",
        "impressions",
        "clicks",
    }
    assert expected_cols.issubset(set(df.columns))

    # Assert variant assignments are correct
    assert set(df["variant"].unique()) == {"control", "treatment"}

    # Assert non-negativity
    assert (df["pre_revenue"] >= 0).all()
    assert (df["revenue"] >= 0).all()
    assert ((df["converted"] == 0) | (df["converted"] == 1)).all()
