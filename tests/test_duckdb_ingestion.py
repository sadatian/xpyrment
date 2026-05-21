"""Unit tests for High-Performance Parquet & DuckDB Ingestion (Block 62)."""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from xpyrment.run.ingestion import DuckDBIngester


@pytest.fixture
def mock_dataset_paths(tmp_path):
    """Fixture to generate standard mock datasets as Parquet files."""
    rng = np.random.default_rng(42)
    N = 1000

    # 1. Standard dataset
    df = pd.DataFrame({
        "user_id": [f"u_{i}" for i in range(N)],
        "treatment": rng.choice(["control", "treatment"], size=N),
        # Numeric covariates
        "age": rng.normal(loc=35.0, scale=10.0, size=N),
        "income": rng.normal(loc=50000.0, scale=15000.0, size=N),
        # Categorical covariates
        "device": rng.choice(["desktop", "mobile", "tablet"], size=N, p=[0.5, 0.4, 0.1]),
        "country": rng.choice(["US", "CA", "GB"], size=N, p=[0.7, 0.2, 0.1]),
        # Continuous metrics
        "revenue": rng.exponential(scale=10.0, size=N),
        "clicks": rng.poisson(lam=5.0, size=N).astype(float)
    })

    # Shift income slightly in treatment to simulate standard balance checks
    df.loc[df["treatment"] == "treatment", "income"] += 5000.0
    
    # 2. Degenerate variance dataset
    df_degenerate = pd.DataFrame({
        "treatment": ["control", "control", "treatment", "treatment"],
        "const_cov": [5.0, 5.0, 5.0, 5.0],
        "const_metric": [10.0, 10.0, 10.0, 10.0]
    })

    # 3. Small sample size dataset
    df_small = pd.DataFrame({
        "treatment": ["control", "treatment", "treatment"],
        "metric": [1.0, 2.0, 3.0]
    })

    # 4. Multi-arm dataset
    df_multi = pd.DataFrame({
        "treatment": rng.choice(["A", "B", "C"], size=N),
        "cov": rng.normal(size=N),
        "metric": rng.normal(size=N)
    })

    # Save to Parquet files
    path_standard = tmp_path / "standard.parquet"
    path_degenerate = tmp_path / "degenerate.parquet"
    path_small = tmp_path / "small.parquet"
    path_multi = tmp_path / "multi.parquet"
    path_empty = tmp_path / "empty.parquet"

    df.to_parquet(path_standard)
    df_degenerate.to_parquet(path_degenerate)
    df_small.to_parquet(path_small)
    df_multi.to_parquet(path_multi)
    
    # Empty schema dataframe
    pd.DataFrame(columns=["treatment", "age", "revenue"]).to_parquet(path_empty)

    return {
        "standard": str(path_standard),
        "degenerate": str(path_degenerate),
        "small": str(path_small),
        "multi": str(path_multi),
        "empty": str(path_empty)
    }


def test_ingester_lifecycle_and_query(tmp_path):
    """Verifies DuckDB connection lifecycles, file system locks, and raw queries."""
    # 1. Transient memory db
    with DuckDBIngester(":memory:") as ingester:
        res = ingester.query("SELECT 1 as val, 'hello' as greeting")
        assert len(res) == 1
        assert res.iloc[0]["val"] == 1
        assert res.iloc[0]["greeting"] == "hello"

    # 2. Persistent file db
    db_file = tmp_path / "test.db"
    ingester = DuckDBIngester(str(db_file))
    res = ingester.query("SELECT 42 as answer")
    assert res.iloc[0]["answer"] == 42
    ingester.close()

    # The file should exist and lock be released (allowing reuse or deletion)
    assert os.path.exists(db_file)
    try:
        os.remove(db_file)
    except OSError:
        pytest.fail("Database connection was not cleanly closed, lock not released.")


def test_compute_covariate_balance_numeric(mock_dataset_paths):
    """Validates compute_covariate_balance for continuous/numeric covariates."""
    parquet_path = mock_dataset_paths["standard"]

    with DuckDBIngester() as ingester:
        res = ingester.compute_covariate_balance(
            parquet_path=parquet_path,
            treatment_col="treatment",
            covariate_cols=["age", "income"],
            control_group="control",
            treatment_group="treatment"
        )

        assert "age" in res
        assert "income" in res
        
        # Verify age balance (should be highly balanced)
        assert res["age"]["type"] == "numeric"
        assert isinstance(res["age"]["smd"], float)
        assert isinstance(res["age"]["p_value"], float)

        # Verify income balance (should have a significant SMD shift)
        assert res["income"]["type"] == "numeric"
        assert abs(res["income"]["smd"]) > 0.1

        # Double-check mathematically against pandas & scipy
        df = pd.read_parquet(parquet_path)
        ctrl_age = df[df["treatment"] == "control"]["age"].dropna()
        treat_age = df[df["treatment"] == "treatment"]["age"].dropna()

        # In-memory manual calculation
        mean_c, mean_t = ctrl_age.mean(), treat_age.mean()
        var_c, var_t = ctrl_age.var(ddof=1), treat_age.var(ddof=1)
        pooled_sd = np.sqrt((var_c + var_t) / 2.0)
        expected_smd = (mean_t - mean_c) / pooled_sd

        # In-memory Welch t-test
        _, expected_p = stats.ttest_ind(treat_age, ctrl_age, equal_var=False)

        assert np.allclose(res["age"]["smd"], expected_smd, rtol=1e-5, atol=1e-8)
        assert np.allclose(res["age"]["p_value"], expected_p, rtol=1e-5, atol=1e-8)


def test_compute_covariate_balance_categorical(mock_dataset_paths):
    """Validates compute_covariate_balance for categorical covariates."""
    parquet_path = mock_dataset_paths["standard"]

    with DuckDBIngester() as ingester:
        res = ingester.compute_covariate_balance(
            parquet_path=parquet_path,
            treatment_col="treatment",
            covariate_cols=["device", "country"],
            control_group="control",
            treatment_group="treatment"
        )

        assert "device" in res
        assert "country" in res

        assert res["device"]["type"] == "categorical"
        assert isinstance(res["device"]["p_value"], float)

        # Verify against scipy chi2 contingency table
        df = pd.read_parquet(parquet_path)
        contingency = pd.crosstab(df["device"], df["treatment"])
        _, expected_p, _, _ = stats.chi2_contingency(contingency.values)

        assert np.allclose(res["device"]["p_value"], expected_p, rtol=1e-5, atol=1e-8)


def test_compute_covariate_balance_categorical_single_category(tmp_path):
    """Validates that a single categorical category across groups fallback works gracefully."""
    # Pivot contingency with shape (1, 2) which stats.chi2_contingency cannot run on
    df = pd.DataFrame({
        "treatment": ["control", "treatment", "control", "treatment"],
        "device": ["desktop", "desktop", "desktop", "desktop"]
    })
    path = tmp_path / "single_category.parquet"
    df.to_parquet(path)

    with DuckDBIngester() as ingester:
        res = ingester.compute_covariate_balance(
            parquet_path=str(path),
            treatment_col="treatment",
            covariate_cols=["device"],
            control_group="control",
            treatment_group="treatment"
        )
        assert res["device"]["type"] == "categorical"
        assert res["device"]["p_value"] == 1.0  # Fully balanced fallback


def test_compute_welch_statistics(mock_dataset_paths):
    """Validates compute_welch_statistics output format, values, and calculations."""
    parquet_path = mock_dataset_paths["standard"]

    with DuckDBIngester() as ingester:
        metrics = ["revenue", "clicks"]
        res = ingester.compute_welch_statistics(
            parquet_path=parquet_path,
            treatment_col="treatment",
            metric_cols=metrics,
            control_group="control",
            treatment_group="treatment",
            alpha=0.05
        )

        assert "revenue" in res
        assert "clicks" in res

        rev = res["revenue"]
        # Schema verify
        expected_keys = {
            "control_mean", "treatment_mean", "control_var", "treatment_var",
            "control_n", "treatment_n", "t_statistic", "p_value", "df",
            "difference", "ci_lower", "ci_upper", "significant"
        }
        assert set(rev.keys()) == expected_keys

        # Mathematically verify against scipy & numpy
        df = pd.read_parquet(parquet_path)
        ctrl = df[df["treatment"] == "control"]["revenue"].dropna()
        treat = df[df["treatment"] == "treatment"]["revenue"].dropna()

        n_0, n_1 = len(ctrl), len(treat)
        mean_0, mean_1 = ctrl.mean(), treat.mean()
        var_0, var_1 = ctrl.var(ddof=1), treat.var(ddof=1)
        diff = mean_1 - mean_0
        se_diff = np.sqrt(var_0 / n_0 + var_1 / n_1)

        t_stat, p_val = stats.ttest_ind(treat, ctrl, equal_var=False)

        # Welch-Satterthwaite DF
        num = (var_0 / n_0 + var_1 / n_1) ** 2
        den = ((var_0 / n_0) ** 2) / (n_0 - 1) + ((var_1 / n_1) ** 2) / (n_1 - 1)
        expected_df = num / den

        expected_ci_half = stats.t.ppf(0.975, df=expected_df) * se_diff
        expected_ci_lower = diff - expected_ci_half
        expected_ci_upper = diff + expected_ci_half

        # Assert precision
        assert np.allclose(rev["control_mean"], mean_0, rtol=1e-5, atol=1e-8)
        assert np.allclose(rev["treatment_mean"], mean_1, rtol=1e-5, atol=1e-8)
        assert np.allclose(rev["control_var"], var_0, rtol=1e-5, atol=1e-8)
        assert np.allclose(rev["treatment_var"], var_1, rtol=1e-5, atol=1e-8)
        assert rev["control_n"] == n_0
        assert rev["treatment_n"] == n_1
        assert np.allclose(rev["t_statistic"], t_stat, rtol=1e-5, atol=1e-8)
        assert np.allclose(rev["p_value"], p_val, rtol=1e-5, atol=1e-8)
        assert np.allclose(rev["df"], expected_df, rtol=1e-5, atol=1e-8)
        assert np.allclose(rev["difference"], diff, rtol=1e-5, atol=1e-8)
        assert np.allclose(rev["ci_lower"], expected_ci_lower, rtol=1e-5, atol=1e-8)
        assert np.allclose(rev["ci_upper"], expected_ci_upper, rtol=1e-5, atol=1e-8)
        assert rev["significant"] == bool(p_val < 0.05)


def test_missing_files_and_columns(mock_dataset_paths):
    """Validates correct error codes are raised for missing files or schema fields."""
    with DuckDBIngester() as ingester:
        # File path does not exist
        with pytest.raises(FileNotFoundError, match="Parquet file/directory not found"):
            ingester.compute_covariate_balance("nonexistent.parquet", "treatment", ["age"])

        # Column does not exist
        parquet_path = mock_dataset_paths["standard"]
        with pytest.raises(KeyError, match="Treatment column 'nonexistent_col' not found"):
            ingester.compute_covariate_balance(parquet_path, "nonexistent_col", ["age"])

        with pytest.raises(KeyError, match="Covariate column 'nonexistent_cov' not found"):
            ingester.compute_covariate_balance(parquet_path, "treatment", ["age", "nonexistent_cov"])

        with pytest.raises(KeyError, match="Metric column 'nonexistent_metric' not found"):
            ingester.compute_welch_statistics(parquet_path, "treatment", ["nonexistent_metric"])


def test_empty_dataset(mock_dataset_paths):
    """Validates that ingestion fails gracefully on zero-record tables."""
    parquet_path = mock_dataset_paths["empty"]
    with DuckDBIngester() as ingester:
        # Note: Empty dataframe to_parquet creates the columns with no rows.
        # But wait, our DuckDBIngester checks COUNT(*) == 0 and raises ValueError.
        with pytest.raises(ValueError, match="Dataset is empty"):
            ingester.compute_covariate_balance(parquet_path, "treatment", ["age"])

        with pytest.raises(ValueError, match="Dataset is empty"):
            ingester.compute_welch_statistics(parquet_path, "treatment", ["revenue"])


def test_degenerate_variances(mock_dataset_paths):
    """Validates behavior on constant values causing degenerate zero-variances."""
    parquet_path = mock_dataset_paths["degenerate"]
    with DuckDBIngester() as ingester:
        with pytest.raises(ValueError, match="Degenerate variance"):
            ingester.compute_covariate_balance(parquet_path, "treatment", ["const_cov"])

        with pytest.raises(ValueError, match="Degenerate variance"):
            ingester.compute_welch_statistics(parquet_path, "treatment", ["const_metric"])


def test_small_sample_sizes(mock_dataset_paths):
    """Validates that arms with fewer than 2 samples raise informative ValueErrors."""
    parquet_path = mock_dataset_paths["small"]
    with DuckDBIngester() as ingester:
        # Control group has 1 sample (< 2)
        with pytest.raises(ValueError, match="Sample size too small"):
            ingester.compute_welch_statistics(parquet_path, "treatment", ["metric"])


def test_multi_arm_handling(mock_dataset_paths):
    """Validates multi-arm experiment safety checks and group selection."""
    parquet_path = mock_dataset_paths["multi"]
    with DuckDBIngester() as ingester:
        # 1. Multi-arm with no specified control/treatment group should fail
        with pytest.raises(ValueError, match="Multiple treatment arms detected"):
            ingester.compute_covariate_balance(parquet_path, "treatment", ["cov"])

        with pytest.raises(ValueError, match="Multiple treatment arms detected"):
            ingester.compute_welch_statistics(parquet_path, "treatment", ["metric"])

        # 2. Invalid specified groups should fail
        with pytest.raises(ValueError, match="not found in distinct groups"):
            ingester.compute_covariate_balance(parquet_path, "treatment", ["cov"], control_group="A", treatment_group="Z")

        # 3. Specifying groups correctly should succeed
        res = ingester.compute_covariate_balance(
            parquet_path=parquet_path,
            treatment_col="treatment",
            covariate_cols=["cov"],
            control_group="A",
            treatment_group="B"
        )
        assert "cov" in res

        stats_res = ingester.compute_welch_statistics(
            parquet_path=parquet_path,
            treatment_col="treatment",
            metric_cols=["metric"],
            control_group="A",
            treatment_group="C"
        )
        assert "metric" in stats_res
        assert stats_res["metric"]["control_n"] > 0
        assert stats_res["metric"]["treatment_n"] > 0
