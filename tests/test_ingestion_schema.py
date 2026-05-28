import pytest
import pandas as pd
from unittest import mock
import builtins
from xpyrment.run.ingestion import ingest_dataframe

def test_ingest_dataframe_schema_success():
    """Test dynamic schema enforcement with valid data types."""
    pytest.importorskip("pandera.pandas")
    df = pd.DataFrame({
        "id": ["u1", "u2"],
        "joined": ["2026-05-01", "2026-05-02"],
        "metric": [10.5, 20.0],
        "cat": ["A", "B"]
    })

    clean_df = ingest_dataframe(
        df,
        unit_id_col="id",
        time_col="joined",
        metric_cols=["metric"],
        categorical_cols=["cat"]
    )

    # Basic shape / column presence checks
    assert len(clean_df) == 2
    assert set(clean_df.columns) >= {"id", "joined", "metric", "cat"}

    # Type enforcement / coercion checks
    assert pd.api.types.is_float_dtype(clean_df["metric"])
    assert pd.api.types.is_datetime64_ns_dtype(clean_df["joined"])

def test_ingest_dataframe_schema_type_error():
    """Test that schema validation fails when types are incorrect."""
    pytest.importorskip("pandera.pandas")
    df = pd.DataFrame({
        "id": ["u1", "u2"],
        "metric": ["not_a_number", "20.0"] # This should fail since we expect a float
    })

    with pytest.raises(ValueError, match="Dynamic schema validation failed:"):
        ingest_dataframe(df, unit_id_col="id", metric_cols=["metric"])

def test_ingest_dataframe_with_custom_schema():
    """Test that user can provide a custom pandera schema."""
    pa = pytest.importorskip("pandera.pandas")
    schema = pa.DataFrameSchema({
        "score": pa.Column(int, pa.Check.ge(0))
    })

    df = pd.DataFrame({"score": [10, 20]})
    clean_df = ingest_dataframe(df, schema=schema)
    assert len(clean_df) == 2

    # Invalid data for custom schema
    df_invalid = pd.DataFrame({"score": [-5, 10]})
    with pytest.raises(ValueError, match="Schema validation failed:"):
        ingest_dataframe(df_invalid, schema=schema)

def test_ingest_dataframe_with_custom_schema_multiple_errors():
    """Test that multiple custom schema violations are aggregated correctly."""
    pa = pytest.importorskip("pandera.pandas")
    schema = pa.DataFrameSchema({
        "score": pa.Column(int, pa.Check.ge(0)),
        "bonus": pa.Column(float, pa.Check.le(1.0)),
    })

    # Both rows violate both columns, leading to multiple schema errors
    df_invalid = pd.DataFrame({
        "score": [-1, -5],
        "bonus": [1.5, 2.0],
    })

    with pytest.raises(ValueError, match="Schema validation failed:"):
        ingest_dataframe(df_invalid, schema=schema)


def test_ingest_dataframe_missing_pandera():
    """Test clear error message when pandera is not installed."""
    df = pd.DataFrame({"id": ["u1"]})

    original_import = builtins.__import__
    def mock_import(name, *args, **kwargs):
        if name == "pandera" or name == "pandera.pandas" or name == "pandera.errors":
            raise ImportError("No module named 'pandera'")
        return original_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=mock_import):
        # We must explicitly provide a schema to trigger the ImportError now.
        with pytest.raises(ImportError, match="pandera is required"):
            ingest_dataframe(df, schema="mock_schema", unit_id_col="id")

        # When no schema is provided, we expect it to return normally (skipping schema enforcement)
        clean_df = ingest_dataframe(df, unit_id_col="id")
        assert len(clean_df) == 1
