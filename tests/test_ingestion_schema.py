import pytest
import pandas as pd
from unittest import mock
import builtins
import pandera.pandas as pa
from xpyrment.run.ingestion import ingest_dataframe

def test_ingest_dataframe_schema_success():
    """Test dynamic schema enforcement with valid data types."""
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
    assert len(clean_df) == 2
    assert "metric" in clean_df.columns

def test_ingest_dataframe_schema_type_error():
    """Test that schema validation fails when types are incorrect."""
    df = pd.DataFrame({
        "id": ["u1", "u2"],
        "metric": ["not_a_number", "20.0"] # This should fail since we expect a float
    })

    with pytest.raises(ValueError, match="Dynamic schema validation failed:"):
        ingest_dataframe(df, unit_id_col="id", metric_cols=["metric"])

def test_ingest_dataframe_with_custom_schema():
    """Test that user can provide a custom pandera schema."""
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

def test_ingest_dataframe_missing_pandera():
    """Test clear error message when pandera is not installed."""
    df = pd.DataFrame({"id": ["u1"]})

    original_import = builtins.__import__
    def mock_import(name, *args, **kwargs):
        if name == "pandera" or name == "pandera.pandas":
            raise ImportError("No module named 'pandera'")
        return original_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=mock_import):
        with pytest.raises(ImportError, match="pandera is required"):
            ingest_dataframe(df, unit_id_col="id")
