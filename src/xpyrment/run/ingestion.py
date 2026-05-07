"""Data ingestion adapters, validation gates, and normalization utilities.

This module provides standard connectors and validation gates for feeding raw client-side or
server-side datasets (such as SQL tables or CSVs) into the `xpyrment` experimental workflow.
"""

import pandas as pd


def load_from_sql(query: str, connection_string: str) -> pd.DataFrame:
    r"""Loads experimental telemetry and assignment logs from an external relational SQL database.

    Fetches exposure matrices, pre-period metrics, and covariate vectors using high-performance
    database adapters.

    Args:
        query (str): The SQL retrieval query (e.g., `"SELECT user_id, variant, revenue FROM experimental_ledger"`).
        connection_string (str): The database connection URI.

    Returns:
        pd.DataFrame: A cleaned pandas DataFrame containing the queried records.
    """
    import sqlite3

    if ":memory:" in connection_string or "sqlite" in connection_string or connection_string == "":
        db_path = ":memory:" if (connection_string == "" or ":memory:" in connection_string) else connection_string.replace("sqlite:///", "")
        conn = sqlite3.connect(db_path)
        try:
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except Exception as e:
            conn.close()
            raise e
    else:
        try:
            import sqlalchemy
            engine = sqlalchemy.create_engine(connection_string)
            df = pd.read_sql_query(query, engine)
            return df
        except ImportError:
            raise ImportError("sqlalchemy is required to connect to non-SQLite databases.")


def ingest_dataframe(
    df: pd.DataFrame,
    unit_id_col: str = None,
    time_col: str = None,
    metric_cols: list = None,
    categorical_cols: list = None
) -> pd.DataFrame:
    """Ingests, validates, and copies an in-memory pandas DataFrame into the xpyrment lifecycle.

    Performs localized validation checks on the pandas DataFrame, ensuring all required column signatures
    are mapped correctly.

    Args:
        df (pd.DataFrame): The raw source DataFrame.
        unit_id_col (str): Column representing unit identifiers (nulls will be dropped).
        time_col (str): Column representing event timestamps (will be parsed to datetime).
        metric_cols (list): Continuous metric columns (nulls will be imputed to 0.0).
        categorical_cols (list): Categorical covariate columns (nulls will be imputed to "UNKNOWN").

    Returns:
        pd.DataFrame: An audited, isolated copy of the DataFrame ready for downstream operations.
    """
    df_clean = df.copy()

    # 1. Primary Key Integrities
    if unit_id_col is not None:
        if unit_id_col not in df_clean.columns:
            raise KeyError(f"unit_id column '{unit_id_col}' not found in DataFrame.")
        # Drop rows with null unit_id
        df_clean = df_clean.dropna(subset=[unit_id_col])

    # 2. Chronological Alignment
    if time_col is not None:
        if time_col not in df_clean.columns:
            raise KeyError(f"time column '{time_col}' not found in DataFrame.")
        df_clean[time_col] = pd.to_datetime(df_clean[time_col])

    # 3. Missing Value Imputation
    if metric_cols is not None:
        for m in metric_cols:
            if m not in df_clean.columns:
                raise KeyError(f"Metric column '{m}' not found in DataFrame.")
            df_clean[m] = df_clean[m].fillna(0.0)

    if categorical_cols is not None:
        for c in categorical_cols:
            if c not in df_clean.columns:
                raise KeyError(f"Categorical column '{c}' not found in DataFrame.")
            df_clean[c] = df_clean[c].fillna("UNKNOWN")

    # TODO: Add schema enforcement using Pydantic models or Pandera DataFrame schemas.
    # TODO: Implement out-of-core chunked ingestion or Dask integration for datasets exceeding local RAM capacities.
    return df_clean
