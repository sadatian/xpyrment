import pandas as pd


def load_from_sql(query: str, connection_string: str) -> pd.DataFrame:
    """Simulates loading data from an external SQL database."""
    # TODO: Implement SQL adapter
    return pd.DataFrame()


def ingest_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ingests a pandas DataFrame into the xpyrment lifecycle."""
    return df.copy()
