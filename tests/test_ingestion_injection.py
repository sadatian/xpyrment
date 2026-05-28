import pytest
import pytest
try:
    import duckdb
except ImportError:
    pytest.skip("duckdb not installed", allow_module_level=True)
import pandas as pd
from xpyrment.run.ingestion import DuckDBIngester
import os

def test_duckdb_path_injection(tmp_path):
    # Setup normal file
    normal_file = tmp_path / "normal.parquet"
    pd.DataFrame({"treatment": ["A", "B", "A", "B"], "metric": [1.0, 2.0, 3.0, 4.0]}).to_parquet(normal_file)

    # We cannot create a file with a single quote using the vulnerable syntax because the issue
    # is that malicious strings get evaluated as SQL commands.
    # The vulnerability occurs when a user supplies a path like "normal.parquet' UNION ... "
    # Even if the file exists or doesn't exist, if it's evaluated as SQL it can be dangerous.
    # Let's create a file with a single quote in its name and test it works, which demonstrates
    # correct escaping.
    malicious_filename = tmp_path / "malicious'name.parquet"
    pd.DataFrame({"treatment": ["A", "B", "A", "B"], "metric": [1.0, 2.0, 3.0, 4.0]}).to_parquet(malicious_filename)

    with DuckDBIngester() as ingester:
        res = ingester.compute_welch_statistics(
            parquet_path=str(malicious_filename),
            treatment_col="treatment",
            metric_cols=["metric"]
        )
        assert "metric" in res
