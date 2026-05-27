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


class DuckDBIngester:
    r"""High-performance out-of-core data ingestion and computation adapter using DuckDB.

    Provides streaming, memory-efficient statistical computations on parquet files/folders,
    such as covariate balance checks and Welch's t-test statistics, completely avoiding
    loading entire massive datasets into RAM.

    ??? mathbox "Mathematical Specifications"

        1. **Standardized Mean Difference (SMD)** for continuous/numeric covariates:
           Let $\bar{X}_1$ and $\bar{X}_0$ be the sample means of a covariate $X$ in the treatment and control groups,
           and let $s_1^2$ and $s_0^2$ be their sample variances.
           $$
           \text{SMD} = \frac{\bar{X}_1 - \bar{X}_0}{\sqrt{\frac{s_1^2 + s_0^2}{2}}}
           $$
        2. **Welch's $t$-test** for unequal variances:
           Let $N_0, N_1$ be the sample sizes, $\bar{X}_0, \bar{X}_1$ be the sample means, and $s_0^2, s_1^2$ be the sample variances.
           $$
           t = \frac{\bar{X}_1 - \bar{X}_0}{\sqrt{\frac{s_0^2}{N_0} + \frac{s_1^2}{N_1}}}
           $$
           The Welch-Satterthwaite degrees of freedom $\nu$ is calculated as:
           $$
           \nu = \frac{\left( \frac{s_0^2}{N_0} + \frac{s_1^2}{N_1} \right)^2}{\frac{\left( \frac{s_0^2}{N_0} \right)^2}{N_0 - 1} + \frac{\left( \frac{s_1^2}{N_1} \right)^2}{N_1 - 1}}
           $$
           The two-sided $p$-value is:
           $$
           p = 2 \times \left(1 - F_{t, \nu}(|t|)\right)
           $$
           where $F_{t, \nu}$ is the cumulative distribution function (CDF) of the Student's $t$-distribution with $\nu$ degrees of freedom.
        3. **Pearson $\chi^2$ Test of Independence** for categorical covariates:
           $$
           \chi^2 = \sum_{i=1}^R \sum_{j=1}^C \frac{(O_{i,j} - E_{i,j})^2}{E_{i,j}}
           $$
           with degrees of freedom:
           $$
           \text{df} = (R - 1)(C - 1)
           $$
           where $O_{i,j}$ and $E_{i,j}$ are the observed and expected frequency counts, respectively.
    """

    def __init__(self, db_path: str = ":memory:"):
        """Initializes the DuckDBIngester and opens a connection.

        Args:
            db_path (str): The file path to a persistent DuckDB database, or ':memory:' for transient sessions.

        Raises:
            ImportError: If duckdb or pyarrow are not installed.
        """
        try:
            import duckdb
            import pyarrow
        except ImportError:
            raise ImportError(
                "Both 'duckdb' and 'pyarrow' packages are required for DuckDBIngester. "
                "Please install them via: pip install duckdb pyarrow"
            )

        self.db_path = db_path
        self._conn = duckdb.connect(database=self.db_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        """Closes the underlying DuckDB connection if open to release locks."""
        if hasattr(self, "_conn") and self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def query(self, sql_query: str) -> pd.DataFrame:
        """Executes a raw SQL query against DuckDB and returns the result as a pandas DataFrame.

        Args:
            sql_query (str): A standard SQL query.

        Returns:
            pd.DataFrame: The queried records.
        """
        if self._conn is None:
            raise RuntimeError("DuckDB connection is closed.")
        return self._conn.execute(sql_query).df()

    def compute_covariate_balance(
        self,
        parquet_path: str,
        treatment_col: str,
        covariate_cols: list,
        control_group: str = None,
        treatment_group: str = None
    ) -> dict:
        r"""Computes covariate balance statistics out-of-core using DuckDB.

        Performs streaming, out-of-core scans to calculate Standardized Mean Difference (SMD)
        for continuous/numeric covariates and Pearson Chi-Square tests of independence for
        categorical covariates.

        Args:
            parquet_path (str): Absolute or relative path to the Parquet file or directory.
            treatment_col (str): Column name identifying experimental groups/arms.
            covariate_cols (list): List of column names representing categorical or continuous pre-experiment covariates.
            control_group (str, optional): The label/value of the control group. Defaults to None.
            treatment_group (str, optional): The label/value of the treatment group. Defaults to None.

        Returns:
            dict: A dictionary mapping each covariate to its balance statistics:
                - For numeric: {"type": "numeric", "smd": float, "p_value": float}
                - For categorical: {"type": "categorical", "p_value": float}

        Raises:
            FileNotFoundError: If the parquet path does not exist.
            KeyError: If columns are not found in the Parquet schema.
            ValueError: If the dataset is empty, has fewer than 2 distinct treatment arms,
                        or contains invalid/degenerate data.
        """
        import os
        from pathlib import Path
        import numpy as np
        from scipy import stats

        # 1. Verify file existence
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet file/directory not found at path: {parquet_path}")

        path_str = str(Path(parquet_path).resolve()).replace("\\", "/")

        # 2. Schema pre-validation & column presence verification
        try:
            schema_df = self.query(f"DESCRIBE SELECT * FROM read_parquet('{path_str}')")
        except Exception as e:
            raise ValueError(f"Failed to parse Parquet schema at {parquet_path}: {e}")

        col_types = dict(zip(schema_df["column_name"], schema_df["column_type"]))

        if treatment_col not in col_types:
            raise KeyError(f"Treatment column '{treatment_col}' not found in Parquet schema.")

        for cov in covariate_cols:
            if cov not in col_types:
                raise KeyError(f"Covariate column '{cov}' not found in Parquet schema.")

        # 3. Check if dataset is empty and get total row count
        count_df = self.query(f"SELECT COUNT(*) as cnt FROM read_parquet('{path_str}')")
        if count_df.empty or count_df.iloc[0]["cnt"] == 0:
            raise ValueError("Dataset is empty.")

        # 4. Retrieve distinct treatment arms and validate groups
        safe_treatment_col = '"' + treatment_col.replace('"', '""') + '"'
        groups_df = self.query(
            f"SELECT DISTINCT {safe_treatment_col} FROM read_parquet('{path_str}') WHERE {safe_treatment_col} IS NOT NULL"
        )
        groups = sorted(groups_df[treatment_col].tolist())
        if len(groups) < 2:
            raise ValueError(
                f"Balance check requires at least 2 distinct groups in '{treatment_col}'. Found {len(groups)}."
            )

        if control_group is not None and treatment_group is not None:
            if control_group not in groups or treatment_group not in groups:
                raise ValueError(
                    f"Specified control_group '{control_group}' or treatment_group '{treatment_group}' "
                    f"not found in distinct groups: {groups}."
                )
            comp_groups = [control_group, treatment_group]
        elif len(groups) == 2:
            comp_groups = groups
        else:
            raise ValueError(
                f"Multiple treatment arms detected: {groups}. "
                f"You must specify both 'control_group' and 'treatment_group' parameters."
            )

        # Helper to convert python value to SQL literal
        def to_sql_val(val):
            if isinstance(val, str):
                # Escape single quotes by doubling them in SQL strings
                safe_val = val.replace("'", "''")
                return f"'{safe_val}'"
            return str(val)

        # Helper to quote identifiers for DuckDB to prevent SQL injection
        def quote_identifier(col_name: str) -> str:
            return '"' + col_name.replace('"', '""') + '"'

        # 5. Partition covariates into numeric vs categorical
        def is_duckdb_numeric(col_name: str) -> bool:
            t = col_types.get(col_name)
            if not t:
                return False
            t_upper = str(t).upper()
            for kw in ["INT", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC", "REAL"]:
                if kw in t_upper:
                    return True
            return False

        numeric_covs = []
        categorical_covs = []
        for cov in covariate_cols:
            if is_duckdb_numeric(cov):
                numeric_covs.append(cov)
            else:
                categorical_covs.append(cov)

        results = {}

        # 6. Compute statistics for continuous/numeric covariates
        if numeric_covs:
            # Build and run optimized group aggregation SQL query for all numeric covariates in a single scan
            select_parts = []
            for cov in numeric_covs:
                safe_cov = quote_identifier(cov)
                # We also need to quote the aliases so we can reliably fetch them
                safe_count_alias = quote_identifier(f"count_{cov}")
                safe_mean_alias = quote_identifier(f"mean_{cov}")
                safe_var_alias = quote_identifier(f"var_{cov}")
                select_parts.append(f"COUNT({safe_cov}) as {safe_count_alias}")
                select_parts.append(f"AVG({safe_cov}) as {safe_mean_alias}")
                select_parts.append(f"VAR_SAMP({safe_cov}) as {safe_var_alias}")

            safe_treatment_col = quote_identifier(treatment_col)

            sql = f"""
                SELECT
                    {safe_treatment_col},
                    {', '.join(select_parts)}
                FROM read_parquet('{path_str}')
                WHERE {safe_treatment_col} IN ({to_sql_val(comp_groups[0])}, {to_sql_val(comp_groups[1])})
                GROUP BY {safe_treatment_col}
            """
            group_stats_df = self.query(sql)
            
            row_0 = group_stats_df[group_stats_df[treatment_col] == comp_groups[0]].iloc[0] if comp_groups[0] in group_stats_df[treatment_col].values else None
            row_1 = group_stats_df[group_stats_df[treatment_col] == comp_groups[1]].iloc[0] if comp_groups[1] in group_stats_df[treatment_col].values else None

            for cov in numeric_covs:
                n_0 = int(row_0[f"count_{cov}"]) if (row_0 is not None and not pd.isna(row_0[f"count_{cov}"])) else 0
                mean_0 = float(row_0[f"mean_{cov}"]) if (row_0 is not None and not pd.isna(row_0[f"mean_{cov}"])) else 0.0
                var_0 = float(row_0[f"var_{cov}"]) if (row_0 is not None and not pd.isna(row_0[f"var_{cov}"])) else 0.0

                n_1 = int(row_1[f"count_{cov}"]) if (row_1 is not None and not pd.isna(row_1[f"count_{cov}"])) else 0
                mean_1 = float(row_1[f"mean_{cov}"]) if (row_1 is not None and not pd.isna(row_1[f"mean_{cov}"])) else 0.0
                var_1 = float(row_1[f"var_{cov}"]) if (row_1 is not None and not pd.isna(row_1[f"var_{cov}"])) else 0.0

                # Guard against degenerate / small sample size edge cases
                if n_0 < 2 or n_1 < 2:
                    raise ValueError(f"Sample size too small for covariate '{cov}': control_N={n_0}, treatment_N={n_1}. Must be >= 2.")

                if var_0 < 0.0 or var_1 < 0.0:
                    raise ValueError(f"Negative variance detected for covariate '{cov}'.")

                if var_0 == 0.0 and var_1 == 0.0:
                    raise ValueError(f"Degenerate variance (zero variance) in treatment arms for covariate '{cov}'.")

                pooled_sd = np.sqrt((var_0 + var_1) / 2.0)
                if pooled_sd == 0.0:
                    smd = 0.0
                else:
                    smd = (mean_1 - mean_0) / pooled_sd

                # Welch's t-test
                se_diff = np.sqrt(var_0 / n_0 + var_1 / n_1)
                diff = mean_1 - mean_0
                if se_diff > 0.0:
                    t_stat = diff / se_diff
                    num = (var_0 / n_0 + var_1 / n_1) ** 2
                    den = ((var_0 / n_0) ** 2) / (n_0 - 1) + ((var_1 / n_1) ** 2) / (n_1 - 1)
                    df_val = num / den if den > 0 else (n_0 + n_1 - 2)
                    p_val = 2 * (1.0 - stats.t.cdf(np.abs(t_stat), df=df_val))
                else:
                    p_val = 1.0

                results[cov] = {
                    "type": "numeric",
                    "smd": float(smd),
                    "p_value": float(p_val)
                }

        # 7. Compute statistics for categorical covariates
        for cov in categorical_covs:
            safe_cov = quote_identifier(cov)
            safe_treatment_col = quote_identifier(treatment_col)
            sql = f"""
                SELECT
                    {safe_cov},
                    {safe_treatment_col},
                    COUNT(*) as cnt
                FROM read_parquet('{path_str}')
                WHERE {safe_treatment_col} IN ({to_sql_val(comp_groups[0])}, {to_sql_val(comp_groups[1])}) AND {safe_cov} IS NOT NULL
                GROUP BY {safe_cov}, {safe_treatment_col}
            """
            cat_df = self.query(sql)

            if not cat_df.empty:
                contingency = cat_df.pivot(index=cov, columns=treatment_col, values="cnt").fillna(0)
                for g in [comp_groups[0], comp_groups[1]]:
                    if g not in contingency.columns:
                        contingency[g] = 0.0
                
                # CRITICAL: chi2_contingency requires df > 0 -> shape must be >= (2, 2)
                if contingency.shape[0] >= 2 and contingency.shape[1] >= 2 and contingency.values.sum() > 0:
                    chi2_res = stats.chi2_contingency(contingency.values)
                    p_val = chi2_res.pvalue
                else:
                    p_val = 1.0  # Completely balanced (e.g., only 1 category exists across all arms)
            else:
                p_val = 1.0

            results[cov] = {
                "type": "categorical",
                "p_value": float(p_val)
            }

        return results

    def compute_welch_statistics(
        self,
        parquet_path: str,
        treatment_col: str,
        metric_cols: list,
        control_group: str = None,
        treatment_group: str = None,
        alpha: float = 0.05
    ) -> dict:
        r"""Computes Welch's t-test statistics for experimental metrics out-of-core using DuckDB.

        Calculates means, sample variances, sample sizes, standard errors, Welch's t-statistic,
        degrees of freedom, p-value, confidence intervals, and statistical significance.

        Args:
            parquet_path (str): Absolute or relative path to the Parquet file or directory.
            treatment_col (str): Column name identifying experimental groups/arms.
            metric_cols (list): List of continuous metric column names.
            control_group (str, optional): The label/value of the control group. Defaults to None.
            treatment_group (str, optional): The label/value of the treatment group. Defaults to None.
            alpha (float): Significance level for confidence interval calculation. Defaults to 0.05.

        Returns:
            dict: A dictionary mapping each metric name to a sub-dictionary of Welch's statistics.

        Raises:
            FileNotFoundError: If the parquet path does not exist.
            KeyError: If columns are not found in the Parquet schema.
            ValueError: If the dataset is empty, has fewer than 2 distinct treatment arms,
                        or contains zero-variance/degenerate data in treatment arms.
        """
        import os
        from pathlib import Path
        import numpy as np
        from scipy import stats

        # 1. Verify file existence
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet file/directory not found at path: {parquet_path}")

        path_str = str(Path(parquet_path).resolve()).replace("\\", "/")

        # 2. Schema pre-validation & column presence verification
        try:
            schema_df = self.query(f"DESCRIBE SELECT * FROM read_parquet('{path_str}')")
        except Exception as e:
            raise ValueError(f"Failed to parse Parquet schema at {parquet_path}: {e}")

        col_types = dict(zip(schema_df["column_name"], schema_df["column_type"]))

        if treatment_col not in col_types:
            raise KeyError(f"Treatment column '{treatment_col}' not found in Parquet schema.")

        for m in metric_cols:
            if m not in col_types:
                raise KeyError(f"Metric column '{m}' not found in Parquet schema.")

        # 3. Check if dataset is empty
        count_df = self.query(f"SELECT COUNT(*) as cnt FROM read_parquet('{path_str}')")
        if count_df.empty or count_df.iloc[0]["cnt"] == 0:
            raise ValueError("Dataset is empty.")

        # 4. Retrieve distinct treatment arms and validate groups
        safe_treatment_col = '"' + treatment_col.replace('"', '""') + '"'
        groups_df = self.query(
            f"SELECT DISTINCT {safe_treatment_col} FROM read_parquet('{path_str}') WHERE {safe_treatment_col} IS NOT NULL"
        )
        groups = sorted(groups_df[treatment_col].tolist())
        if len(groups) < 2:
            raise ValueError(
                f"Welch statistics calculation requires at least 2 distinct groups in '{treatment_col}'. Found {len(groups)}."
            )

        if control_group is not None and treatment_group is not None:
            if control_group not in groups or treatment_group not in groups:
                raise ValueError(
                    f"Specified control_group '{control_group}' or treatment_group '{treatment_group}' "
                    f"not found in distinct groups: {groups}."
                )
            comp_groups = [control_group, treatment_group]
        elif len(groups) == 2:
            comp_groups = groups
        else:
            raise ValueError(
                f"Multiple treatment arms detected: {groups}. "
                f"You must specify both 'control_group' and 'treatment_group' parameters."
            )

        # Helper to convert python value to SQL literal
        def to_sql_val(val):
            if isinstance(val, str):
                safe_val = val.replace("'", "''")
                return f"'{safe_val}'"
            return str(val)

        # Helper to quote identifiers for DuckDB to prevent SQL injection
        def quote_identifier(col_name: str) -> str:
            return '"' + col_name.replace('"', '""') + '"'

        # 5. Construct SQL query to aggregate all metrics at once
        select_parts = []
        for m in metric_cols:
            safe_m = quote_identifier(m)
            safe_count_alias = quote_identifier(f"count_{m}")
            safe_mean_alias = quote_identifier(f"mean_{m}")
            safe_var_alias = quote_identifier(f"var_{m}")

            select_parts.append(f"COUNT({safe_m}) as {safe_count_alias}")
            select_parts.append(f"AVG({safe_m}) as {safe_mean_alias}")
            select_parts.append(f"VAR_SAMP({safe_m}) as {safe_var_alias}")

        safe_treatment_col = quote_identifier(treatment_col)

        sql = f"""
            SELECT
                {safe_treatment_col},
                {', '.join(select_parts)}
            FROM read_parquet('{path_str}')
            WHERE {safe_treatment_col} IN ({to_sql_val(comp_groups[0])}, {to_sql_val(comp_groups[1])})
            GROUP BY {safe_treatment_col}
        """
        stats_df = self.query(sql)

        row_0 = stats_df[stats_df[treatment_col] == comp_groups[0]].iloc[0] if comp_groups[0] in stats_df[treatment_col].values else None
        row_1 = stats_df[stats_df[treatment_col] == comp_groups[1]].iloc[0] if comp_groups[1] in stats_df[treatment_col].values else None

        results = {}
        for m in metric_cols:
            n_0 = int(row_0[f"count_{m}"]) if (row_0 is not None and not pd.isna(row_0[f"count_{m}"])) else 0
            mean_0 = float(row_0[f"mean_{m}"]) if (row_0 is not None and not pd.isna(row_0[f"mean_{m}"])) else 0.0
            var_0 = float(row_0[f"var_{m}"]) if (row_0 is not None and not pd.isna(row_0[f"var_{m}"])) else 0.0

            n_1 = int(row_1[f"count_{m}"]) if (row_1 is not None and not pd.isna(row_1[f"count_{m}"])) else 0
            mean_1 = float(row_1[f"mean_{m}"]) if (row_1 is not None and not pd.isna(row_1[f"mean_{m}"])) else 0.0
            var_1 = float(row_1[f"var_{m}"]) if (row_1 is not None and not pd.isna(row_1[f"var_{m}"])) else 0.0

            # Guard against degenerate / small sample sizes
            if n_0 < 2 or n_1 < 2:
                raise ValueError(f"Sample size too small for metric '{m}': control_N={n_0}, treatment_N={n_1}. Must be >= 2.")

            if var_0 < 0.0 or var_1 < 0.0:
                raise ValueError(f"Negative variance detected for metric '{m}'.")

            if var_0 == 0.0 and var_1 == 0.0:
                raise ValueError(f"Degenerate variance (zero variance) in treatment arms for metric '{m}'.")

            diff = mean_1 - mean_0
            se_diff = np.sqrt(var_0 / n_0 + var_1 / n_1)

            if se_diff > 0.0:
                t_stat = diff / se_diff
                num = (var_0 / n_0 + var_1 / n_1) ** 2
                den = ((var_0 / n_0) ** 2) / (n_0 - 1) + ((var_1 / n_1) ** 2) / (n_1 - 1)
                df_val = num / den if den > 0 else (n_0 + n_1 - 2)
                p_val = 2 * (1.0 - stats.t.cdf(np.abs(t_stat), df=df_val))

                # Confidence interval calculation
                ci_half = stats.t.ppf(1.0 - alpha / 2.0, df=df_val) * se_diff
                ci_lower = diff - ci_half
                ci_upper = diff + ci_half
                significant = bool(p_val < alpha)
            else:
                t_stat = 0.0
                p_val = 1.0
                df_val = float(n_0 + n_1 - 2)
                ci_lower = diff
                ci_upper = diff
                significant = False

            results[m] = {
                "control_mean": float(mean_0),
                "treatment_mean": float(mean_1),
                "control_var": float(var_0),
                "treatment_var": float(var_1),
                "control_n": int(n_0),
                "treatment_n": int(n_1),
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
                "df": float(df_val),
                "difference": float(diff),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "significant": significant
            }

        return results

