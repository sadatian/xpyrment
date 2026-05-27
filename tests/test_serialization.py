import json
import numpy as np
import pandas as pd

from xpyrment.core.serialization import make_serializable
from xpyrment.analyze.orchestrator import setup
from xpyrment.quasi.diff_in_diff import DifferenceInDifferences, ParallelTrendsPlaceboTest
from xpyrment.quasi.instrumental_variables import InstrumentalVariables2SLS


def test_basic_make_serializable():
    """Asserts that various scientific and nested structures are correctly cleaned to JSON-serializable types."""
    nested_data = {
        "float_val": np.float64(3.14159),
        "int_val": np.int64(42),
        "bool_val": np.bool_(True),
        "nan_val": np.float64(np.nan),
        "inf_val": np.float64(np.inf),
        "array_val": np.array([1.1, 2.2, 3.3]),
        "list_of_ints": [np.int32(10), np.int64(20)],
    }

    cleaned = make_serializable(nested_data)

    # 1. Type assertions
    assert isinstance(cleaned["float_val"], float)
    assert isinstance(cleaned["int_val"], int)
    assert isinstance(cleaned["bool_val"], bool)
    assert cleaned["nan_val"] == "nan"
    assert cleaned["inf_val"] == "inf"
    assert isinstance(cleaned["array_val"], list)
    assert all(isinstance(x, float) for x in cleaned["array_val"])
    assert all(isinstance(x, int) for x in cleaned["list_of_ints"])

    # 2. Assert actually serializable via standard json.dumps
    json_str = json.dumps(cleaned)
    assert "3.14159" in json_str
    assert "42" in json_str
    assert "true" in json_str
    assert '"nan"' in json_str
    assert '"inf"' in json_str


def test_analysis_result_serialization():
    """Validates AnalysisResult state representation serialization and reproducibility checks."""
    df = pd.DataFrame({
        "variant": ["control"] * 30 + ["treatment"] * 30,
        "revenue": np.random.default_rng(1).normal(10, 2, 60),
        "pre_revenue": np.random.default_rng(1).normal(9, 2, 60),
        "user_id": range(60),
    })

    exp = setup(df, treatment_col="variant", id_col="user_id", covariates=["pre_revenue"])
    exp.register_metric("revenue", metric_type="mean")

    res = exp.run_analysis()
    
    # Check to_dict()
    res_dict = res.to_dict()
    assert "alpha" in res_dict
    assert "metrics" in res_dict
    assert "covariate_balance" in res_dict
    assert res_dict["alpha"] == 0.05
    assert len(res_dict["metrics"]) == 1

    # Check to_json()
    res_json = res.to_json(indent=2)
    assert "alpha" in res_json
    assert "metrics" in res_json
    # Parse back
    loaded = json.loads(res_json)
    assert loaded["alpha"] == 0.05


def test_estimator_serialization():
    """Tests to_dict and to_json across DiD, Placebo, and 2SLS estimators."""
    # 1. Difference-in-Differences
    did = DifferenceInDifferences()
    y = np.array([10.0, 12.0, 11.0, 15.0])
    treatment = np.array([0, 1, 0, 1])
    post = np.array([0, 0, 1, 1])
    did.fit(y, treatment, post)

    did_dict = did.to_dict()
    assert "treatment_effect" in did_dict
    assert "standard_error" in did_dict
    assert "p_value" in did_dict
    assert "summary_results" in did_dict

    did_json = did.to_json()
    assert '"treatment_effect"' in did_json

    # 2. Placebo Test
    placebo = ParallelTrendsPlaceboTest(significance_level=0.05)
    time = np.array([1, 2, 1, 2, 1, 2])
    y_p = np.array([10, 11, 10, 11, 12, 13])
    t_p = np.array([0, 0, 1, 1, 1, 1])
    placebo.fit_placebo_test(y_p, t_p, time, treatment_start_time=3.0)

    p_dict = placebo.to_dict()
    assert "alpha" in p_dict
    assert "results" in p_dict
    assert p_dict["results"]["trends_parallel"] == True

    p_json = placebo.to_json()
    assert "trends_parallel" in p_json

    # 3. Instrumental Variables (2SLS)
    iv = InstrumentalVariables2SLS()
    outcome = np.array([5.0, 6.0, 5.5, 7.0, 8.0, 7.5])
    treatment_rec = np.array([0, 1, 0, 1, 1, 0])
    instrument = np.array([0, 0, 0, 1, 1, 1])
    iv.fit(outcome, treatment_rec, instrument)

    iv_dict = iv.to_dict()
    assert "l2_penalty" in iv_dict
    assert "complier_average_causal_effect" in iv_dict
    assert "stage1_weak_instrument_f_statistic" in iv_dict

    iv_json = iv.to_json()
    assert "complier_average_causal_effect" in iv_json
