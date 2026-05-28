import time
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from xpyrment.analyze.orchestrator import AnalysisResult

def generate_mock_data(n_rows):
    return [
        {
            "metric_name": f"metric_{i}",
            "metric_type": "continuous",
            "control_mean": np.random.rand() * 10,
            "treatment_mean": np.random.rand() * 10,
            "relative_lift": np.random.randn() if np.random.rand() > 0.1 else np.nan,
            "rel_ci_lower": np.random.randn(),
            "rel_ci_upper": np.random.randn(),
            "p_value": np.random.rand() if np.random.rand() > 0.05 else np.nan,
            "power": np.random.rand() if np.random.rand() > 0.1 else np.nan,
            "cuped_applied": np.random.choice([True, False]),
            "variance_reduction": np.random.rand() if np.random.rand() > 0.2 else np.nan,
        }
        for i in range(n_rows)
    ]

def run_benchmark():
    n_rows = 100000
    print(f"Generating {n_rows} rows of mock data...")
    data = generate_mock_data(n_rows)
    result = AnalysisResult(data)

    print("Running benchmark...")

    # Warmup
    _ = result.summary()

    start_time = time.perf_counter()
    iterations = 5
    for _ in range(iterations):
        _ = result.summary()
    end_time = time.perf_counter()

    avg_time = (end_time - start_time) / iterations
    print(f"Average time over {iterations} iterations: {avg_time:.4f} seconds")

if __name__ == "__main__":
    run_benchmark()
