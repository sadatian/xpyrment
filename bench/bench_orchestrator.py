import time
import pandas as pd
import numpy as np
import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from xpyrment.analyze.orchestrator import AnalysisResult

# Use a dedicated RNG so benchmark runs are deterministic.
rng = np.random.default_rng(seed=42)

def generate_mock_data(n_rows):
    return [
        {
            "metric_name": f"metric_{i}",
            "metric_type": "continuous",
            "control_mean": rng.random() * 10,
            "treatment_mean": rng.random() * 10,
            "relative_lift": rng.normal() if rng.random() > 0.1 else np.nan,
            "rel_ci_lower": rng.normal(),
            "rel_ci_upper": rng.normal(),
            "p_value": rng.random() if rng.random() > 0.05 else np.nan,
            "power": rng.random() if rng.random() > 0.1 else np.nan,
            "cuped_applied": rng.choice([True, False]),
            "variance_reduction": rng.random() if rng.random() > 0.2 else np.nan,
        }
        for i in range(n_rows)
    ]

def run_benchmark():
    parser = argparse.ArgumentParser(description="Benchmark AnalysisResult.summary()")
    parser.add_argument("--rows", type=int, default=100000, help="Number of rows to generate (default 100000)")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations for benchmarking (default 5)")
    args = parser.parse_args()

    n_rows = args.rows
    iterations = args.iterations
    print(f"Generating {n_rows} rows of mock data...")
    data = generate_mock_data(n_rows)
    result = AnalysisResult(data)

    print(f"Running benchmark ({iterations} iterations)...")

    # Warmup
    _ = result.summary()

    start_time = time.perf_counter()
    for _ in range(iterations):
        _ = result.summary()
    end_time = time.perf_counter()

    avg_time = (end_time - start_time) / iterations
    print(f"Average time over {iterations} iterations: {avg_time:.4f} seconds")

if __name__ == "__main__":
    run_benchmark()
