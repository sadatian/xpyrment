import itertools
import numpy as np
import pandas as pd
import pytest

from xpyrment.plan.power import design_experiment, generate_power_curve_data
from xpyrment.design.splits import TrafficSplitter
from xpyrment.design.stratification import stratified_randomization
from xpyrment.design.doe.fractional_factorial import FractionalFactorialDesign
from xpyrment.design.doe.ccd import CentralCompositeDesign
from xpyrment.design.doe.box_behnken import BoxBehnkenDesign
from xpyrment.design.doe.plackett_burman import PlackettBurmanDesign
from xpyrment.design.doe.taguchi import TaguchiDesign
from xpyrment.design.doe.dsd import DefinitiveScreeningDesign
from xpyrment.design.doe.d_optimal import DOptimalDesign
from xpyrment.design.doe.lhs import LatinHypercubeDesign
from xpyrment.design.doe.mixture import MixtureDesign
from xpyrment.design.doe.switchback import SwitchbackDesign
from xpyrment.design.doe.evop import EVOPDesign
from xpyrment.design.randomization import hash_assign
from xpyrment.design.doe.full_factorial import FullFactorialDesign


def test_design_experiment_proportion():
    """Tests sample size calculations for proportion metrics."""
    res = design_experiment(
        metric_type="proportion",
        baseline_value=0.10,
        mde=0.01,
        mde_type="absolute",
        alpha=0.05,
        power=0.80,
    )
    details = res.details
    assert details["metric_type"] == "proportion"
    assert details["baseline_value"] == 0.10
    assert details["mde_absolute"] == 0.01
    assert details["sample_size_per_variant"] > 0


def test_design_experiment_mean():
    """Tests sample size calculations for continuous mean metrics."""
    res = design_experiment(
        metric_type="mean",
        baseline_value=10.0,
        standard_deviation=5.0,
        mde=0.05,
        mde_type="relative",
        alpha=0.05,
        power=0.80,
    )
    details = res.details
    assert details["metric_type"] == "mean"
    assert details["baseline_value"] == 10.0
    assert pytest.approx(details["mde_absolute"]) == 0.5
    assert details["sample_size_per_variant"] > 0


def test_design_experiment_cuped_savings():
    """Tests that a positive pre-period correlation reduces sample sizes accordingly."""
    res_no_cuped = design_experiment(
        metric_type="mean",
        baseline_value=10.0,
        standard_deviation=5.0,
        mde=0.05,
        alpha=0.05,
        power=0.80,
    )

    res_cuped = design_experiment(
        metric_type="mean",
        baseline_value=10.0,
        standard_deviation=5.0,
        mde=0.05,
        alpha=0.05,
        power=0.80,
        pre_post_correlation=0.6,
    )

    assert "cuped_sample_size_per_variant" in res_cuped.details
    # 1 - 0.6^2 = 0.64 variance factor -> 36% sample size savings
    assert pytest.approx(res_cuped.details["cuped_savings"]) == 0.36
    assert (
        res_cuped.details["cuped_sample_size_per_variant"]
        < res_no_cuped.details["sample_size_per_variant"]
    )


def test_generate_power_curve_data():
    """Tests structure and sorting properties of generated power curve coordinates."""
    curve = generate_power_curve_data(
        metric_type="mean",
        baseline_value=10.0,
        standard_deviation=5.0,
        pre_post_correlation=0.5,
    )
    assert "mde_relative" in curve
    assert "sample_size_per_variant" in curve
    assert "cuped_sample_size_per_variant" in curve
    assert len(curve["mde_relative"]) == 50


def test_traffic_splitter_custom_ramp():
    """Tests TrafficSplitter allocation rules and ramp-up schedule validations."""
    allocations = {"control": 0.40, "treatment": 0.40}
    
    # Valid setup with holdout
    splitter = TrafficSplitter(allocations=allocations, holdout_percentage=0.20)
    assert splitter.holdout_percentage == 0.20
    assert splitter.get_ramp_schedule() == [0.01, 0.10, 0.50, 1.0]

    # Custom valid ramp schedule
    custom_ramp = [0.05, 0.20, 0.50, 1.0]
    splitter_custom = TrafficSplitter(allocations=allocations, holdout_percentage=0.20, ramp_schedule=custom_ramp)
    assert splitter_custom.get_ramp_schedule() == custom_ramp

    # Invalid schedules must raise ValueError
    with pytest.raises(ValueError):
        TrafficSplitter(allocations=allocations, holdout_percentage=0.20, ramp_schedule=[0.1, 0.05, 1.0])  # Non-monotonic
    with pytest.raises(ValueError):
        TrafficSplitter(allocations=allocations, holdout_percentage=0.20, ramp_schedule=[0.1, 0.5])  # Doesn't end in 1.0
    with pytest.raises(ValueError):
        TrafficSplitter(allocations=allocations, holdout_percentage=0.20, ramp_schedule=[-0.1, 1.0])  # Out of bounds


def test_stratified_randomization_balance():
    """Tests that stratified_randomization perfectly balances assignments within strata."""
    # Create sample dataset with heterogeneous groups
    data = {
        "user_id": list(range(1, 101)),
        "country": ["US"] * 60 + ["EU"] * 40,
        "device": ["mobile"] * 30 + ["desktop"] * 30 + ["mobile"] * 20 + ["desktop"] * 20,
    }
    df = pd.DataFrame(data)

    variants = ["control", "treatment_a", "treatment_b"]
    assigned_df = stratified_randomization(
        df=df,
        strata_cols=["country", "device"],
        variants=variants,
        treatment_col="assigned_group",
        random_state=42
    )

    # Output row count must match input
    assert len(assigned_df) == len(df)
    assert "assigned_group" in assigned_df.columns

    # Verify balance within each stratum combination
    for (country, device), group in assigned_df.groupby(["country", "device"]):
        counts = group["assigned_group"].value_counts()
        # The difference in variant counts inside each homogeneous cohort must be at most 1
        assert max(counts) - min(counts) <= 1


def test_fractional_factorial_generation():
    """Tests generation of a 2^(5-1) Resolution V Fractional Factorial Design."""
    factors = {
        "A": [10.0, 20.0],
        "B": [0.0, 1.0],
        "C": [-1.0, 1.0],
        "D": [5.0, 10.0],
        "E": [100.0, 200.0]
    }
    # Standard generator for 1/2 fraction of 5 factors
    design = FractionalFactorialDesign(factors, generator_string="E = A * B * C * D")
    df = design.generate()

    # Size must be 2^(5-1) = 16 runs
    assert len(df) == 16
    assert list(df.columns) == ["A", "B", "C", "D", "E"]

    # In coded space, E must be the product of A, B, C, and D
    # Let's map back to coded space [-1, 1] for verification
    coded_df = pd.DataFrame()
    for col in df.columns:
        low, high = factors[col]
        coded_df[col] = df[col].map({low: -1.0, high: 1.0})

    expected_E = coded_df["A"] * coded_df["B"] * coded_df["C"] * coded_df["D"]
    pd.testing.assert_series_equal(coded_df["E"], expected_E, check_names=False)


def test_fractional_factorial_level_validation():
    """Asserts that supplying non-2-level factors to Fractional Factorial raises ValueError."""
    factors_3_level = {"A": [1.0, 2.0, 3.0], "B": [0.0, 1.0]}
    with pytest.raises(ValueError, match="strictly require exactly 2 levels"):
        FractionalFactorialDesign(factors_3_level, generator_string="B = A")


def test_dsd_level_validation():
    """Asserts that supplying non-3-level factors to DSD raises ValueError."""
    factors_2_level = {"A": [-1.0, 1.0], "B": [-1.0, 1.0]}
    with pytest.raises(ValueError, match="strictly require exactly 3 levels"):
        DefinitiveScreeningDesign(factors_2_level)


def test_ccd_generation():
    """Tests face-centered and rotatable central composite designs."""
    factors = {
        "temperature": [100.0, 200.0],
        "pressure": [15.0, 30.0]
    }

    # CCF (Face-Centered) has alpha = 1.0
    design_ccf = CentralCompositeDesign(factors, alpha="face-centered")
    df_ccf = design_ccf.generate()

    # For k=2, N = 2^2 + 2(2) + center points.
    # Default center points should be 4 (standard for k=2 rotatability/orthogonality balance)
    assert len(df_ccf) == 12  # 4 cube + 4 star + 4 center
    
    # Check that temperature values only consist of 100, 150, and 200 (since alpha=1.0)
    assert set(df_ccf["temperature"].unique()) == {100.0, 150.0, 200.0}

    # Rotatable CCD has alpha = (N_cube)^(1/4) = 4^(1/4) = 1.4142...
    design_rot = CentralCompositeDesign(factors, alpha="rotatable")
    df_rot = design_rot.generate()
    assert len(df_rot) == 12

    # In rotatable, star points exceed the original bounds
    temps = df_rot["temperature"].unique()
    assert len(temps) == 5  # low-star, low, mid, high, high-star
    assert min(temps) < 100.0
    assert max(temps) > 200.0


def test_bbd_generation():
    """Tests Box-Behnken Design generation constraints and shape."""
    factors = {
        "A": [10.0, 20.0],
        "B": [100.0, 200.0],
        "C": [0.1, 0.5]
    }

    design = BoxBehnkenDesign(factors)
    df = design.generate()

    # For k=3 factors: N = 2*3*(2) + center points. Default centers = 3
    # N = 12 + 3 = 15 runs
    assert len(df) == 15
    assert list(df.columns) == ["A", "B", "C"]

    # Verify that no run has all extreme high/low levels simultaneously
    # (Box-Behnken does not have corner points, i.e., no row where absolute coded value of all columns is 1)
    for _, row in df.iterrows():
        coded_values = []
        for col in df.columns:
            low, high = factors[col]
            mid = (low + high) / 2
            if pytest.approx(row[col]) == low:
                coded_values.append(-1.0)
            elif pytest.approx(row[col]) == high:
                coded_values.append(1.0)
            elif pytest.approx(row[col]) == mid:
                coded_values.append(0.0)
        
        # At least one factor must be at its center level (0.0) for every single run
        assert 0.0 in coded_values


def test_plackett_burman_generation():
    """Tests Plackett-Burman Design matrix generation and orthogonality."""
    factors = {
        f"factor_{i}": [0.0, 10.0] for i in range(1, 11)  # 10 factors
    }

    # Since 10 factors are requested, smallest N multiple of 4 is N=12
    design = PlackettBurmanDesign(factors)
    df = design.generate()

    assert len(df) == 12
    assert len(df.columns) == 10

    # Ensure all values are strictly low (0.0) or high (10.0)
    for col in df.columns:
        assert set(df[col].unique()).issubset({0.0, 10.0})

    # Let's verify column orthogonality in coded space [-1, 1]
    coded_df = pd.DataFrame()
    for col in df.columns:
        low, high = factors[col]
        coded_df[col] = df[col].map({low: -1.0, high: 1.0})

    # Columns must be orthogonal, i.e., dot product of any two distinct columns is zero (or close)
    for col_a, col_b in itertools.combinations(df.columns, 2):
        dot_product = np.dot(coded_df[col_a], coded_df[col_b])
        assert abs(dot_product) < 1e-5


def test_taguchi_generation():
    """Tests Taguchi Design L9 orthogonal array correctness and pair balance."""
    factors = {
        "A": [1.0, 2.0, 3.0],
        "B": [10.0, 20.0, 30.0],
        "C": [100.0, 200.0, 300.0],
        "D": [0.1, 0.2, 0.3],
    }

    design = TaguchiDesign(factors, array_name="L9")
    df = design.generate()

    # L9 array must have exactly 9 runs and 4 columns
    assert len(df) == 9
    assert len(df.columns) == 4

    # Mathematical property: Orthogonality (Pair-wise balance)
    # For any two columns, all 9 combinations of levels (1-3) must appear exactly once
    for col_a, col_b in itertools.combinations(df.columns, 2):
        pairs = list(zip(df[col_a], df[col_b]))
        assert len(pairs) == 9
        assert len(set(pairs)) == 9  # Unique count must be 9


def test_dsd_generation():
    """Tests Definitive Screening Design (DSD) linear orthogonality and fold-over symmetry."""
    # Test even number of factors (k=4 -> N = 2k + 1 = 9 runs)
    factors_even = {f"F{i}": [-1.0, 0.0, 1.0] for i in range(1, 5)}
    design_even = DefinitiveScreeningDesign(factors_even)
    df_even = design_even.generate()
    assert len(df_even) == 9

    # Test odd number of factors (k=5 -> N = 2k + 3 = 13 runs)
    factors_odd = {f"F{i}": [-1.0, 0.0, 1.0] for i in range(1, 6)}
    design_odd = DefinitiveScreeningDesign(factors_odd)
    df_odd = design_odd.generate()
    assert len(df_odd) == 13

    # Map odd factors to coded space for algebraic verification
    coded_df = pd.DataFrame()
    for col in df_odd.columns:
        low, mid, high = factors_odd[col]
        half_range = (high - low) / 2
        coded_df[col] = (df_odd[col] - mid) / half_range

    # 1. Verification of Main Effects Orthogonality
    # Sum of dot product of distinct columns in coded space must be exactly 0.0
    for col_a, col_b in itertools.combinations(coded_df.columns, 2):
        dot_product = np.dot(coded_df[col_a], coded_df[col_b])
        assert abs(dot_product) < 1e-10

    # 2. Verification of Fold-Over Symmetry
    # For every row except the center point (all zeros), there must be a corresponding
    # row that is its exact sign-inverted opposite
    non_center_rows = coded_df[(coded_df != 0).any(axis=1)]
    center_rows = coded_df[(coded_df == 0).all(axis=1)]
    assert len(center_rows) == 1  # Exactly one center point

    for idx, row in non_center_rows.iterrows():
        opposite = -row
        found = False
        for o_idx, o_row in non_center_rows.iterrows():
            if np.allclose(opposite, o_row):
                found = True
                break
        assert found, f"Fold-over opposite not found for row: {row.tolist()}"


def test_d_optimal_generation():
    """Tests D-Optimal Coordinate Exchange optimization and singularity checks."""
    factors = {
        "temperature": [100.0, 150.0, 200.0],
        "speed": [10.0, 20.0, 30.0],
        "load": [5.0, 10.0, 15.0]
    }
    num_runs = 12
    design = DOptimalDesign(factors, num_runs=num_runs)
    df = design.generate()

    assert len(df) == num_runs
    assert list(df.columns) == ["temperature", "speed", "load"]

    # Verify that X^T X is non-singular
    # Build linear design matrix X (including an intercept column)
    X = np.hstack([np.ones((num_runs, 1)), df.values])
    information_matrix = np.dot(X.T, X)
    det = np.linalg.det(information_matrix)
    
    # Determinant must be strictly positive (non-singular design)
    assert det > 1e-5


def test_lhs_generation():
    """Tests Latin Hypercube Sampling projection property (1 sample per interval)."""
    factors = {
        "X1": [10.0, 50.0],
        "X2": [100.0, 200.0]
    }
    num_samples = 20
    design = LatinHypercubeDesign(factors, num_samples=num_samples)
    df = design.generate()

    assert len(df) == num_samples

    # Mathematically verify the LHS projection property:
    # If we divide the range of each factor into N equal intervals,
    # each interval must contain exactly one point.
    for col in df.columns:
        low, high = factors[col]
        intervals = np.linspace(low, high, num_samples + 1)
        counts = []
        for i in range(num_samples):
            # Check how many points fall into interval [intervals[i], intervals[i+1]]
            pt_count = df[col].between(intervals[i] - 1e-9, intervals[i+1] + 1e-9).sum()
            counts.append(pt_count)
        
        # Every interval must contain exactly 1 point
        assert set(counts) == {1}


def test_mixture_generation():
    """Tests Mixture Design simplex lattice and sum-to-one constraint."""
    factors = {
        "water": [0.0, 1.0],
        "oil": [0.0, 1.0],
        "emulsifier": [0.0, 1.0]
    }
    
    design = MixtureDesign(factors)
    df = design.generate()

    # The sum of ingredients in every run must be exactly 1.0
    for _, row in df.iterrows():
        total = sum(row)
        assert pytest.approx(total) == 1.0

    # Every component must be non-negative
    assert (df >= 0.0).all().all()
    assert (df <= 1.0).all().all()


def test_switchback_generation():
    """Tests Switchback Design crossover balancing and washout flags."""
    factors = {
        "dispatch_algorithm": ["greedy", "predictive"]
    }
    design = SwitchbackDesign(factors, unit_window_hours=4)
    regions = ["Region_A", "Region_B"]
    df = design.generate(regions=regions, num_periods=8, washout_minutes=30)

    # Output columns: region, period, start_hour, end_hour, washout_active, dispatch_algorithm
    assert "region" in df.columns
    assert "period" in df.columns
    assert "washout_active" in df.columns
    assert "dispatch_algorithm" in df.columns

    # Verify that the total number of rows matches regions * periods
    assert len(df) == len(regions) * 8

    # Verify crossover balance: across regions, dispatch algorithms are switched
    for period in range(1, 9):
        period_df = df[df["period"] == period]
        assert len(period_df) == 2
        # Algorithms should be opposite (crossover) to prevent systematic temporal bias
        variants = period_df["dispatch_algorithm"].tolist()
        assert set(variants) == {"greedy", "predictive"}


def test_evop_generation():
    """Tests Evolutionary Operation (EVOP) tiny perturbations around baseline."""
    # current settings are A=150.0, B=50.0. Tiny steps are delta_A=5.0, delta_B=2.0
    factors = {
        "temperature": [150.0],  # Center level
        "pressure": [50.0]       # Center level
    }

    # Custom deltas passed for tiny perturbations
    design = EVOPDesign(factors)
    # Generate 3 cycles for Phase 1
    df = design.generate(center_settings={"temperature": 150.0, "pressure": 50.0}, deltas={"temperature": 5.0, "pressure": 2.0}, num_cycles=3)

    # 2 factors -> 2^2 + 1 = 5 runs per cycle. 3 cycles -> 15 runs
    assert len(df) == 15
    assert "Cycle" in df.columns
    assert "Phase" in df.columns

    # Check bounds of temperature (must strictly be within [145.0, 155.0])
    assert set(df["temperature"].unique()) == {145.0, 150.0, 155.0}
    assert set(df["pressure"].unique()) == {48.0, 50.0, 52.0}


def test_switchback_washout_optimization():
    """Validates auto-regressive AR(p) washout optimization inside SwitchbackDesign."""
    from xpyrment.design.doe.switchback import SwitchbackDesign
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    N = 100
    
    # Chronological sequence of elapsed times to exhibit a smooth physical carryover decay
    time_elapsed = np.sort(rng.choice([0, 5, 10, 15, 20, 25, 30, 40, 50, 60], size=N))
    treatment = rng.choice([0, 1], size=N)
    
    # Initialize metric with standard values plus noise
    metric = np.zeros(N)
    metric[0] = 10.0 + 2.5 * treatment[0] + rng.normal(0, 0.05)
    
    # Inject strong auto-regressive carryover (correlation) for times < 20 minutes
    # and keep it small / stable for times >= 20 minutes.
    for i in range(1, N):
        if time_elapsed[i] < 20:
            # Strong sequential carryover
            metric[i] = 10.0 + 2.5 * treatment[i] + 0.8 * (metric[i-1] - (10.0 + 2.5 * treatment[i-1])) + rng.normal(0, 0.05)
        else:
            # Independent observations
            metric[i] = 10.0 + 2.5 * treatment[i] + rng.normal(0, 0.05)

    df = pd.DataFrame({
        "time": time_elapsed,
        "metric": metric,
        "treatment": treatment
    })

    design = SwitchbackDesign(factors={"dispatch_algorithm": ["greedy", "predictive"]})
    optimal_washout = design.optimize_washout(
        df=df,
        time_col="time",
        metric_col="metric",
        treatment_col="treatment",
        max_p=1,
        stability_threshold=0.1
    )

    # Autocorrelation dies out starting at 20 minutes (10 is also highly stable compared to 0)
    assert optimal_washout in [10, 20, 30]


def test_carryover_decomposition():
    """Validates Intertemporal Carryover Decomposition direct effect and decay rate estimations."""
    from xpyrment.design.doe.carryover import CarryoverDecomposition

    rng = np.random.default_rng(42)
    N = 150
    times = np.zeros(N)
    for i in range(1, N):
        times[i] = times[i-1] + rng.uniform(0.5, 2.5)

    # Independent random Bernoulli assignments to eliminate collinearity
    treatments = rng.choice([0.0, 1.0], size=N)

    # True parameters: baseline = 10.0, direct = 2.5, carryover = 1.8, lambda = 0.5
    baseline = 10.0
    direct = 2.5
    carryover = 1.8
    lmb = 0.5

    outcomes = np.zeros(N)
    outcomes[0] = baseline + rng.normal(scale=0.01)

    for t in range(1, N):
        dt = times[t] - times[t-1]
        carry_term = treatments[t-1] * np.exp(-lmb * dt)
        outcomes[t] = (
            baseline 
            + direct * treatments[t] 
            + carryover * carry_term 
            + rng.normal(scale=0.01)
        )

    decomposer = CarryoverDecomposition(l2_penalty=1e-5)
    decomposer.fit(outcomes, treatments, times)

    summary = decomposer.summary
    assert "decay_constant_lambda" in summary
    assert "baseline" in summary["coefficients"]

    # Assert accurate recovery within numerical bounds
    assert decomposer.beta_baseline_ == pytest.approx(10.0, abs=0.1)
    assert decomposer.beta_direct_ == pytest.approx(2.5, abs=0.1)
    assert decomposer.beta_carryovers_[0] == pytest.approx(1.8, abs=0.1)
    assert decomposer.lambdas_[0] == pytest.approx(0.5, abs=0.1)


def test_carryover_multi_lag_and_profile():
    """Validates multi-stage lag structures and profile likelihood confidence intervals."""
    from xpyrment.design.doe.carryover import CarryoverDecomposition
    
    np.random.seed(42)
    n = 50
    times = np.arange(n)
    treatments = np.random.randint(0, 2, n)
    # Generate outcomes with carryover from T-1 and T-2
    outcomes = 10 + 2 * treatments + 1 * np.roll(treatments, 1) * np.exp(-0.5) + 0.5 * np.roll(treatments, 2) * np.exp(-0.5 * 2) + np.random.normal(0, 0.1, n)
    
    cd = CarryoverDecomposition(max_lags=2)
    cd.fit(outcomes, treatments, times) 
    
    summary = cd.summary
    assert "decay_constant_lambda" in summary
    assert "lambda_95_ci" in summary
    assert "carryover_effect_lag_1" in summary["coefficients"]
    assert "carryover_effect_lag_2" in summary["coefficients"]
    assert summary["coefficients"]["direct_treatment_effect"] > 0




def test_hash_assign():
    """Tests deterministic hashing for unit-to-variant assignments."""
    assert hash_assign(123, 'salt', ['A', 'B']) == hash_assign(123, 'salt', ['A', 'B'])
    assert hash_assign(123, 'salt1', ['A', 'B']) != hash_assign(123, 'salt2', ['A', 'B'])
    with pytest.raises(ValueError):
        hash_assign(123, 'salt', [])


def test_full_factorial_design():
    """Tests generation of a basic 2^k Full Factorial Design."""
    design = FullFactorialDesign({'F1': [0, 1], 'F2': [0, 1]})
    df = design.generate()
    assert df.shape == (4, 2)
    assert set(df['F1']) == {0, 1}
    assert set(df['F2']) == {0, 1}

def test_taguchi_l12_generation():
    """Tests Taguchi Design L12 orthogonal array correctness."""
    factors = {f"F{i}": [1.0, 2.0] for i in range(11)}
    design = TaguchiDesign(factors, array_name="L12")
    df = design.generate()

    assert len(df) == 12
    assert len(df.columns) == 11

    # Verify column balance (each column has 6 low and 6 high)
    for col in df.columns:
        assert len(df[col].unique()) == 2
        assert df[col].value_counts().iloc[0] == 6

    # Verify that if fewer factors are provided it still works
    factors_short = {f"F{i}": [1.0, 2.0] for i in range(5)}
    design_short = TaguchiDesign(factors_short, array_name="L12")
    df_short = design_short.generate()
    assert len(df_short) == 12
    assert len(df_short.columns) == 5

    # Assert validation
    factors_invalid = {"A": [1.0, 2.0, 3.0]}
    design_invalid = TaguchiDesign(factors_invalid, array_name="L12")
    with pytest.raises(ValueError, match="must have exactly 2 levels"):
        design_invalid.generate()

    # Assert >11 factors
    factors_too_many = {f"F{i}": [1.0, 2.0] for i in range(12)}
    design_too_many = TaguchiDesign(factors_too_many, array_name="L12")
    with pytest.raises(ValueError, match="supports at most 11 factors"):
        design_too_many.generate()


def test_taguchi_l16_generation():
    """Tests Taguchi Design L16 orthogonal array correctness."""
    factors = {f"F{i}": [1.0, 2.0] for i in range(15)}
    design = TaguchiDesign(factors, array_name="L16")
    df = design.generate()

    assert len(df) == 16
    assert len(df.columns) == 15


def test_taguchi_l16_invalid_levels():
    """L16 factors must each have exactly 2 levels."""
    # Single factor with an invalid number of levels
    factors_invalid = {"A": [1.0]}
    design_invalid = TaguchiDesign(factors_invalid, array_name="L16")

    with pytest.raises(ValueError, match="must have exactly 2 levels"):
        design_invalid.generate()


def test_taguchi_l16_too_many_factors():
    """L16 supports at most 15 two-level factors."""
    # 16 factors exceeds the maximum supported by the L16 array
    factors_too_many = {f"F{i}": [1.0, 2.0] for i in range(16)}
    design_too_many = TaguchiDesign(factors_too_many, array_name="L16")

    with pytest.raises(ValueError, match="supports at most 15 factors"):
        design_too_many.generate()


def test_taguchi_l18_generation():
    """Tests Taguchi Design L18 mixed-level orthogonal array correctness."""
    # L18 supports 1 factor with 2 levels, and 7 factors with 3 levels.
    factors = {"F0": [1.0, 2.0]}
    factors.update({f"F{i}": [1.0, 2.0, 3.0] for i in range(1, 8)})

    design = TaguchiDesign(factors, array_name="L18")
    df = design.generate()

    assert len(df) == 18
    assert len(df.columns) == 8

    # Assert valid mixed level validation
    factors_invalid_first = {"F0": [1.0, 2.0, 3.0]}
    design_invalid = TaguchiDesign(factors_invalid_first, array_name="L18")
    with pytest.raises(ValueError, match="must have exactly 2 levels"):
        design_invalid.generate()

    factors_invalid_second = {"F0": [1.0, 2.0], "F1": [1.0, 2.0]}
    design_invalid_2 = TaguchiDesign(factors_invalid_second, array_name="L18")
    with pytest.raises(ValueError, match="must have exactly 3 levels"):
        design_invalid_2.generate()


def test_taguchi_l18_raises_on_too_many_factors():
    """L18 should raise when more than the maximum number of factors is provided."""
    # 1 two-level factor + 8 three-level factors = 9 total factors (> 8 allowed).
    factors = {"F0": [1.0, 2.0]}
    factors.update({f"F{i}": [1.0, 2.0, 3.0] for i in range(1, 9)})

    design = TaguchiDesign(factors, array_name="L18")
    with pytest.raises(ValueError, match="supports at most 8 factors"):
        design.generate()


def test_taguchi_l18_generation_with_fewer_factors():
    """L18 should still generate 18 runs when fewer than 8 factors are provided."""
    # Use F0 (2 levels) plus 3 three-level factors (total 4 factors < 8).
    factors = {"F0": [1.0, 2.0]}
    factors.update({f"F{i}": [1.0, 2.0, 3.0] for i in range(1, 4)})

    design = TaguchiDesign(factors, array_name="L18")
    df = design.generate()

    assert len(df) == 18
    assert len(df.columns) == len(factors)
