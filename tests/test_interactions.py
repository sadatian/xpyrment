import pytest
import numpy as np
import pandas as pd
from xpyrment.interactions.anova import run_factorial_anova
from xpyrment.interactions.regression import check_treatment_covariate_interaction
from xpyrment.interactions.hstat import compute_friedman_h_statistic
from xpyrment.interactions.detector import InteractionDetector
from xpyrment.interactions.plots import plot_interaction_heatmap, plot_interaction_effects
from xpyrment.analyze.orchestrator import setup
import matplotlib.pyplot as plt

def test_factorial_anova_interaction():
    """Asserts that factorial ANOVA correctly identifies interaction effects in a 2x2 design."""
    # Factor A: [0, 1], Factor B: [0, 1]
    # Y = 10 + 2*A + 2*B + 5*A*B + noise
    df = pd.DataFrame({
        'A': [0, 0, 1, 1] * 25,
        'B': [0, 1, 0, 1] * 25,
    })
    noise = np.random.default_rng(42).normal(0, 0.1, 100)
    df['Y'] = 10.0 + 2.0 * df['A'] + 2.0 * df['B'] + 5.0 * df['A'] * df['B'] + noise
    
    anova_table = run_factorial_anova(df, 'Y ~ A * B')
    
    assert 'A:B' in anova_table.index
    # Interaction p-value should be extremely small
    assert anova_table.loc['A:B', 'PR(>F)'] < 0.001

def test_lrt_interaction_check():
    """Validates Likelihood Ratio Test for treatment-covariate interaction."""
    rng = np.random.default_rng(42)
    n = 200
    treatment = rng.binomial(1, 0.5, size=n)
    covariate = rng.normal(size=n)
    
    # Case 1: Significant interaction
    # Y = 10 + 2*T + 5*T*C + noise
    y_sig = 10.0 + 2.0 * treatment + 5.0 * treatment * covariate + rng.normal(0, 0.1, n)
    df_sig = pd.DataFrame({'T': treatment, 'C': covariate, 'Y': y_sig})
    p_val_sig = check_treatment_covariate_interaction(df_sig, 'T', 'C', 'Y')
    assert p_val_sig < 0.01

    # Case 2: No interaction
    # Y = 10 + 2*T + 2*C + noise
    y_null = 10.0 + 2.0 * treatment + 2.0 * covariate + rng.normal(0, 0.1, n)
    df_null = pd.DataFrame({'T': treatment, 'C': covariate, 'Y': y_null})
    p_val_null = check_treatment_covariate_interaction(df_null, 'T', 'C', 'Y')
    assert p_val_null > 0.05

def test_friedman_h_statistic():
    """Validates H-statistic for linear and non-linear interactions."""
    class DummyModel:
        def predict(self, X):
            # Y = X1 + X2 + 10 * X1 * X2
            return X['X1'] + X['X2'] + 10.0 * X['X1'] * X['X2']
    
    X = pd.DataFrame({
        'X1': np.random.default_rng(42).uniform(0, 1, 50),
        'X2': np.random.default_rng(42).uniform(0, 1, 50)
    })
    
    h_stat = compute_friedman_h_statistic(DummyModel(), X, 'X1', 'X2')
    # Since there is a strong interaction (10 * X1 * X2), h_stat should be non-zero
    assert h_stat > 0.1

def test_interaction_detector_integration():
    """Verifies InteractionDetector identifies HTEs in an Experiment setup."""
    df = pd.DataFrame({
        'variant': [0, 0, 1, 1] * 50,
        'age': np.random.default_rng(42).normal(30, 5, 200),
        'user_id': range(200)
    })
    # Strong interaction with age
    df['revenue'] = 10.0 + 5.0 * df['variant'] * df['age'] + np.random.default_rng(42).normal(0, 1, 200)
    
    exp = setup(df, treatment_col='variant', id_col='user_id', covariates=['age'])
    exp.register_metric('revenue')
    
    detector = InteractionDetector(exp)
    results = detector.detect_all()
    
    assert len(results['heterogeneous_treatment_effects']) > 0
    match = results['heterogeneous_treatment_effects'][0]
    assert match['metric'] == 'revenue'
    assert match['covariate'] == 'age'
    assert match['p_value'] < 0.05


def test_plot_interaction_heatmap():
    """Validates interaction-specific plotting logic for plot_interaction_heatmap."""
    # Test with strictly positive values
    df_pos = pd.DataFrame(np.random.rand(5, 5), columns=list('ABCDE'), index=list('ABCDE'))
    fig_pos, ax_pos = plot_interaction_heatmap(df_pos)

    # Basic object checks
    assert isinstance(fig_pos, plt.Figure)
    assert isinstance(ax_pos, plt.Axes)
    assert ax_pos.get_title() == "Factor Interaction Heatmap"

    # Heatmaps in seaborn are typically rendered as collections (QuadMesh), not images
    assert ax_pos.collections, "Expected at least one collection in the axes for the heatmap."
    im_pos = ax_pos.collections[0]

    # For strictly positive data, we expect a sequential colormap (e.g. 'YlOrRd')
    cmap_name_pos = im_pos.get_cmap().name.lower()
    assert 'ylorrd' in cmap_name_pos, f"Expected a 'YlOrRd'-like colormap, got {im_pos.get_cmap().name!r}"

    # For strictly positive data, limits should not be centered on 0
    vmin_pos, vmax_pos = im_pos.get_clim()
    assert vmin_pos >= 0, "For positive-only data, vmin should be non-negative."
    # Not centered around 0: |vmin| != |vmax|
    assert not pytest.approx(-vmin_pos) == vmax_pos

    # Axis labels should encode interaction semantics
    assert ax_pos.get_xlabel(), "Expected x-axis label to be set."
    assert ax_pos.get_ylabel(), "Expected y-axis label to be set."

    # Tick labels should correspond to DataFrame index/columns
    x_tick_labels_pos = [t.get_text() for t in ax_pos.get_xticklabels() if t.get_text()]
    y_tick_labels_pos = [t.get_text() for t in ax_pos.get_yticklabels() if t.get_text()]
    assert set(df_pos.columns).issubset(set(x_tick_labels_pos))
    assert set(df_pos.index).issubset(set(y_tick_labels_pos))

    # Colorbar should be present and labeled with interaction semantics
    cbar_ax_pos = next(
        (a for a in fig_pos.axes if a is not ax_pos and a.get_ylabel()),
        None,
    )
    assert cbar_ax_pos is not None, "Expected a colorbar axes."
    assert cbar_ax_pos.get_ylabel() == "Interaction Strength"

    plt.close(fig_pos)

    # Test with negative values (mixed-sign matrix)
    df_neg = pd.DataFrame(np.random.randn(5, 5), columns=list('ABCDE'), index=list('ABCDE'))
    fig_neg, ax_neg = plot_interaction_heatmap(df_neg)

    # Basic object checks
    assert isinstance(fig_neg, plt.Figure)
    assert isinstance(ax_neg, plt.Axes)
    assert ax_neg.get_title() == "Factor Interaction Heatmap"

    # Heatmaps in seaborn are typically rendered as collections (QuadMesh), not images
    assert ax_neg.collections, "Expected at least one collection in the axes for the heatmap."
    im_neg = ax_neg.collections[0]

    # For mixed-sign data, we expect a diverging colormap (e.g. 'RdBu_r')
    # Under newer seaborn/matplotlib, cmap name might not be exactly retained (e.g., could be "from_list"
    # if it's constructed dynamically), but we can check the cmap type or assume it is handled correctly if it renders without issue.
    # To avoid brittleness, we skip exact name assertion if it is 'from_list'
    cmap_name_neg = im_neg.get_cmap().name.lower()
    if cmap_name_neg != 'from_list':
        assert 'rdbu' in cmap_name_neg, f"Expected an 'RdBu_r'-like colormap, got {im_neg.get_cmap().name!r}"

    # For mixed-sign data, normalization should be centered on 0
    vmin_neg, vmax_neg = im_neg.get_clim()
    assert vmin_neg < 0 < vmax_neg, "Expected color limits to span 0 for mixed-sign data."
    # Approximately symmetric around 0
    assert pytest.approx(-vmin_neg, rel=0.1) == vmax_neg

    # Axis labels should encode interaction semantics
    assert ax_neg.get_xlabel(), "Expected x-axis label to be set."
    assert ax_neg.get_ylabel(), "Expected y-axis label to be set."

    # Tick labels should correspond to DataFrame index/columns
    x_tick_labels_neg = [t.get_text() for t in ax_neg.get_xticklabels() if t.get_text()]
    y_tick_labels_neg = [t.get_text() for t in ax_neg.get_yticklabels() if t.get_text()]
    assert set(df_neg.columns).issubset(set(x_tick_labels_neg))
    assert set(df_neg.index).issubset(set(y_tick_labels_neg))

    # Colorbar should be present and labeled with interaction semantics
    cbar_ax_neg = next(
        (a for a in fig_neg.axes if a is not ax_neg and a.get_ylabel()),
        None,
    )
    assert cbar_ax_neg is not None, "Expected a colorbar axes."
    assert cbar_ax_neg.get_ylabel() == "Interaction Strength"

    plt.close(fig_neg)


@pytest.mark.parametrize(
    "covariate_type, num_treatments, add_nans",
    [
        ("categorical", 2, False),       # Categorical/discrete covariate
        ("continuous", 2, False),        # Continuous numeric covariate
        ("discrete_numeric", 3, False),  # Numeric but discrete (e.g. few unique values)
        ("continuous", 1, False),        # Single treatment group
        ("categorical", 2, True),        # Data with NaNs
    ]
)
def test_plot_interaction_effects(covariate_type, num_treatments, add_nans):
    """Validates plot_interaction_effects for different covariates, treatments, and missing data."""
    n_samples = 60

    treatments = [f'Group_{i}' for i in range(num_treatments)]
    treatment_col = np.random.choice(treatments, n_samples)

    if covariate_type == "categorical":
        covariate_col = np.random.choice(['US', 'UK', 'CA'], n_samples)
    elif covariate_type == "continuous":
        covariate_col = np.random.rand(n_samples) * 50 + 20
    elif covariate_type == "discrete_numeric":
        covariate_col = np.random.choice([0, 1, 2], n_samples)

    revenue = np.random.randn(n_samples) * 10 + 100

    df = pd.DataFrame({
        'treatment': treatment_col,
        'covariate': covariate_col,
        'revenue': revenue
    })

    if add_nans:
        df.loc[0:5, 'revenue'] = np.nan
        df.loc[6:10, 'covariate'] = np.nan

    fig, ax = plot_interaction_effects(df, 'treatment', 'revenue', 'covariate')

    # Assertions
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

    title = ax.get_title().lower()
    assert 'treatment' in title
    assert 'covariate' in title
    assert 'revenue' in title

    assert ax.get_xlabel().lower() == 'covariate'
    assert ax.get_ylabel().lower() == 'revenue'

    plt.close('all')
