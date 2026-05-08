import pytest
import numpy as np
from xpyrment.personalize.meta_learners import RidgeRegressor, SLearner, TLearner, XLearner, ElasticNetRegressor
from xpyrment.personalize.causal_forest import CausalTree, CausalForest


def test_ridge_regressor():
    """Validates analytical Ridge regression fitting accuracy and prediction formats."""
    rng = np.random.default_rng(42)
    X = rng.normal(0.0, 1.0, size=(100, 2))
    # True relationship: y = 5.0 + 2.5 * x_0 - 1.2 * x_1
    y = 5.0 + 2.5 * X[:, 0] - 1.2 * X[:, 1] + rng.normal(0.0, 0.1, size=100)

    reg = RidgeRegressor(alpha=1.0)
    reg.fit(X, y)

    # Asserts coefficients match close to true values
    assert reg.beta[0] == pytest.approx(5.0, abs=0.1)  # Intercept
    assert reg.beta[1] == pytest.approx(2.5, abs=0.1)  # Beta 1
    assert reg.beta[2] == pytest.approx(-1.2, abs=0.1)  # Beta 2

    # Asserts output shapes
    preds = reg.predict(X)
    assert preds.shape == (100,)


def test_meta_learners_uplift():
    """Validates CATE estimation outputs for S-Learner, T-Learner, and X-Learner."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 3))
    # Unbalanced assignment: 70% treatment, 30% control
    treatment = rng.binomial(1, 0.7, size=200)
    
    # Custom treatment effect: +2.0 lift for users with x_0 > 0.0, else 0.0 lift
    true_lift = np.where(X[:, 0] > 0.0, 2.0, 0.0)
    y = 10.0 + 1.5 * X[:, 1] + treatment * true_lift + rng.normal(scale=0.1, size=200)

    # 1. Test S-Learner
    s_learner = SLearner(alpha=1.0)
    s_learner.fit(X, treatment, y)
    s_cate = s_learner.estimate_effect(X)
    assert s_cate.shape == (200,)
    # Verify directional treatment effect prediction correctness
    assert np.mean(s_cate[X[:, 0] > 0.0]) > np.mean(s_cate[X[:, 0] <= 0.0])

    # 2. Test T-Learner
    t_learner = TLearner(alpha=1.0)
    t_learner.fit(X, treatment, y)
    t_cate = t_learner.estimate_effect(X)
    assert t_cate.shape == (200,)
    assert np.mean(t_cate[X[:, 0] > 0.0]) > np.mean(t_cate[X[:, 0] <= 0.0])

    # 3. Test X-Learner (Highly effective for this unbalanced 70/30 layout)
    x_learner = XLearner(alpha=1.0)
    x_learner.fit(X, treatment, y)
    x_cate = x_learner.estimate_effect(X)
    assert x_cate.shape == (200,)
    assert np.mean(x_cate[X[:, 0] > 0.0]) > np.mean(x_cate[X[:, 0] <= 0.0])
    assert x_learner.propensity_score == pytest.approx(0.7, abs=0.15)


def test_causal_tree_and_forest():
    """Validates honest Athey-Imbens causal split logic and random forest bootstrapping."""
    rng = np.random.default_rng(42)
    # Binary splitting feature x_0
    X = np.zeros((100, 2))
    X[:50, 0] = -1.0
    X[50:, 0] = 1.0
    X += rng.normal(scale=0.01, size=(100, 2))  # Add slight jitter

    # Alternate assignment: treatment and control balanced
    treatment = np.tile([1, 0], 50)

    # Lift of +5.0 when x_0 > 0.0, and lift of -2.0 when x_0 <= 0.0
    true_lift = np.where(X[:, 0] > 0.0, 5.0, -2.0)
    y = 20.0 + treatment * true_lift + rng.normal(scale=0.1, size=100)

    # 1. Test Causal Tree partitioning
    tree = CausalTree(max_depth=2, min_samples_leaf=10)
    tree.fit(X, treatment, y)
    tree_cate = tree.predict(X)

    assert tree_cate.shape == (100,)
    # Verify split logic isolated the correct partitions
    assert np.mean(tree_cate[X[:, 0] > 0.0]) == pytest.approx(5.0, abs=0.25)
    assert np.mean(tree_cate[X[:, 0] <= 0.0]) == pytest.approx(-2.0, abs=0.25)

    # 2. Test Causal Forest bootstrap ensemble
    forest = CausalForest(n_estimators=5, max_depth=2, min_samples_leaf=5)
    forest.fit(X, treatment, y, rng=rng)
    forest_cate = forest.estimate_effect(X)

    assert forest_cate.shape == (100,)
    assert np.mean(forest_cate[X[:, 0] > 0.0]) > 0.0
    assert np.mean(forest_cate[X[:, 0] <= 0.0]) < 0.0


def test_elastic_net_regressor():
    """Validates L1/L2 coordinate descent of ElasticNetRegressor for sparsity & weight selection."""
    rng = np.random.default_rng(42)
    n = 100
    p = 10

    # High-dimensional dataset where only first 2 features have true signal (sparse setting)
    X = rng.normal(size=(n, p))
    y = 5.0 + 3.0 * X[:, 0] - 2.0 * X[:, 1] + rng.normal(scale=0.1, size=n)

    # Elastic Net Regressor with small penalty to verify convergence to true coefficients
    en = ElasticNetRegressor(alpha1=0.02, alpha2=0.01, max_iter=1000, tol=1e-5)
    en.fit(X, y)

    # Assert model predicted intercept and relevant coefficients accurately
    assert en.beta_0 == pytest.approx(5.0, abs=0.2)
    assert en.beta[0] == pytest.approx(3.0, abs=0.2)
    assert en.beta[1] == pytest.approx(-2.0, abs=0.2)

    # Assert irrelevant features' weights are close to 0 (sparsity effect)
    for j in range(2, p):
        assert abs(en.beta[j]) < 0.2


def test_elastic_net_meta_learners():
    """Validates CATE estimation of S, T, X Learners configured with Elastic Net base estimators."""
    rng = np.random.default_rng(42)
    n = 150
    p = 5

    X = rng.normal(size=(n, p))
    treatment = rng.binomial(1, 0.5, size=n)
    
    # Lift of +3.0 only for users with x_0 > 0
    true_lift = np.where(X[:, 0] > 0.0, 3.0, 0.0)
    y = 12.0 + 1.0 * X[:, 1] + treatment * true_lift + rng.normal(scale=0.1, size=n)

    # Instantiate meta-learners with ElasticNetRegressor
    s_en = SLearner(base_learner_class=ElasticNetRegressor, alpha1=0.1, alpha2=0.1)
    t_en = TLearner(base_learner_class=ElasticNetRegressor, alpha1=0.1, alpha2=0.1)
    x_en = XLearner(base_learner_class=ElasticNetRegressor, alpha1=0.1, alpha2=0.1)

    for learner in [s_en, t_en, x_en]:
        learner.fit(X, treatment, y)
        cate = learner.estimate_effect(X)

        assert cate.shape == (n,)
        # Verify CATE estimates reflect the high-impact subgroup
        assert np.mean(cate[X[:, 0] > 0.0]) > np.mean(cate[X[:, 0] <= 0.0])


def test_double_machine_learning():
    """Validates Robinson's Double Machine Learning (DML) treatment effect estimation with K-Fold cross-fitting."""
    from xpyrment.personalize.double_ml import DoubleMachineLearning
    import numpy as np

    rng = np.random.default_rng(42)
    n = 200
    p = 5

    X = rng.normal(size=(n, p))
    # Confounded treatment assignment: treatment depends on X[:, 0]
    prop_score = 1.0 / (1.0 + np.exp(-1.5 * X[:, 0]))
    treatment = rng.binomial(1, prop_score).astype(float)

    # True treatment effect is exactly 2.5
    # Outcome has heavy confounder contribution from X[:, 0]
    y = 10.0 + 5.0 * X[:, 0] + 2.5 * treatment + rng.normal(scale=0.1, size=n)

    dml = DoubleMachineLearning(n_folds=3, model_y_kwargs={"alpha": 0.1}, model_t_kwargs={"alpha": 0.1})
    dml.fit(X, treatment, y, seed=42)

    # Assert estimated treatment effect is highly accurate (controlling for confounding via DML residuals)
    assert dml.treatment_effect == pytest.approx(2.5, abs=0.25)
    assert dml.standard_error > 0.0
    assert dml.p_value < 1e-3


