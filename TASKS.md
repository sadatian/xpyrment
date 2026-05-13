# Active Development Backlog & Task Board

This is the central task board for the **`xpyrment`** project maintenance. It is designed to be fully integrated with **GitHub Issues**, **Pull Requests**, and **GitHub Project Boards**.

> [!NOTE]
> You can convert any checkbox task below directly into a tracked GitHub Issue or link it to a PR using the GitHub UI, or reference the task checklist when closing issues!

---

## 📋 Block A: Systematic Requirements Audit & Edge Case Verification
* **Objective**: Systematically audit all analytical models against mathematical boundary conditions, singular matrix configurations, and empty arrays to prevent production crashes.
* **Status**: 🔄 In Progress

- [x] **A.1 CUPED Singular Covariate Matrices**
  - [x] Implement protective condition when the covariate variance is zero or collinear.
  - [x] Ensure proper fallback to standard Difference-in-Means when regression adjustments are unsolvable.
  - [x] Add regression tests under `tests/test_metrics.py`.
- [x] **A.2 Sample Ratio Mismatch (SRM) Extreme Low Sample Sizes**
  - [x] Guard SRM chi-square tests when cell counts are zero or extremely low (e.g., < 5).
  - [x] Log helpful diagnostics warnings instead of crashing with division-by-zero or math domain errors.
- [x] **A.3 BCa Bootstrap Extreme Resampling Bounds**
  - [x] Handle scenarios where the bootstrap distributions are perfectly degenerate or have zero variance.
  - [x] Add test cases simulating zero variance metrics in `tests/test_bootstrap_harden.py`.
- [x] **A.4 Design of Experiments Level Limits & Edge Configurations**
  - [x] Add validators to raise descriptive exceptions when level counts do not align with fractional factorial or definitive screening generator matrices.

---

## 📋 Block B: Documentation Validation & Warnings Scrub
* **Objective**: Guarantee a 100% clean documentation compilation with correct mathematical symbols and zero formatting alerts.
* **Status**: 📅 Planned

- [x] **B.1 MathJax Character Rendering Sweep**
  - [x] Audit LaTeX formulas in both code docstrings and `.md` pages to ensure perfect compatibility with Material for MkDocs.
- [x] **B.2 Unescaped Code Snippets in Docstrings**
  - [x] Double-check docstrings of newly added modules (`quasi`, `plan`) to ensure all variables and parameters are fenced correctly.
- [x] **B.3 Dynamic Macros Verification**
  - [x] Verify that all automated macros (`list_doe_designs`, `list_metrics`, `cli_help`) evaluate cleanly on fresh builds.

---

## 🚀 Primary Future Goals & Performance Optimization
* **Objective**: Accelerate statistical computations and multi-armed bandit simulation runs through vectorization and parallelism.
* **Status**: 📅 Long-term Roadmap

- [x] **P.1 Multithreading & Parallelization**
  - [x] Parallelize the long-running simulation loops inside the Thompson Sampling and UCB bandit agents.
  - [x] Vectorize Monte Carlo resampling steps within the bootstrap modules.
- [x] **P.2 High-Performance Pre-built Methods (NumPy First)**
  - [x] Audit performance hotspots and replace standard native loops with vectorized NumPy array broadcasting.
  - [x] Optimize covariate balancing algorithms through SciPy optimization solvers.
- [x] **P.3 Execution Profiling Suite**
  - [x] Create a lightweight memory and execution profiler script to monitor analysis runtimes over large datasets.
- [ ] **bayesian.py**: Implement conjugate Gamma-Poisson model pairing for discrete count metrics (such as page views or clicks).
- [ ] **carryover.py**: Add a profile likelihood fallback solver to compute joint asymptotic confidence intervals for both lambda and the beta parameters.
- [ ] **carryover.py**: Extend the carryover decomposition to handle multi-stage lag structures (e.g., T_{t-2} and T_{t-3}) with distinct decay vectors.
- [ ] **causal_forest.py**: Implement randomized feature subspace selection (m_try) at each split to increase forest diversity.
- [ ] **causal_forest.py**: Support true honest splitting (partitioning on half the training set, estimating outcomes on the other half).
- [ ] **cluster.py**: Implement modularity-based partitioning (e.g., Louvain or Spectral partitioning) for denser, complex networks.
- [ ] **cluster.py**: Support cluster-size balancing mechanisms to prevent extreme power decay caused by highly asymmetric community structures.
- [ ] **confounding.py**: Implement sequential D-optimal design updates that minimize the trace of the alias matrix projection error
- [ ] **copula.py**: Add multivariate p-value corrections (e.g. step-down procedures) to control Family-Wise Error Rate (FWER) under copula.
- [ ] **copula.py**: Support parametric copula families (such as Clayton or Gumbel) to capture asymmetric tail dependencies.
- [ ] **d_optimal.py**: Add alternative optimality criteria such as A-Optimality (trace of inverse information matrix) and G-Optimality (minimizing maximum prediction variance).
- [ ] **d_optimal.py**: Implement fast rank-1 update formulas (using Sherman-Morrison) to compute determinants in O(1) instead of recalculating full SVD in O(p^3).
- [ ] **detector.py**: Implement dispatcher
- [ ] **diff_in_diff.py**: Implement cluster-robust standard errors to handle correlated errors across repeat-measure cohort panels.
- [ ] **diff_in_diff.py**: Support incorporating external covariate matrices into OLS adjustments.
- [ ] **double_ml.py**: Support estimating Heterogeneous Treatment Effects (CATE) via local polynomial residual-on-residual regression
- [ ] **dsd.py**: Add validation checks to confirm no active two-factor interaction terms are fully aliased with main effects.
- [ ] **dsd.py**: Implement algorithmic synthesis of conference matrices of arbitrary even orders using quadratic residues.
- [ ] **dtr.py**: Implement doubly robust Q-learning corrections to protect against Q-model specification biases.
- [ ] **dtr.py**: Support non-linear Q-functions using basis expansion (e.g. B-splines) for continuous history states.
- [ ] **epsilon_greedy.py**: Extend with contextual multi-armed bandit variants using online regression base learners.
- [ ] **epsilon_greedy.py**: Support a configurable minimum epsilon floor (e.g., min_epsilon=0.01) to prevent exploration from decaying completely to zero.
- [ ] **evop.py**: Add Box-Hunter evolutionary optimization cycle calculation curves to automatically compute main/interaction effects and errors.
- [ ] **evop.py**: Integrate automated stopping rules when treatment boundary shift reaches optimum levels (Simplex Evolutionary Operation optimization).
- [ ] **extreme.py**: Add a profile likelihood fallback optimizer to support shape parameter estimation when xi is outside the MOM boundary [0, 0.5].
- [ ] **extreme.py**: Implement automated threshold choice heuristics (e.g., using Hill plots or Gertensgarbe's sequential tests).
- [ ] **federated.py**: Implement threshold decryption where the private key lambda is divided into shares (lambda_1, lambda_2)
- [ ] **frequentist.py**: Add Brunner-Munzel test as a robust alternative to Mann-Whitney U when variances are highly unequal.
- [ ] **frequentist.py**: Implement Fisher's Exact test and G-test of independence for high-precision categorical conversions.
- [ ] **hstat.py**: Implement H-statistic calculations
- [ ] **hte.py**: Implement causal tree or subgroup t-test sweep
- [ ] **identity.py**: Implement parallelized union-find component graph traversal using multi-threaded batch resolution for large-scale production logs.
- [ ] **infinite_mixture.py**: Extend the collapsed Gibbs sampler to multivariate Normal-Inverse-Wishart conjugate mixtures.
- [ ] **infinite_mixture.py**: Implement a Variational Inference (VI) coordinate ascent solver (Blei-Jordan, 2006) to accelerate clustering speed on massive scale datasets.
- [ ] **ingestion.py**: Add schema enforcement using Pydantic models or Pandera DataFrame schemas.
- [ ] **ingestion.py**: Implement out-of-core chunked ingestion or Dask integration for datasets exceeding local RAM capacities.
- [ ] **meta_learners.py**: Implement cyclic coordinate descent path optimization (warm starts over a regularization grid lambda) to compute the complete Elastic Net path efficiently.
- [ ] **meta_regression.py**: Add Restricted Maximum Likelihood (REML) iteration solver as an alternative to the closed-form DerSimonian-Laird estimator.
- [ ] **meta_regression.py**: Support empirical Bayes shrinkage estimations of individual study-level random effects (u_j).
- [ ] **mixture.py**: Add support for McLean-Anderson or coordinate exchange constraints to handle upper/lower bounds on individual ingredients (e.g., ingredient A must be between 10% and 30%).
- [ ] **mixture.py**: Implement Simplex Centroid designs to supplement Simplex Lattice with interior-point checks.
- [ ] **monitor.py**: Add real-time anomaly detection alerts (such as moving-average threshold alerts) to notify users of sudden traffic drops or abnormal shifts.
- [ ] **monitor.py**: Integrate Slack and Email webhook messaging adapters to broadcast live traffic monitoring reports.
- [ ] **multi_objective.py**: Add hypervolume-based probability of improvement (HV-PI) as an alternative multi-objective acquisition metric.
- [ ] **multi_objective.py**: Implement 3D+ hypervolume partitioning algorithms (e.g., Overmars and Yap) to support arbitrary objective dimensions.
- [ ] **non_stationary.py**: Extend the reward model to support non-stationary continuous conjugate priors (Normal-Inverse-Gamma for linear payoffs).
- [ ] **non_stationary.py**: Implement a dynamic tracking update for the discount factor gamma using meta-learning (e.g., bandit-over-bandits).
- [ ] **novelty.py**: Add support for weighted least squares (WLS) where observation weights scale with daily binned traffic volumes.
- [ ] **novelty.py**: Implement non-linear decay estimation (such as exponential decay curves) using non-linear least squares optimization.
- [ ] **ope.py**: Implement Marginalized Importance Sampling (MIS) for sequential multi-step decision processes.
- [ ] **ope.py**: Integrate asymptotic confidence interval estimators based on empirical Bernstein inequalities.
- [ ] **optimal_transport.py**: Add asymptotic bootstrapping of Wasserstein boundaries to conduct non-parametric distribution equality tests.
- [ ] **optimal_transport.py**: Extend the optimal transport solver to multidimensional metric profiles using Sinkhorn-Knopp entropic regularization.
- [ ] **p_curve.py**: Create visualization plots comparing observed significant p-value densities against uniform null curves.
- [ ] **p_curve.py**: Implement analytical estimation of the underlying statistical power curve based on non-central distribution fits.
- [ ] **partition.py**: Implement asynchronous lock-free update rules (Hogwild!) to parallelize label changes on massive graphs.
- [ ] **partition.py**: Support Louvain modularity criteria (Q-index) as an alternative objective to balance small local clusters.
- [ ] **plots.py**: Implement plotting code
- [ ] **privacy.py**: Implement Renyi Differential Privacy (RDP) accounting to support tight composition over multi-pass queries.
- [ ] **privacy.py**: Support private covariance matrix noise injection based on the Wishart mechanism or advanced output perturbation.
- [ ] **regression.py**: Implement interactive regression model
- [ ] **rolling_synthetic_control.py**: Add interactive covariate balance weight constraints (V-matrix optimizations) within the rolling SLSQP loss functions.
- [ ] **rolling_synthetic_control.py**: Implement out-of-fold temporal cross-validation to select rolling window size H and regularization hyper-parameters (lambda_l1, lambda_l2) dynamically.
- [ ] **router.py**: Implement full intelligent router
- [ ] **sdid.py**: Implement placebo-based inference and standard error estimation using block bootstrap.
- [ ] **sensitivity.py**: Add a contour-plot generation utility that maps adjusted effect boundaries over a 2D grid of r2_d_u and r2_y_u.
- [ ] **sensitivity.py**: Support Wilcoxon signed-rank sum statistic bounds as a distribution-free alternative to sign-test Rosenbaum bounds.
- [ ] **sequential.py**: Implement exact multivariate normal integration (e.g. using Genz-Bretz algorithms) to solve multi-look joint covariance critical bounds.
- [ ] **sequential.py**: Implement sequential boundary functions (mSPRT or Pocock)
- [ ] **sequential.py**: Support binding and non-binding futility boundaries using beta-spending formulations to support early stopping for futility.
- [ ] **shap.py**: Implement optional shap dependency check and interaction calculation
- [ ] **snmm.py**: Add wild bootstrap inference over sequential stages to compute joint confidence intervals for beta vectors.
- [ ] **snmm.py**: Implement doubly robust sequential g-estimation incorporating stage-specific baseline outcome models.
- [ ] **spillover.py**: Implement bootstrap or Horvitz-Thompson variance solvers to provide standard errors and p-values for DTE and ISE.
- [ ] **spillover.py**: Support fractional neighborhood exposure thresholds (e.g., classifying nodes as exposed only when >20% of their neighbors are treated).
- [ ] **spillover.py**: Support multi-hop network exposures (e.g., 2-hop exposures where a node is influenced by friends-of-friends) to capture deeper peer cascades.
- [ ] **stopping.py**: Add Group Sequential Design boundaries (such as O'Brien-Fleming or Pocock spend functions) for traditional multi-stage peeking.
- [ ] **stopping.py**: Integrate Bayesian sequential early-stopping checks leveraging Expected Loss threshold limits.
- [ ] **streaming.py**: Support dynamic forgetting factors (exponential decay weighting) to allow the streaming model to track non-stationary regimes in high-frequency event streams.
- [ ] **switchback.py**: Add option to optimize period switchover frequencies to minimize carrying-over spillover effects.
- [ ] **switchback.py**: Implement Latin Square multi-period and multi-variant Latin Square crossover balancing to optimize more than 2 variants.
- [ ] **synthetic_control.py**: Implement placebo in-space/in-time permutation testing to construct analytical inference p-values.
- [ ] **synthetic_control.py**: Support optimizing weights on auxiliary predictor covariate matrices in addition to target outcome histories.
- [ ] **synthetic_control.py**: Support regularized/penalized Synthetic Control (e.g., L1/L2 weights penalization) to handle high-dimensional donor pools where J > T_pre.
- [ ] **taguchi.py**: Add automatic lookup support for L12, L16, and L18 mixed-level orthogonal arrays.
- [ ] **taguchi.py**: Integrate signal-to-noise ratio (SNR) loss analysis plots for parameter robust design.
- [ ] **thompson.py**: Implement Dirichlet-Multinomial Thompson Sampling to support categorical/multinomial feedback.
- [ ] **thompson.py**: Implement batched/delayed reward Thompson Sampling updates using Gaussian Process models to handle settings where feedback is slow or clustered.
- [ ] **thompson.py**: Support Normal-Inverse-Gamma conjugate priors for continuous rewards with unknown variance.
- [ ] **transformations.py**: Implement full delta normalization
- [ ] **tuning.py**: Implement automatic kernel lengthscale optimization (marginal likelihood maximization) via Brent's method or gradient descent on GP log likelihood.
- [ ] **ucb.py**: Implement sliding-window UCB versions to better handle non-stationary environments.
- [ ] **ucb.py**: Support standard-deviation-based variance scaling (UCB1-Tuned) to adapt to reward variability.
- [ ] **variance_reduction.py**: Add automatic pre-period alignment diagnostics to confirm the covariate is truly independent of treatment assignments.
- [ ] **variance_reduction.py**: Implement multi-covariate CUPAC (Controlled-experiments Using Pre-Experiment Data and Machine Learning) to support non-linear ML-based predictions as covariates.
## 📋 Block T: Repository TODOs
* **Objective**: Address all pending TODOs scattered across the codebase.
* **Status**: 📅 Planned

- [ ] **balance.py**: Add Kolmogorov-Smirnov distance validation checks on continuous covariates to verify full distribution shape alignment beyond mean and variance.
- [ ] **balance.py**: Integrate Mahalanobis distance multivariate covariance balance tests to verify joint multi-feature balance.
- [ ] **bayesian.py**: Add numerical integration solvers to compute PBB and Expected Loss exactly without relying on Monte Carlo simulations.
- [ ] **bayesian.py**: Implement conjugate Gamma-Poisson model pairing for discrete count metrics (such as page views or clicks).
- [ ] **carryover.py**: Add a profile likelihood fallback solver to compute joint asymptotic confidence intervals for both lambda and the beta parameters.
- [ ] **carryover.py**: Extend the carryover decomposition to handle multi-stage lag structures (e.g., T_{t-2} and T_{t-3}) with distinct decay vectors.
- [ ] **causal_forest.py**: Implement randomized feature subspace selection (m_try) at each split to increase forest diversity.
- [ ] **causal_forest.py**: Support true honest splitting (partitioning on half the training set, estimating outcomes on the other half).
- [ ] **cluster.py**: Implement modularity-based partitioning (e.g., Louvain or Spectral partitioning) for denser, complex networks.
- [ ] **cluster.py**: Support cluster-size balancing mechanisms to prevent extreme power decay caused by highly asymmetric community structures.
- [ ] **confounding.py**: Implement sequential D-optimal design updates that minimize the trace of the alias matrix projection error
- [ ] **copula.py**: Add multivariate p-value corrections (e.g. step-down procedures) to control Family-Wise Error Rate (FWER) under copula.
- [ ] **copula.py**: Support parametric copula families (such as Clayton or Gumbel) to capture asymmetric tail dependencies.
- [ ] **d_optimal.py**: Add alternative optimality criteria such as A-Optimality (trace of inverse information matrix) and G-Optimality (minimizing maximum prediction variance).
- [ ] **d_optimal.py**: Implement fast rank-1 update formulas (using Sherman-Morrison) to compute determinants in O(1) instead of recalculating full SVD in O(p^3).
- [ ] **detector.py**: Implement dispatcher
- [ ] **diff_in_diff.py**: Implement cluster-robust standard errors to handle correlated errors across repeat-measure cohort panels.
- [ ] **diff_in_diff.py**: Support incorporating external covariate matrices into OLS adjustments.
- [ ] **double_ml.py**: Support estimating Heterogeneous Treatment Effects (CATE) via local polynomial residual-on-residual regression
- [ ] **dsd.py**: Add validation checks to confirm no active two-factor interaction terms are fully aliased with main effects.
- [ ] **dsd.py**: Implement algorithmic synthesis of conference matrices of arbitrary even orders using quadratic residues.
- [ ] **dtr.py**: Implement doubly robust Q-learning corrections to protect against Q-model specification biases.
- [ ] **dtr.py**: Support non-linear Q-functions using basis expansion (e.g. B-splines) for continuous history states.
- [ ] **epsilon_greedy.py**: Extend with contextual multi-armed bandit variants using online regression base learners.
- [ ] **epsilon_greedy.py**: Support a configurable minimum epsilon floor (e.g., min_epsilon=0.01) to prevent exploration from decaying completely to zero.
- [ ] **evop.py**: Add Box-Hunter evolutionary optimization cycle calculation curves to automatically compute main/interaction effects and errors.
- [ ] **evop.py**: Integrate automated stopping rules when treatment boundary shift reaches optimum levels (Simplex Evolutionary Operation optimization).
- [ ] **extreme.py**: Add a profile likelihood fallback optimizer to support shape parameter estimation when xi is outside the MOM boundary [0, 0.5].
- [ ] **extreme.py**: Implement automated threshold choice heuristics (e.g., using Hill plots or Gertensgarbe's sequential tests).
- [ ] **federated.py**: Implement threshold decryption where the private key lambda is divided into shares (lambda_1, lambda_2)
- [ ] **frequentist.py**: Add Brunner-Munzel test as a robust alternative to Mann-Whitney U when variances are highly unequal.
- [ ] **frequentist.py**: Implement Fisher's Exact test and G-test of independence for high-precision categorical conversions.
- [ ] **hstat.py**: Implement H-statistic calculations
- [ ] **hte.py**: Implement causal tree or subgroup t-test sweep
- [ ] **identity.py**: Implement parallelized union-find component graph traversal using multi-threaded batch resolution for large-scale production logs.
- [ ] **infinite_mixture.py**: Extend the collapsed Gibbs sampler to multivariate Normal-Inverse-Wishart conjugate mixtures.
- [ ] **infinite_mixture.py**: Implement a Variational Inference (VI) coordinate ascent solver (Blei-Jordan, 2006) to accelerate clustering speed on massive scale datasets.
- [ ] **ingestion.py**: Add schema enforcement using Pydantic models or Pandera DataFrame schemas.
- [ ] **ingestion.py**: Implement out-of-core chunked ingestion or Dask integration for datasets exceeding local RAM capacities.
- [ ] **instrumental_variables.py**: Implement robust Huber-White sandwich standard error estimators to handle heteroskedasticity in the second-stage residuals.
- [ ] **instrumental_variables.py**: Support multi-valued discrete and continuous treatment indicators under general control-function approaches.
- [ ] **its.py**: Add dynamic lag order selection using AIC/BIC information criteria to optimize the Newey-West HAC spectral bandwidth.
- [ ] **its.py**: Support autoregressive integrated moving average (ARIMA) error components integrated with the segmented OLS model.
- [ ] **lhs.py**: Add correlation-minimization algorithms (such as Owen's randomized LHS) to reduce collinearity between factors.
- [ ] **lhs.py**: Implement Maximin Latin Hypercube optimization (shuffling columns to maximize minimum pairwise Euclidean distance).
- [ ] **markov.py**: Add continuous-time Markov intensity matrix (Q) estimations to model exact duration stay times within states.
- [ ] **markov.py**: Implement bootstrap confidence interval approximations for stationary distribution probability shifts.
- [ ] **matching.py**: Support high-performance KD-Tree indexing to accelerate nearest-neighbor caliper searches on multi-million row observational datasets.
- [ ] **matrix_completion.py**: Add temporal and unit-level fixed effects (Y_{t, n} = L_{t, n} + alpha_i + beta_t) integrated with the SVT update loops.
- [ ] **matrix_completion.py**: Implement randomized SVD projections (Halko et al., 2011) to accelerate singular value thresholding on large scale matrices.
- [ ] **meta_analysis.py**: Implement study-level meta-regression adjustments supporting auxiliary study covariates (e.g., historical run duration).
- [ ] **meta_analysis.py**: Support Trim-and-Fill algorithms to estimate and adjust pooled estimates for funnel plot asymmetry / publication bias.
- [ ] **meta_analysis.py**: Support alternative random-effects variance estimators (such as Hedges-Olkin or Sidik-Jonkman) to compare against DerSimonian-Laird.
- [ ] **meta_learners.py**: Implement cyclic coordinate descent path optimization (warm starts over a regularization grid lambda) to compute the complete Elastic Net path efficiently.
- [ ] **meta_regression.py**: Add Restricted Maximum Likelihood (REML) iteration solver as an alternative to the closed-form DerSimonian-Laird estimator.
- [ ] **meta_regression.py**: Support empirical Bayes shrinkage estimations of individual study-level random effects (u_j).
- [ ] **mixture.py**: Add support for McLean-Anderson or coordinate exchange constraints to handle upper/lower bounds on individual ingredients (e.g., ingredient A must be between 10% and 30%).
- [ ] **mixture.py**: Implement Simplex Centroid designs to supplement Simplex Lattice with interior-point checks.
- [ ] **monitor.py**: Add real-time anomaly detection alerts (such as moving-average threshold alerts) to notify users of sudden traffic drops or abnormal shifts.
- [ ] **monitor.py**: Integrate Slack and Email webhook messaging adapters to broadcast live traffic monitoring reports.
- [ ] **multi_objective.py**: Add hypervolume-based probability of improvement (HV-PI) as an alternative multi-objective acquisition metric.
- [ ] **multi_objective.py**: Implement 3D+ hypervolume partitioning algorithms (e.g., Overmars and Yap) to support arbitrary objective dimensions.
- [ ] **non_stationary.py**: Extend the reward model to support non-stationary continuous conjugate priors (Normal-Inverse-Gamma for linear payoffs).
- [ ] **non_stationary.py**: Implement a dynamic tracking update for the discount factor gamma using meta-learning (e.g., bandit-over-bandits).
- [ ] **novelty.py**: Add support for weighted least squares (WLS) where observation weights scale with daily binned traffic volumes.
- [ ] **novelty.py**: Implement non-linear decay estimation (such as exponential decay curves) using non-linear least squares optimization.
- [ ] **ope.py**: Implement Marginalized Importance Sampling (MIS) for sequential multi-step decision processes.
- [ ] **ope.py**: Integrate asymptotic confidence interval estimators based on empirical Bernstein inequalities.
- [ ] **optimal_transport.py**: Add asymptotic bootstrapping of Wasserstein boundaries to conduct non-parametric distribution equality tests.
- [ ] **optimal_transport.py**: Extend the optimal transport solver to multidimensional metric profiles using Sinkhorn-Knopp entropic regularization.
- [ ] **p_curve.py**: Create visualization plots comparing observed significant p-value densities against uniform null curves.
- [ ] **p_curve.py**: Implement analytical estimation of the underlying statistical power curve based on non-central distribution fits.
- [ ] **partition.py**: Implement asynchronous lock-free update rules (Hogwild!) to parallelize label changes on massive graphs.
- [ ] **partition.py**: Support Louvain modularity criteria (Q-index) as an alternative objective to balance small local clusters.
- [ ] **plots.py**: Implement plotting code
- [ ] **privacy.py**: Implement Renyi Differential Privacy (RDP) accounting to support tight composition over multi-pass queries.
- [ ] **privacy.py**: Support private covariance matrix noise injection based on the Wishart mechanism or advanced output perturbation.
- [ ] **regression.py**: Implement interactive regression model
- [ ] **rolling_synthetic_control.py**: Add interactive covariate balance weight constraints (V-matrix optimizations) within the rolling SLSQP loss functions.
- [ ] **rolling_synthetic_control.py**: Implement out-of-fold temporal cross-validation to select rolling window size H and regularization hyper-parameters (lambda_l1, lambda_l2) dynamically.
- [ ] **router.py**: Implement full intelligent router
- [ ] **sdid.py**: Implement placebo-based inference and standard error estimation using block bootstrap.
- [ ] **sensitivity.py**: Add a contour-plot generation utility that maps adjusted effect boundaries over a 2D grid of r2_d_u and r2_y_u.
- [ ] **sensitivity.py**: Support Wilcoxon signed-rank sum statistic bounds as a distribution-free alternative to sign-test Rosenbaum bounds.
- [ ] **sequential.py**: Implement exact multivariate normal integration (e.g. using Genz-Bretz algorithms) to solve multi-look joint covariance critical bounds.
- [ ] **sequential.py**: Implement sequential boundary functions (mSPRT or Pocock)
- [ ] **sequential.py**: Support binding and non-binding futility boundaries using beta-spending formulations to support early stopping for futility.
- [ ] **shap.py**: Implement optional shap dependency check and interaction calculation
- [ ] **snmm.py**: Add wild bootstrap inference over sequential stages to compute joint confidence intervals for beta vectors.
- [ ] **snmm.py**: Implement doubly robust sequential g-estimation incorporating stage-specific baseline outcome models.
- [ ] **spillover.py**: Implement bootstrap or Horvitz-Thompson variance solvers to provide standard errors and p-values for DTE and ISE.
- [ ] **spillover.py**: Support fractional neighborhood exposure thresholds (e.g., classifying nodes as exposed only when >20% of their neighbors are treated).
- [ ] **spillover.py**: Support multi-hop network exposures (e.g., 2-hop exposures where a node is influenced by friends-of-friends) to capture deeper peer cascades.
- [ ] **stopping.py**: Add Group Sequential Design boundaries (such as O'Brien-Fleming or Pocock spend functions) for traditional multi-stage peeking.
- [ ] **stopping.py**: Integrate Bayesian sequential early-stopping checks leveraging Expected Loss threshold limits.
- [ ] **streaming.py**: Support dynamic forgetting factors (exponential decay weighting) to allow the streaming model to track non-stationary regimes in high-frequency event streams.
- [ ] **switchback.py**: Add option to optimize period switchover frequencies to minimize carrying-over spillover effects.
- [ ] **switchback.py**: Implement Latin Square multi-period and multi-variant Latin Square crossover balancing to optimize more than 2 variants.
- [ ] **synthetic_control.py**: Implement placebo in-space/in-time permutation testing to construct analytical inference p-values.
- [ ] **synthetic_control.py**: Support optimizing weights on auxiliary predictor covariate matrices in addition to target outcome histories.
- [ ] **synthetic_control.py**: Support regularized/penalized Synthetic Control (e.g., L1/L2 weights penalization) to handle high-dimensional donor pools where J > T_pre.
- [ ] **taguchi.py**: Add automatic lookup support for L12, L16, and L18 mixed-level orthogonal arrays.
- [ ] **taguchi.py**: Integrate signal-to-noise ratio (SNR) loss analysis plots for parameter robust design.
- [ ] **thompson.py**: Implement Dirichlet-Multinomial Thompson Sampling to support categorical/multinomial feedback.
- [ ] **thompson.py**: Implement batched/delayed reward Thompson Sampling updates using Gaussian Process models to handle settings where feedback is slow or clustered.
- [ ] **thompson.py**: Support Normal-Inverse-Gamma conjugate priors for continuous rewards with unknown variance.
- [ ] **transformations.py**: Implement full delta normalization
- [ ] **tuning.py**: Implement automatic kernel lengthscale optimization (marginal likelihood maximization) via Brent's method or gradient descent on GP log likelihood.
- [ ] **ucb.py**: Implement sliding-window UCB versions to better handle non-stationary environments.
- [ ] **ucb.py**: Support standard-deviation-based variance scaling (UCB1-Tuned) to adapt to reward variability.
- [ ] **variance_reduction.py**: Add automatic pre-period alignment diagnostics to confirm the covariate is truly independent of treatment assignments.
- [ ] **variance_reduction.py**: Implement multi-covariate CUPAC (Controlled-experiments Using Pre-Experiment Data and Machine Learning) to support non-linear ML-based predictions as covariates.
- [x] **aa_test.py**: Add parallel execution or vectorization for large-scale multi-run simulations to reduce processing time under 100k iterations.
- [x] **aa_test.py**: Integrate false discovery rate (FDR) control and family-wise error rate verification diagnostics to confirm multi-metric simulation alpha thresholds.
- [x] **anova.py**: Implement statsmodels OLS and anova_lm integration
- [x] **audit.py**: Add RSA/ECDSA digital signatures to each block to cryptographically bind executed events to specific authorized users.
- [x] **audit.py**: Implement a backup automated distributed consensus sync log (such as SQLite-backed replication) for tamper-proof persistence.
