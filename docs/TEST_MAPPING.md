# Xpyrment Test-to-Source Mapping

This document provides a comprehensive mapping between the test suite and the source code it validates.

| Test File | Test Entity (Class/Function) | Source File | Notes |
|-----------|-----------------------------|-------------|-------|
| `test_core.py` | `test_experiment_state_transitions` | `src/xpyrment/core/experiment.py`, `src/xpyrment/core/state.py` | Validates Experiment state machine and PhaseOrderError. |
| `test_core.py` | `test_metric_addition_restrictions` | `src/xpyrment/core/experiment.py` | Verifies phase-gating for metric additions. |
| `test_core.py` | `test_experiment_registry_hashing` | `src/xpyrment/core/registry.py` | Tests SHA-256 spec verification. |
| `test_core.py` | `test_hypothesis_spec` | `src/xpyrment/plan/hypothesis.py` | Validates hypothesis properties. |
| `test_core.py` | `test_preregistration_card` | `src/xpyrment/plan/preregistration.py` | Tests card verification logic. |
| `test_metrics.py` | `test_mean_metric_calculation` | `src/xpyrment/metrics/taxonomy.py` | Tests MeanMetric calculation. |
| `test_metrics.py` | `test_proportion_metric_calculation` | `src/xpyrment/metrics/taxonomy.py` | Tests ProportionMetric calculation. |
| `test_metrics.py` | `test_ratio_metric_delta_method` | `src/xpyrment/metrics/taxonomy.py` | Tests RatioMetric with Delta Method. |
| `test_metrics.py` | `test_mean_metric_cuped` | `src/xpyrment/metrics/taxonomy.py` | Verifies CUPED variance reduction in MeanMetric. |
| `test_metrics.py` | `test_reproduce_issue_cuped_singular_covariate` | `src/xpyrment/metrics/taxonomy.py` | Tests CUPED fallback on singular covariates. |
| `test_metrics.py` | `test_guardrail_metric` | `src/xpyrment/metrics/guardrails.py` | Tests breach detection logic. |
| `test_metrics.py` | `test_transformations` | `src/xpyrment/metrics/transformations.py` | Tests log and delta transformations. |
| `test_analysis.py` | `test_end_to_end_setup_and_analysis` | `src/xpyrment/analyze/orchestrator.py` | Full experiment workflow validation. |
| `test_analysis.py` | `test_multiple_comparison_corrections` | `src/xpyrment/analyze/orchestrator.py` | Validates Bonferroni and other corrections. |
| `test_analysis.py` | `test_frequentist_welch_and_mann_whitney` | `src/xpyrment/analyze/inference/frequentist.py` | Mathematical validation of T-test and U-test. |
| `test_analysis.py` | `test_apply_cuped_variance_deflation` | `src/xpyrment/analyze/variance_reduction.py` | Tests low-level CUPED implementation. |
| `test_analysis.py` | `test_bayesian_inference_conjugate_beta_binomial` | `src/xpyrment/analyze/inference/bayesian.py` | Beta-Binomial conjugate posterior tests. |
| `test_analysis.py` | `test_bayesian_inference_conjugate_normal_normal` | `src/xpyrment/analyze/inference/bayesian.py` | Normal-Normal conjugate posterior tests. |
| `test_analysis.py` | `test_streaming_ols` | `src/xpyrment/analyze/streaming.py` | Woodbury updates and prediction accuracy. |
| `test_analysis.py` | `test_alias_resolver` | `src/xpyrment/analyze/confounding.py` | Alias matrix and de-biasing tests. |
| `test_analysis.py` | `test_copula_multi_metric_inference` | `src/xpyrment/analyze/copula.py` | Gaussian Copula joint shift tests. |
| `test_analysis.py` | `test_markov_journey_transition_homogeneity` | `src/xpyrment/analyze/markov.py` | Transition matrix and homogeneity tests. |
| `test_analysis.py` | `test_route_inference_engine` | `src/xpyrment/analyze/inference/router.py` | Routing logic for metric/test types. |
| `test_balance.py` | `test_covariate_balance_checker` | `src/xpyrment/quasi/balance.py` | Tests CovariateBalanceChecker and Love Plot generation. |
| `test_bandit.py` | `test_epsilon_greedy_bandit` | `src/xpyrment/bandit/epsilon_greedy.py` | Exploration-exploitation balance and decay. |
| `test_bandit.py` | `test_ucb1_bandit` | `src/xpyrment/bandit/ucb.py` | UCB1 initialization and logarithmic bounds. |
| `test_bandit.py` | `test_thompson_sampling_binary` | `src/xpyrment/bandit/thompson.py` | Beta-Binomial conjugate posterior updating. |
| `test_bandit.py` | `test_thompson_sampling_continuous` | `src/xpyrment/bandit/thompson.py` | Normal-Normal conjugate posterior updating. |
| `test_bandit.py" | `test_gp_regressor` | `src/xpyrment/bandit/tuning.py` | GaussianProcessRegressor mean/variance predictions. |
| `test_bandit.py` | `test_bandit_hyperparameter_tuner` | `src/xpyrment/bandit/tuning.py`, `src/xpyrment/bandit/__init__.py` | Bayesian Optimization for bandit tuning. |
| `test_bandit.py` | `test_off_policy_evaluation` | `src/xpyrment/bandit/ope.py` | IPS, SN-IPS, and Double Robust (DR) estimators. |
| `test_bandit.py` | `test_multi_objective_tuning` | `src/xpyrment/bandit/multi_objective.py` | Hypervolume, Pareto frontier, and EHVI. |
| `test_bandit.py` | `test_non_stationary_bandits` | `src/xpyrment/bandit/non_stationary.py` | Discounted and Sliding-Window TS. |
| `test_bootstrap_harden.py` | `test_bootstrap_basic_percentile_and_bca` | `src/xpyrment/analyze/inference/bootstrap.py` | Percentile and BCa bootstrap validation. |
| `test_bootstrap_harden.py` | `test_bootstrap_reproducibility` | `src/xpyrment/analyze/inference/bootstrap.py` | Random seed reproducibility check. |
| `test_bootstrap_harden.py` | `test_bootstrap_zero_variance_fallback` | `src/xpyrment/analyze/inference/bootstrap.py` | Constant array handling. |
| `test_bootstrap_harden.py` | `test_bootstrap_degenerate_resamples` | `src/xpyrment/analyze/inference/bootstrap.py` | Degenerate replicate fallback. |
| `test_bootstrap_harden.py` | `test_bootstrap_large_chunked_execution` | `src/xpyrment/analyze/inference/bootstrap.py` | Large-scale chunked vectorized processing. |
| `test_bootstrap_harden.py` | `test_bootstrap_empty_validation` | `src/xpyrment/analyze/inference/bootstrap.py` | Empty array error handling. |
| `test_bootstrap_harden.py` | `test_bootstrap_invalid_method` | `src/xpyrment/analyze/inference/bootstrap.py` | Invalid method error handling. |
| `test_clean_edge_cases.py` | `test_clean_array_basic` | `src/xpyrment/validate/clean.py` | Basic array cleaning and edge cases. |
| `test_clean_edge_cases.py` | `test_validate_estimation_inputs_mismatches` | `src/xpyrment/validate/clean.py` | Length mismatch and sample count validation. |
| `test_clean_edge_cases.py` | `test_verify_collinearity_singular` | `src/xpyrment/validate/clean.py` | Rank-deficiency and singularity detection. |
| `test_clean_edge_cases.py` | `test_did_estimator_edge_cases` | `src/xpyrment/quasi/diff_in_diff.py` | DiD robustness against NaNs and collinearity. |
| `test_clean_edge_cases.py` | `test_iv_estimator_edge_cases` | `src/xpyrment/quasi/instrumental_variables.py` | IV robustness against zero-variance inputs. |
| `test_cli.py` | `test_calculate_required_power` | `src/xpyrment/cli.py` | Power calculation math and bounds. |
| `test_cli.py` | `test_cli_power_command` | `src/xpyrment/cli.py` | CLI 'power' command execution. |
| `test_cli.py` | `test_cli_balance_command` | `src/xpyrment/cli.py` | CLI 'balance' command execution. |
| `test_cli.py` | `test_cli_regress_command` | `src/xpyrment/cli.py` | CLI 'regress' command execution. |
| `test_cli.py` | `test_cli_nonexistent_csv_handling` | `src/xpyrment/cli.py` | Missing CSV error handling. |
| `test_corrections.py` | `test_multiple_testing_corrections` | `src/xpyrment/analyze/corrections.py` | BH, BY, and Hochberg corrections. |
| `test_design.py` | `test_design_experiment_proportion` | `src/xpyrment/plan/power.py` | Sample size for proportions. |
| `test_design.py` | `test_design_experiment_mean` | `src/xpyrment/plan/power.py` | Sample size for continuous means. |
| `test_design.py` | `test_design_experiment_cuped_savings` | `src/xpyrment/plan/power.py` | CUPED sample size reduction logic. |
| `test_design.py` | `test_generate_power_curve_data` | `src/xpyrment/plan/power.py` | Power curve coordinate generation. |
| `test_design.py` | `test_traffic_splitter_custom_ramp` | `src/xpyrment/design/splits.py` | Traffic allocation and ramp schedules. |
| `test_design.py` | `test_stratified_randomization_balance` | `src/xpyrment/design/stratification.py` | Strata-based randomization balancing. |
| `test_design.py` | `test_fractional_factorial_generation` | `src/xpyrment/design/doe/fractional_factorial.py` | 2^(k-p) design matrix generation. |
| `test_design.py` | `test_ccd_generation` | `src/xpyrment/design/doe/ccd.py` | Central Composite Design (Face-Centered/Rotatable). |
| `test_design.py` | `test_bbd_generation` | `src/xpyrment/design/doe/box_behnken.py` | Box-Behnken Design generation. |
| `test_design.py` | `test_plackett_burman_generation` | `src/xpyrment/design/doe/plackett_burman.py` | Orthogonal screening design generation. |
| `test_design.py` | `test_taguchi_generation` | `src/xpyrment/design/doe/taguchi.py` | L9 Orthogonal Array and pair balance. |
| `test_design.py` | `test_dsd_generation` | `src/xpyrment/design/doe/dsd.py` | Definitive Screening Design (DSD) properties. |
| `test_design.py` | `test_d_optimal_generation` | `src/xpyrment/design/doe/d_optimal.py` | Coordinate Exchange optimization for D-optimality. |
| `test_design.py` | `test_lhs_generation` | `src/xpyrment/design/doe/lhs.py` | Latin Hypercube Sampling projection properties. |
| `test_design.py` | `test_mixture_generation` | `src/xpyrment/design/doe/mixture.py` | Mixture design sum-to-one constraints. |
| `test_design.py` | `test_switchback_generation` | `src/xpyrment/design/doe/switchback.py` | Temporal crossover balancing and washouts. |
| `test_design.py` | `test_evop_generation` | `src/xpyrment/design/doe/evop.py` | Evolutionary Operation (EVOP) perturbations. |
| `test_design.py` | `test_carryover_decomposition` | `src/xpyrment/design/doe/carryover.py` | Direct effect and decay rate estimation. |
| `test_design.py` | `test_hash_assign` | `src/xpyrment/design/randomization.py` | Deterministic unit-to-variant hashing. |
| `test_design.py` | `test_full_factorial_design` | `src/xpyrment/design/doe/full_factorial.py` | 2^k Full Factorial Design generation. |
| `test_dtr.py` | `test_q_factor_model` | `src/xpyrment/personalize/dtr.py` | Q-Learning model parameter recovery. |
| `test_dtr.py` | `test_dtr_backward_induction` | `src/xpyrment/personalize/dtr.py` | Multistage backward induction recommendations. |
| `test_extreme.py` | `test_extreme_value_tail_estimator` | `src/xpyrment/analyze/extreme.py` | GPD scale/shape and Expected Shortfall. |
| `test_governance.py` | `test_meta_analysis` | `src/xpyrment/governance/meta_analysis.py` | Fixed and Random Effects pooling estimators. |
| `test_governance.py` | `test_p_curve_analysis` | `src/xpyrment/governance/p_curve.py` | Skewness checks for evidential value vs p-hacking. |
| `test_infinite_mixture.py` | `test_infinite_dirichlet_clustering` | `src/xpyrment/personalize/infinite_mixture.py` | Dirichlet Process Mixture clustering. |
| `test_instrumental_variables.py` | `test_instrumental_variables_2sls` | `src/xpyrment/quasi/instrumental_variables.py` | 2SLS Complier Average Causal Effect (CACE). |
| `test_interactions.py` | `test_factorial_anova_interaction` | `src/xpyrment/interactions/anova.py` | ANOVA table with interaction terms (A:B). |
| `test_interactions.py` | `test_lrt_interaction_check` | `src/xpyrment/interactions/regression.py` | Likelihood Ratio Test for HTE detection. |
| `test_interactions.py` | `test_friedman_h_statistic` | `src/xpyrment/interactions/hstat.py` | Friedman's H-statistic for non-linear interactions. |
| `test_interactions.py` | `test_interaction_detector_integration` | `src/xpyrment/interactions/detector.py` | End-to-end HTE scanning via Experiment orchestrator. |
| `test_interpret.py` | `test_launch_recommendation_logic` | `src/xpyrment/interpret/decision.py` | Ship/No-Ship/Inconclusive decision boundaries. |
| `test_interpret.py` | `test_cohens_d_calculation` | `src/xpyrment/interpret/effect_size.py` | Standardized effect size (Cohen's d). |
| `test_interpret.py` | `test_subgroup_hte_scan` | `src/xpyrment/interpret/hte.py` | HTE detection and lift reporting by subgroup. |
| `test_interpret.py` | `test_practical_significance_check` | `src/xpyrment/interpret/significance.py` | Practical vs. Statistical significance (MVE). |
| `test_its.py" | `test_interrupted_time_series` | `src/xpyrment/analyze/its.py` | ITS level and slope shift with HAC SEs. |
| `test_meta_regression.py` | `test_meta_regression_knapp_hartung` | `src/xpyrment/analyze/meta_regression.py` | Knapp-Hartung SEs and tau-squared estimation. |
| `test_network.py` | `test_cluster_randomizer` | `src/xpyrment/network/cluster.py` | Graph partitioning (LPA) and cluster randomization. |
| `test_network.py` | `test_neighborhood_exposure_model` | `src/xpyrment/network/spillover.py` | Neighborhood exposure classification (DTE/ISE). |
| `test_network.py` | `test_identity_resolution` | `src/xpyrment/network/identity.py` | Identity stitching (DSU) across identifiers. |
| `test_network.py` | `test_federated_pooling` | `src/xpyrment/network/federated.py` | Paillier cryptosystem and federated pooling. |
| `test_network.py` | `test_entropy_balanced_partition` | `src/xpyrment/network/partition.py` | Entropy-balanced graph partitioning. |
| `test_optimal_transport.py` | `test_quantile_treatment_effects` | `src/xpyrment/quasi/optimal_transport.py` | Distributional quantile treatment effects (QTE). |
| `test_optimal_transport.py` | `test_wasserstein_distance` | `src/xpyrment/quasi/optimal_transport.py` | 1D Wasserstein distance between distributions. |
| `test_orchestrator_harden.py` | `test_fluent_and_covariates` | `src/xpyrment/analyze/orchestrator.py` | Fluent API and automated CUPED/Love Plot routing. |
| `test_orchestrator_harden.py` | `test_covariate_imbalance_warning` | `src/xpyrment/analyze/orchestrator.py` | Automatic warnings for imbalanced covariates. |
| `test_orchestrator_harden.py` | `test_topological_metric_dag_evaluation` | `src/xpyrment/analyze/registry.py` | MetricRegistry DAG topological sorting. |
| `test_outliers.py` | `test_winsorization_engine` | `src/xpyrment/analyze/outliers.py` | Outlier capping (Winsorization) logic. |
| `test_personalize.py` | `test_ridge_regressor` | `src/xpyrment/personalize/meta_learners.py` | Ridge regression coefficient recovery. |
| `test_personalize.py` | `test_meta_learners_uplift` | `src/xpyrment/personalize/meta_learners.py` | S, T, and X-Learner CATE estimation. |
| `test_personalize.py` | `test_causal_tree_and_forest` | `src/xpyrment/personalize/causal_forest.py` | Honest Causal Tree splits and Forest ensembles. |
| `test_personalize.py` | `test_elastic_net_regressor` | `src/xpyrment/personalize/meta_learners.py` | Coordinate descent for Elastic Net sparsity. |
| `test_personalize.py` | `test_double_machine_learning` | `src/xpyrment/personalize/double_ml.py` | Robinson's DML with K-Fold cross-fitting. |
| `test_placebo.py` | `test_parallel_trends_placebo_test` | `src/xpyrment/quasi/diff_in_diff.py` | DiD parallel trends validation via placebo start. |
| `test_power.py` | `test_analytical_power_calculator` | `src/xpyrment/design/power.py` | Analytical power, MDE, and sample size recovery. |
| `test_power.py` | `test_clustered_design_power_adjustment` | `src/xpyrment/design/power.py` | Variance Inflation Factor (VIF) for cluster designs. |
| `test_privacy.py` | `test_dp_sensitivity` | `src/xpyrment/network/privacy.py` | Mean sensitivity calculation for DP. |
| `test_privacy.py` | `test_laplace_and_gaussian_mechanisms` | `src/xpyrment/network/privacy.py` | Laplace and Gaussian noise addition for DP. |
| `test_profiler.py` | `test_profiler` | `src/xpyrment/profiler.py` | Validates profiling output (time/memory). |
| `test_quasi.py` | `test_difference_in_differences` | `src/xpyrment/quasi/diff_in_diff.py` | DiD effect size and parallel trends validation. |
| `test_quasi.py` | `test_synthetic_control_optimization` | `src/xpyrment/quasi/synthetic_control.py` | Synthetic Control convex weight optimization. |
| `test_quasi.py` | `test_mahalanobis_distance` | `src/xpyrment/quasi/matching.py` | Mahalanobis distance calculation for matching. |
| `test_quasi.py` | `test_propensity_score_matching` | `src/xpyrment/quasi/matching.py` | Propensity score matching (caliper logit). |
| `test_quasi.py` | `test_coarsened_exact_matching` | `src/xpyrment/quasi/matching.py` | CEM binning and balancing weights. |
| `test_quasi.py` | `test_synthetic_difference_in_differences` | `src/xpyrment/quasi/sdid.py` | SDID unit and time weight estimation. |
| `test_quasi.py` | `test_snmm_causal_inference` | `src/xpyrment/quasi/snmm.py` | Structural Nested Mean Model (G-estimation). |
| `test_quasi.py` | `test_panel_matrix_completion` | `src/xpyrment/quasi/matrix_completion.py` | Nuclear norm minimization for matrix completion. |
| `test_quasi.py` | `test_causal_sensitivity_analysis` | `src/xpyrment/quasi/sensitivity.py` | Rosenbaum bounds and partial R^2 sensitivity. |
| `test_ratio.py` | `test_ratio_metric_delta_method` | `src/xpyrment/analyze/ratio.py` | Delta Method SEs for ratio metrics. |
| `test_registry.py` | `test_metric_registry_evaluation` | `src/xpyrment/analyze/registry.py` | Evaluates raw and derived metrics in a registry. |
| `test_registry.py` | `test_metric_registry_cycles` | `src/xpyrment/analyze/registry.py` | Cycle detection in metric dependency graphs. |
| `test_report.py` | `test_experiment_card_serialization` | `src/xpyrment/report/card.py" | JSON/Dict serialization for experiment cards. |
| `test_report.py` | `test_audit_trail_cryptographic_chain` | `src/xpyrment/report/audit.py` | SHA-256 chain and integrity verification. |
| `test_report.py` | `test_visualizations_plot_generation` | `src/xpyrment/report/export.py` | Forest plots and Power curve visualizations. |
| `test_report.py` | `test_experiment_report_generator` | `src/xpyrment/report/generator.py` | Automated Markdown and HTML report generation. |
| `test_rolling_synthetic_control.py` | `test_rolling_synthetic_control` | `src/xpyrment/quasi/rolling_synthetic_control.py` | Simplex-constrained rolling synthetic controls. |
| `test_run.py` | `test_assignment_logger_deduplication` | `src/xpyrment/run/assignment.py` | First-touch attribution deduplication. |
| `test_run.py` | `test_ingest_dataframe_auditing` | `src/xpyrment/run/ingestion.py` | Data cleaning, imputation, and datetime gates. |
| `test_run.py` | `test_live_monitor_cumulative_traffic` | `src/xpyrment/run/monitor.py` | Temporal binning and cumulative traffic monitoring. |
| `test_run.py` | `test_stopping_rules_msprt` | `src/xpyrment/run/stopping.py` | mSPRT martingale likelihood ratio stopping rules. |
| `test_run.py` | `test_estimate_duration_days` | `src/xpyrment/plan/duration.py` | Sample size to duration estimation. |
| `test_run.py` | `test_load_from_sql_mock` | `src/xpyrment/run/ingestion.py` | SQL ingestion integration check. |
| `test_sequential.py` | `test_sequential_spending_obf` | `src/xpyrment/analyze/sequential.py` | O'Brien-Fleming alpha spending function. |
| `test_sequential.py` | `test_sequential_boundaries_pocock` | `src/xpyrment/analyze/sequential.py` | Pocock aggressive early alpha spending. |
| `test_serialization.py` | `test_basic_make_serializable` | `src/xpyrment/core/serialization.py` | JSON-safe cleaning of NumPy/Pandas types. |
| `test_serialization.py` | `test_analysis_result_serialization` | `src/xpyrment/analyze/orchestrator.py` | AnalysisResults state persistence logic. |
| `test_serialization.py` | `test_estimator_serialization` | `src/xpyrment/quasi/` | Serialization for DiD, 2SLS, and Placebo estimators. |
| `test_simulation.py` | `test_generate_ab_data` | `src/xpyrment/simulation.py` | Multi-metric synthetic A/B data generation. |
| `test_simulation.py` | `test_experiment_simulator_panel` | `src/xpyrment/simulation.py` | Complex panel data generation with non-compliance. |
| `test_simulation.py` | `test_experiment_simulator_monte_carlo` | `src/xpyrment/simulation.py` | Monte Carlo simulations for bias/MSE estimation. |
| `test_srm.py` | `test_srm_retrospective_perfect` | `src/xpyrment/analyze/srm.py` | Retrospective Chi-Squared SRM detection. |
| `test_srm.py` | `test_srm_sequential_perfect` | `src/xpyrment/analyze/srm.py` | Sequential SPRT-based SRM guardrails. |
| `test_subgroup.py` | `test_subgroup_heterogeneity_discoverer` | `src/xpyrment/personalize/subgroup.py` | Decision-tree based subgroup segment discovery. |
| `test_telemetry.py` | `test_json_formatter_and_configuration` | `src/xpyrment/core/telemetry.py` | Structured JSON logging configuration. |
| `test_telemetry.py` | `test_execution_profiler_context_manager` | `src/xpyrment/core/telemetry.py` | In-process stage timing and memory profiling. |
| `test_validate.py" | `test_check_srm_no_mismatch` | `src/xpyrment/validate/srm.py` | Validation gate for SRM detections. |
| `test_validate.py` | `test_check_covariate_balance` | `src/xpyrment/validate/balance.py` | SMD-based covariate balance validation gates. |
| `test_validate.py` | `test_run_aa_test_validation` | `src/xpyrment/validate/aa_test.py` | A/A simulation and KS uniform p-value check. |
| `test_validate.py` | `test_check_novelty_effects` | `src/xpyrment/validate/novelty.py` | Interaction slope regression for novelty detection. |
