# -*- coding: utf-8 -*-
"""
Unit tests for analysis/sensitivity.py (ML + ranking sensitivity).
"""
import numpy as np
import pandas as pd
import pytest


def _make_forecast_result(n_entities=8, n_features=5, n_models=3, seed=0):
    rng = np.random.RandomState(seed)
    entities = [f"P{i}" for i in range(n_entities)]
    feats = [f"f{i}" for i in range(n_features)]
    models = [f"model_{i}" for i in range(n_models)]

    class _FR:
        pass

    fr = _FR()
    fr.predictions = pd.DataFrame(
        rng.rand(n_entities, 1), index=entities, columns=["c0"]
    )
    fr.uncertainty = pd.DataFrame(
        rng.rand(n_entities, 1) * 0.1, index=entities, columns=["c0"]
    )
    fr.prediction_intervals = {
        "main": pd.DataFrame(
            np.column_stack([
                fr.predictions.values[:, 0] - 0.1,
                fr.predictions.values[:, 0] + 0.1,
            ]),
            index=entities,
            columns=["lower", "upper"],
        )
    }
    fr.feature_importance = pd.DataFrame(
        rng.rand(n_features, n_models) / n_features,
        index=feats, columns=models,
    )
    fr.model_contributions = {m: 1.0 / n_models for m in models}
    fr.model_performance = {m: {"r2": float(rng.uniform(0.5, 0.9))} for m in models}
    fr.cross_validation_scores = {m: list(rng.uniform(0.4, 0.9, 5)) for m in models}
    fr.holdout_performance = None
    fr.training_info = {}
    return fr


def _make_er_result(n_entities=8, n_grades=4, seed=0):
    rng = np.random.RandomState(seed)
    from ranking.evidential_reasoning.base import BeliefDistribution

    entities = [f"P{i}" for i in range(n_entities)]
    grades = [f"G{i}" for i in range(n_grades)]
    belief_dists = {}
    for e in entities:
        raw = rng.dirichlet(np.ones(n_grades))
        belief_dists[e] = BeliefDistribution(grades=grades, beliefs=raw)

    final_scores = pd.Series({e: belief_dists[e].average_utility() for e in entities})
    final_ranking = final_scores.rank(ascending=False).astype(int)

    class _ER:
        pass

    er = _ER()
    er.belief_distributions = belief_dists
    er.final_scores = final_scores
    er.final_ranking = final_ranking
    er.criterion_method_scores = {}
    er.aggregation_weights = {f"method_{i}": 1.0 / 3 for i in range(3)}
    return er


class TestMLSensitivityAnalysis:

    def test_returns_ml_sensitivity_result(self):
        from analysis.sensitivity import MLSensitivityAnalysis, MLSensitivityResult
        fr = _make_forecast_result()
        result = MLSensitivityAnalysis(n_bootstrap=20, seed=0).analyze(fr)
        assert isinstance(result, MLSensitivityResult)

    def test_feature_importance_keys_match_features(self):
        from analysis.sensitivity import MLSensitivityAnalysis
        fr = _make_forecast_result(n_features=4)
        result = MLSensitivityAnalysis(n_bootstrap=20, seed=1).analyze(fr)
        assert set(result.feature_importance_cv.keys()) == set(fr.feature_importance.index.tolist())

    def test_feature_importance_cv_nonneg(self):
        from analysis.sensitivity import MLSensitivityAnalysis
        fr = _make_forecast_result()
        result = MLSensitivityAnalysis(n_bootstrap=20, seed=2).analyze(fr)
        for feat, cv in result.feature_importance_cv.items():
            assert cv >= 0.0, f"Feature {feat}: CV={cv}"

    def test_loo_model_impact_has_all_models(self):
        from analysis.sensitivity import MLSensitivityAnalysis
        fr = _make_forecast_result(n_models=3)
        result = MLSensitivityAnalysis(n_bootstrap=10, seed=3).analyze(fr)
        for model in fr.model_contributions:
            assert model in result.loo_model_impact

    def test_overall_robustness_in_unit_interval(self):
        from analysis.sensitivity import MLSensitivityAnalysis
        fr = _make_forecast_result()
        result = MLSensitivityAnalysis(n_bootstrap=20, seed=4).analyze(fr)
        assert 0.0 <= result.overall_robustness <= 1.0

    def test_temporal_stability_in_unit_interval(self):
        from analysis.sensitivity import MLSensitivityAnalysis
        fr = _make_forecast_result()
        result = MLSensitivityAnalysis(n_bootstrap=10, seed=5).analyze(fr)
        assert 0.0 <= result.temporal_prediction_stability <= 1.0

    def test_interval_coverage_sensitivity_in_unit_interval(self):
        from analysis.sensitivity import MLSensitivityAnalysis
        fr = _make_forecast_result()
        result = MLSensitivityAnalysis(n_bootstrap=10, seed=6).analyze(fr)
        assert 0.0 <= result.interval_coverage_sensitivity <= 1.0

    def test_summary_is_string(self):
        from analysis.sensitivity import MLSensitivityAnalysis
        fr = _make_forecast_result()
        result = MLSensitivityAnalysis(n_bootstrap=10, seed=7).analyze(fr)
        s = result.summary()
        assert isinstance(s, str)
        assert "ML FORECASTING" in s

    def test_empty_feature_importance_doesnt_crash(self):
        from analysis.sensitivity import MLSensitivityAnalysis
        fr = _make_forecast_result()
        fr.feature_importance = None
        result = MLSensitivityAnalysis(n_bootstrap=10, seed=8).analyze(fr)
        assert result is not None

    def test_no_cv_scores_graceful(self):
        from analysis.sensitivity import MLSensitivityAnalysis
        fr = _make_forecast_result()
        fr.cross_validation_scores = {}
        result = MLSensitivityAnalysis(n_bootstrap=10, seed=9).analyze(fr)
        assert result is not None


class TestERSensitivityAnalysis:

    def test_returns_er_sensitivity_result(self):
        from analysis.sensitivity import ERSensitivityAnalysis, ERSensitivityResult
        er = _make_er_result()
        result = ERSensitivityAnalysis(n_simulations=50, seed=0).analyze(er)
        assert isinstance(result, ERSensitivityResult)

    def test_overall_er_robustness_in_unit_interval(self):
        from analysis.sensitivity import ERSensitivityAnalysis
        er = _make_er_result()
        result = ERSensitivityAnalysis(n_simulations=50, seed=1).analyze(er)
        assert 0.0 <= result.overall_er_robustness <= 1.0

    def test_utility_sensitivity_has_all_entities(self):
        from analysis.sensitivity import ERSensitivityAnalysis
        er = _make_er_result(n_entities=6)
        result = ERSensitivityAnalysis(n_simulations=30, seed=2).analyze(er)
        for entity in er.belief_distributions.keys():
            assert str(entity) in result.utility_interval_width

    def test_utility_interval_width_nonneg(self):
        from analysis.sensitivity import ERSensitivityAnalysis
        er = _make_er_result()
        result = ERSensitivityAnalysis(n_simulations=30, seed=3).analyze(er)
        for e, w in result.utility_interval_width.items():
            assert w >= 0.0, f"Entity {e}: width={w}"

    def test_mean_belief_entropy_nonneg(self):
        from analysis.sensitivity import ERSensitivityAnalysis
        er = _make_er_result()
        result = ERSensitivityAnalysis(n_simulations=30, seed=4).analyze(er)
        assert result.mean_belief_entropy >= 0.0

    def test_high_uncertainty_entities_is_list(self):
        from analysis.sensitivity import ERSensitivityAnalysis
        er = _make_er_result()
        result = ERSensitivityAnalysis(n_simulations=30, seed=5).analyze(er)
        assert isinstance(result.high_uncertainty_entities, list)

    def test_summary_is_string(self):
        from analysis.sensitivity import ERSensitivityAnalysis
        er = _make_er_result()
        result = ERSensitivityAnalysis(n_simulations=30, seed=6).analyze(er)
        s = result.summary()
        assert isinstance(s, str)
        assert "EVIDENTIAL REASONING" in s

    def test_empty_belief_dists_graceful(self):
        from analysis.sensitivity import ERSensitivityAnalysis
        er = _make_er_result()
        er.belief_distributions = {}
        er.final_scores = pd.Series(dtype=float)
        result = ERSensitivityAnalysis(n_simulations=20, seed=7).analyze(er)
        assert result is not None

    def test_empty_aggregation_weights_graceful(self):
        from analysis.sensitivity import ERSensitivityAnalysis
        er = _make_er_result()
        er.aggregation_weights = None
        result = ERSensitivityAnalysis(n_simulations=20, seed=8).analyze(er)
        assert result is not None


class TestCombinedSensitivityResult:

    def test_overall_robustness_average_of_ml_and_er(self):
        from analysis.sensitivity import (
            CombinedSensitivityResult, MLSensitivityAnalysis, ERSensitivityAnalysis,
        )
        fr = _make_forecast_result()
        er = _make_er_result()
        ml_r = MLSensitivityAnalysis(n_bootstrap=20, seed=0).analyze(fr)
        er_r = ERSensitivityAnalysis(n_simulations=30, seed=0).analyze(er)
        combined = CombinedSensitivityResult(ml_sensitivity=ml_r, er_sensitivity=er_r)
        expected = (ml_r.overall_robustness + er_r.overall_er_robustness) / 2
        assert abs(combined.overall_robustness - expected) < 1e-9

    def test_overall_robustness_ml_only(self):
        from analysis.sensitivity import CombinedSensitivityResult, MLSensitivityAnalysis
        fr = _make_forecast_result()
        ml_r = MLSensitivityAnalysis(n_bootstrap=20, seed=0).analyze(fr)
        combined = CombinedSensitivityResult(ml_sensitivity=ml_r)
        assert abs(combined.overall_robustness - ml_r.overall_robustness) < 1e-9

    def test_overall_robustness_none_is_0_5(self):
        from analysis.sensitivity import CombinedSensitivityResult
        combined = CombinedSensitivityResult()
        assert combined.overall_robustness == 0.5

    def test_run_sensitivity_analysis_returns_combined(self):
        from analysis.sensitivity import run_sensitivity_analysis, CombinedSensitivityResult
        fr = _make_forecast_result()
        er = _make_er_result()
        result = run_sensitivity_analysis(
            forecast_result=fr, er_result=er, n_simulations=30, n_bootstrap=10, seed=0
        )
        assert isinstance(result, CombinedSensitivityResult)


class TestACIUpdateRule:

    def _perfect_model(self):
        class _ZeroModel:
            def predict(self, X):
                return np.zeros(len(X))
        return _ZeroModel()

    def test_alpha_decreases_on_first_miss(self):
        from forecasting.conformal import ConformalPredictor

        n = 80
        n_cal = max(5, int(n * 0.25))
        n_init = max(3, n_cal // 2)
        y = np.concatenate([
            np.zeros(n - n_cal),
            np.zeros(n_init),
            np.full(n_cal - n_init, 1000.0),
        ])
        X = np.ones((n, 1))
        initial_alpha = 0.05

        cp = ConformalPredictor(method="adaptive", alpha=initial_alpha, gamma=0.95)
        cp.calibrate(self._perfect_model(), X, y)

        history = cp._aci_history
        assert len(history) > 0, "ACI history must not be empty"

        missed = [h for h in history if not h["covered"]]
        assert len(missed) > 0, "No missed step found despite huge residuals"

        first_miss_alpha = missed[0]["alpha_t"]
        assert first_miss_alpha < initial_alpha, (
            f"ACI missed step: expected alpha_t < {initial_alpha}, got {first_miss_alpha:.4f}"
        )

    def test_alpha_increases_when_always_covered(self):
        from forecasting.conformal import ConformalPredictor

        class _PerfectModel:
            def __init__(self, X, y):
                self._lookup = {tuple(row): val for row, val in zip(X, y)}
            def predict(self, X_in):
                return np.array([self._lookup.get(tuple(r), 0.0) for r in X_in])

        n = 100
        rng = np.random.RandomState(0)
        X = rng.randn(n, 2)
        y = rng.randn(n)
        initial_alpha = 0.05

        cp = ConformalPredictor(method="adaptive", alpha=initial_alpha, gamma=0.95)
        cp.calibrate(_PerfectModel(X, y), X, y)

        history = cp._aci_history
        assert len(history) > 0, "ACI history should not be empty for n=100"

        covered_steps = [h for h in history[1:] if h["covered"]]
        if covered_steps:
            for step in covered_steps:
                assert step["alpha_t"] >= initial_alpha - 1e-9, (
                    f"Expected alpha_t >= {initial_alpha} after covered step, "
                    f"got {step['alpha_t']:.6f}"
                )
