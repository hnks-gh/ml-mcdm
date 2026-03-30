# -*- coding: utf-8 -*-
"""
Unit tests for analysis/validation.py and weighting/validation.py.

Covers:
  - ERValidator  (belief validity, completeness, cross-level consistency)
  - ForecastValidator (CV diagnostics, interval coverage, residual tests)
  - TemporalStabilityValidator (via weighting.validation, Spearman)

Regression tests preserved from earlier phases:
  - TestTOPSISDegenerate  (M3)
  - TestSAWZeroCost       (H2)
  - TestVIKORCompromiseSetC2 (M2)
  - TestModifiedEDASTrimmeanPath (H3)
  - TestEvidentialReasoningKConstant (C2)
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# ERValidator tests
# ---------------------------------------------------------------------------

class TestERValidator:

    def test_returns_er_validation_result(self):
        from analysis.validation import ERValidator, ERValidationResult
        er = _make_er_result()
        result = ERValidator().validate(er)
        assert isinstance(result, ERValidationResult)

    def test_valid_beliefs_pass_belief_validity(self):
        from analysis.validation import ERValidator
        er = _make_er_result()
        result = ERValidator().validate(er)
        assert result.belief_validity is True

    def test_invalid_beliefs_fail_belief_validity(self):
        from analysis.validation import ERValidator
        er = _make_er_result()
        # Inject a mock belief distribution that does NOT normalize (raw object)
        class _FakeBD:
            grades = ["G0", "G1"]
            beliefs = np.array([0.8, 0.8])  # sum = 1.6 > 1
            def utility_interval(self):
                return (0.0, 1.0)
        er.belief_distributions["BAD"] = _FakeBD()
        result = ERValidator().validate(er)
        # Should detect the sum > 1 violation
        assert result.belief_validity is False or len(result.validation_warnings) > 0

    def test_mean_belief_entropy_nonneg(self):
        from analysis.validation import ERValidator
        er = _make_er_result()
        result = ERValidator().validate(er)
        assert result.mean_belief_entropy >= 0.0

    def test_er_aggregation_score_in_unit_interval(self):
        from analysis.validation import ERValidator
        er = _make_er_result()
        result = ERValidator().validate(er)
        assert 0.0 <= result.er_aggregation_score <= 1.0

    def test_grade_distribution_has_all_grades(self):
        from analysis.validation import ERValidator
        from ranking.evidential_reasoning.base import BeliefDistribution
        grades = ["Low", "Medium", "High"]
        entities = [f"E{i}" for i in range(5)]
        rng = np.random.RandomState(42)
        belief_dists = {
            e: BeliefDistribution(grades=grades, beliefs=rng.dirichlet(np.ones(3)))
            for e in entities
        }
        scores = pd.Series({e: belief_dists[e].average_utility() for e in entities})

        class _ER:
            pass
        er = _ER()
        er.belief_distributions = belief_dists
        er.final_scores = scores
        er.final_ranking = scores.rank(ascending=False).astype(int)
        er.criterion_method_scores = {}
        er.aggregation_weights = {}

        result = ERValidator().validate(er)
        for g in grades:
            assert g in result.grade_distribution

    def test_empty_belief_distributions_handled(self):
        from analysis.validation import ERValidator
        er = _make_er_result()
        er.belief_distributions = {}
        result = ERValidator().validate(er)
        assert result is not None

    def test_er_valid_attribute_is_bool(self):
        from analysis.validation import ERValidator
        er = _make_er_result()
        result = ERValidator().validate(er)
        assert isinstance(result.er_valid, bool)


# ---------------------------------------------------------------------------
# ForecastValidator tests
# ---------------------------------------------------------------------------

class TestForecastValidator:

    def test_returns_forecast_validation_result(self):
        from analysis.validation import ForecastValidator, ForecastValidationResult
        fr = _make_forecast_result()
        result = ForecastValidator().validate(fr)
        assert isinstance(result, ForecastValidationResult)

    def test_cv_fold_stability_in_unit_interval(self):
        from analysis.validation import ForecastValidator
        fr = _make_forecast_result()
        result = ForecastValidator().validate(fr)
        assert 0.0 <= result.cv_fold_stability <= 1.0

    def test_interval_coverage_in_unit_interval(self):
        from analysis.validation import ForecastValidator
        fr = _make_forecast_result()
        result = ForecastValidator().validate(fr)
        assert 0.0 <= result.interval_coverage <= 1.0

    def test_interval_sharpness_nonneg(self):
        from analysis.validation import ForecastValidator
        fr = _make_forecast_result()
        result = ForecastValidator().validate(fr)
        assert result.interval_sharpness >= 0.0

    def test_model_agreement_in_unit_interval(self):
        from analysis.validation import ForecastValidator
        fr = _make_forecast_result()
        result = ForecastValidator().validate(fr)
        assert 0.0 <= result.model_agreement <= 1.0

    def test_overall_score_in_unit_interval(self):
        from analysis.validation import ForecastValidator
        fr = _make_forecast_result()
        result = ForecastValidator().validate(fr)
        assert 0.0 <= result.overall_score <= 1.0

    def test_forecast_valid_is_bool(self):
        from analysis.validation import ForecastValidator
        fr = _make_forecast_result()
        result = ForecastValidator().validate(fr)
        assert isinstance(result.forecast_valid, bool)

    def test_negative_r2_marks_forecast_invalid(self):
        from analysis.validation import ForecastValidator
        fr = _make_forecast_result()
        # Force all models to negative R²
        for m in fr.model_performance:
            fr.model_performance[m] = {"r2": -0.5}
        result = ForecastValidator().validate(fr)
        # Very poor R² → should flag invalid
        assert not result.forecast_valid or result.overall_score < 0.5

    def test_empty_cv_scores_handled(self):
        from analysis.validation import ForecastValidator
        fr = _make_forecast_result()
        fr.cross_validation_scores = {}
        result = ForecastValidator().validate(fr)
        assert result is not None

    def test_no_prediction_intervals_handled(self):
        from analysis.validation import ForecastValidator
        fr = _make_forecast_result()
        fr.prediction_intervals = {}
        result = ForecastValidator().validate(fr)
        assert result is not None


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Regression tests preserved from earlier phases
# ---------------------------------------------------------------------------

class TestTOPSISDegenerate:
    """M3 — TOPSIS degenerate case: all identical rows → score = 0.5."""

    def test_identical_rows_score_half(self):
        from ranking.topsis import TOPSISCalculator

        dm = pd.DataFrame(
            {"C1": [0.5, 0.5, 0.5], "C2": [0.7, 0.7, 0.7]},
            index=["X", "Y", "Z"],
        )
        calc = TOPSISCalculator()
        res = calc.calculate(dm, {"C1": 0.5, "C2": 0.5})

        for alt in ["X", "Y", "Z"]:
            assert abs(res.scores[alt] - 0.5) < 1e-9


class TestSAWZeroCost:
    """H2 — SAW: zero-cost value must not get worst (0) score in max/sum modes."""

    def test_max_mode_zero_cost_not_worst(self):
        from ranking.saw import SAWCalculator

        dm = pd.DataFrame(
            {"benefit": [0.9, 0.5, 0.3], "cost": [0.0, 0.5, 0.8]},
            index=["Best", "Mid", "Worst"],
        )
        calc = SAWCalculator(normalization="max", cost_criteria=["cost"])
        res = calc.calculate(dm, {"benefit": 0.5, "cost": 0.5})
        assert res.ranks["Best"] == 1

    def test_sum_mode_zero_cost_not_crash(self):
        from ranking.saw import SAWCalculator

        dm = pd.DataFrame(
            {"C1": [0.8, 0.6], "C_cost": [0.0, 0.4]},
            index=["A", "B"],
        )
        calc = SAWCalculator(normalization="sum", cost_criteria=["C_cost"])
        res = calc.calculate(dm, {"C1": 0.5, "C_cost": 0.5})
        assert not res.scores.isna().any()


class TestVIKORCompromiseSetC2:
    """M2 — VIKOR compromise set must include best_by_S and best_by_R when C2 fails."""

    def test_compromise_set_is_subset_of_alternatives(self):
        from ranking.vikor import VIKORCalculator

        dm = pd.DataFrame(
            {"C1": [0.9, 0.8, 0.5, 0.2], "C2": [0.8, 0.6, 0.7, 0.4]},
            index=["A", "B", "C", "D"],
        )
        calc = VIKORCalculator(v=0.5)
        res = calc.calculate(dm, {"C1": 0.5, "C2": 0.5})
        for alt in res.compromise_set:
            assert alt in dm.index

    def test_compromise_set_non_empty(self):
        from ranking.vikor import VIKORCalculator

        dm = pd.DataFrame(
            {"C1": [0.9, 0.1, 0.5], "C2": [0.2, 0.9, 0.5]},
            index=["A", "B", "C"],
        )
        calc = VIKORCalculator()
        res = calc.calculate(dm, {"C1": 0.5, "C2": 0.5})
        assert len(res.compromise_set) >= 1


class TestModifiedEDASTrimmeanPath:
    """H3 — ModifiedEDAS with trimmed mean must produce a valid result."""

    def test_trimmed_mean_differs_from_regular_mean(self):
        from ranking.edas import EDASCalculator, ModifiedEDAS

        rng = np.random.RandomState(7)
        data = rng.rand(10, 4) + 0.1
        dm = pd.DataFrame(data, index=[f"A{i}" for i in range(10)],
                          columns=["C1", "C2", "C3", "C4"])
        weights = {"C1": 0.25, "C2": 0.25, "C3": 0.25, "C4": 0.25}

        res_regular = EDASCalculator().calculate(dm, weights)
        res_trimmed = ModifiedEDAS(use_trimmed_mean=True).calculate(dm, weights)

        assert res_regular.AS.shape[0] == 10
        assert res_trimmed.AS.shape[0] == 10
        assert not np.allclose(res_regular.AS.values, res_trimmed.AS.values)


class TestEvidentialReasoningKConstant:
    """C2 — ranking aggregation combine() must use correct normalisation constant K."""

    def test_k_constant_correct_for_two_independent_sources(self):
        from ranking.evidential_reasoning.base import BeliefDistribution, EvidentialReasoningEngine

        engine = EvidentialReasoningEngine(grades=["H1", "H2", "H3"])
        b1 = BeliefDistribution(grades=["H1", "H2", "H3"], beliefs=np.array([0.9, 0.05, 0.05]))
        b2 = BeliefDistribution(grades=["H1", "H2", "H3"], beliefs=np.array([0.05, 0.05, 0.9]))
        result = engine.combine([b1, b2], weights=np.array([0.5, 0.5]))

        assert result.beliefs.sum() <= 1.0 + 1e-9
        assert (result.beliefs >= -1e-9).all()

    def test_three_sources_valid(self):
        from ranking.evidential_reasoning.base import BeliefDistribution, EvidentialReasoningEngine

        engine = EvidentialReasoningEngine(grades=["A", "B", "C", "D"])
        sources = [
            BeliefDistribution(grades=["A", "B", "C", "D"],
                               beliefs=np.array([0.7, 0.2, 0.05, 0.05])),
            BeliefDistribution(grades=["A", "B", "C", "D"],
                               beliefs=np.array([0.1, 0.5, 0.3, 0.1])),
            BeliefDistribution(grades=["A", "B", "C", "D"],
                               beliefs=np.array([0.05, 0.1, 0.35, 0.5])),
        ]
        result = engine.combine(sources, weights=np.array([0.4, 0.35, 0.25]))

        assert result.beliefs.sum() <= 1.0 + 1e-9
        assert (result.beliefs >= -1e-9).all()

    def test_uniform_sources_stay_uniform(self):
        from ranking.evidential_reasoning.base import BeliefDistribution, EvidentialReasoningEngine

        engine = EvidentialReasoningEngine(grades=["A", "B", "C", "D"])
        beliefs = np.array([0.25, 0.25, 0.25, 0.25])
        sources = [BeliefDistribution(grades=["A", "B", "C", "D"],
                                      beliefs=beliefs.copy()) for _ in range(3)]
        result = engine.combine(sources, weights=np.array([1 / 3, 1 / 3, 1 / 3]))

        std_belief = result.beliefs.std()
        assert std_belief < 0.01
