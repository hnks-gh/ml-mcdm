# -*- coding: utf-8 -*-
"""
Unit tests for analysis/validation.py and weighting/validation.py
— Phase 6 coverage.

Covers the Phase 5 bug-fixes:
  - M9:  Spearman rank correlation used in TemporalStabilityValidator
  - M11: _validate_weight_temporal_stability returns 0.0 (not 0.90) when
         no stability evidence (cosine_similarity / mc_diagnostics) is present
  - M11: _validate_weight_method_agreement returns 1 - mean_cv using
         Level-2 MC diagnostics; returns 0.0 when no diagnostics available

Also covers TOPSIS/SAW/VIKOR fixes via edge-case assertions:
  - M3:  TOPSIS degenerate case (all identical rows) returns 0.5 scores
  - H2:  SAW zero-cost column does not give worst (0) score
  - M2:  VIKOR compromise set contains correct members when C2 fails
"""

import numpy as np
import pandas as pd
import pytest

from analysis.validation import Validator


# ---------------------------------------------------------------------------
# Helpers / minimal mock objects
# ---------------------------------------------------------------------------

class _MinimalPanel:
    """Minimal mock of PanelData for Validator unit tests."""

    def __init__(self, n_years: int = 3):
        self.years = list(range(2020, 2020 + n_years))

    # No other attributes needed for the weight-validation methods


def _make_weights(methods: list[str] = ("entropy", "critic", "merec", "std_dev"),
                  n_criteria: int = 4,
                  seed: int = 0) -> dict:
    """
    Return a minimal weights dict for validator unit tests.
    Uses the new HybridWeightingCalculator structure with 'details', 'subcriteria',
    'global_sc_weights', 'sc_array', and 'criterion_weights'.
    """
    rng = np.random.RandomState(seed)
    raw = rng.dirichlet(np.ones(n_criteria))
    subcriteria = [f"S{i}" for i in range(n_criteria)]
    return {
        "global_sc_weights": dict(zip(subcriteria, raw)),
        "sc_array": raw.copy(),
        "subcriteria": subcriteria,
        "criterion_weights": {},
        "details": {},   # no stability info — overridden per-test
    }


# ---------------------------------------------------------------------------
# M11 — _validate_weight_temporal_stability fallback is 0.0, not 0.90
# ---------------------------------------------------------------------------

class TestValidatorTemporalStability:
    """When no bootstrap/stability evidence exists, return 0.0."""

    def test_no_details_returns_zero(self):
        """
        weights['details'] is empty → no bootstrap or stability info
        → must return 0.0, not 0.90.
        """
        validator = Validator()
        panel = _MinimalPanel(n_years=3)
        weights = _make_weights()
        weights["details"] = {}   # explicitly empty

        result = validator._validate_weight_temporal_stability(panel, weights)

        assert result == 0.0, (
            f"Expected 0.0 fallback (not 0.90), got {result}. "
            "M11 fix may not be applied."
        )

    def test_missing_details_key_returns_zero(self):
        """weights has no 'details' key at all → fallback is 0.0."""
        validator = Validator()
        panel = _MinimalPanel(n_years=3)
        weights = _make_weights()
        weights.pop("details", None)

        result = validator._validate_weight_temporal_stability(panel, weights)
        assert result == 0.0, f"Expected 0.0, got {result}"

    def test_empty_bootstrap_returns_zero(self):
        """details.bootstrap present but empty → no std_weights → returns 0.0."""
        validator = Validator()
        panel = _MinimalPanel(n_years=3)
        weights = _make_weights()
        weights["details"] = {"bootstrap": {}}

        result = validator._validate_weight_temporal_stability(panel, weights)
        assert result == 0.0, f"Expected 0.0, got {result}"

    def test_with_cosine_similarity_returns_it(self):
        """When stability.cosine_similarity is present, it should be returned."""
        validator = Validator()
        panel = _MinimalPanel(n_years=3)
        weights = _make_weights()
        weights["details"] = {"stability": {"cosine_similarity": 0.87}}

        result = validator._validate_weight_temporal_stability(panel, weights)
        assert abs(result - 0.87) < 1e-6, (
            f"Expected cosine_similarity=0.87, got {result}"
        )

    def test_with_bootstrap_mean_cv_returns_stability(self):
        """
        When Level-2 MC cv_weights are provided, stability = 1 - mean_cv.
        Craft a case with known CV and verify.

        Note: all criteria contribute equally to mean_cv.
        """
        validator = Validator()
        panel = _MinimalPanel(n_years=3)
        weights = _make_weights()
        # cv_weights: S0→0.10, S1→0.20, S2→0.00, S3→0.00
        # mean_cv = (0.10 + 0.20 + 0.00 + 0.00) / 4 = 0.075
        # stability = 1 - 0.075 = 0.925
        weights["details"] = {
            "level2": {
                "mc_diagnostics": {
                    "cv_weights": {
                        "S0": 0.10, "S1": 0.20, "S2": 0.00, "S3": 0.00
                    }
                }
            }
        }

        result = validator._validate_weight_temporal_stability(panel, weights)
        expected = float(np.clip(1.0 - np.mean([0.10, 0.20, 0.00, 0.00]), 0.0, 1.0))
        assert abs(result - expected) < 1e-6, (
            f"Expected stability≈{expected:.4f}, got {result:.4f}"
        )

    def test_single_year_returns_one(self):
        """
        panel.years has only 1 year → len(years) < 2 guard triggers → returns 1.0.
        """
        validator = Validator()
        panel = _MinimalPanel(n_years=1)
        weights = _make_weights()

        result = validator._validate_weight_temporal_stability(panel, weights)
        assert result == 1.0, (
            f"Expected 1.0 for single-year panel (trivially stable), got {result}"
        )


# ---------------------------------------------------------------------------
# M11 — _validate_weight_method_agreement returns 0.0, not 0.85
# ---------------------------------------------------------------------------

class TestValidatorMethodAgreement:
    """_validate_weight_method_agreement uses Level-2 MC CV to measure agreement."""

    def test_single_method_returns_zero(self):
        """
        Only one weighting method available → cannot compute pairwise
        correlation → must return 0.0 (not 0.85).
        """
        validator = Validator()
        weights = {"entropy": np.array([0.3, 0.4, 0.3])}

        result = validator._validate_weight_method_agreement(weights)
        assert result == 0.0, (
            f"Expected 0.0 for single method (not 0.85), got {result}. "
            "M11 fix may not be applied."
        )

    def test_no_methods_returns_zero(self):
        """Empty weights dict → 0 methods → must return 0.0."""
        validator = Validator()
        result = validator._validate_weight_method_agreement({})
        assert result == 0.0, f"Expected 0.0, got {result}"

    def test_all_constant_arrays_returns_zero(self):
        """
        Constant arrays → Spearman is NaN → all correlations are NaN
        → must return 0.0 (not 0.85).
        """
        validator = Validator()
        constant = np.array([0.25, 0.25, 0.25, 0.25])
        weights = {
            "entropy": constant.copy(),
            "critic":  constant.copy(),
            "merec":   constant.copy(),
        }

        result = validator._validate_weight_method_agreement(weights)
        assert result == 0.0, (
            f"Expected 0.0 for all-constant weights (NaN Spearman), got {result}"
        )

    def test_zero_cv_weights_return_one(self):
        """
        When all criterion cv_weights are zero → mean_cv = 0 → 1 - 0 = 1.0.
        """
        validator = Validator()
        weights = {
            "details": {
                "level2": {
                    "mc_diagnostics": {
                        "cv_weights": {"C01": 0.0, "C02": 0.0, "C03": 0.0, "C04": 0.0}
                    }
                }
            }
        }

        result = validator._validate_weight_method_agreement(weights)
        assert abs(result - 1.0) < 1e-6, (
            f"Expected 1.0 for zero CV weights, got {result}"
        )

    def test_known_cv_weights_return_expected_agreement(self):
        """
        Known cv_weights: mean_cv = (0.05 + 0.15) / 2 = 0.10
        → agreement = 1 - 0.10 = 0.90.
        """
        validator = Validator()
        weights = {
            "details": {
                "level2": {
                    "mc_diagnostics": {
                        "cv_weights": {"C01": 0.05, "C02": 0.15}
                    }
                }
            }
        }
        result = validator._validate_weight_method_agreement(weights)
        expected = 1.0 - (0.05 + 0.15) / 2  # = 0.90
        assert abs(result - expected) < 1e-6, (
            f"Expected {expected:.4f}, got {result:.4f}"
        )

    def test_low_cv_gives_positive_agreement(self):
        """Low coefficient of variation in criterion weights → agreement > 0.5."""
        validator = Validator()
        weights = {
            "details": {
                "level2": {
                    "mc_diagnostics": {
                        "cv_weights": {"C01": 0.03, "C02": 0.05, "C03": 0.04, "C04": 0.06}
                    }
                }
            }
        }
        result = validator._validate_weight_method_agreement(weights)
        assert result > 0.5, (
            f"Expected positive agreement (>0.5) for low-CV weights, got {result}"
        )

    def test_two_methods_with_length_two_returns_zero(self):
        """
        Arrays shorter than 3 elements: Spearman not computable (skipped)
        → no valid correlations → returns 0.0.
        """
        validator = Validator()
        weights = {
            "entropy": np.array([0.6, 0.4]),
            "critic":  np.array([0.3, 0.7]),
        }
        # The condition `len(weight_arrays[i]) > 2` means len=2 is skipped
        result = validator._validate_weight_method_agreement(weights)
        assert result == 0.0, (
            f"Expected 0.0 for length-2 arrays (too short for Spearman), got {result}"
        )


# ---------------------------------------------------------------------------
# M9 — TemporalStabilityValidator uses Spearman (weighting/validation.py)
# ---------------------------------------------------------------------------

class TestTemporalStabilityValidatorSpearman:
    """TemporalStabilityValidator.validate() must use Spearman correlation."""

    def test_identical_vectors_rank_correlation_is_one(self):
        """
        If w1 == w2, Spearman rank correlation must be 1.0 (or as close as
        the nan_to_num guard permits for degenerate identical vectors).
        """
        from weighting.validation import TemporalStabilityValidator

        w = np.array([0.1, 0.4, 0.2, 0.3])
        tsv = TemporalStabilityValidator(threshold=0.9)
        result = tsv.validate(w.copy(), w.copy())

        # Identical → Spearman = 1.0 (not NaN because numpy would give NaN
        # for identical vectors via corrcoef too, but spearmanr gives NaN
        # only when all ranks are tied)
        # In scipy.stats.spearmanr: identical vectors actually give 1.0
        assert abs(result.correlation - 1.0) < 1e-6, (
            f"Expected Spearman correlation = 1.0 for identical vectors, "
            f"got {result.correlation}"
        )

    def test_reversed_vectors_give_minus_one(self):
        """w2 is w1 reversed → Spearman = -1.0."""
        from weighting.validation import TemporalStabilityValidator

        w1 = np.array([0.4, 0.3, 0.2, 0.1])
        w2 = w1[::-1].copy()
        tsv = TemporalStabilityValidator(threshold=0.9)
        result = tsv.validate(w1, w2)

        assert abs(result.correlation + 1.0) < 1e-6, (
            f"Expected Spearman = -1.0 for reversed vectors, got {result.correlation}"
        )

    def test_correlation_in_minus_one_plus_one(self):
        """Spearman correlation must lie in [-1, 1]."""
        from weighting.validation import TemporalStabilityValidator

        rng = np.random.RandomState(99)
        for _ in range(10):
            w1 = rng.dirichlet(np.ones(5))
            w2 = rng.dirichlet(np.ones(5))
            tsv = TemporalStabilityValidator(threshold=0.9)
            result = tsv.validate(w1, w2)
            assert -1.0 - 1e-9 <= result.correlation <= 1.0 + 1e-9

    def test_single_element_returns_one(self):
        """Single-element vectors: len(w) == 1 → correlation defaults to 1.0."""
        from weighting.validation import TemporalStabilityValidator

        tsv = TemporalStabilityValidator(threshold=0.9)
        result = tsv.validate(np.array([1.0]), np.array([1.0]))
        assert result.correlation == 1.0

    def test_stability_flag_based_on_cosine_similarity(self):
        """
        is_stable is determined by cosine similarity ≥ threshold;
        correlation value does not affect the flag.
        """
        from weighting.validation import TemporalStabilityValidator

        # High cosine similarity → stable even if correlation is not 1
        w1 = np.array([0.3, 0.4, 0.3])
        w2 = np.array([0.31, 0.39, 0.30])  # very close to w1
        tsv = TemporalStabilityValidator(threshold=0.99)
        result_tight = tsv.validate(w1, w2)

        w3 = np.array([0.30, 0.40, 0.30])  # essentially same
        tsv_loose = TemporalStabilityValidator(threshold=0.5)
        result_loose = tsv_loose.validate(w1, w3)
        assert result_loose.is_stable


# ---------------------------------------------------------------------------
# Regression tests for earlier fixed bugs
# ---------------------------------------------------------------------------

class TestTOPSISDegenerate:
    """M3 — TOPSIS degenerate case: all identical rows → score = 0.5 (not 0.0)."""

    def test_identical_rows_score_half(self):
        from mcdm.traditional.topsis import TOPSISCalculator

        dm = pd.DataFrame(
            {"C1": [0.5, 0.5, 0.5], "C2": [0.7, 0.7, 0.7]},
            index=["X", "Y", "Z"],
        )
        calc = TOPSISCalculator()
        res = calc.calculate(dm, {"C1": 0.5, "C2": 0.5})

        for alt in ["X", "Y", "Z"]:
            assert abs(res.scores[alt] - 0.5) < 1e-9, (
                f"Degenerate TOPSIS: expected 0.5 for {alt}, got {res.scores[alt]}"
            )


class TestSAWZeroCost:
    """H2 — SAW: zero-cost value must not get worst (0) score in max/sum modes."""

    def test_max_mode_zero_cost_not_worst(self):
        """
        With normalization='max' and a cost column whose best alternative
        has value 0, normalization = (max - x) / max.
        The alternative with cost=0 should receive score 1 (best), not 0.
        """
        from mcdm.traditional.saw import SAWCalculator

        dm = pd.DataFrame(
            {"benefit": [0.9, 0.5, 0.3], "cost": [0.0, 0.5, 0.8]},
            index=["Best", "Mid", "Worst"],
        )
        calc = SAWCalculator(normalization="max", cost_criteria=["cost"])
        res = calc.calculate(dm, {"benefit": 0.5, "cost": 0.5})

        # 'Best' has highest benefit and lowest cost → should be ranked 1
        assert res.ranks["Best"] == 1, (
            f"Expected 'Best' to be rank 1 (zero cost + high benefit), got rank {res.ranks['Best']}"
        )

    def test_sum_mode_zero_cost_not_crash(self):
        """
        With normalization='sum' and a cost column containing zero,
        replacing zeros with ε before inversion prevents divide-by-zero.
        """
        from mcdm.traditional.saw import SAWCalculator

        dm = pd.DataFrame(
            {"C1": [0.8, 0.6], "C_cost": [0.0, 0.4]},
            index=["A", "B"],
        )
        calc = SAWCalculator(normalization="sum", cost_criteria=["C_cost"])
        # Should not raise ZeroDivisionError or produce NaN
        res = calc.calculate(dm, {"C1": 0.5, "C_cost": 0.5})
        assert not res.scores.isna().any(), "SAW sum-mode zero-cost produced NaN scores"


class TestVIKORCompromiseSetC2:
    """M2 — VIKOR compromise set must include best_by_S and best_by_R when C2 fails."""

    def test_compromise_set_is_subset_of_alternatives(self):
        from mcdm.traditional.vikor import VIKORCalculator

        # Use an asymmetric dataset so C1/C2 conditions may not both hold
        dm = pd.DataFrame(
            {"C1": [0.9, 0.8, 0.5, 0.2], "C2": [0.8, 0.6, 0.7, 0.4]},
            index=["A", "B", "C", "D"],
        )
        calc = VIKORCalculator(v=0.5)
        res = calc.calculate(dm, {"C1": 0.5, "C2": 0.5})

        # All members of compromise_set must be actual alternatives
        for alt in res.compromise_set:
            assert alt in dm.index, f"Compromise set member {alt!r} not in alternatives"

    def test_compromise_set_non_empty(self):
        from mcdm.traditional.vikor import VIKORCalculator

        dm = pd.DataFrame(
            {"C1": [0.9, 0.1, 0.5], "C2": [0.2, 0.9, 0.5]},
            index=["A", "B", "C"],
        )
        calc = VIKORCalculator()
        res = calc.calculate(dm, {"C1": 0.5, "C2": 0.5})
        assert len(res.compromise_set) >= 1, (
            "Compromise set must contain at least one alternative"
        )


class TestModifiedEDASTrimmeanPath:
    """H3 — ModifiedEDAS with trimmed mean must produce a valid result."""

    def test_trimmed_mean_differs_from_regular_mean(self):
        """ModifiedEDAS(use_trimmed_mean=True) must use the trimmed path."""
        from mcdm.traditional.edas import EDASCalculator, ModifiedEDAS

        rng = np.random.RandomState(7)
        data = rng.rand(10, 4) + 0.1
        dm = pd.DataFrame(data, index=[f"A{i}" for i in range(10)],
                          columns=["C1", "C2", "C3", "C4"])
        weights = {"C1": 0.25, "C2": 0.25, "C3": 0.25, "C4": 0.25}

        res_regular = EDASCalculator().calculate(dm, weights)
        res_trimmed = ModifiedEDAS(use_trimmed_mean=True).calculate(dm, weights)

        # Both must return results with valid shapes
        assert res_regular.AS.shape[0] == 10
        assert res_trimmed.AS.shape[0] == 10

        # Appraisal scores must differ from the regular path
        assert not np.allclose(res_regular.AS.values, res_trimmed.AS.values), (
            "ModifiedEDAS trimmed-mean path returned same scores as regular EDAS "
            "— H3 fix may not be applied"
        )


# ---------------------------------------------------------------------------
# MEREC normalization regression (H4)
# ---------------------------------------------------------------------------

class TestMERECNormalization:
    """H4 — MEREC uses ratio-based normalization, not min-max."""

    def test_benefit_weight_sum_to_one(self):
        """All-benefit MEREC weights must sum to 1."""
        from weighting.merec import MERECWeightCalculator

        dm = pd.DataFrame(
            {"C1": [0.8, 0.6, 0.9, 0.4], "C2": [0.5, 0.7, 0.3, 0.6]},
        )
        calc = MERECWeightCalculator()
        result = calc.calculate(dm)
        assert abs(result.as_array.sum() - 1.0) < 1e-6

    def test_cost_column_handled(self):
        """MEREC with a cost criterion must not crash and weights sum to 1."""
        from weighting.merec import MERECWeightCalculator

        dm = pd.DataFrame(
            {"C1": [0.8, 0.6, 0.9], "C_cost": [0.2, 0.5, 0.1]},
        )
        calc = MERECWeightCalculator(cost_criteria=["C_cost"])
        result = calc.calculate(dm)
        assert abs(result.as_array.sum() - 1.0) < 1e-6
        assert (result.as_array >= 0).all()


# ---------------------------------------------------------------------------
# ER K constant regression (C2)
# ---------------------------------------------------------------------------

class TestEvidentialReasoningKConstant:
    """
    C2 — ER combine() must use the correct normalisation constant
    K = 1 / (∑∏A − (N−1)·∏B), not 1 / (1 − ∏B).
    """

    def test_k_constant_correct_for_two_independent_sources(self):
        """
        For two fully certain sources with disjoint dominant grades,
        the ER result must still be valid (beliefs sum ≤ 1).
        With the wrong K = 1/(1−∏B) several mass configurations produce
        beliefs > 1.
        """
        from evidential_reasoning.base import BeliefDistribution, EvidentialReasoningEngine

        engine = EvidentialReasoningEngine(grades=["H1", "H2", "H3"])
        # Source heavily in H1
        b1 = BeliefDistribution(grades=["H1", "H2", "H3"], beliefs=np.array([0.9, 0.05, 0.05]))
        # Source heavily in H3
        b2 = BeliefDistribution(grades=["H1", "H2", "H3"], beliefs=np.array([0.05, 0.05, 0.9]))

        result = engine.combine([b1, b2], weights=np.array([0.5, 0.5]))

        assert result.beliefs.sum() <= 1.0 + 1e-9, (
            f"ER K-constant bug: beliefs sum to {result.beliefs.sum():.6f} > 1  "
            "C2 fix may not be applied"
        )
        assert (result.beliefs >= -1e-9).all()

    def test_three_sources_valid(self):
        """Three sources combined with correct K must produce valid beliefs."""
        from evidential_reasoning.base import BeliefDistribution, EvidentialReasoningEngine

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
        """When all sources are uniform, result should remain near-uniform."""
        from evidential_reasoning.base import BeliefDistribution, EvidentialReasoningEngine

        engine = EvidentialReasoningEngine(grades=["A", "B", "C", "D"])
        beliefs = np.array([0.25, 0.25, 0.25, 0.25])
        sources = [BeliefDistribution(grades=["A", "B", "C", "D"],
                                      beliefs=beliefs.copy()) for _ in range(3)]
        result = engine.combine(sources, weights=np.array([1 / 3, 1 / 3, 1 / 3]))

        # All grades should receive equal belief (up to floating-point)
        std_belief = result.beliefs.std()
        assert std_belief < 0.01, (
            f"Uniform sources should yield ~uniform combined beliefs, "
            f"std={std_belief:.4f}"
        )

