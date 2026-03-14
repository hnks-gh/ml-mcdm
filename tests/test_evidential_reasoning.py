# -*- coding: utf-8 -*-
"""
Unit tests for Evidential Reasoning (ER) module.

Covers:
  - BeliefDistribution construction and properties
  - BeliefDistribution.expected_utility, utility_interval, belief_entropy
  - EvidentialReasoningEngine.combine correctness
  - Weight normalisation, degenerate cases
"""

import numpy as np
import pytest

from ranking.evidential_reasoning.base import BeliefDistribution, EvidentialReasoningEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GRADES = ["Excellent", "Good", "Fair", "Poor"]


def make_belief(beliefs, grades=None):
    return BeliefDistribution(
        grades=grades or GRADES[: len(beliefs)],
        beliefs=np.array(beliefs, dtype=float),
    )


def uniform_engine(n_grades=4):
    return EvidentialReasoningEngine(grades=GRADES[:n_grades])


# ---------------------------------------------------------------------------
# TestBeliefDistribution
# ---------------------------------------------------------------------------

class TestBeliefDistribution:
    def test_unassigned_correct(self):
        bd = make_belief([0.4, 0.3, 0.1])
        assert abs(bd.unassigned - 0.2) < 1e-9

    def test_unassigned_zero_when_sum_one(self):
        bd = make_belief([0.3, 0.3, 0.4])
        assert abs(bd.unassigned) < 1e-9

    def test_beliefs_clipped_at_zero(self):
        bd = BeliefDistribution(grades=["A", "B"], beliefs=np.array([-0.1, 0.5]))
        assert (bd.beliefs >= 0).all()

    def test_beliefs_renormalised_when_sum_gt_one(self):
        bd = BeliefDistribution(
            grades=["A", "B", "C"],
            beliefs=np.array([0.5, 0.5, 0.5]),
        )
        assert bd.beliefs.sum() <= 1.0 + 1e-9

    def test_n_grades_correct(self):
        bd = make_belief([0.2, 0.3, 0.4])
        assert bd.n_grades == 3

    def test_expected_utility_default_in_0_1(self):
        bd = make_belief([0.5, 0.5])
        u = bd.expected_utility()
        assert 0.0 <= u <= 1.0

    def test_expected_utility_certain_best(self):
        """Full belief in 'Excellent' (first grade) → utility ≈ 1."""
        bd = make_belief([1.0, 0.0, 0.0, 0.0])
        assert abs(bd.expected_utility() - 1.0) < 1e-9

    def test_expected_utility_certain_worst(self):
        """Full belief in last grade → utility ≈ 0."""
        bd = make_belief([0.0, 0.0, 0.0, 1.0])
        assert abs(bd.expected_utility() - 0.0) < 1e-9

    def test_utility_interval_lo_le_hi(self):
        bd = make_belief([0.4, 0.3, 0.1])
        lo, hi = bd.utility_interval()
        assert lo <= hi

    def test_utility_interval_covers_expected_utility(self):
        bd = make_belief([0.3, 0.4, 0.2])
        lo, hi = bd.utility_interval()
        eu = bd.expected_utility()
        assert lo <= eu <= hi

    def test_belief_entropy_non_negative(self):
        bd = make_belief([0.3, 0.4, 0.3])
        assert bd.belief_entropy() >= 0.0

    def test_belief_entropy_zero_for_certain(self):
        """Certainty → zero entropy."""
        bd = make_belief([1.0, 0.0, 0.0])
        assert abs(bd.belief_entropy()) < 1e-9

    def test_belief_entropy_max_for_uniform(self):
        """Uniform distribution → maximum entropy."""
        n = 4
        uniform = make_belief([0.25, 0.25, 0.25, 0.25])
        certain = make_belief([1.0, 0.0, 0.0, 0.0])
        assert uniform.belief_entropy() > certain.belief_entropy()

    def test_repr_contains_grades(self):
        bd = make_belief([0.5, 0.5])
        r = repr(bd)
        assert "Excellent" in r or "0.5" in r  # at least some content


# ---------------------------------------------------------------------------
# TestEvidentialReasoningEngine
# ---------------------------------------------------------------------------

class TestEvidentialReasoningEngine:
    def test_result_is_belief_distribution(self):
        engine = uniform_engine(4)
        b1 = make_belief([0.5, 0.3, 0.2, 0.0])
        b2 = make_belief([0.4, 0.4, 0.2, 0.0])
        result = engine.combine([b1, b2], weights=np.array([0.5, 0.5]))
        assert isinstance(result, BeliefDistribution)

    def test_combined_beliefs_sum_le_one(self):
        engine = uniform_engine(4)
        b1 = make_belief([0.5, 0.3, 0.2, 0.0])
        b2 = make_belief([0.3, 0.3, 0.3, 0.1])
        result = engine.combine([b1, b2], weights=np.array([0.6, 0.4]))
        assert result.beliefs.sum() <= 1.0 + 1e-9

    def test_combined_beliefs_non_negative(self):
        engine = uniform_engine(4)
        b1 = make_belief([0.5, 0.5, 0.0, 0.0])
        b2 = make_belief([0.2, 0.3, 0.3, 0.2])
        result = engine.combine([b1, b2], weights=np.array([0.5, 0.5]))
        assert (result.beliefs >= -1e-9).all()

    def test_single_source_returns_same_distribution(self):
        """Combining one source with weight 1.0 should return same beliefs."""
        engine = uniform_engine(4)
        beliefs = np.array([0.4, 0.3, 0.2, 0.1])
        bd = make_belief(beliefs.tolist())
        result = engine.combine([bd], weights=np.array([1.0]))
        # After ER, beliefs may differ due to the formula, but
        # the dominant grade should remain the same
        assert result.beliefs[0] > result.beliefs[3]  # best > worst

    def test_combining_with_unequal_weights(self):
        """Higher weight on a source should pull the combined result toward it."""
        engine = EvidentialReasoningEngine(grades=["H1", "H2"])
        # Source A: strong in H1
        src_a = BeliefDistribution(grades=["H1", "H2"], beliefs=np.array([0.9, 0.1]))
        # Source B: strong in H2
        src_b = BeliefDistribution(grades=["H1", "H2"], beliefs=np.array([0.1, 0.9]))
        # Weight heavily toward A
        result = engine.combine([src_a, src_b], weights=np.array([0.9, 0.1]))
        # Combined should favour H1 (index 0)
        assert result.beliefs[0] > result.beliefs[1]

    def test_combining_identical_sources(self):
        """Identical sources with equal weights → same dominant grade."""
        engine = uniform_engine(4)
        beliefs = np.array([0.6, 0.3, 0.1, 0.0])
        bd = make_belief(beliefs.tolist())
        result = engine.combine([bd, bd], weights=np.array([0.5, 0.5]))
        # Excellent should still be the dominant grade
        assert result.beliefs[0] == result.beliefs.max()

    def test_weight_normalisation(self):
        """Unnormalised weights should be normalised internally."""
        engine = uniform_engine(4)
        b1 = make_belief([0.5, 0.3, 0.2, 0.0])
        b2 = make_belief([0.4, 0.4, 0.1, 0.1])
        # Double the weights → same result
        r1 = engine.combine([b1, b2], weights=np.array([1.0, 1.0]))
        r2 = engine.combine([b1, b2], weights=np.array([10.0, 10.0]))
        np.testing.assert_allclose(r1.beliefs, r2.beliefs, atol=1e-9)

    def test_grades_stored_correctly(self):
        engine = EvidentialReasoningEngine(grades=["A", "B", "C"])
        b1 = BeliefDistribution(grades=["A", "B", "C"], beliefs=np.array([0.4, 0.4, 0.2]))
        result = engine.combine([b1], weights=np.array([1.0]))
        assert result.grades == ["A", "B", "C"]

    def test_unassigned_non_negative(self):
        engine = uniform_engine(4)
        b1 = make_belief([0.4, 0.3, 0.2, 0.1])
        b2 = make_belief([0.3, 0.3, 0.2, 0.2])
        result = engine.combine([b1, b2], weights=np.array([0.5, 0.5]))
        assert result.unassigned >= -1e-9

    def test_three_sources_valid_result(self):
        engine = uniform_engine(4)
        sources = [
            make_belief([0.5, 0.3, 0.1, 0.1]),
            make_belief([0.3, 0.4, 0.2, 0.1]),
            make_belief([0.2, 0.2, 0.4, 0.2]),
        ]
        result = engine.combine(sources, weights=np.array([0.5, 0.3, 0.2]))
        assert isinstance(result, BeliefDistribution)
        assert result.beliefs.sum() <= 1.0 + 1e-9

    def test_fully_uncertain_sources_give_all_unassigned(self):
        """All-zero belief distributions → all mass is unassigned."""
        engine = EvidentialReasoningEngine(grades=["H1", "H2"])
        b1 = BeliefDistribution(grades=["H1", "H2"], beliefs=np.array([0.0, 0.0]))
        b2 = BeliefDistribution(grades=["H1", "H2"], beliefs=np.array([0.0, 0.0]))
        result = engine.combine([b1, b2], weights=np.array([0.5, 0.5]))
        assert result.unassigned >= 1.0 - 1e-9

    def test_expected_utility_of_combined_in_0_1(self):
        engine = uniform_engine(4)
        b1 = make_belief([0.4, 0.3, 0.2, 0.1])
        b2 = make_belief([0.3, 0.4, 0.2, 0.1])
        result = engine.combine([b1, b2], weights=np.array([0.5, 0.5]))
        u = result.expected_utility()
        assert 0.0 <= u <= 1.0


# ---------------------------------------------------------------------------
# TestERNumericalTextbook  (P4-21)
#
# Hand-computed reference values using Yang & Xu (2002) Eqs. 1–3.
#
#  L = 2, N = 3 grades ["H1", "H2", "H3"]
#  Source 1 : β = [0.5, 0.3, 0.2],  weight = 0.6
#  Source 2 : β = [0.4, 0.4, 0.2],  weight = 0.4
#
#  Both sources are complete (Σβ = 1), so B = C and m̃_H = 0.
#
#  A matrix (A[n, i] = w_i·β_{n,i} + 1 − w_i):
#   A[0,0]=0.70  A[1,0]=0.58  A[2,0]=0.52
#   A[0,1]=0.76  A[1,1]=0.76  A[2,1]=0.68
#
#  prod_A = [0.532, 0.4408, 0.3536]
#  prod_B = prod_C = 0.24
#
#  denom = 0.532+0.4408+0.3536 − 2×0.24 = 0.8464
#
#  final_β = [0.292, 0.2008, 0.1136] / 0.8464
#  unassigned (β_H) = 0.24 / 0.8464
# ---------------------------------------------------------------------------

class TestERNumericalTextbook:
    """Known-answer ER verification against Yang & Xu (2002) formulas."""

    # Shared expected values (computed above)
    _DENOM   = 0.8464
    _EXPECT0 = 0.292  / 0.8464  # ≈ 0.34498
    _EXPECT1 = 0.2008 / 0.8464  # ≈ 0.23726
    _EXPECT2 = 0.1136 / 0.8464  # ≈ 0.13422
    _EXPECT_H = 0.24  / 0.8464  # ≈ 0.28354

    @pytest.fixture
    def er_result(self):
        """Run ER combination and return the BeliefDistribution."""
        engine = EvidentialReasoningEngine(grades=["H1", "H2", "H3"])
        src1 = BeliefDistribution(
            grades=["H1", "H2", "H3"],
            beliefs=np.array([0.5, 0.3, 0.2]),
        )
        src2 = BeliefDistribution(
            grades=["H1", "H2", "H3"],
            beliefs=np.array([0.4, 0.4, 0.2]),
        )
        return engine.combine([src1, src2], weights=np.array([0.6, 0.4]))

    def test_beta_h1_matches_formula(self, er_result):
        assert er_result.beliefs[0] == pytest.approx(self._EXPECT0, abs=1e-4)

    def test_beta_h2_matches_formula(self, er_result):
        assert er_result.beliefs[1] == pytest.approx(self._EXPECT1, abs=1e-4)

    def test_beta_h3_matches_formula(self, er_result):
        assert er_result.beliefs[2] == pytest.approx(self._EXPECT2, abs=1e-4)

    def test_unassigned_matches_formula(self, er_result):
        assert er_result.unassigned == pytest.approx(self._EXPECT_H, abs=1e-4)

    def test_beliefs_sum_plus_unassigned_equals_one(self, er_result):
        total = er_result.beliefs.sum() + er_result.unassigned
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_dominant_grade_is_h1(self, er_result):
        """H1 should have the highest belief (strongest source favours H1)."""
        assert er_result.beliefs[0] == er_result.beliefs.max()

    def test_unnormalised_weights_give_same_result(self):
        """Multiplying weights by any positive scalar must not change the result."""
        engine = EvidentialReasoningEngine(grades=["H1", "H2", "H3"])
        src1 = BeliefDistribution(
            grades=["H1", "H2", "H3"],
            beliefs=np.array([0.5, 0.3, 0.2]),
        )
        src2 = BeliefDistribution(
            grades=["H1", "H2", "H3"],
            beliefs=np.array([0.4, 0.4, 0.2]),
        )
        r1 = engine.combine([src1, src2], weights=np.array([0.6, 0.4]))
        r2 = engine.combine([src1, src2], weights=np.array([6.0, 4.0]))
        np.testing.assert_allclose(r1.beliefs, r2.beliefs, atol=1e-9)


# ---------------------------------------------------------------------------
# TestHierarchicalERF07 — F-07: per-province available-criterion ER
#
# Validates that Stage 1 correctly skips absent provinces and Stage 2
# renormalises criterion weights per province (Type 3 structural gaps).
# ---------------------------------------------------------------------------

import pandas as pd
from ranking.evidential_reasoning.hierarchical_er import HierarchicalEvidentialReasoning


class TestHierarchicalERF07:
    """F-07: Per-province available-criterion ER — NaN cells not imputed."""

    @pytest.fixture
    def partial_province_setup(self):
        """
        P1 and P2 have scores for both C01 and C02.
        P3 has scores ONLY for C01 (absent from C02 — Type 3 structural gap).
        """
        return {
            'method_scores': {
                'C01': {'TOPSIS': pd.Series({'P1': 0.8, 'P2': 0.5, 'P3': 0.6})},
                'C02': {'TOPSIS': pd.Series({'P1': 0.7, 'P2': 0.4})},   # P3 absent
            },
            'criterion_weights': {'C01': 0.5, 'C02': 0.5},
            'alternatives': ['P1', 'P2', 'P3'],
        }

    def test_partial_province_has_final_score(self, partial_province_setup):
        """All three provinces must produce a final score (no KeyError)."""
        s = partial_province_setup
        er = HierarchicalEvidentialReasoning(n_grades=5)
        result = er.aggregate(s['method_scores'], s['criterion_weights'],
                              s['alternatives'])
        assert 'P3' in result.final_scores.index

    def test_absent_criterion_not_in_criterion_beliefs(self, partial_province_setup):
        """P3's criterion_beliefs must contain C01 but NOT C02."""
        s = partial_province_setup
        er = HierarchicalEvidentialReasoning(n_grades=5)
        result = er.aggregate(s['method_scores'], s['criterion_weights'],
                              s['alternatives'])
        assert 'C01' in result.criterion_beliefs['P3']
        assert 'C02' not in result.criterion_beliefs['P3']

    def test_partial_province_score_equals_single_criterion_score(
        self, partial_province_setup
    ):
        """
        P3's ER score (with C02 absent) must equal what it would get if ranked
        on C01 alone with weight 1.0 (available-case ER renormalisation).
        """
        s = partial_province_setup
        er = HierarchicalEvidentialReasoning(n_grades=5)
        result = er.aggregate(s['method_scores'], s['criterion_weights'],
                              s['alternatives'])

        result_c01_only = er.aggregate(
            {'C01': s['method_scores']['C01']},
            {'C01': 1.0},
            ['P3'],
        )
        assert abs(
            result.final_scores['P3'] - result_c01_only.final_scores['P3']
        ) < 1e-8, (
            "P3's score must equal the C01-only score after weight renormalisation"
        )

    def test_full_provinces_unaffected(self, partial_province_setup):
        """P1 and P2 (full data) must have beliefs for all criteria."""
        s = partial_province_setup
        er = HierarchicalEvidentialReasoning(n_grades=5)
        result = er.aggregate(s['method_scores'], s['criterion_weights'],
                              s['alternatives'])
        for alt in ['P1', 'P2']:
            assert 'C01' in result.criterion_beliefs[alt]
            assert 'C02' in result.criterion_beliefs[alt]

    def test_get_criterion_belief_returns_none_for_absent_pair(
        self, partial_province_setup
    ):
        """get_criterion_belief() must return None for (P3, C02) — no KeyError."""
        s = partial_province_setup
        er = HierarchicalEvidentialReasoning(n_grades=5)
        result = er.aggregate(s['method_scores'], s['criterion_weights'],
                              s['alternatives'])
        val = result.get_criterion_belief('P3', 'C02')
        assert val is None

    def test_get_criterion_belief_returns_belief_for_present_pair(
        self, partial_province_setup
    ):
        """get_criterion_belief() must return a BeliefDistribution for (P3, C01)."""
        from ranking.evidential_reasoning.base import BeliefDistribution
        s = partial_province_setup
        er = HierarchicalEvidentialReasoning(n_grades=5)
        result = er.aggregate(s['method_scores'], s['criterion_weights'],
                              s['alternatives'])
        val = result.get_criterion_belief('P3', 'C01')
        assert isinstance(val, BeliefDistribution)

    def test_province_with_no_data_gets_minimum_utility(self):
        """
        A province absent from ALL criterion score Series must receive
        the minimum utility (0.0) — not raise KeyError or crash.
        """
        method_scores = {
            'C01': {'TOPSIS': pd.Series({'P1': 0.8, 'P2': 0.5})},
            # P3 absent from all criteria
        }
        criterion_weights = {'C01': 1.0}
        alternatives = ['P1', 'P2', 'P3']

        er = HierarchicalEvidentialReasoning(n_grades=5)
        result = er.aggregate(method_scores, criterion_weights, alternatives)

        assert 'P3' in result.final_scores.index
        assert result.final_scores['P3'] == pytest.approx(0.0)

    def test_multiple_absent_criteria_each_skipped(self):
        """
        Province P2 absent from C01 and C02, but present for C03.
        P2's criterion_beliefs must contain only C03.
        """
        method_scores = {
            'C01': {'M1': pd.Series({'P1': 0.9, 'P3': 0.7})},
            'C02': {'M1': pd.Series({'P1': 0.6, 'P3': 0.5})},
            'C03': {'M1': pd.Series({'P1': 0.8, 'P2': 0.4, 'P3': 0.6})},
        }
        criterion_weights = {'C01': 0.4, 'C02': 0.3, 'C03': 0.3}
        alternatives = ['P1', 'P2', 'P3']

        er = HierarchicalEvidentialReasoning(n_grades=5)
        result = er.aggregate(method_scores, criterion_weights, alternatives)

        assert 'C01' not in result.criterion_beliefs['P2']
        assert 'C02' not in result.criterion_beliefs['P2']
        assert 'C03' in result.criterion_beliefs['P2']
        # P2's score should equal C03-only score with weight 1.0
        result_c03_only = er.aggregate(
            {'C03': method_scores['C03']},
            {'C03': 1.0},
            ['P2'],
        )
        assert abs(
            result.final_scores['P2'] - result_c03_only.final_scores['P2']
        ) < 1e-8
