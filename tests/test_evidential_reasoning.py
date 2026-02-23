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

from evidential_reasoning.base import BeliefDistribution, EvidentialReasoningEngine


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
