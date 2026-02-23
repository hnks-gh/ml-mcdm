# -*- coding: utf-8 -*-
"""
Unit tests for the six traditional MCDM methods.

Covers:
    - TOPSIS  — rank inversion, cost criteria, normalization modes
    - VIKOR   — compromise conditions, v=0/1 extremes
    - EDAS    — PDA/NDA structure, average solution
    - SAW     — three normalization modes, cost inversion
    - COPRAS  — utility sum = 100 %, benefit / cost split
    - PROMETHEE — net-flow antisymmetry, zero-sum property
    - ModifiedEDAS — trimmed-mean path returns a result
    - WeightResult.as_array — insertion order preserved
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def dm4():
    """4-alternative × 3-criteria decision matrix with known best alternative."""
    return pd.DataFrame(
        {
            "C1": [0.9, 0.7, 0.5, 0.3],   # benefit, A is best
            "C2": [0.8, 0.6, 0.7, 0.4],   # benefit
            "C3": [0.2, 0.5, 0.8, 0.9],   # cost, A is best (lowest)
        },
        index=["A", "B", "C", "D"],
    )


@pytest.fixture
def equal_weights():
    return {"C1": 1 / 3, "C2": 1 / 3, "C3": 1 / 3}


@pytest.fixture
def dm6():
    """6-alternative × 4-criteria matrix (all benefit)."""
    rng = np.random.RandomState(0)
    data = rng.rand(6, 4) + 0.1
    return pd.DataFrame(data, index=list("ABCDEF"),
                        columns=["C1", "C2", "C3", "C4"])


@pytest.fixture
def uniform_weights4():
    return {"C1": 0.25, "C2": 0.25, "C3": 0.25, "C4": 0.25}


# ---------------------------------------------------------------------------
# TOPSIS
# ---------------------------------------------------------------------------

class TestTOPSIS:
    def test_ranks_return_all_alternatives(self, dm4, equal_weights):
        from mcdm.traditional.topsis import TOPSISCalculator

        calc = TOPSISCalculator(cost_criteria=["C3"])
        res = calc.calculate(dm4, equal_weights)
        assert set(res.ranks.index) == set(dm4.index)
        assert sorted(res.ranks.values.tolist()) == [1, 2, 3, 4]

    def test_best_alternative_known(self, dm4, equal_weights):
        """A dominates all others on all criteria → rank 1."""
        from mcdm.traditional.topsis import TOPSISCalculator

        calc = TOPSISCalculator(cost_criteria=["C3"])
        res = calc.calculate(dm4, equal_weights)
        assert res.ranks["A"] == 1

    def test_worst_alternative_known(self, dm4, equal_weights):
        from mcdm.traditional.topsis import TOPSISCalculator

        calc = TOPSISCalculator(cost_criteria=["C3"])
        res = calc.calculate(dm4, equal_weights)
        assert res.ranks["D"] == 4

    def test_closeness_in_unit_interval(self, dm4, equal_weights):
        from mcdm.traditional.topsis import TOPSISCalculator

        calc = TOPSISCalculator(cost_criteria=["C3"])
        res = calc.calculate(dm4, equal_weights)
        assert (res.scores >= 0).all() and (res.scores <= 1).all()

    def test_distance_positivity(self, dm4, equal_weights):
        from mcdm.traditional.topsis import TOPSISCalculator

        calc = TOPSISCalculator(cost_criteria=["C3"])
        res = calc.calculate(dm4, equal_weights)
        assert (res.d_positive >= 0).all()
        assert (res.d_negative >= 0).all()

    def test_vector_normalization(self, dm6, uniform_weights4):
        """Vector norm with all-benefit data should not crash."""
        from mcdm.traditional.topsis import TOPSISCalculator

        calc = TOPSISCalculator(normalization="vector")
        res = calc.calculate(dm6, uniform_weights4)
        assert res.ranks.shape[0] == 6

    def test_minmax_normalization(self, dm6, uniform_weights4):
        from mcdm.traditional.topsis import TOPSISCalculator

        calc = TOPSISCalculator(normalization="minmax")
        res = calc.calculate(dm6, uniform_weights4)
        assert res.ranks.shape[0] == 6

    def test_weight_result_input(self, dm4):
        """TOPSISCalculator should accept a WeightResult object."""
        from mcdm.traditional.topsis import TOPSISCalculator
        from weighting.base import WeightResult

        wr = WeightResult(
            weights={"C1": 0.4, "C2": 0.3, "C3": 0.3},
            method="test",
            details={},
        )
        calc = TOPSISCalculator(cost_criteria=["C3"])
        res = calc.calculate(dm4, wr)
        assert res.ranks["A"] == 1

    def test_identical_rows_tied_rank(self):
        """Two identical rows should receive the same score."""
        from mcdm.traditional.topsis import TOPSISCalculator

        dm = pd.DataFrame(
            {"C1": [0.5, 0.5, 0.9], "C2": [0.5, 0.5, 0.1]},
            index=["X", "Y", "Z"],
        )
        calc = TOPSISCalculator()
        res = calc.calculate(dm, {"C1": 0.5, "C2": 0.5})
        assert abs(res.scores["X"] - res.scores["Y"]) < 1e-10


# ---------------------------------------------------------------------------
# VIKOR
# ---------------------------------------------------------------------------

class TestVIKOR:
    def test_compromise_solution_is_an_alternative(self, dm4, equal_weights):
        from mcdm.traditional.vikor import VIKORCalculator

        calc = VIKORCalculator(cost_criteria=["C3"])
        res = calc.calculate(dm4, equal_weights)
        assert res.compromise_solution in dm4.index

    def test_q_values_bounded(self, dm4, equal_weights):
        """Q values should lie in [0, 1]."""
        from mcdm.traditional.vikor import VIKORCalculator

        calc = VIKORCalculator(cost_criteria=["C3"])
        res = calc.calculate(dm4, equal_weights)
        assert (res.Q >= -1e-9).all() and (res.Q <= 1 + 1e-9).all()

    def test_best_alternative_q_near_zero(self, dm4, equal_weights):
        from mcdm.traditional.vikor import VIKORCalculator

        calc = VIKORCalculator(cost_criteria=["C3"])
        res = calc.calculate(dm4, equal_weights)
        assert res.Q["A"] < 0.1

    def test_v_zero_maximises_regret(self, dm6, uniform_weights4):
        """v=0 means only individual regret counts; all alternatives ranked."""
        from mcdm.traditional.vikor import VIKORCalculator

        calc = VIKORCalculator(v=0.0)
        res = calc.calculate(dm6, uniform_weights4)
        assert len(res.ranks_Q) == 6

    def test_v_one_maximises_group_utility(self, dm6, uniform_weights4):
        from mcdm.traditional.vikor import VIKORCalculator

        calc = VIKORCalculator(v=1.0)
        res = calc.calculate(dm6, uniform_weights4)
        assert set(res.ranks_Q.values.tolist()) == set(range(1, 6 + 1))

    def test_compromise_set_subset_of_alternatives(self, dm4, equal_weights):
        from mcdm.traditional.vikor import VIKORCalculator

        calc = VIKORCalculator(cost_criteria=["C3"])
        res = calc.calculate(dm4, equal_weights)
        assert all(a in dm4.index for a in res.compromise_set)

    def test_ranks_complete_no_gaps(self, dm6, uniform_weights4):
        from mcdm.traditional.vikor import VIKORCalculator

        calc = VIKORCalculator()
        res = calc.calculate(dm6, uniform_weights4)
        assert sorted(res.ranks_Q.values.tolist()) == list(range(1, 7))


# ---------------------------------------------------------------------------
# EDAS
# ---------------------------------------------------------------------------

class TestEDAS:
    def test_pda_nda_non_negative(self, dm4, equal_weights):
        from mcdm.traditional.edas import EDASCalculator

        calc = EDASCalculator(cost_criteria=["C3"])
        res = calc.calculate(dm4, equal_weights)
        assert (res.PDA >= -1e-9).all().all()
        assert (res.NDA >= -1e-9).all().all()

    def test_appraisal_score_in_unit_interval(self, dm4, equal_weights):
        from mcdm.traditional.edas import EDASCalculator

        calc = EDASCalculator(cost_criteria=["C3"])
        res = calc.calculate(dm4, equal_weights)
        assert (res.AS >= -1e-9).all() and (res.AS <= 1 + 1e-9).all()

    def test_average_solution_is_column_mean(self, dm4, equal_weights):
        """Average solution should equal column mean for benefit criteria."""
        from mcdm.traditional.edas import EDASCalculator

        calc = EDASCalculator()   # all benefit
        res = calc.calculate(dm4, equal_weights)
        for col in dm4.columns:
            assert abs(res.average_solution[col] - dm4[col].mean()) < 1e-9

    def test_best_alternative_highest_score(self, dm4, equal_weights):
        """A has best values on all criteria → rank 1."""
        from mcdm.traditional.edas import EDASCalculator

        calc = EDASCalculator(cost_criteria=["C3"])
        res = calc.calculate(dm4, equal_weights)
        assert res.ranks["A"] == 1

    def test_modified_edas_trimmed_mean_returns_result(self, dm6, uniform_weights4):
        """ModifiedEDAS with use_trimmed_mean=True must return a result (not None)."""
        from mcdm.traditional.edas import ModifiedEDAS

        calc = ModifiedEDAS(use_trimmed_mean=True)
        res = calc.calculate(dm6, uniform_weights4)
        assert res is not None
        assert res.AS.shape[0] == 6

    def test_pda_nda_zero_for_average_alternative(self):
        """An alternative exactly at the column mean has PDA=NDA=0."""
        from mcdm.traditional.edas import EDASCalculator

        mean_val = 0.5
        dm = pd.DataFrame(
            {"C1": [0.3, mean_val, 0.7], "C2": [0.6, mean_val, 0.4]},
            index=["X", "M", "Z"],
        )
        calc = EDASCalculator()
        res = calc.calculate(dm, {"C1": 0.5, "C2": 0.5})
        assert abs(res.PDA.loc["M"].sum()) < 1e-6
        assert abs(res.NDA.loc["M"].sum()) < 1e-6


# ---------------------------------------------------------------------------
# SAW
# ---------------------------------------------------------------------------

class TestSAW:
    def test_minmax_scores_in_unit_interval(self, dm4, equal_weights):
        from mcdm.traditional.saw import SAWCalculator

        calc = SAWCalculator(normalization="minmax", cost_criteria=["C3"])
        res = calc.calculate(dm4, equal_weights)
        assert (res.scores >= -1e-9).all() and (res.scores <= 1 + 1e-9).all()

    def test_best_alternative_rank1(self, dm4, equal_weights):
        from mcdm.traditional.saw import SAWCalculator

        calc = SAWCalculator(normalization="minmax", cost_criteria=["C3"])
        res = calc.calculate(dm4, equal_weights)
        assert res.ranks["A"] == 1

    def test_max_normalization(self, dm6, uniform_weights4):
        from mcdm.traditional.saw import SAWCalculator

        calc = SAWCalculator(normalization="max")
        res = calc.calculate(dm6, uniform_weights4)
        assert sorted(res.ranks.values.tolist()) == list(range(1, 7))

    def test_sum_normalization_benefit(self, dm6, uniform_weights4):
        from mcdm.traditional.saw import SAWCalculator

        calc = SAWCalculator(normalization="sum")
        res = calc.calculate(dm6, uniform_weights4)
        assert res.ranks.shape[0] == 6

    def test_sum_normalization_cost_inverts(self):
        """With sum normalization and a cost criterion, lower raw value → better rank."""
        from mcdm.traditional.saw import SAWCalculator

        dm = pd.DataFrame(
            {"benefit": [0.9, 0.5], "cost": [0.1, 0.9]},
            index=["Good", "Bad"],
        )
        calc = SAWCalculator(normalization="sum", cost_criteria=["cost"])
        res = calc.calculate(dm, {"benefit": 0.5, "cost": 0.5})
        assert res.ranks["Good"] < res.ranks["Bad"]

    def test_equal_alternatives_same_score(self):
        """Identical rows must yield identical scores."""
        from mcdm.traditional.saw import SAWCalculator

        dm = pd.DataFrame(
            {"C1": [0.5, 0.5], "C2": [0.7, 0.7]},
            index=["X", "Y"],
        )
        calc = SAWCalculator()
        res = calc.calculate(dm, {"C1": 0.5, "C2": 0.5})
        assert abs(res.scores["X"] - res.scores["Y"]) < 1e-10

    def test_ranks_are_contiguous_integers(self, dm4, equal_weights):
        from mcdm.traditional.saw import SAWCalculator

        calc = SAWCalculator(cost_criteria=["C3"])
        res = calc.calculate(dm4, equal_weights)
        assert sorted(res.ranks.values.tolist()) == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# COPRAS
# ---------------------------------------------------------------------------

class TestCOPRAS:
    def test_utility_sum_100(self, dm4, equal_weights):
        """Best alternative's utility degree must be 100 % (COPRAS definition)."""
        from mcdm.traditional.copras import COPRASCalculator

        calc = COPRASCalculator(cost_criteria=["C3"])
        res = calc.calculate(dm4, equal_weights)
        assert abs(res.utility_degree.max() - 100.0) < 0.1

    def test_best_utility_is_100(self, dm4, equal_weights):
        from mcdm.traditional.copras import COPRASCalculator

        calc = COPRASCalculator(cost_criteria=["C3"])
        res = calc.calculate(dm4, equal_weights)
        assert abs(res.utility_degree.max() - 100.0) < 0.1

    def test_best_rank_is_1(self, dm4, equal_weights):
        from mcdm.traditional.copras import COPRASCalculator

        calc = COPRASCalculator(cost_criteria=["C3"])
        res = calc.calculate(dm4, equal_weights)
        assert res.ranks["A"] == 1

    def test_s_plus_s_minus_positive(self, dm4, equal_weights):
        from mcdm.traditional.copras import COPRASCalculator

        calc = COPRASCalculator(cost_criteria=["C3"])
        res = calc.calculate(dm4, equal_weights)
        assert (res.S_plus >= 0).all()
        assert (res.S_minus >= 0).all()

    def test_all_benefit_no_cost(self, dm6, uniform_weights4):
        """With only benefit criteria S_minus should be zero for all."""
        from mcdm.traditional.copras import COPRASCalculator

        calc = COPRASCalculator()
        res = calc.calculate(dm6, uniform_weights4)
        assert (res.S_minus < 1e-9).all()

    def test_ranks_complete(self, dm6, uniform_weights4):
        from mcdm.traditional.copras import COPRASCalculator

        calc = COPRASCalculator()
        res = calc.calculate(dm6, uniform_weights4)
        assert sorted(res.ranks.values.tolist()) == list(range(1, 7))


# ---------------------------------------------------------------------------
# PROMETHEE
# ---------------------------------------------------------------------------

class TestPROMETHEE:
    def test_net_flow_antisymmetry(self, dm4, equal_weights):
        """Φ_net should sum to ~0 (antisymmetry of outranking relation)."""
        from mcdm.traditional.promethee import PROMETHEECalculator

        calc = PROMETHEECalculator(cost_criteria=["C3"])
        res = calc.calculate(dm4, equal_weights)
        assert abs(res.phi_net.sum()) < 1e-9

    def test_positive_minus_negative_gives_net(self, dm4, equal_weights):
        from mcdm.traditional.promethee import PROMETHEECalculator

        calc = PROMETHEECalculator(cost_criteria=["C3"])
        res = calc.calculate(dm4, equal_weights)
        reconstructed = res.phi_positive - res.phi_negative
        np.testing.assert_allclose(
            res.phi_net.values, reconstructed.values, atol=1e-9
        )

    def test_best_alternative_highest_net(self, dm4, equal_weights):
        from mcdm.traditional.promethee import PROMETHEECalculator

        calc = PROMETHEECalculator(cost_criteria=["C3"])
        res = calc.calculate(dm4, equal_weights)
        assert res.phi_net.idxmax() == "A"

    def test_ranks_complete(self, dm4, equal_weights):
        from mcdm.traditional.promethee import PROMETHEECalculator

        calc = PROMETHEECalculator(cost_criteria=["C3"])
        res = calc.calculate(dm4, equal_weights)
        assert sorted(res.ranks_promethee_ii.values.tolist()) == [1, 2, 3, 4]

    def test_positive_flow_positive(self, dm4, equal_weights):
        from mcdm.traditional.promethee import PROMETHEECalculator

        calc = PROMETHEECalculator(cost_criteria=["C3"])
        res = calc.calculate(dm4, equal_weights)
        assert (res.phi_positive >= -1e-9).all()

    def test_vshape_preference_function(self, dm6, uniform_weights4):
        from mcdm.traditional.promethee import PROMETHEECalculator

        calc = PROMETHEECalculator(preference_function="vshape")
        res = calc.calculate(dm6, uniform_weights4)
        assert res.phi_net.shape[0] == 6


# ---------------------------------------------------------------------------
# WeightResult — regression for Tier 2 fix (column order)
# ---------------------------------------------------------------------------

class TestWeightResultColumnOrder:
    def test_as_array_preserves_insertion_order(self):
        from weighting.base import WeightResult

        wr = WeightResult(
            weights={"C3": 0.3, "C1": 0.5, "C2": 0.2},
            method="test",
            details={},
        )
        arr = wr.as_array
        # Should follow insertion order: C3, C1, C2
        np.testing.assert_allclose(arr, [0.3, 0.5, 0.2])

    def test_as_array_sums_to_one(self):
        from weighting.base import WeightResult

        wr = WeightResult(
            weights={"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
            method="equal",
            details={},
        )
        assert abs(wr.as_array.sum() - 1.0) < 1e-10

    def test_as_series_matches_weights_dict(self):
        from weighting.base import WeightResult

        wr = WeightResult(
            weights={"X": 0.6, "Y": 0.4},
            method="test",
            details={},
        )
        s = wr.as_series
        assert s["X"] == pytest.approx(0.6)
        assert s["Y"] == pytest.approx(0.4)
