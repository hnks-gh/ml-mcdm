# -*- coding: utf-8 -*-
"""
Unit tests for IFS MCDM methods.

Covers:
  - IFN arithmetic and properties
  - IFSDecisionMatrix construction and derived matrices
  - IFS_SAW, IFS_TOPSIS, IFS_VIKOR, IFS_EDAS, IFS_COPRAS, IFS_PROMETHEE
"""

import numpy as np
import pandas as pd
import pytest

from mcdm.ifs.base import IFN, IFSDecisionMatrix
from mcdm.ifs.ifs_saw import IFS_SAW
from mcdm.ifs.ifs_topsis import IFS_TOPSIS
from mcdm.ifs.ifs_vikor import IFS_VIKOR
from mcdm.ifs.ifs_edas import IFS_EDAS
from mcdm.ifs.ifs_copras import IFS_COPRAS
from mcdm.ifs.ifs_promethee import IFS_PROMETHEE


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

def make_ifs_matrix(n_alts: int = 4, n_crit: int = 3, rng_seed: int = 0) -> IFSDecisionMatrix:
    """Build a synthetic IFSDecisionMatrix with valid μ + ν ≤ 1."""
    rng = np.random.RandomState(rng_seed)
    alts = [f"A{i}" for i in range(n_alts)]
    crits = [f"C{j}" for j in range(n_crit)]
    matrix = {}
    for a in alts:
        matrix[a] = {}
        for c in crits:
            mu = rng.uniform(0.1, 0.5)
            nu = rng.uniform(0.1, min(0.45, 0.9 - mu))
            matrix[a][c] = IFN(mu, nu)
    return IFSDecisionMatrix(matrix, alts, crits)


def equal_weights(criteria):
    """Return equal-weight dict for a list of criteria."""
    n = len(criteria)
    return {c: 1.0 / n for c in criteria}


# ---------------------------------------------------------------------------
# TestIFN
# ---------------------------------------------------------------------------

class TestIFN:
    def test_pi_non_negative(self):
        ifn = IFN(0.4, 0.3)
        assert ifn.pi >= 0.0

    def test_pi_correct(self):
        ifn = IFN(0.4, 0.3)
        assert abs(ifn.pi - 0.3) < 1e-9

    def test_mu_nu_sum_le_one(self):
        ifn = IFN(0.5, 0.4)
        # After IFS constraint enforcement μ + ν ≤ 1
        assert ifn.mu + ifn.nu <= 1.0 + 1e-9

    def test_score_equals_mu_minus_nu(self):
        ifn = IFN(0.6, 0.2)
        assert abs(ifn.score() - (0.6 - 0.2)) < 1e-9

    def test_accuracy_equals_mu_plus_nu(self):
        ifn = IFN(0.6, 0.2)
        assert abs(ifn.accuracy() - (0.6 + 0.2)) < 1e-9

    def test_lt_ordering(self):
        low = IFN(0.2, 0.5)   # score = -0.3
        high = IFN(0.8, 0.1)  # score =  0.7
        assert low < high

    def test_eq_same_values(self):
        a = IFN(0.3, 0.4)
        b = IFN(0.3, 0.4)
        assert a == b

    def test_hamming_distance_zero_same(self):
        ifn = IFN(0.4, 0.3)
        # hamming distance to itself must be 0
        d = IFN.hamming_distance(ifn, ifn)
        assert d == 0.0

    def test_normalized_euclidean_non_negative(self):
        a = IFN(0.5, 0.2)
        b = IFN(0.3, 0.4)
        dist = IFN.normalized_euclidean(a, b)
        assert dist >= 0.0

    def test_power_method(self):
        """IFN.power(2) should satisfy μ^2, 1-(1-ν)^2."""
        ifn = IFN(0.4, 0.3)
        p = ifn.power(2)
        assert abs(p.mu - 0.4 ** 2) < 1e-9
        assert abs(p.nu - (1 - (1 - 0.3) ** 2)) < 1e-9

    def test_clipping_invalid_inputs(self):
        """IFN should clip and not raise on mu+nu > 1."""
        ifn = IFN(0.7, 0.6)   # sum = 1.3 → should be normalised
        assert ifn.mu + ifn.nu <= 1.0 + 1e-9
        assert ifn.pi >= -1e-9


# ---------------------------------------------------------------------------
# TestIFSDecisionMatrix
# ---------------------------------------------------------------------------

class TestIFSDecisionMatrix:
    def test_shape_preserved(self):
        dm = make_ifs_matrix(4, 3)
        assert len(dm.alternatives) == 4
        assert len(dm.criteria) == 3

    def test_to_score_matrix_shape(self):
        dm = make_ifs_matrix(4, 3)
        S = dm.to_score_matrix()
        assert S.shape == (4, 3)

    def test_to_mu_matrix_values_match(self):
        dm = make_ifs_matrix(4, 3)
        mu_mat = dm.to_mu_matrix()
        for a in dm.alternatives:
            for c in dm.criteria:
                assert abs(mu_mat.loc[a, c] - dm.matrix[a][c].mu) < 1e-12

    def test_to_nu_matrix_values_match(self):
        dm = make_ifs_matrix(4, 3)
        nu_mat = dm.to_nu_matrix()
        for a in dm.alternatives:
            for c in dm.criteria:
                assert abs(nu_mat.loc[a, c] - dm.matrix[a][c].nu) < 1e-12

    def test_to_pi_matrix_equals_1_minus_mu_minus_nu(self):
        dm = make_ifs_matrix(4, 3)
        pi_mat = dm.to_pi_matrix()
        for a in dm.alternatives:
            for c in dm.criteria:
                ifn = dm.matrix[a][c]
                expected_pi = 1.0 - ifn.mu - ifn.nu
                assert abs(pi_mat.loc[a, c] - expected_pi) < 1e-9

    def test_score_matrix_range(self):
        """Scores should be in [-1, 1]."""
        dm = make_ifs_matrix(5, 4)
        S = dm.to_score_matrix()
        assert (S >= -1.0 - 1e-9).all().all()
        assert (S <= 1.0 + 1e-9).all().all()

    def test_from_crisp_factory(self):
        """IFSDecisionMatrix.from_crisp should produce valid matrix."""
        data = pd.DataFrame(
            {"C1": [0.8, 0.5, 0.3], "C2": [0.2, 0.6, 0.9]},
            index=["A", "B", "C"],
        )
        dm = IFSDecisionMatrix.from_crisp(data, default_pi=0.1)
        assert dm.alternatives == ["A", "B", "C"]
        assert dm.criteria == ["C1", "C2"]
        for a in dm.alternatives:
            for c in dm.criteria:
                ifn = dm.matrix[a][c]
                assert ifn.mu + ifn.nu <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# TestIFS_SAW
# ---------------------------------------------------------------------------

class TestIFS_SAW:
    def test_ranks_complete(self):
        dm = make_ifs_matrix(4, 3)
        calc = IFS_SAW()
        result = calc.calculate(dm, equal_weights(dm.criteria))
        assert set(result.ranks.values) == {1, 2, 3, 4}

    def test_ranks_length(self):
        dm = make_ifs_matrix(5, 3)
        result = IFS_SAW().calculate(dm, equal_weights(dm.criteria))
        assert len(result.ranks) == 5

    def test_scores_series_length(self):
        dm = make_ifs_matrix(4, 3)
        result = IFS_SAW().calculate(dm, equal_weights(dm.criteria))
        assert len(result.scores) == 4

    def test_cost_criteria_lowers_high_nu_score(self):
        """Alternative with high ν on cost criterion should rank better."""
        alts = ["A", "B"]
        criits = ["C"]
        # A: mu=0.7, nu=0.1 → score=0.6 (good for benefit, bad for cost)
        # B: mu=0.2, nu=0.6 → score=-0.4 (good for cost)
        dm = IFSDecisionMatrix(
            {"A": {"C": IFN(0.7, 0.1)}, "B": {"C": IFN(0.2, 0.6)}},
            alts, criits,
        )
        result = IFS_SAW(cost_criteria=["C"]).calculate(dm, {"C": 1.0})
        # B has lower mu and higher nu → better for cost → rank 1
        assert result.ranks["B"] < result.ranks["A"]

    def test_equal_alternatives_same_score(self):
        ifn = IFN(0.4, 0.3)
        alts = ["A", "B"]
        dm = IFSDecisionMatrix(
            {"A": {"C1": ifn, "C2": ifn}, "B": {"C1": ifn, "C2": ifn}},
            alts, ["C1", "C2"],
        )
        result = IFS_SAW().calculate(dm, {"C1": 0.5, "C2": 0.5})
        assert abs(result.scores["A"] - result.scores["B"]) < 1e-9

    def test_weights_preserved(self):
        dm = make_ifs_matrix(3, 3)
        w = equal_weights(dm.criteria)
        result = IFS_SAW().calculate(dm, w)
        assert set(result.weights.keys()) == set(dm.criteria)


# ---------------------------------------------------------------------------
# TestIFS_TOPSIS
# ---------------------------------------------------------------------------

class TestIFS_TOPSIS:
    def test_ranks_complete(self):
        dm = make_ifs_matrix(4, 3)
        result = IFS_TOPSIS().calculate(dm, equal_weights(dm.criteria))
        assert set(result.ranks.values) == {1, 2, 3, 4}

    def test_closeness_in_0_1(self):
        dm = make_ifs_matrix(5, 4)
        result = IFS_TOPSIS().calculate(dm, equal_weights(dm.criteria))
        assert (result.scores >= -1e-9).all()
        assert (result.scores <= 1.0 + 1e-9).all()

    def test_d_positive_non_negative(self):
        dm = make_ifs_matrix(4, 3)
        result = IFS_TOPSIS().calculate(dm, equal_weights(dm.criteria))
        assert (result.d_positive >= -1e-9).all()

    def test_d_negative_non_negative(self):
        dm = make_ifs_matrix(4, 3)
        result = IFS_TOPSIS().calculate(dm, equal_weights(dm.criteria))
        assert (result.d_negative >= -1e-9).all()

    def test_best_has_lowest_d_positive(self):
        dm = make_ifs_matrix(4, 3)
        result = IFS_TOPSIS().calculate(dm, equal_weights(dm.criteria))
        best_alt = result.ranks.idxmin()
        assert result.d_positive[best_alt] == result.d_positive.min()

    def test_ideal_anti_ideal_stored(self):
        dm = make_ifs_matrix(4, 3)
        result = IFS_TOPSIS().calculate(dm, equal_weights(dm.criteria))
        assert result.ideal is not None
        assert result.anti_ideal is not None

    def test_cost_criteria_accepted(self):
        dm = make_ifs_matrix(4, 3)
        result = IFS_TOPSIS(cost_criteria=["C0"]).calculate(
            dm, equal_weights(dm.criteria)
        )
        assert len(result.ranks) == 4


# ---------------------------------------------------------------------------
# TestIFS_VIKOR
# ---------------------------------------------------------------------------

class TestIFS_VIKOR:
    def test_ranks_q_complete(self):
        dm = make_ifs_matrix(4, 3)
        result = IFS_VIKOR().calculate(dm, equal_weights(dm.criteria))
        assert set(result.ranks_Q.values) == {1, 2, 3, 4}

    def test_Q_non_negative(self):
        dm = make_ifs_matrix(5, 3)
        result = IFS_VIKOR().calculate(dm, equal_weights(dm.criteria))
        assert (result.Q >= -1e-9).all()

    def test_compromise_set_non_empty(self):
        dm = make_ifs_matrix(4, 3)
        result = IFS_VIKOR().calculate(dm, equal_weights(dm.criteria))
        assert len(result.compromise_set) >= 1

    def test_compromise_set_is_subset_of_alternatives(self):
        dm = make_ifs_matrix(4, 3)
        result = IFS_VIKOR().calculate(dm, equal_weights(dm.criteria))
        assert set(result.compromise_set).issubset(set(dm.alternatives))

    def test_v_equals_zero(self):
        """v=0 → pure group utility (S-based ranking)."""
        dm = make_ifs_matrix(4, 3)
        result = IFS_VIKOR(v=0.0).calculate(dm, equal_weights(dm.criteria))
        assert len(result.ranks_Q) == 4

    def test_v_equals_one(self):
        """v=1 → pure individual regret (R-based ranking)."""
        dm = make_ifs_matrix(4, 3)
        result = IFS_VIKOR(v=1.0).calculate(dm, equal_weights(dm.criteria))
        assert len(result.ranks_Q) == 4

    def test_s_and_r_non_negative(self):
        dm = make_ifs_matrix(4, 3)
        result = IFS_VIKOR().calculate(dm, equal_weights(dm.criteria))
        assert (result.S >= -1e-9).all()
        assert (result.R >= -1e-9).all()


# ---------------------------------------------------------------------------
# TestIFS_EDAS
# ---------------------------------------------------------------------------

class TestIFS_EDAS:
    def test_ranks_complete(self):
        dm = make_ifs_matrix(4, 3)
        result = IFS_EDAS().calculate(dm, equal_weights(dm.criteria))
        assert set(result.ranks.values) == {1, 2, 3, 4}

    def test_AS_in_0_1(self):
        dm = make_ifs_matrix(5, 4)
        result = IFS_EDAS().calculate(dm, equal_weights(dm.criteria))
        assert (result.AS >= -1e-9).all()
        assert (result.AS <= 1.0 + 1e-9).all()

    def test_PDA_non_negative(self):
        dm = make_ifs_matrix(4, 3)
        result = IFS_EDAS().calculate(dm, equal_weights(dm.criteria))
        assert (result.PDA >= -1e-9).all().all()

    def test_NDA_non_negative(self):
        dm = make_ifs_matrix(4, 3)
        result = IFS_EDAS().calculate(dm, equal_weights(dm.criteria))
        assert (result.NDA >= -1e-9).all().all()

    def test_best_has_rank_one(self):
        dm = make_ifs_matrix(4, 3)
        result = IFS_EDAS().calculate(dm, equal_weights(dm.criteria))
        best_alt = result.AS.idxmax()
        assert result.ranks[best_alt] == 1

    def test_average_solution_length(self):
        dm = make_ifs_matrix(4, 3)
        result = IFS_EDAS().calculate(dm, equal_weights(dm.criteria))
        assert len(result.average_solution) == len(dm.criteria)


# ---------------------------------------------------------------------------
# TestIFS_COPRAS
# ---------------------------------------------------------------------------

class TestIFS_COPRAS:
    def test_ranks_complete(self):
        dm = make_ifs_matrix(4, 3)
        result = IFS_COPRAS().calculate(dm, equal_weights(dm.criteria))
        assert set(result.ranks.values) == {1, 2, 3, 4}

    def test_utility_degree_max_is_100(self):
        dm = make_ifs_matrix(4, 3)
        result = IFS_COPRAS().calculate(dm, equal_weights(dm.criteria))
        assert abs(result.utility_degree.max() - 100.0) < 1e-6

    def test_utility_degree_max_is_100_relative(self):
        """Max utility degree normalized → best utility ≥ min utility."""
        dm = make_ifs_matrix(5, 3)
        result = IFS_COPRAS().calculate(dm, equal_weights(dm.criteria))
        assert result.utility_degree.max() >= result.utility_degree.min()

    def test_best_has_rank_one(self):
        dm = make_ifs_matrix(4, 3)
        result = IFS_COPRAS().calculate(dm, equal_weights(dm.criteria))
        best_alt = result.utility_degree.idxmax()
        assert result.ranks[best_alt] == 1

    def test_s_plus_non_negative_benefit_only(self):
        """When no cost criteria, S- should be zero (or very small)."""
        dm = make_ifs_matrix(4, 3)
        result = IFS_COPRAS(cost_criteria=[]).calculate(
            dm, equal_weights(dm.criteria)
        )
        # All benefit → S_minus should be ~0
        assert (result.S_minus >= -1e-9).all()

    def test_returns_q_series(self):
        dm = make_ifs_matrix(4, 3)
        result = IFS_COPRAS().calculate(dm, equal_weights(dm.criteria))
        assert isinstance(result.Q, pd.Series)


# ---------------------------------------------------------------------------
# TestIFS_PROMETHEE
# ---------------------------------------------------------------------------

class TestIFS_PROMETHEE:
    def test_ranks_complete(self):
        dm = make_ifs_matrix(4, 3)
        result = IFS_PROMETHEE().calculate(dm, equal_weights(dm.criteria))
        assert set(result.ranks.values) == {1, 2, 3, 4}

    def test_net_flow_sums_to_zero(self):
        dm = make_ifs_matrix(4, 3)
        result = IFS_PROMETHEE().calculate(dm, equal_weights(dm.criteria))
        assert abs(result.phi_net.sum()) < 1e-9

    def test_phi_positive_non_negative(self):
        dm = make_ifs_matrix(4, 3)
        result = IFS_PROMETHEE().calculate(dm, equal_weights(dm.criteria))
        assert (result.phi_positive >= -1e-9).all()

    def test_phi_negative_non_negative(self):
        dm = make_ifs_matrix(4, 3)
        result = IFS_PROMETHEE().calculate(dm, equal_weights(dm.criteria))
        assert (result.phi_negative >= -1e-9).all()

    def test_best_has_max_net_flow(self):
        dm = make_ifs_matrix(4, 3)
        result = IFS_PROMETHEE().calculate(dm, equal_weights(dm.criteria))
        best_alt = result.ranks.idxmin()
        assert result.phi_net[best_alt] == result.phi_net.max()

    def test_preference_matrix_shape(self):
        n = 4
        dm = make_ifs_matrix(n, 3)
        result = IFS_PROMETHEE().calculate(dm, equal_weights(dm.criteria))
        assert result.preference_matrix.shape == (n, n)

    def test_phi_net_equals_phi_plus_minus_phi_neg(self):
        dm = make_ifs_matrix(4, 3)
        r = IFS_PROMETHEE().calculate(dm, equal_weights(dm.criteria))
        np.testing.assert_allclose(
            r.phi_net.values,
            (r.phi_positive - r.phi_negative).values,
            atol=1e-9,
        )
