# -*- coding: utf-8 -*-
"""
Textbook-verified numerical tests for MCDM methods.

Every expected value is hand-computed from first principles and verified
analytically.  The same 3 × 2 decision matrix is used across all methods
so cross-method ranking agreement can also be checked.

Decision matrix (all benefit criteria):

        C1   C2
  A1     1    5
  A2     3    3
  A3     5    1

Weights:  w = (0.3, 0.7)

With these asymmetric weights favouring C2, the expected ranking is
A1 > A2 > A3 for all five methods.
"""

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Common fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def dm():
    """3-alternative × 2-criteria decision matrix (all benefit)."""
    return pd.DataFrame(
        {"C1": [1.0, 3.0, 5.0], "C2": [5.0, 3.0, 1.0]},
        index=["A1", "A2", "A3"],
    )


@pytest.fixture
def weights():
    return {"C1": 0.3, "C2": 0.7}


# ===================================================================
# TOPSIS — vector normalisation, all benefit
# ===================================================================
#
# Column norms:  ||C1|| = ||C2|| = sqrt(35)
#
# Weighted normalised values (v_ij = w_j · x_ij / sqrt(35)):
#   A1: (0.3/√35,  3.5/√35)
#   A2: (0.9/√35,  2.1/√35)
#   A3: (1.5/√35,  0.7/√35)
#
# Ideal:      A+ = (1.5/√35, 3.5/√35)
# Anti-ideal: A- = (0.3/√35, 0.7/√35)
#
# D+(A1) = 1.2/√35,   D-(A1) = 2.8/√35   →  C(A1) = 2.8/4.0 = 0.70
# D+(A2) = √2.32/√35, D-(A2) = √2.32/√35 →  C(A2) = 0.50
# D+(A3) = 2.8/√35,   D-(A3) = 1.2/√35   →  C(A3) = 1.2/4.0 = 0.30
# ===================================================================

class TestTOPSISTextbook:
    def test_closeness_coefficients(self, dm, weights):
        from ranking.topsis import TOPSISCalculator

        calc = TOPSISCalculator(normalization="vector")  # default
        res = calc.calculate(dm, weights)

        assert res.scores["A1"] == pytest.approx(0.70, abs=1e-6)
        assert res.scores["A2"] == pytest.approx(0.50, abs=1e-6)
        assert res.scores["A3"] == pytest.approx(0.30, abs=1e-6)

    def test_distances(self, dm, weights):
        from ranking.topsis import TOPSISCalculator

        calc = TOPSISCalculator(normalization="vector")
        res = calc.calculate(dm, weights)

        s35 = np.sqrt(35.0)
        assert res.d_positive["A1"] == pytest.approx(1.2 / s35, rel=1e-6)
        assert res.d_negative["A1"] == pytest.approx(2.8 / s35, rel=1e-6)
        assert res.d_positive["A3"] == pytest.approx(2.8 / s35, rel=1e-6)
        assert res.d_negative["A3"] == pytest.approx(1.2 / s35, rel=1e-6)
        # A2: D+ = D- (midpoint)
        assert res.d_positive["A2"] == pytest.approx(res.d_negative["A2"], rel=1e-6)

    def test_ranking_order(self, dm, weights):
        from ranking.topsis import TOPSISCalculator

        calc = TOPSISCalculator(normalization="vector")
        res = calc.calculate(dm, weights)

        assert res.ranks["A1"] == 1
        assert res.ranks["A2"] == 2
        assert res.ranks["A3"] == 3

    def test_ideal_solution(self, dm, weights):
        from ranking.topsis import TOPSISCalculator

        calc = TOPSISCalculator(normalization="vector")
        res = calc.calculate(dm, weights)

        s35 = np.sqrt(35.0)
        assert res.ideal_solution["C1"] == pytest.approx(1.5 / s35, rel=1e-6)
        assert res.ideal_solution["C2"] == pytest.approx(3.5 / s35, rel=1e-6)


# ===================================================================
# VIKOR — v = 0.5, all benefit
# ===================================================================
#
# f* = (5, 5),  f- = (1, 1),  range = 4 for both
#
# S(A1) = 0.3·(5-1)/4 + 0.7·(5-5)/4 = 0.3
# S(A2) = 0.3·(5-3)/4 + 0.7·(5-3)/4 = 0.5
# S(A3) = 0.3·(5-5)/4 + 0.7·(5-1)/4 = 0.7
#
# R(A1) = max(0.3·1, 0.7·0) = 0.3
# R(A2) = max(0.3·0.5, 0.7·0.5) = 0.35
# R(A3) = max(0.3·0, 0.7·1) = 0.7
#
# S*=0.3, S-=0.7, R*=0.3, R-=0.7
#
# Q(A1) = 0.5·0/0.4 + 0.5·0/0.4 = 0.0
# Q(A2) = 0.5·0.2/0.4 + 0.5·0.05/0.4 = 0.25+0.0625 = 0.3125
# Q(A3) = 0.5·1 + 0.5·1 = 1.0
#
# Ranking by Q: A1(1), A2(2), A3(3)
# DQ = 1/(3-1) = 0.5.  Q(A2)-Q(A1)=0.3125 < 0.5 → C1 fails
# C2: A1 is best by S and R → C2 holds
# Compromise set = {A1, A2}
# ===================================================================

class TestVIKORTextbook:
    def test_s_values(self, dm, weights):
        from ranking.vikor import VIKORCalculator

        calc = VIKORCalculator(v=0.5)
        res = calc.calculate(dm, weights)

        assert res.S["A1"] == pytest.approx(0.3, abs=1e-9)
        assert res.S["A2"] == pytest.approx(0.5, abs=1e-9)
        assert res.S["A3"] == pytest.approx(0.7, abs=1e-9)

    def test_r_values(self, dm, weights):
        from ranking.vikor import VIKORCalculator

        calc = VIKORCalculator(v=0.5)
        res = calc.calculate(dm, weights)

        assert res.R["A1"] == pytest.approx(0.3, abs=1e-9)
        assert res.R["A2"] == pytest.approx(0.35, abs=1e-9)
        assert res.R["A3"] == pytest.approx(0.7, abs=1e-9)

    def test_q_values(self, dm, weights):
        from ranking.vikor import VIKORCalculator

        calc = VIKORCalculator(v=0.5)
        res = calc.calculate(dm, weights)

        assert res.Q["A1"] == pytest.approx(0.0, abs=1e-9)
        assert res.Q["A2"] == pytest.approx(0.3125, abs=1e-9)
        assert res.Q["A3"] == pytest.approx(1.0, abs=1e-9)

    def test_ranking_order(self, dm, weights):
        from ranking.vikor import VIKORCalculator

        calc = VIKORCalculator(v=0.5)
        res = calc.calculate(dm, weights)

        assert res.ranks_Q["A1"] == 1
        assert res.ranks_Q["A2"] == 2
        assert res.ranks_Q["A3"] == 3

    def test_compromise_set_c1_fails(self, dm, weights):
        """C1 fails (Q(a2)-Q(a1)=0.3125 < DQ=0.5), so {A1,A2} in set."""
        from ranking.vikor import VIKORCalculator

        calc = VIKORCalculator(v=0.5)
        res = calc.calculate(dm, weights)

        assert res.advantage_condition == False  # noqa: E712 — np.bool_
        assert res.stability_condition == True   # noqa: E712
        assert set(res.compromise_set) == {"A1", "A2"}

    def test_compromise_solution_is_best(self, dm, weights):
        from ranking.vikor import VIKORCalculator

        calc = VIKORCalculator(v=0.5)
        res = calc.calculate(dm, weights)

        assert res.compromise_solution == "A1"


# ===================================================================
# EDAS — all benefit
# ===================================================================
#
# AV = (3, 3)
#
# PDA (benefit):                    NDA (benefit):
#   A1: (0,   2/3)                    A1: (2/3, 0  )
#   A2: (0,   0  )                    A2: (0,   0  )
#   A3: (2/3, 0  )                    A3: (0,   2/3)
#
# SP = [0.3·0+0.7·2/3,   0,   0.3·2/3+0.7·0] = [7/15, 0, 1/5]
# SN = [0.3·2/3+0.7·0,   0,   0.3·0+0.7·2/3] = [1/5,  0, 7/15]
#
# max(SP) = 7/15,  max(SN) = 7/15
# NSP = [1.0,  0.0,  3/7]
# NSN = [4/7,  1.0,  0.0]
#
# AS  = [(1+4/7)/2,  (0+1)/2,  (3/7+0)/2] = [11/14, 1/2, 3/14]
#      ≈ [0.78571,  0.50000,  0.21429]
#
# Ranking: A1(1), A2(2), A3(3)
# ===================================================================

class TestEDASTextbook:
    def test_average_solution(self, dm, weights):
        from ranking.edas import EDASCalculator

        calc = EDASCalculator()
        res = calc.calculate(dm, weights)

        assert res.average_solution["C1"] == pytest.approx(3.0, abs=1e-9)
        assert res.average_solution["C2"] == pytest.approx(3.0, abs=1e-9)

    def test_pda_values(self, dm, weights):
        from ranking.edas import EDASCalculator

        calc = EDASCalculator()
        res = calc.calculate(dm, weights)

        assert res.PDA.loc["A1", "C1"] == pytest.approx(0.0, abs=1e-9)
        assert res.PDA.loc["A1", "C2"] == pytest.approx(2.0 / 3, abs=1e-9)
        assert res.PDA.loc["A2", "C1"] == pytest.approx(0.0, abs=1e-9)
        assert res.PDA.loc["A2", "C2"] == pytest.approx(0.0, abs=1e-9)
        assert res.PDA.loc["A3", "C1"] == pytest.approx(2.0 / 3, abs=1e-9)
        assert res.PDA.loc["A3", "C2"] == pytest.approx(0.0, abs=1e-9)

    def test_nda_values(self, dm, weights):
        from ranking.edas import EDASCalculator

        calc = EDASCalculator()
        res = calc.calculate(dm, weights)

        assert res.NDA.loc["A1", "C1"] == pytest.approx(2.0 / 3, abs=1e-9)
        assert res.NDA.loc["A1", "C2"] == pytest.approx(0.0, abs=1e-9)
        assert res.NDA.loc["A3", "C2"] == pytest.approx(2.0 / 3, abs=1e-9)

    def test_appraisal_scores(self, dm, weights):
        from ranking.edas import EDASCalculator

        calc = EDASCalculator()
        res = calc.calculate(dm, weights)

        assert res.AS["A1"] == pytest.approx(11.0 / 14, abs=1e-5)
        assert res.AS["A2"] == pytest.approx(0.5, abs=1e-5)
        assert res.AS["A3"] == pytest.approx(3.0 / 14, abs=1e-5)

    def test_ranking_order(self, dm, weights):
        from ranking.edas import EDASCalculator

        calc = EDASCalculator()
        res = calc.calculate(dm, weights)

        assert res.ranks["A1"] == 1
        assert res.ranks["A2"] == 2
        assert res.ranks["A3"] == 3

    def test_sp_sn_values(self, dm, weights):
        from ranking.edas import EDASCalculator

        calc = EDASCalculator()
        res = calc.calculate(dm, weights)

        assert res.SP["A1"] == pytest.approx(7.0 / 15, abs=1e-9)
        assert res.SP["A2"] == pytest.approx(0.0, abs=1e-9)
        assert res.SP["A3"] == pytest.approx(1.0 / 5, abs=1e-9)

        assert res.SN["A1"] == pytest.approx(1.0 / 5, abs=1e-9)
        assert res.SN["A2"] == pytest.approx(0.0, abs=1e-9)
        assert res.SN["A3"] == pytest.approx(7.0 / 15, abs=1e-9)


# ===================================================================
# COPRAS — all benefit (no cost criteria → S- = 0)
# ===================================================================
#
# Sum-normalisation:  col_sums = (9, 9)
#   r_ij = x_ij / col_sum_j
#
# Weighted normalised:  d_ij = r_ij · w_j
#   A1: (0.3/9, 0.7·5/9)  = (1/30,  7/18)
#   A2: (0.3·3/9, 0.7·3/9) = (1/10,  7/30)
#   A3: (0.3·5/9, 0.7/9)   = (1/6,   7/90)
#
# S+ = sum of benefit weighted values per row:
#   S+(A1) = 1/30 + 7/18 = 3/90 + 35/90 = 38/90
#   S+(A2) = 1/10 + 7/30 = 3/30 + 7/30  = 10/30 = 1/3
#   S+(A3) = 1/6  + 7/90 = 15/90 + 7/90 = 22/90
#
# S- = 0  (all benefit)
# Q = S+  (when all benefit)
#
# Utility degree: N_i = (Q_i / Q_max) × 100
#   Q_max = 38/90
#   N(A1) = 100.0
#   N(A2) = (1/3)/(38/90)·100 = 30/38·100 ≈ 78.947
#   N(A3) = (22/90)/(38/90)·100 = 22/38·100 ≈ 57.895
#
# Ranking: A1(1), A2(2), A3(3)
# ===================================================================

class TestCOPRASTextbook:
    def test_s_plus_values(self, dm, weights):
        from ranking.copras import COPRASCalculator

        calc = COPRASCalculator()  # all benefit
        res = calc.calculate(dm, weights)

        assert res.S_plus["A1"] == pytest.approx(38.0 / 90, abs=1e-9)
        assert res.S_plus["A2"] == pytest.approx(1.0 / 3, abs=1e-9)
        assert res.S_plus["A3"] == pytest.approx(22.0 / 90, abs=1e-9)

    def test_s_minus_zero_for_all_benefit(self, dm, weights):
        from ranking.copras import COPRASCalculator

        calc = COPRASCalculator()
        res = calc.calculate(dm, weights)

        assert res.S_minus["A1"] == pytest.approx(0.0, abs=1e-9)
        assert res.S_minus["A2"] == pytest.approx(0.0, abs=1e-9)
        assert res.S_minus["A3"] == pytest.approx(0.0, abs=1e-9)

    def test_utility_degree(self, dm, weights):
        from ranking.copras import COPRASCalculator

        calc = COPRASCalculator()
        res = calc.calculate(dm, weights)

        assert res.utility_degree["A1"] == pytest.approx(100.0, abs=0.01)
        assert res.utility_degree["A2"] == pytest.approx(3000.0 / 38, abs=0.01)
        assert res.utility_degree["A3"] == pytest.approx(2200.0 / 38, abs=0.01)

    def test_ranking_order(self, dm, weights):
        from ranking.copras import COPRASCalculator

        calc = COPRASCalculator()
        res = calc.calculate(dm, weights)

        assert res.ranks["A1"] == 1
        assert res.ranks["A2"] == 2
        assert res.ranks["A3"] == 3

    def test_best_utility_is_100(self, dm, weights):
        from ranking.copras import COPRASCalculator

        calc = COPRASCalculator()
        res = calc.calculate(dm, weights)

        assert res.utility_degree.max() == pytest.approx(100.0, abs=0.01)


# ===================================================================
# PROMETHEE II — "usual" preference function, all benefit
# ===================================================================
#
# Min-max normalisation (benefit):
#   range = 4 for both columns
#   A1: (0.0, 1.0),  A2: (0.5, 0.5),  A3: (1.0, 0.0)
#
# Usual preference: P(d) = 1 if d > 0, else 0
#
# Pairwise aggregated preference  π(a,b) = Σ w_j·P_j(a,b):
#   π(A1,A2) = 0.3·0 + 0.7·1 = 0.7
#   π(A1,A3) = 0.3·0 + 0.7·1 = 0.7
#   π(A2,A1) = 0.3·1 + 0.7·0 = 0.3
#   π(A2,A3) = 0.3·0 + 0.7·1 = 0.7
#   π(A3,A1) = 0.3·1 + 0.7·0 = 0.3
#   π(A3,A2) = 0.3·1 + 0.7·0 = 0.3
#
# Flows (n−1 = 2):
#   Φ+(A1) = (0.7 + 0.7)/2 = 0.7
#   Φ+(A2) = (0.3 + 0.7)/2 = 0.5
#   Φ+(A3) = (0.3 + 0.3)/2 = 0.3
#
#   Φ−(A1) = (0.3 + 0.3)/2 = 0.3
#   Φ−(A2) = (0.7 + 0.3)/2 = 0.5
#   Φ−(A3) = (0.7 + 0.7)/2 = 0.7
#
#   Φ_net(A1) = 0.7 − 0.3 =  0.4
#   Φ_net(A2) = 0.5 − 0.5 =  0.0
#   Φ_net(A3) = 0.3 − 0.7 = −0.4
#
# Σ Φ_net = 0  ✓ (antisymmetry)
# Ranking: A1(1), A2(2), A3(3)
# ===================================================================

class TestPROMETHEETextbook:
    def test_phi_positive(self, dm, weights):
        from ranking.promethee import PROMETHEECalculator

        calc = PROMETHEECalculator(preference_function="usual")
        res = calc.calculate(dm, weights)

        assert res.phi_positive["A1"] == pytest.approx(0.7, abs=1e-9)
        assert res.phi_positive["A2"] == pytest.approx(0.5, abs=1e-9)
        assert res.phi_positive["A3"] == pytest.approx(0.3, abs=1e-9)

    def test_phi_negative(self, dm, weights):
        from ranking.promethee import PROMETHEECalculator

        calc = PROMETHEECalculator(preference_function="usual")
        res = calc.calculate(dm, weights)

        assert res.phi_negative["A1"] == pytest.approx(0.3, abs=1e-9)
        assert res.phi_negative["A2"] == pytest.approx(0.5, abs=1e-9)
        assert res.phi_negative["A3"] == pytest.approx(0.7, abs=1e-9)

    def test_phi_net(self, dm, weights):
        from ranking.promethee import PROMETHEECalculator

        calc = PROMETHEECalculator(preference_function="usual")
        res = calc.calculate(dm, weights)

        assert res.phi_net["A1"] == pytest.approx(0.4, abs=1e-9)
        assert res.phi_net["A2"] == pytest.approx(0.0, abs=1e-9)
        assert res.phi_net["A3"] == pytest.approx(-0.4, abs=1e-9)

    def test_net_flow_sums_to_zero(self, dm, weights):
        from ranking.promethee import PROMETHEECalculator

        calc = PROMETHEECalculator(preference_function="usual")
        res = calc.calculate(dm, weights)

        assert abs(res.phi_net.sum()) < 1e-12

    def test_ranking_order(self, dm, weights):
        from ranking.promethee import PROMETHEECalculator

        calc = PROMETHEECalculator(preference_function="usual")
        res = calc.calculate(dm, weights)

        assert res.ranks_promethee_ii["A1"] == 1
        assert res.ranks_promethee_ii["A2"] == 2
        assert res.ranks_promethee_ii["A3"] == 3

    def test_preference_matrix(self, dm, weights):
        """Verify the aggregated preference matrix π(a,b)."""
        from ranking.promethee import PROMETHEECalculator

        calc = PROMETHEECalculator(preference_function="usual")
        res = calc.calculate(dm, weights)

        pi = res.preference_matrix
        assert pi.loc["A1", "A2"] == pytest.approx(0.7, abs=1e-9)
        assert pi.loc["A1", "A3"] == pytest.approx(0.7, abs=1e-9)
        assert pi.loc["A2", "A1"] == pytest.approx(0.3, abs=1e-9)
        assert pi.loc["A2", "A3"] == pytest.approx(0.7, abs=1e-9)
        assert pi.loc["A3", "A1"] == pytest.approx(0.3, abs=1e-9)
        assert pi.loc["A3", "A2"] == pytest.approx(0.3, abs=1e-9)
        # Diagonal is zero
        assert pi.loc["A1", "A1"] == pytest.approx(0.0, abs=1e-12)


# ===================================================================
# Cross-method ranking agreement
# ===================================================================

class TestCrossMethodAgreement:
    """All five methods should agree on ranking for this simple example."""

    def test_all_methods_rank_a1_first(self, dm, weights):
        from ranking.topsis import TOPSISCalculator
        from ranking.vikor import VIKORCalculator
        from ranking.edas import EDASCalculator
        from ranking.copras import COPRASCalculator
        from ranking.promethee import PROMETHEECalculator

        topsis = TOPSISCalculator().calculate(dm, weights)
        vikor = VIKORCalculator().calculate(dm, weights)
        edas = EDASCalculator().calculate(dm, weights)
        copras = COPRASCalculator().calculate(dm, weights)
        promethee = PROMETHEECalculator(
            preference_function="usual"
        ).calculate(dm, weights)

        assert topsis.ranks["A1"] == 1
        assert vikor.ranks_Q["A1"] == 1
        assert edas.ranks["A1"] == 1
        assert copras.ranks["A1"] == 1
        assert promethee.ranks_promethee_ii["A1"] == 1

    def test_all_methods_rank_a3_last(self, dm, weights):
        from ranking.topsis import TOPSISCalculator
        from ranking.vikor import VIKORCalculator
        from ranking.edas import EDASCalculator
        from ranking.copras import COPRASCalculator
        from ranking.promethee import PROMETHEECalculator

        topsis = TOPSISCalculator().calculate(dm, weights)
        vikor = VIKORCalculator().calculate(dm, weights)
        edas = EDASCalculator().calculate(dm, weights)
        copras = COPRASCalculator().calculate(dm, weights)
        promethee = PROMETHEECalculator(
            preference_function="usual"
        ).calculate(dm, weights)

        assert topsis.ranks["A3"] == 3
        assert vikor.ranks_Q["A3"] == 3
        assert edas.ranks["A3"] == 3
        assert copras.ranks["A3"] == 3
        assert promethee.ranks_promethee_ii["A3"] == 3


# ===================================================================
# Cross-method dominance using shared conftest fixtures  (P4-29)
#
# dm3x3 / w_equal_3 are defined in tests/conftest.py.
# A1 dominates A3 in all three criteria, so every method that produces
# a complete ranking must assign rank(A1) < rank(A3).
# ===================================================================

class TestSharedFixtureCrossMethod:
    """All six MCDM methods agree: A1 ranks above A3 under equal weights."""

    def test_topsis_a1_beats_a3(self, dm3x3, w_equal_3):
        from ranking.topsis import TOPSISCalculator
        res = TOPSISCalculator().calculate(dm3x3, w_equal_3)
        assert res.ranks["A1"] < res.ranks["A3"]

    def test_vikor_a1_beats_a3(self, dm3x3, w_equal_3):
        from ranking.vikor import VIKORCalculator
        res = VIKORCalculator().calculate(dm3x3, w_equal_3)
        assert res.ranks_Q["A1"] < res.ranks_Q["A3"]

    def test_edas_a1_beats_a3(self, dm3x3, w_equal_3):
        from ranking.edas import EDASCalculator
        res = EDASCalculator().calculate(dm3x3, w_equal_3)
        assert res.ranks["A1"] < res.ranks["A3"]

    def test_saw_a1_beats_a3(self, dm3x3, w_equal_3):
        from ranking.saw import SAWCalculator
        res = SAWCalculator().calculate(dm3x3, w_equal_3)
        assert res.ranks["A1"] < res.ranks["A3"]

    def test_copras_a1_beats_a3(self, dm3x3, w_equal_3):
        from ranking.copras import COPRASCalculator
        res = COPRASCalculator().calculate(dm3x3, w_equal_3)
        assert res.ranks["A1"] < res.ranks["A3"]

    def test_promethee_a1_beats_a3(self, dm3x3, w_equal_3):
        from ranking.promethee import PROMETHEECalculator
        res = PROMETHEECalculator().calculate(dm3x3, w_equal_3)
        assert res.ranks_promethee_ii["A1"] < res.ranks_promethee_ii["A3"]
