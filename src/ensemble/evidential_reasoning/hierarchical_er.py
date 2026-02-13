# -*- coding: utf-8 -*-
"""
Hierarchical Evidential Reasoning for MCDM Ensemble
=====================================================

Two-stage ER aggregation that respects the hierarchical criterion
structure (28 subcriteria → 8 criteria → final score).

Architecture
------------
Stage 1 — **Within-Criterion Aggregation**
    For each criterion C_k  (k = 1 … 8):
        • Collect 12 MCDM method scores (6 traditional + 6 IFS).
        • Convert each method score to a belief distribution.
        • Combine 12 beliefs using ER with equal method weights.
        → Produces one criterion-level belief per province.

Stage 2 — **Global Aggregation**
    For each province:
        • Input: 8 criterion-level beliefs.
        • Combine using ER with criterion weights (from GTWC phase).
        → Final belief distribution  →  expected utility  → ranking.

References
----------
[1] Yang, J.B. & Xu, D.L. (2002). IEEE Trans. SMC – Part A, 32(3).
[2] Xu, D.L. et al. (2006). "Intelligent decision system." EJOR 170(1).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from .base import BeliefDistribution, EvidentialReasoningEngine


@dataclass
class HierarchicalERResult:
    """
    Result container for two-stage ER ranking.

    Attributes
    ----------
    final_ranking : pd.Series
        1-based ranks  (1 = best).
    final_scores : pd.Series
        Average-utility scores from the fused belief.
    final_beliefs : Dict[str, BeliefDistribution]
        Province → fused belief distribution.
    criterion_beliefs : Dict[str, Dict[str, BeliefDistribution]]
        Province → criterion → belief distribution.
    method_rankings : Dict[str, Dict[str, pd.Series]]
        Criterion → method → ranking Series.
    uncertainty : pd.DataFrame
        Province × {belief_entropy, utility_interval_width}.
    kendall_w : float
        Agreement among the 12 base methods.
    """

    final_ranking: pd.Series
    final_scores: pd.Series
    final_beliefs: Dict[str, BeliefDistribution]
    criterion_beliefs: Dict[str, Dict[str, BeliefDistribution]]
    method_rankings: Dict[str, Dict[str, pd.Series]]
    method_scores: Dict[str, Dict[str, pd.Series]]
    uncertainty: pd.DataFrame
    kendall_w: float

    def top_n(self, n: int = 10) -> pd.DataFrame:
        return pd.DataFrame({
            'Score': self.final_scores,
            'Rank': self.final_ranking,
        }).nsmallest(n, 'Rank')

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            "HIERARCHICAL EVIDENTIAL REASONING RESULTS",
            f"{'='*60}",
            f"Alternatives: {len(self.final_ranking)}",
            f"Criteria groups: {len(next(iter(self.criterion_beliefs.values())))}",
            f"Kendall's W: {self.kendall_w:.4f}",
            f"\nTop 10:",
        ]
        top = self.top_n(10)
        for i, (idx, row) in enumerate(top.iterrows(), 1):
            lines.append(f"  {i}. {idx}: Score={row['Score']:.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)


class HierarchicalEvidentialReasoning:
    """
    Two-stage ER aggregator for hierarchical MCDM.

    Parameters
    ----------
    n_grades : int
        Number of evaluation grades (default 5).
    grade_labels : list, optional
        Custom grade labels.
    method_weight_scheme : str
        How to assign method weights in Stage 1:
        'equal'   — all 12 methods identical weight.
        'rank_performance' — weight by inverse average rank distance.
    """

    GRADE_DEFAULTS = ['Excellent', 'Good', 'Fair', 'Poor', 'Bad']

    def __init__(self,
                 n_grades: int = 5,
                 grade_labels: Optional[List[str]] = None,
                 method_weight_scheme: str = 'equal'):
        self.n_grades = n_grades
        self.grades = grade_labels or self.GRADE_DEFAULTS[:n_grades]
        self.method_weight_scheme = method_weight_scheme
        self.er_engine = EvidentialReasoningEngine(grades=self.grades)

    # ==================================================================
    # Public API
    # ==================================================================

    def aggregate(self,
                  method_scores: Dict[str, Dict[str, pd.Series]],
                  criterion_weights: Dict[str, float],
                  alternatives: List[str],
                  ) -> HierarchicalERResult:
        """
        Execute two-stage ER aggregation.

        Parameters
        ----------
        method_scores : dict
            Structure:  {criterion_id: {method_name: pd.Series(scores)}}
            Each Series is indexed by alternative labels,
            values in [0, 1] (higher → better).
        criterion_weights : dict
            {criterion_id: weight}  (positive, will be normalised).
        alternatives : list of str
            Province / alternative labels in canonical order.

        Returns
        -------
        HierarchicalERResult
        """
        criteria_ids = sorted(criterion_weights.keys())
        n_alt = len(alternatives)

        # ------------------------------------------------------------------
        # Stage 1: Within-criterion ER
        # ------------------------------------------------------------------
        criterion_beliefs: Dict[str, Dict[str, BeliefDistribution]] = {
            alt: {} for alt in alternatives
        }
        all_method_rankings: Dict[str, Dict[str, pd.Series]] = {}

        for crit in criteria_ids:
            crit_method_scores = method_scores[crit]
            method_names = list(crit_method_scores.keys())
            n_methods = len(method_names)

            # Method weights for Stage 1
            method_w = self._compute_method_weights(crit_method_scores,
                                                     alternatives)

            # Convert and combine for every province
            all_method_rankings[crit] = {}
            for m_name in method_names:
                ranks = crit_method_scores[m_name].rank(ascending=False).astype(int)
                all_method_rankings[crit][m_name] = ranks

            for alt in alternatives:
                beliefs_list: List[BeliefDistribution] = []
                w_list: List[float] = []
                for m_idx, m_name in enumerate(method_names):
                    score = float(crit_method_scores[m_name].get(alt, 0.5))
                    bd = self.er_engine.score_to_belief(score)
                    beliefs_list.append(bd)
                    w_list.append(method_w[m_idx])

                combined = self.er_engine.combine(beliefs_list,
                                                   np.array(w_list))
                criterion_beliefs[alt][crit] = combined

        # ------------------------------------------------------------------
        # Stage 2: Global ER
        # ------------------------------------------------------------------
        final_beliefs: Dict[str, BeliefDistribution] = {}
        final_scores_list: List[float] = []

        crit_w_arr = np.array([criterion_weights[c] for c in criteria_ids])

        for alt in alternatives:
            crit_beliefs = [criterion_beliefs[alt][c] for c in criteria_ids]
            global_belief = self.er_engine.combine(crit_beliefs, crit_w_arr)
            final_beliefs[alt] = global_belief
            final_scores_list.append(global_belief.average_utility())

        final_scores = pd.Series(final_scores_list, index=alternatives,
                                 name='ER_Score')
        final_ranking = final_scores.rank(ascending=False).astype(int)
        final_ranking.name = 'ER_Rank'

        # ------------------------------------------------------------------
        # Uncertainty metrics
        # ------------------------------------------------------------------
        entropy_list = []
        width_list = []
        for alt in alternatives:
            fb = final_beliefs[alt]
            entropy_list.append(fb.belief_entropy())
            lo, hi = fb.utility_interval()
            width_list.append(hi - lo)

        uncertainty = pd.DataFrame({
            'belief_entropy': entropy_list,
            'utility_interval_width': width_list,
        }, index=alternatives)

        # ------------------------------------------------------------------
        # Kendall's W (inter-method agreement across all criteria)
        # ------------------------------------------------------------------
        kw = self._compute_kendall_w(all_method_rankings, alternatives)

        return HierarchicalERResult(
            final_ranking=final_ranking,
            final_scores=final_scores,
            final_beliefs=final_beliefs,
            criterion_beliefs=criterion_beliefs,
            method_rankings=all_method_rankings,
            method_scores=method_scores,
            uncertainty=uncertainty,
            kendall_w=kw,
        )

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _compute_method_weights(self,
                                crit_method_scores: Dict[str, pd.Series],
                                alternatives: List[str],
                                ) -> np.ndarray:
        """
        Compute per-method weights for Stage 1 combination.
        """
        n_methods = len(crit_method_scores)

        if self.method_weight_scheme == 'equal':
            return np.ones(n_methods) / n_methods

        # 'rank_performance': weight by consistency (inverse CV of ranks)
        method_names = list(crit_method_scores.keys())
        ranks_matrix = np.zeros((n_methods, len(alternatives)))
        for i, m in enumerate(method_names):
            ranks_matrix[i] = crit_method_scores[m].rank(
                ascending=False).reindex(alternatives).values

        # Average rank position
        mean_ranks = ranks_matrix.mean(axis=1)
        cvs = ranks_matrix.std(axis=1) / (mean_ranks + 1e-12)
        inv_cv = 1.0 / (cvs + 1e-12)
        return inv_cv / inv_cv.sum()

    @staticmethod
    def _compute_kendall_w(method_rankings: Dict[str, Dict[str, pd.Series]],
                           alternatives: List[str]) -> float:
        """Kendall's W across all method-criterion rankings."""
        all_ranks = []
        for crit, methods in method_rankings.items():
            for m_name, ranks_s in methods.items():
                r = ranks_s.reindex(alternatives).values.astype(float)
                all_ranks.append(r)
        if len(all_ranks) < 2:
            return 1.0
        rank_mat = np.vstack(all_ranks)
        n_methods, n_alt = rank_mat.shape
        rank_sums = rank_mat.sum(axis=0)
        mean_rs = rank_sums.mean()
        S = np.sum((rank_sums - mean_rs) ** 2)
        S_max = (n_methods ** 2 * (n_alt ** 3 - n_alt)) / 12
        return S / S_max if S_max > 0 else 1.0
