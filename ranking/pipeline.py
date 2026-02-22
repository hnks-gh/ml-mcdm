# -*- coding: utf-8 -*-
"""
Hierarchical Ranking Pipeline
==============================

Two-stage ranking system that runs 12 MCDM methods (6 traditional +
6 IFS) within each criterion group, then aggregates results using
Evidential Reasoning.

Architecture
------------
Stage 1 — Within-Criterion Ranking
    For each criterion C_k (k = 1…8):
        • Extract subcriteria data for C_k.
        • Build crisp decision matrix → run 6 traditional methods.
        • Build IFS decision matrix (temporal variance) → run 6 IFS methods.
        • Normalize all 12 method scores to [0, 1].

Stage 2 — Global Aggregation via Evidential Reasoning
    • Convert method scores to belief distributions (5 grades).
    • Stage 1 ER: combine 12 methods per criterion.
    • Stage 2 ER: combine 8 criterion beliefs with criterion weights.
    • Final ranking from average utility of fused belief.

References
----------
[1] Atanassov, K.T. (1986). Intuitionistic Fuzzy Sets.
[2] Yang, J.B. & Xu, D.L. (2002). Evidential Reasoning algorithm.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging

from ..data_loader import PanelData, HierarchyMapping
from ..mcdm.traditional import (
    TOPSISCalculator, VIKORCalculator, PROMETHEECalculator,
    COPRASCalculator, EDASCalculator,
)
from ..mcdm.traditional.saw import SAWCalculator
from ..mcdm.ifs import (
    IFN, IFSDecisionMatrix,
    IFS_SAW, IFS_TOPSIS, IFS_VIKOR,
    IFS_PROMETHEE, IFS_COPRAS, IFS_EDAS,
)
from ..evidential_reasoning import (
    HierarchicalEvidentialReasoning, HierarchicalERResult,
    BeliefDistribution, EvidentialReasoningEngine,
)

logger = logging.getLogger('ml_mcdm')


# =========================================================================
# Result container
# =========================================================================

@dataclass
class HierarchicalRankingResult:
    """
    Result container for the full hierarchical ranking pipeline.

    Attributes
    ----------
    er_result : HierarchicalERResult
        Full ER aggregation output (rankings, beliefs, uncertainty).
    criterion_method_scores : dict
        {criterion: {method: pd.Series}} — raw normalized scores.
    criterion_method_ranks : dict
        {criterion: {method: pd.Series}} — per-method ranks.
    criterion_weights_used : dict
        {criterion: float} — weights fed to Stage 2.
    subcriteria_weights_used : dict
        {criterion: {subcrit: float}} — subcriteria weights within each group.
    ifs_diagnostics : dict
        {criterion: {alt: {subcrit: IFN}}} — sample IFS values for inspection.
    methods_used : list
        Names of the 12 MCDM methods.
    target_year : int
        Year for which ranking was computed.
    """

    er_result: HierarchicalERResult
    criterion_method_scores: Dict[str, Dict[str, pd.Series]]
    criterion_method_ranks: Dict[str, Dict[str, pd.Series]]
    criterion_weights_used: Dict[str, float]
    subcriteria_weights_used: Dict[str, Dict[str, float]]
    ifs_diagnostics: Dict[str, Any]
    methods_used: List[str]
    target_year: int

    # Convenience delegation to ER result
    @property
    def final_ranking(self) -> pd.Series:
        return self.er_result.final_ranking

    @property
    def final_scores(self) -> pd.Series:
        return self.er_result.final_scores

    @property
    def kendall_w(self) -> float:
        return self.er_result.kendall_w

    def top_n(self, n: int = 10) -> pd.DataFrame:
        return self.er_result.top_n(n)

    def summary(self) -> str:
        return self.er_result.summary()


# =========================================================================
# Pipeline
# =========================================================================

class HierarchicalRankingPipeline:
    """
    Orchestrates two-stage hierarchical ranking
    (12 MCDM methods + Evidential Reasoning).

    Parameters
    ----------
    n_grades : int
        Number of ER evaluation grades (default 5).
    method_weight_scheme : str
        How to weight method contributions in Stage 1 ER:
        ``'equal'`` or ``'rank_performance'``.
    ifs_spread_factor : float
        Multiplier on temporal σ for IFS hesitancy construction.
    cost_criteria : list, optional
        Subcriteria codes where lower values are preferred.
    """

    TRADITIONAL_METHODS = ['TOPSIS', 'VIKOR', 'PROMETHEE', 'COPRAS', 'EDAS', 'SAW']
    IFS_METHODS = ['IFS_TOPSIS', 'IFS_VIKOR', 'IFS_PROMETHEE',
                   'IFS_COPRAS', 'IFS_EDAS', 'IFS_SAW']
    ALL_METHODS = TRADITIONAL_METHODS + IFS_METHODS

    def __init__(
        self,
        n_grades: int = 5,
        method_weight_scheme: str = 'equal',
        ifs_spread_factor: float = 1.0,
        cost_criteria: Optional[List[str]] = None,
    ):
        self.n_grades = n_grades
        self.method_weight_scheme = method_weight_scheme
        self.ifs_spread_factor = ifs_spread_factor
        self.cost_criteria = cost_criteria or []

        # Initialise ER aggregator
        self.er_aggregator = HierarchicalEvidentialReasoning(
            n_grades=n_grades,
            method_weight_scheme=method_weight_scheme,
        )

    # ==================================================================
    # Public API
    # ==================================================================

    def rank(
        self,
        panel_data: PanelData,
        subcriteria_weights: Dict[str, float],
        target_year: Optional[int] = None,
        ifs_overrides: Optional[Dict[str, 'IFSDecisionMatrix']] = None,
    ) -> HierarchicalRankingResult:
        """
        Execute the full two-stage hierarchical ranking.

        Parameters
        ----------
        panel_data : PanelData
            Hierarchical panel dataset.
        subcriteria_weights : dict
            {SC01: 0.035, SC02: 0.041, …} — fused subcriteria weights.
        target_year : int, optional
            Year to rank (default: latest year).
        ifs_overrides : dict, optional
            {criterion_id: IFSDecisionMatrix} — pre-built (e.g. perturbed)
            IFS matrices to use *instead* of constructing from temporal
            variance.  Used by sensitivity analysis.

        Returns
        -------
        HierarchicalRankingResult
        """
        if target_year is None:
            target_year = max(panel_data.years)

        hierarchy = panel_data.hierarchy
        alternatives = panel_data.provinces

        logger.info(f"Hierarchical ranking for year {target_year}")
        logger.info(f"  {len(alternatives)} alternatives, "
                    f"{len(hierarchy.all_criteria)} criteria groups, "
                    f"12 MCDM methods")

        # ------------------------------------------------------------------
        # Derive criterion-level weights (sum subcriteria weights per group)
        # ------------------------------------------------------------------
        criterion_weights, group_subcrit_weights = self._derive_hierarchical_weights(
            subcriteria_weights, hierarchy
        )

        # ------------------------------------------------------------------
        # Prepare data
        # ------------------------------------------------------------------
        current_data = panel_data.subcriteria_cross_section[target_year]
        historical_std = self._compute_historical_std(panel_data)
        global_range = self._compute_global_range(panel_data)

        # ------------------------------------------------------------------
        # Stage 1: Run 12 methods per criterion
        # ------------------------------------------------------------------
        all_method_scores: Dict[str, Dict[str, pd.Series]] = {}
        all_method_ranks: Dict[str, Dict[str, pd.Series]] = {}
        ifs_diagnostics: Dict[str, Any] = {}

        for crit_id in sorted(hierarchy.all_criteria):
            subcrit_cols = hierarchy.criteria_to_subcriteria.get(crit_id, [])
            subcrit_cols = [sc for sc in subcrit_cols if sc in current_data.columns]

            if len(subcrit_cols) == 0:
                logger.warning(f"  {crit_id}: no subcriteria available, skipping")
                continue

            logger.info(f"  {crit_id}: {len(subcrit_cols)} subcriteria "
                       f"({', '.join(subcrit_cols)})")

            # Subcriteria-level weights for this group
            local_weights = group_subcrit_weights.get(crit_id, {})

            # Extract data slices
            df_crit = current_data[subcrit_cols].copy()
            std_crit = historical_std[subcrit_cols].copy() if all(
                sc in historical_std.columns for sc in subcrit_cols
            ) else pd.DataFrame(0.0, index=alternatives, columns=subcrit_cols)
            range_crit = global_range[subcrit_cols] if all(
                sc in global_range.index for sc in subcrit_cols
            ) else pd.Series(1.0, index=subcrit_cols)

            # Run traditional + IFS methods
            ifs_override = (ifs_overrides or {}).get(crit_id)
            crit_scores, crit_ranks, ifs_diag = self._run_methods_for_criterion(
                df_crit, std_crit, range_crit, local_weights, alternatives,
                ifs_matrix_override=ifs_override,
            )

            all_method_scores[crit_id] = crit_scores
            all_method_ranks[crit_id] = crit_ranks
            ifs_diagnostics[crit_id] = ifs_diag

        # ------------------------------------------------------------------
        # Stage 2: ER aggregation
        # ------------------------------------------------------------------
        logger.info("  Running Evidential Reasoning aggregation...")
        er_result = self.er_aggregator.aggregate(
            method_scores=all_method_scores,
            criterion_weights=criterion_weights,
            alternatives=alternatives,
        )

        logger.info(f"  Kendall's W = {er_result.kendall_w:.4f}")
        logger.info(f"  Top: {er_result.final_ranking.idxmin()} "
                    f"(score={er_result.final_scores.max():.4f})")

        return HierarchicalRankingResult(
            er_result=er_result,
            criterion_method_scores=all_method_scores,
            criterion_method_ranks=all_method_ranks,
            criterion_weights_used=criterion_weights,
            subcriteria_weights_used=group_subcrit_weights,
            ifs_diagnostics=ifs_diagnostics,
            methods_used=self.ALL_METHODS,
            target_year=target_year,
        )

    # ==================================================================
    # Internal: method execution per criterion
    # ==================================================================

    def _run_methods_for_criterion(
        self,
        df: pd.DataFrame,
        historical_std: pd.DataFrame,
        global_range: pd.Series,
        subcrit_weights: Dict[str, float],
        alternatives: List[str],
        ifs_matrix_override: Optional['IFSDecisionMatrix'] = None,
    ) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series], Dict]:
        """
        Run 12 MCDM methods on subcriteria data for one criterion group.

        Returns
        -------
        scores : {method_name: pd.Series}  (normalized to [0, 1])
        ranks  : {method_name: pd.Series}
        ifs_diag : sample IFS diagnostics
        """
        original_criteria = df.columns.tolist()
        original_alternatives = alternatives
        
        # ===== ADAPTIVE ZERO HANDLING =====
        # Step 1: Exclude provinces (alternatives) with all-zero data
        row_sums = df.sum(axis=1)
        valid_rows = row_sums > 0
        df = df[valid_rows].copy()
        alternatives = [alt for alt, valid in zip(alternatives, valid_rows) if valid]
        
        # Step 2: Exclude subcriteria with all-zero values
        col_sums = df.sum(axis=0)
        valid_cols = col_sums > 0
        df = df.loc[:, valid_cols].copy()
        
        excluded_alternatives = [alt for alt, valid in zip(original_alternatives, valid_rows) if not valid]
        excluded_criteria = [c for c, valid in zip(original_criteria, valid_cols) if not valid]
        
        if len(excluded_alternatives) > 0:
            logger.info(f"    → Excluded {len(excluded_alternatives)} provinces with zero data: "
                       f"{', '.join(excluded_alternatives[:5])}{'...' if len(excluded_alternatives) > 5 else ''}")
        if len(excluded_criteria) > 0:
            logger.info(f"    → Excluded {len(excluded_criteria)} zero-valued subcriteria: "
                       f"{', '.join(excluded_criteria)}")
        
        # Update related data structures
        if len(df) < 2 or len(df.columns) < 1:
            logger.warning(f"    → Insufficient data after zero filtering: "
                          f"{len(df)} provinces, {len(df.columns)} subcriteria. Skipping.")
            # Return empty results for all original alternatives
            empty_scores = {}
            empty_ranks = {}
            for method in self.ALL_METHODS:
                empty_scores[method] = pd.Series(0.5, index=original_alternatives, name=method)
                empty_ranks[method] = pd.Series(
                    list(range(1, len(original_alternatives) + 1)), 
                    index=original_alternatives, 
                    name=f"{method}_Rank"
                )
            return empty_scores, empty_ranks, {}
        
        historical_std = historical_std.loc[df.index, df.columns]
        global_range = global_range[df.columns]
        subcrit_weights = {c: subcrit_weights.get(c, 1.0 / len(df.columns)) for c in df.columns}
        
        criteria = df.columns.tolist()
        cost_local = [c for c in criteria if c in self.cost_criteria]

        scores: Dict[str, pd.Series] = {}
        ranks: Dict[str, pd.Series] = {}

        # ----- Normalize crisp data to [0, 1] via min-max -----
        df_norm = self._minmax_normalize(df, cost_criteria=cost_local)

        # ----- Build IFS matrix from temporal variance (or use override) -----
        if ifs_matrix_override is not None:
            ifs_matrix = ifs_matrix_override
        else:
            ifs_matrix = IFSDecisionMatrix.from_temporal_variance(
                current_data=df_norm,
                historical_std=historical_std,
                global_range=global_range,
                spread_factor=self.ifs_spread_factor,
            )

        # Sample diagnostics (first 3 alternatives × first 2 subcriteria)
        ifs_diag = {}
        for alt in alternatives[:3]:
            ifs_diag[alt] = {}
            for crit in criteria[:2]:
                if alt in ifs_matrix.matrix and crit in ifs_matrix.matrix[alt]:
                    ifn = ifs_matrix.get(alt, crit)
                    ifs_diag[alt][crit] = {
                        'mu': ifn.mu, 'nu': ifn.nu, 'pi': ifn.pi
                    }

        # ===== TRADITIONAL METHODS =====
        trad_results = self._run_traditional(df_norm, subcrit_weights, cost_local)
        for name, res in trad_results.items():
            s = self._normalize_scores(res['scores'], higher_is_better=res['higher_better'])
            scores[name] = s
            ranks[name] = res['ranks']

        # ===== IFS METHODS =====
        ifs_results = self._run_ifs(ifs_matrix, subcrit_weights, cost_local)
        for name, res in ifs_results.items():
            s = self._normalize_scores(res['scores'], higher_is_better=res['higher_better'])
            scores[name] = s
            ranks[name] = res['ranks']
        
        # ===== RESTORE EXCLUDED ALTERNATIVES =====
        # Fill excluded provinces with median scores/ranks to avoid bias
        if len(excluded_alternatives) > 0:
            for method_name in scores.keys():
                # Use median score for excluded provinces
                median_score = scores[method_name].median()
                for excluded_alt in excluded_alternatives:
                    scores[method_name].loc[excluded_alt] = median_score
                
                # Assign median rank for excluded provinces
                median_rank = int(np.ceil(len(original_alternatives) / 2))
                for excluded_alt in excluded_alternatives:
                    ranks[method_name].loc[excluded_alt] = median_rank
            
            # Reindex to original alternatives order
            for method_name in scores.keys():
                scores[method_name] = scores[method_name].reindex(original_alternatives)
                ranks[method_name] = ranks[method_name].reindex(original_alternatives)

        return scores, ranks, ifs_diag

    # ------------------------------------------------------------------
    # Traditional method runners
    # ------------------------------------------------------------------

    def _run_traditional(
        self,
        df: pd.DataFrame,
        weights: Dict[str, float],
        cost_criteria: List[str],
    ) -> Dict[str, Dict]:
        """Run 6 traditional MCDM methods. Returns raw scores + ranks."""
        results = {}

        # TOPSIS
        try:
            topsis = TOPSISCalculator(normalization='vector',
                                       cost_criteria=cost_criteria)
            r = topsis.calculate(df, weights)
            results['TOPSIS'] = {
                'scores': r.scores, 'ranks': r.ranks, 'higher_better': True
            }
        except Exception as e:
            logger.warning(f"    TOPSIS failed: {e}")

        # VIKOR
        try:
            vikor = VIKORCalculator(v=0.5, cost_criteria=cost_criteria)
            r = vikor.calculate(df, weights)
            # VIKOR Q: lower is better
            results['VIKOR'] = {
                'scores': r.Q, 'ranks': r.final_ranks, 'higher_better': False
            }
        except Exception as e:
            logger.warning(f"    VIKOR failed: {e}")

        # PROMETHEE
        try:
            promethee = PROMETHEECalculator(
                preference_function='vshape',
                preference_threshold=0.3,
                indifference_threshold=0.1,
                cost_criteria=cost_criteria
            )
            r = promethee.calculate(df, weights)
            results['PROMETHEE'] = {
                'scores': r.phi_net, 'ranks': r.ranks_promethee_ii,
                'higher_better': True
            }
        except Exception as e:
            logger.warning(f"    PROMETHEE failed: {e}")

        # COPRAS
        try:
            copras = COPRASCalculator(cost_criteria=cost_criteria)
            r = copras.calculate(df, weights)
            results['COPRAS'] = {
                'scores': r.utility_degree, 'ranks': r.ranks,
                'higher_better': True
            }
        except Exception as e:
            logger.warning(f"    COPRAS failed: {e}")

        # EDAS
        try:
            edas = EDASCalculator(cost_criteria=cost_criteria)
            r = edas.calculate(df, weights)
            results['EDAS'] = {
                'scores': r.AS, 'ranks': r.ranks, 'higher_better': True
            }
        except Exception as e:
            logger.warning(f"    EDAS failed: {e}")

        # SAW
        try:
            saw = SAWCalculator(normalization='minmax',
                                cost_criteria=cost_criteria)
            r = saw.calculate(df, weights)
            results['SAW'] = {
                'scores': r.scores, 'ranks': r.ranks, 'higher_better': True
            }
        except Exception as e:
            logger.warning(f"    SAW failed: {e}")

        return results

    # ------------------------------------------------------------------
    # IFS method runners
    # ------------------------------------------------------------------

    def _run_ifs(
        self,
        ifs_matrix: IFSDecisionMatrix,
        weights: Dict[str, float],
        cost_criteria: List[str],
    ) -> Dict[str, Dict]:
        """Run 6 IFS-MCDM methods. Returns raw scores + ranks."""
        results = {}

        # IFS-TOPSIS
        try:
            calc = IFS_TOPSIS(distance_metric='normalized_euclidean',
                              cost_criteria=cost_criteria)
            r = calc.calculate(ifs_matrix, weights)
            results['IFS_TOPSIS'] = {
                'scores': r.scores, 'ranks': r.ranks, 'higher_better': True
            }
        except Exception as e:
            logger.warning(f"    IFS_TOPSIS failed: {e}")

        # IFS-VIKOR
        try:
            calc = IFS_VIKOR(v=0.5, cost_criteria=cost_criteria)
            r = calc.calculate(ifs_matrix, weights)
            results['IFS_VIKOR'] = {
                'scores': r.Q, 'ranks': r.ranks_Q, 'higher_better': False
            }
        except Exception as e:
            logger.warning(f"    IFS_VIKOR failed: {e}")

        # IFS-PROMETHEE
        try:
            calc = IFS_PROMETHEE(preference_threshold=0.3,
                                 cost_criteria=cost_criteria)
            r = calc.calculate(ifs_matrix, weights)
            results['IFS_PROMETHEE'] = {
                'scores': r.phi_net, 'ranks': r.ranks, 'higher_better': True
            }
        except Exception as e:
            logger.warning(f"    IFS_PROMETHEE failed: {e}")

        # IFS-COPRAS
        try:
            calc = IFS_COPRAS(cost_criteria=cost_criteria)
            r = calc.calculate(ifs_matrix, weights)
            results['IFS_COPRAS'] = {
                'scores': r.utility_degree, 'ranks': r.ranks,
                'higher_better': True
            }
        except Exception as e:
            logger.warning(f"    IFS_COPRAS failed: {e}")

        # IFS-EDAS
        try:
            calc = IFS_EDAS(cost_criteria=cost_criteria)
            r = calc.calculate(ifs_matrix, weights)
            results['IFS_EDAS'] = {
                'scores': r.AS, 'ranks': r.ranks, 'higher_better': True
            }
        except Exception as e:
            logger.warning(f"    IFS_EDAS failed: {e}")

        # IFS-SAW
        try:
            calc = IFS_SAW()
            r = calc.calculate(ifs_matrix, weights)
            results['IFS_SAW'] = {
                'scores': r.scores, 'ranks': r.ranks, 'higher_better': True
            }
        except Exception as e:
            logger.warning(f"    IFS_SAW failed: {e}")

        return results

    # ==================================================================
    # Weight derivation
    # ==================================================================

    def _derive_hierarchical_weights(
        self,
        subcriteria_weights: Dict[str, float],
        hierarchy: HierarchyMapping,
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        Derive criterion-level and within-group subcriteria weights.

        Criterion weight = sum of subcriteria weights in the group.
        Within-group weights = subcriteria weight / criterion weight.

        Parameters
        ----------
        subcriteria_weights : dict
            {SC01: w, SC02: w, …}
        hierarchy : HierarchyMapping

        Returns
        -------
        criterion_weights : {C01: w, …}
        group_weights : {C01: {SC01: w_local, …}, …}
        """
        criterion_weights = {}
        group_weights = {}

        for crit_id, subcrit_list in hierarchy.criteria_to_subcriteria.items():
            group_w = sum(subcriteria_weights.get(sc, 0.0) for sc in subcrit_list)
            criterion_weights[crit_id] = group_w

            # Normalize within-group
            local = {}
            for sc in subcrit_list:
                w_sc = subcriteria_weights.get(sc, 0.0)
                local[sc] = w_sc / group_w if group_w > 0 else 1.0 / len(subcrit_list)
            group_weights[crit_id] = local

        # Normalize criterion weights to sum to 1
        total = sum(criterion_weights.values())
        if total > 0:
            criterion_weights = {k: v / total for k, v in criterion_weights.items()}

        logger.info(f"  Criterion weights: " +
                   ", ".join(f"{k}={v:.3f}" for k, v in sorted(criterion_weights.items())))

        return criterion_weights, group_weights

    # ==================================================================
    # Data preparation helpers
    # ==================================================================

    def _compute_historical_std(self, panel_data: PanelData) -> pd.DataFrame:
        """Compute per-province, per-subcriterion std across years."""
        frames = []
        for year in panel_data.years:
            df = panel_data.subcriteria_cross_section[year]
            df = df.copy()
            df['_year'] = year
            frames.append(df)

        all_data = pd.concat(frames)
        subcrit_cols = [c for c in all_data.columns if c != '_year']
        return all_data.groupby(all_data.index)[subcrit_cols].std().fillna(0.0)

    def _compute_global_range(self, panel_data: PanelData) -> pd.Series:
        """Compute per-subcriterion range across all years and provinces."""
        frames = []
        for year in panel_data.years:
            frames.append(panel_data.subcriteria_cross_section[year])

        all_data = pd.concat(frames)
        subcrit_cols = all_data.columns.tolist()
        return all_data[subcrit_cols].max() - all_data[subcrit_cols].min()

    @staticmethod
    def _minmax_normalize(
        df: pd.DataFrame,
        cost_criteria: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Min-max normalize to [0, 1]. Cost criteria are inverted."""
        result = df.copy().astype(float)
        cost_criteria = cost_criteria or []

        for col in result.columns:
            col_min = result[col].min()
            col_max = result[col].max()
            rng = col_max - col_min

            if rng < 1e-12:
                result[col] = 0.5  # constant column
            elif col in cost_criteria:
                result[col] = (col_max - result[col]) / rng
            else:
                result[col] = (result[col] - col_min) / rng

        return result

    @staticmethod
    def _normalize_scores(
        scores: pd.Series,
        higher_is_better: bool = True,
    ) -> pd.Series:
        """Normalize a score Series to [0, 1] (1 = best)."""
        s = scores.astype(float)
        if not higher_is_better:
            s = -s  # invert so higher = better

        s_min = s.min()
        s_max = s.max()
        rng = s_max - s_min

        if rng < 1e-12:
            return pd.Series(0.5, index=scores.index, name=scores.name)

        normalized = (s - s_min) / rng
        normalized.name = scores.name
        return normalized
