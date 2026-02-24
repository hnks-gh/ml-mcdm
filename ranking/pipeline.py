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

from data_loader import PanelData, HierarchyMapping
from mcdm.traditional import (
    TOPSISCalculator, VIKORCalculator, PROMETHEECalculator,
    COPRASCalculator, EDASCalculator,
)
from mcdm.traditional.saw import SAWCalculator
from mcdm.ifs import (
    IFN, IFSDecisionMatrix,
    IFS_SAW, IFS_TOPSIS, IFS_VIKOR,
    IFS_PROMETHEE, IFS_COPRAS, IFS_EDAS,
)
from evidential_reasoning import (
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

        Dynamic exclusion: the ``YearContext`` stored inside *panel_data*
        dictates which provinces and sub-criteria are treated as *existing*
        for *target_year*.  Missing entities are completely absent from the
        ranking — they are not assigned placeholder scores.

        Parameters
        ----------
        panel_data : PanelData
            Hierarchical panel dataset (must contain ``year_contexts``).
        subcriteria_weights : dict
            {SC_code: weight} — fused sub-criteria weights from GTWC.
        target_year : int, optional
            Year to rank (default: latest year).
        ifs_overrides : dict, optional
            Pre-built IFS matrices for sensitivity analysis.

        Returns
        -------
        HierarchicalRankingResult
        """
        if target_year is None:
            target_year = max(panel_data.years)

        hierarchy   = panel_data.hierarchy

        # ------------------------------------------------------------------
        # Determine active alternatives via YearContext
        # ------------------------------------------------------------------
        ctx = panel_data.year_contexts.get(target_year)
        if ctx is not None:
            alternatives = ctx.active_provinces   # only active provinces
            logger.info(
                f"Hierarchical ranking for year {target_year} — "
                f"YearContext active: "
                f"{len(alternatives)} provinces, "
                f"{len(ctx.active_subcriteria)} sub-criteria, "
                f"{len(ctx.active_criteria)} criteria"
            )
            if ctx.excluded_provinces:
                logger.info(
                    f"  Excluded provinces ({len(ctx.excluded_provinces)}): "
                    f"{', '.join(ctx.excluded_provinces)}"
                )
            if ctx.excluded_subcriteria:
                logger.info(
                    f"  Excluded sub-criteria ({len(ctx.excluded_subcriteria)}): "
                    f"{', '.join(ctx.excluded_subcriteria)}"
                )
            # Filter subcriteria_weights to only year-active SCs so that
            # missing SCs do not inflate criterion-level weights
            active_sc_weights = {
                sc: w for sc, w in subcriteria_weights.items()
                if sc in ctx.active_subcriteria
            }
        else:
            alternatives      = panel_data.provinces
            active_sc_weights = subcriteria_weights
            logger.info(
                f"Hierarchical ranking for year {target_year} "
                f"(no YearContext — using full province list)"
            )

        logger.info(
            f"  {len(alternatives)} alternatives, "
            f"{len(hierarchy.all_criteria)} criteria groups (pre-exclusion), "
            f"12 MCDM methods"
        )

        # ------------------------------------------------------------------
        # Derive criterion-level weights from year-active SC weights
        # ------------------------------------------------------------------
        criterion_weights, group_subcrit_weights = self._derive_hierarchical_weights(
            active_sc_weights, hierarchy, ctx=ctx
        )

        # ------------------------------------------------------------------
        # Prepare shared data
        # ------------------------------------------------------------------
        current_data   = panel_data.subcriteria_cross_section[target_year]
        historical_std = self._compute_historical_std(panel_data)
        global_range   = self._compute_global_range(panel_data)

        # ------------------------------------------------------------------
        # Stage 1: Run 12 methods per criterion
        # ------------------------------------------------------------------
        all_method_scores: Dict[str, Dict[str, pd.Series]] = {}
        all_method_ranks:  Dict[str, Dict[str, pd.Series]] = {}
        ifs_diagnostics:   Dict[str, Any] = {}

        for crit_id in sorted(hierarchy.all_criteria):
            # Determine active SCs and provinces for this criterion-year
            if ctx is not None:
                subcrit_cols     = ctx.criterion_subcriteria.get(crit_id, [])
                crit_alternatives = ctx.criterion_alternatives.get(crit_id, [])
            else:
                subcrit_cols = [
                    sc for sc in hierarchy.criteria_to_subcriteria.get(crit_id, [])
                    if sc in current_data.columns
                ]
                crit_alternatives = alternatives

            if not subcrit_cols:
                logger.warning(f"  {crit_id}: no active sub-criteria — skipped")
                continue
            if not crit_alternatives:
                logger.warning(
                    f"  {crit_id}: no provinces with complete data — skipped"
                )
                continue

            logger.info(
                f"  {crit_id}: {len(crit_alternatives)} provinces, "
                f"{len(subcrit_cols)} sub-criteria "
                f"({', '.join(subcrit_cols)})"
            )

            # Obtain a clean (NaN-free) decision matrix for this criterion-year
            if ctx is not None:
                df_crit = panel_data.get_criterion_matrix(target_year, crit_id)
            else:
                df_crit = current_data[subcrit_cols].copy()
                df_crit = df_crit.loc[
                    [p for p in crit_alternatives if p in df_crit.index]
                ]

            if df_crit.empty:
                logger.warning(f"  {crit_id}: empty decision matrix — skipped")
                continue

            local_weights = group_subcrit_weights.get(crit_id, {})

            std_crit = (
                historical_std[subcrit_cols].copy()
                if all(sc in historical_std.columns for sc in subcrit_cols)
                else pd.DataFrame(
                    0.0, index=df_crit.index, columns=subcrit_cols
                )
            )
            range_crit = (
                global_range[subcrit_cols]
                if all(sc in global_range.index for sc in subcrit_cols)
                else pd.Series(1.0, index=subcrit_cols)
            )

            ifs_override = (ifs_overrides or {}).get(crit_id)
            crit_scores, crit_ranks, ifs_diag = self._run_methods_for_criterion(
                df_crit, std_crit, range_crit, local_weights,
                df_crit.index.tolist(),   # only criterion-active provinces
                ifs_matrix_override=ifs_override,
            )

            all_method_scores[crit_id] = crit_scores
            all_method_ranks[crit_id]  = crit_ranks
            ifs_diagnostics[crit_id]   = ifs_diag

        # ------------------------------------------------------------------
        # Stage 2: ER aggregation
        # ------------------------------------------------------------------
        # The ER engine uses `.get(alt, 0.5)` for alternatives absent from a
        # criterion's score Series — mapping to a uniform (max-uncertainty)
        # belief distribution.  Only `alternatives` (active provinces) are
        # ranked; fully excluded provinces never appear here.
        logger.info("  Running Evidential Reasoning aggregation...")
        er_result = self.er_aggregator.aggregate(
            method_scores=all_method_scores,
            criterion_weights=criterion_weights,
            alternatives=alternatives,
        )

        logger.info(f"  Kendall's W = {er_result.kendall_w:.4f}")
        if len(er_result.final_ranking) > 0:
            best = er_result.final_scores.idxmax()
            logger.info(
                f"  Top: {best} "
                f"(score={er_result.final_scores.max():.4f})"
            )

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

    def rank_fast(
        self,
        precomputed_scores: Dict[str, Dict[str, pd.Series]],
        subcriteria_weights: Dict[str, float],
        hierarchy: 'HierarchyMapping',
        alternatives: List[str],
    ) -> 'HierarchicalERResult':
        """
        Lightweight ER-only re-ranking using precomputed MCDM scores.

        Skips ALL 12 MCDM method computations and re-runs only the ER
        aggregation step with new criterion weights derived from
        *subcriteria_weights*.  Yields ~50-100x speedup over
        :meth:`rank` for sensitivity-analysis weight perturbations
        where the underlying data and MCDM scores are unchanged.

        Parameters
        ----------
        precomputed_scores : dict
            Structure ``{criterion_id: {method_name: pd.Series(score)}}``
            as returned by
            :attr:`HierarchicalRankingResult.criterion_method_scores`.
        subcriteria_weights : dict
            Perturbed subcriteria weights ``{SC01: w, …}``.
        hierarchy : HierarchyMapping
            Hierarchy definition used to derive criterion-level weights.
        alternatives : list of str
            Province / alternative labels in canonical order.

        Returns
        -------
        HierarchicalERResult
            ER aggregation result with the perturbed weights.
        """
        # Derive criterion-level weights inline (no info logging to avoid spam)
        criterion_weights: Dict[str, float] = {}
        for crit_id, subcrit_list in hierarchy.criteria_to_subcriteria.items():
            criterion_weights[crit_id] = sum(
                subcriteria_weights.get(sc, 0.0) for sc in subcrit_list
            )
        total = sum(criterion_weights.values())
        if total > 0:
            criterion_weights = {k: v / total for k, v in criterion_weights.items()}

        return self.er_aggregator.aggregate(
            method_scores=precomputed_scores,
            criterion_weights=criterion_weights,
            alternatives=alternatives,
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
        Run 12 MCDM methods on a **clean** criterion-level decision matrix.

        The matrix is guaranteed NaN-free by the caller (``rank()`` uses
        :meth:`PanelData.get_criterion_matrix` which applies dynamic exclusion
        via ``YearContext``).  No internal NaN filtering or province
        back-filling is performed here.

        Returns
        -------
        scores  : {method_name: pd.Series}  (normalised [0, 1])
        ranks   : {method_name: pd.Series}
        ifs_diag : diagnostics dict
        """
        criteria = df.columns.tolist()
        cost_local = [c for c in criteria if c in self.cost_criteria]

        # Guard: need at least 2 rows and 1 column
        if len(df) < 2 or len(df.columns) < 1:
            logger.warning(
                f"    Too few rows/cols ({len(df)} rows, "
                f"{len(df.columns)} cols) — returning neutral scores."
            )
            neutral_scores = {}
            neutral_ranks  = {}
            for method in self.ALL_METHODS:
                neutral_scores[method] = pd.Series(
                    0.5, index=alternatives, name=method)
                neutral_ranks[method]  = pd.Series(
                    list(range(1, len(alternatives) + 1)),
                    index=alternatives, name=f"{method}_Rank")
            return neutral_scores, neutral_ranks, {}

        # Align ancillary arrays to the (already-clean) df index/columns
        historical_std = historical_std.reindex(
            index=df.index, columns=df.columns
        ).fillna(0.0)
        global_range   = global_range.reindex(df.columns).fillna(1.0)
        subcrit_weights = {
            c: subcrit_weights.get(c, 1.0 / len(df.columns))
            for c in df.columns
        }

        scores: Dict[str, pd.Series] = {}
        ranks:  Dict[str, pd.Series] = {}

        # ----- Normalize crisp data to [0, 1] via min-max -----
        df_norm = self._minmax_normalize(df, cost_criteria=cost_local)

        # ----- Build IFS matrix from temporal variance (or override) -----
        if ifs_matrix_override is not None:
            ifs_matrix = ifs_matrix_override
        else:
            ifs_matrix = IFSDecisionMatrix.from_temporal_variance(
                current_data=df_norm,
                historical_std=historical_std,
                global_range=global_range,
                spread_factor=self.ifs_spread_factor,
            )

        # Sample diagnostics (first 3 alternatives × first 2 sub-criteria)
        ifs_diag: Dict = {}
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
            s = self._normalize_scores(
                res['scores'], higher_is_better=res['higher_better'])
            scores[name] = s
            ranks[name]  = res['ranks']

        # ===== IFS METHODS =====
        ifs_results = self._run_ifs(ifs_matrix, subcrit_weights, cost_local)
        for name, res in ifs_results.items():
            s = self._normalize_scores(
                res['scores'], higher_is_better=res['higher_better'])
            scores[name] = s
            ranks[name]  = res['ranks']

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
            calc = IFS_SAW(cost_criteria=cost_criteria)
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
        ctx=None,                       # YearContext | None
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        Derive criterion-level and within-group sub-criteria weights.

        Only sub-criteria that are *active* for the target year (per ``ctx``)
        contribute to their parent criterion's weight.  Excluded SCs are
        silently omitted so the criterion weight reflects the information
        actually available for the year.

        Criterion weight  = Σ(active SC weights in group)
        Within-group weight = SC weight / criterion weight

        Parameters
        ----------
        subcriteria_weights : dict
            {SC_code: weight} — year-filtered fused weights.
        hierarchy : HierarchyMapping
        ctx : YearContext or None
            When provided, only ``ctx.criterion_subcriteria[crit_id]`` SCs
            are included per criterion.

        Returns
        -------
        criterion_weights : {C01: w, …}   (normalised to sum 1)
        group_weights     : {C01: {SC01: w_local, …}, …}
        """
        criterion_weights: Dict[str, float] = {}
        group_weights:     Dict[str, Dict[str, float]] = {}

        for crit_id in hierarchy.all_criteria:
            # Use year-specific SC list if context available, else full list
            if ctx is not None:
                subcrit_list = ctx.criterion_subcriteria.get(crit_id, [])
            else:
                subcrit_list = hierarchy.criteria_to_subcriteria[crit_id]

            group_w = sum(subcriteria_weights.get(sc, 0.0) for sc in subcrit_list)
            criterion_weights[crit_id] = group_w

            # Normalise within-group
            local: Dict[str, float] = {}
            for sc in subcrit_list:
                w_sc = subcriteria_weights.get(sc, 0.0)
                local[sc] = w_sc / group_w if group_w > 0 else 1.0 / max(len(subcrit_list), 1)
            group_weights[crit_id] = local

        # Normalise criterion weights to sum to 1
        total = sum(criterion_weights.values())
        if total > 0:
            criterion_weights = {k: v / total for k, v in criterion_weights.items()}

        logger.info(
            "  Criterion weights: "
            + ", ".join(
                f"{k}={v:.3f}" for k, v in sorted(criterion_weights.items())
            )
        )
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
        """Min-max normalize to [0, 1].  Cost criteria are inverted.

        NaN cells (partially missing observations) are filled with 0.5 after
        normalisation so that downstream MCDM algorithms receive a complete,
        numeric matrix.  The neutral mid-point value avoids biasing the
        ranking in either direction for missing sub-criterion scores.
        """
        result = df.copy().astype(float)
        cost_criteria = cost_criteria or []

        for col in result.columns:
            col_min = result[col].min()          # pandas skips NaN by default
            col_max = result[col].max()
            rng = col_max - col_min

            if rng < 1e-12:
                result[col] = 0.5  # constant (or all-NaN) column
            elif col in cost_criteria:
                result[col] = (col_max - result[col]) / rng
            else:
                result[col] = (result[col] - col_min) / rng

        # Fill any NaN cells (partially missing rows) with the neutral 0.5
        # so all MCDM algorithms receive a fully numeric matrix.
        result = result.fillna(0.5)

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
