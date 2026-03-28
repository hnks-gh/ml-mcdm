# -*- coding: utf-8 -*-
"""Hierarchical Ranking Pipeline

**Core principle**: Ranking RESPECTS missing data structure. NO IMPUTATION.

Single-stage ranking system that runs 6 traditional MCDM methods within each
criterion group, then aggregates results using criterion-weighted averaging.
All methods handle partial NaN natively on observed (non-missing) values.

Architecture
-----------
Criterion-Level Ranking (NO IMPUTATION)
    For each criterion C_k (k = 1…8):
        • Filter all-NaN rows/columns (preserve partial NaN)
        • Extract and rank on observed values only
        • Run 6 MCDM methods (TOPSIS, VIKOR, PROMETHEE II, COPRAS, EDAS, SAW)
        • Normalize scores to [0, 1]

Global Aggregation
    For each alternative:
        • Compute criterion-level composite scores (average of method scores)
        • Apply criterion weights to get final composite score
        • Rank alternatives by composite score

Missing Data Handling
---------------------
As of 2026-03-28, no imputation occurs in the ranking phase. All methods
(TOPSIS, VIKOR, PROMETHEE II, COPRAS, EDAS, SAW) handle NaN via:
    • Complete-case distance/preference computations (pairwise on observed)
    • NaN skipping in dimension-wise calculations
    • No artificial 0.5 neutral score imputation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging

from data import PanelData, HierarchyMapping
from .topsis import TOPSISCalculator
from .vikor import VIKORCalculator
from .promethee import PROMETHEECalculator
from .copras import COPRASCalculator
from .edas import EDASCalculator

logger = logging.getLogger('ml_mcdm')


# =========================================================================
# Result container
# =========================================================================

@dataclass
class HierarchicalRankingResult:
    """Container for the result of :class:`HierarchicalRankingPipeline`.

    Attributes
    ----------
    final_ranking : pd.Series
        Rank order (1 = best).
    final_scores : pd.Series
        Composite criterion-weighted mean scores.
    criterion_method_scores : dict
        {criterion: {method: pd.Series}} — raw normalized scores.
    criterion_method_ranks : dict
        {criterion: {method: pd.Series}} — per-method ranks.
    criterion_weights_used : dict
        {criterion: float} — weights applied in aggregation.
    subcriteria_weights_used : dict
        {criterion: {subcrit: float}} — subcriteria weights within each group.
    methods_used : list
        Names of all methods (TOPSIS, VIKOR, PROMETHEE, COPRAS, EDAS, SAW).
    target_year : int
        Year for which ranking was computed.
    """

    final_ranking: 'pd.Series'
    final_scores: 'pd.Series'
    criterion_method_scores: Dict[str, Dict[str, 'pd.Series']]
    criterion_method_ranks: Dict[str, Dict[str, 'pd.Series']]
    criterion_weights_used: Dict[str, float]
    subcriteria_weights_used: Dict[str, Dict[str, float]]
    methods_used: List[str]
    target_year: int

    def top_n(self, n: int = 10) -> Optional['pd.DataFrame']:
        """Return top-N ranked alternatives."""
        if self.final_scores is None or self.final_ranking is None:
            return None
        import pandas as _pd
        df = _pd.DataFrame({
            'Score': self.final_scores,
            'Rank': self.final_ranking,
        })
        return df.nsmallest(n, 'Rank')

    @property
    def kendall_w(self) -> float:
        """
        Compute Kendall's coefficient of concordance W for the 6 MCDM methods.
        
        Measures agreement among method rankings across all criteria.
        W ∈ [0, 1]: 0 = no agreement, 1 = perfect agreement.
        
        Formula: W = 12*S / [k^2*(n^3-n)]
        where:
          k = number of raters (6 MCDM methods)
          n = number of items (alternatives)
          S = sum of squared deviations from mean rank sum
        
        Returns
        -------
        float
            Kendall's W in [0, 1], or NaN if computation not possible.
        """
        try:
            if not self.criterion_method_ranks:
                return float('nan')
            
            # Collect all method ranks across all criteria
            all_ranks = {}  # method -> list of rank arrays
            
            for crit_id in sorted(self.criterion_method_ranks.keys()):
                crit_ranks = self.criterion_method_ranks[crit_id]
                for method in crit_ranks.keys():
                    ranks_series = crit_ranks[method]
                    if method not in all_ranks:
                        all_ranks[method] = []
                    all_ranks[method].append(ranks_series.values)
            
            if len(all_ranks) < 2:
                return float('nan')
            
            # Stack ranks into matrix: n_methods × n_alternatives
            methods = sorted(all_ranks.keys())
            rank_matrix = np.array([
                np.concatenate(all_ranks[m]) if all_ranks[m] else np.array([])
                for m in methods
            ])
            
            if rank_matrix.size == 0:
                return float('nan')
            
            k = rank_matrix.shape[0]  # number of methods
            n = rank_matrix.shape[1]  # number of alternatives
            
            if n < 2:
                return float('nan')
            
            # Compute sum of ranks for each alternative (row mean)
            rank_sums = np.sum(rank_matrix, axis=0)
            mean_rank_sum = np.mean(rank_sums)
            
            # Compute S: sum of squared deviations
            S = np.sum((rank_sums - mean_rank_sum) ** 2)
            
            # Compute Kendall's W
            denom = k * k * (n * n * n - n)
            if denom <= 0:
                return float('nan')
            
            W = 12 * S / denom
            return float(np.clip(W, 0, 1))
        except Exception:
            return float('nan')

    def summary(self) -> str:
        n_alts = len(self.final_scores) if self.final_scores is not None else 0
        return (
            f"Hierarchical ranking — "
            f"{n_alts} alternatives, "
            f"{len(self.criterion_method_scores)} criteria, "
            f"year {self.target_year}."
        )


# =========================================================================
# Pipeline
# =========================================================================

class HierarchicalRankingPipeline:
    """
    Orchestrates hierarchical ranking using 6 traditional MCDM methods
    (TOPSIS, VIKOR, PROMETHEE, COPRAS, EDAS, SAW).

    For each criterion, scores from all 6 methods are computed and then
    aggregated using weighted averaging. Final ranking is produced by
    applying criterion weights and ranking by composite score.

    Parameters
    ----------
    cost_criteria : list, optional
        Subcriteria codes where lower values are preferred.
    """

    # All methods used in ranking
    TRADITIONAL_METHODS = ['TOPSIS', 'VIKOR', 'PROMETHEE', 'COPRAS', 'EDAS', 'SAW']
    ALL_METHODS = TRADITIONAL_METHODS

    def __init__(
        self,
        cost_criteria: Optional[List[str]] = None,
    ):
        self.cost_criteria = cost_criteria or []

    # ==================================================================
    # Public API
    # ==================================================================

    def rank(
        self,
        panel_data: PanelData,
        subcriteria_weights: Dict[str, float],
        target_year: Optional[int] = None,
        criterion_weights: Optional[Dict[str, float]] = None,
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
            {SC_code: weight} — fused sub-criteria weights from hybrid MC ensemble.
        target_year : int, optional
            Year to rank (default: latest year).

        Returns
        -------
        HierarchicalRankingResult
        """
        if target_year is None:
            target_year = max(panel_data.years)

        hierarchy   = panel_data.hierarchy

        logger.info(
            f"[DEBUG] Ranking.rank() called: hierarchy has {len(hierarchy.all_subcriteria)} SCs, "
            f"{len(hierarchy.all_criteria)} criteria"
        )
        logger.info(
            f"[DEBUG] All SCs in hierarchy: {sorted(hierarchy.all_subcriteria)}"
        )

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
            f"6 MCDM methods"
        )

        # ------------------------------------------------------------------
        # Derive criterion-level weights from year-active SC weights
        # ------------------------------------------------------------------
        criterion_weights, group_subcrit_weights = self._derive_hierarchical_weights(
            active_sc_weights, hierarchy, ctx=ctx,
            criterion_weights_override=criterion_weights,
        )

        # ------------------------------------------------------------------
        # Prepare shared data
        # ------------------------------------------------------------------
        current_data   = panel_data.subcriteria_cross_section[target_year]

        # ------------------------------------------------------------------
        # Stage 1: Run 6 methods per criterion
        # ------------------------------------------------------------------
        all_method_scores: Dict[str, Dict[str, pd.Series]] = {}
        all_method_ranks:  Dict[str, Dict[str, pd.Series]] = {}

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
                # No YearContext: province rows with any missing SC are excluded
                # (complete-case strategy; consistent with the primary YearContext path).
                df_crit = df_crit.dropna()

            if df_crit.empty:
                logger.warning(f"  {crit_id}: empty decision matrix — skipped")
                continue

            local_weights = group_subcrit_weights.get(crit_id, {})

            crit_scores, crit_ranks = self._run_methods_for_criterion(
                df_crit, local_weights,
                df_crit.index.tolist(),
            )

            all_method_scores[crit_id] = crit_scores
            all_method_ranks[crit_id]  = crit_ranks
            logger.info(
                f"[DEBUG] {crit_id}: successfully processed "
                f"({len(df_crit)} provinces, {len(df_crit.columns)} SCs)"
            )

        logger.info(
            f"[DEBUG] Stage 1 complete: all_method_scores has {len(all_method_scores)} criteria, "
            f"all_method_ranks has {len(all_method_ranks)} criteria"
        )

        # ------------------------------------------------------------------
        # Aggregation: Criterion-weighted mean of method averages
        # ------------------------------------------------------------------
        logger.info(
            "  Computing criterion-weighted mean composite scores..."
        )
        
        # Collect all alternatives that appeared in ANY criterion's ranking
        all_alternatives_set = set()
        for crit_scores in all_method_scores.values():
            for method_scores in crit_scores.values():
                all_alternatives_set.update(method_scores.index)
        all_alternatives_list = sorted(all_alternatives_set)
        
        if not all_alternatives_list:
            logger.warning("  No alternatives found — returning empty rankings")
            final_scores = pd.Series(dtype=float, name="composite_score")
            final_ranking = pd.Series(dtype=int, name="rank")
        else:
            # Build composite: for each alternative, weighted average of criterion means
            composite_scores_dict = {}
            for alt in all_alternatives_list:
                crit_contributions = []
                for crit_id in sorted(all_method_scores.keys()):
                    crit_weight = criterion_weights.get(crit_id, 0)
                    if crit_weight == 0:
                        continue
                    # Average all method scores for this alternative-criterion
                    method_values = []
                    for method_scores in all_method_scores[crit_id].values():
                        if alt in method_scores.index:
                            method_values.append(method_scores[alt])
                    if method_values:
                        crit_mean = np.nanmean(method_values)
                        crit_contributions.append(crit_weight * crit_mean)
                
                if crit_contributions:
                    composite_scores_dict[alt] = np.sum(crit_contributions)
                else:
                    composite_scores_dict[alt] = 0.0
            
            final_scores = pd.Series(composite_scores_dict, name="composite_score")
            final_ranking = final_scores.rank(ascending=False).astype(int)
            final_ranking.name = "rank"
            
            logger.info(
                f"  Composite ranking: {len(final_ranking)} alternatives scored"
            )
            if len(final_scores) > 0:
                best = final_scores.idxmax()
                logger.info(
                    f"  Top: {best} "
                    f"(score={final_scores.max():.4f})"
                )
        
        return HierarchicalRankingResult(
            final_ranking=final_ranking,
            final_scores=final_scores,
            criterion_method_scores=all_method_scores,
            criterion_method_ranks=all_method_ranks,
            criterion_weights_used=criterion_weights,
            subcriteria_weights_used=group_subcrit_weights,
            methods_used=self.ALL_METHODS,
            target_year=target_year,
        )

    def rank_fast(
        self,
        precomputed_scores: Dict[str, Dict[str, pd.Series]],
        subcriteria_weights: Dict[str, float],
        hierarchy: 'HierarchyMapping',
        alternatives: List[str],
        criterion_weights: Optional[Dict[str, float]] = None,
    ) -> 'HierarchicalRankingResult':
        """
        Lightweight re-ranking using precomputed MCDM scores.

        Skips ALL MCDM method computations and re-runs only the aggregation
        step with new criterion weights. Yields ~50-100x speedup over :meth:`rank`
        for sensitivity-analysis weight perturbations.

        Parameters
        ----------
        precomputed_scores : dict
            {criterion: {method: pd.Series}} of precomputed MCDM scores.
        subcriteria_weights : dict
            Subcriteria weights for deriving criterion-level weights.
        hierarchy : HierarchyMapping
            Criterion hierarchy mapping.
        alternatives : list
            List of alternative names.
        criterion_weights : dict, optional
            Externally computed criterion weights. If None, derived from subcriteria_weights.

        Returns
        -------
        HierarchicalRankingResult
            Re-ranked result using new criterion weights.
        """
        # Derive criterion-level weights inline (no info logging to avoid spam)
        if criterion_weights is not None:
            # Use externally computed weights directly (from Level 2 MC ensemble)
            crit_w = {k: criterion_weights.get(k, 0.0)
                      for k in hierarchy.criteria_to_subcriteria}
            total = sum(crit_w.values())
            if total > 0:
                crit_w = {k: v / total for k, v in crit_w.items()}
        else:
            crit_w = {}
            for crit_id, subcrit_list in hierarchy.criteria_to_subcriteria.items():
                crit_w[crit_id] = sum(
                    subcriteria_weights.get(sc, 0.0) for sc in subcrit_list
                )
            total = sum(crit_w.values())
            if total > 0:
                crit_w = {k: v / total for k, v in crit_w.items()}

        # Aggregate with new weights
        all_alternatives_set = set()
        for crit_scores in precomputed_scores.values():
            for method_scores in crit_scores.values():
                all_alternatives_set.update(method_scores.index)
        all_alternatives_list = sorted(all_alternatives_set)
        
        if not all_alternatives_list:
            final_scores = pd.Series(dtype=float, name="composite_score")
            final_ranking = pd.Series(dtype=int, name="rank")
        else:
            composite_scores_dict = {}
            for alt in all_alternatives_list:
                crit_contributions = []
                for crit_id in sorted(precomputed_scores.keys()):
                    crit_weight = crit_w.get(crit_id, 0)
                    if crit_weight == 0:
                        continue
                    method_values = []
                    for method_scores in precomputed_scores[crit_id].values():
                        if alt in method_scores.index:
                            method_values.append(method_scores[alt])
                    if method_values:
                        crit_mean = np.nanmean(method_values)
                        crit_contributions.append(crit_weight * crit_mean)
                
                if crit_contributions:
                    composite_scores_dict[alt] = np.sum(crit_contributions)
                else:
                    composite_scores_dict[alt] = 0.0
            
            final_scores = pd.Series(composite_scores_dict, name="composite_score")
            final_ranking = final_scores.rank(ascending=False).astype(int)
            final_ranking.name = "rank"

        return HierarchicalRankingResult(
            final_ranking=final_ranking,
            final_scores=final_scores,
            criterion_method_scores=precomputed_scores,
            criterion_method_ranks={},  # Not recomputed in fast path
            criterion_weights_used=crit_w,
            subcriteria_weights_used={},
            methods_used=self.ALL_METHODS,
            target_year=0,
        )

    # ==================================================================
    # Internal: method execution per criterion
    # ==================================================================

    def _run_methods_for_criterion(
        self,
        df: pd.DataFrame,
        subcrit_weights: Dict[str, float],
        alternatives: List[str],
    ) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
        """
        Run 6 MCDM methods on a **clean** criterion-level decision matrix.

        The matrix is guaranteed NaN-free by the caller (``rank()`` uses
        :meth:`PanelData.get_criterion_matrix` which applies dynamic exclusion
        via ``YearContext``).  No internal NaN filtering or province
        back-filling is performed here.

        Methods: TOPSIS, VIKOR, PROMETHEE, COPRAS, EDAS, Base

        Returns
        -------
        scores  : {method_name: pd.Series}  (scores normalised [0, 1]; Base in raw units)
        ranks   : {method_name: pd.Series}
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
            return neutral_scores, neutral_ranks

        subcrit_weights = {
            c: subcrit_weights.get(c, 1.0 / len(df.columns))
            for c in df.columns
        }

        scores: Dict[str, pd.Series] = {}
        ranks:  Dict[str, pd.Series] = {}

        # ----- Normalize crisp data to [0, 1] via min-max -----
        # Cost criteria are inverted during min-max normalisation so that
        # higher values = better for all columns after this step.
        # NOTE: All sub-criteria in the PCI/PAPI governance dataset are
        # benefit-type (higher = better governance).  cost_local will
        # therefore be empty in normal operation.  The cost-inversion path
        # is retained for generality but must NOT be forwarded to individual
        # MCDM methods — they would apply a second inversion on already-
        # inverted data, reversing the intended direction.
        df_norm = self._minmax_normalize(df, cost_criteria=cost_local)

        # ===== TRADITIONAL METHODS =====
        # Pass cost_criteria=[] because direction is already encoded
        # in df_norm by _minmax_normalize (audit fix C1).
        # Pass the original df so _run_traditional can compute the Base
        # raw-sum score from un-normalised values.
        trad_results = self._run_traditional(df_norm, df, subcrit_weights, cost_criteria=[])
        for name, res in trad_results.items():
            s = self._normalize_scores(
                res['scores'], higher_is_better=res['higher_better'])
            scores[name] = s
            ranks[name]  = res['ranks']

        return scores, ranks

    # ------------------------------------------------------------------
    # Traditional method runners
    # ------------------------------------------------------------------

    def _run_traditional(
        self,
        df: pd.DataFrame,
        df_orig: pd.DataFrame,
        weights: Dict[str, float],
        cost_criteria: List[str],
    ) -> Dict[str, Dict]:
        """Run 6 MCDM methods (5 traditional + 1 baseline). Returns raw scores + ranks.

        ``df`` is the min-max normalised decision matrix used by all MCDM
        methods.  ``df_orig`` is the original (un-normalised) matrix used
        exclusively by the Base baseline, which sums raw sub-criteria values
        without any normalisation or weighting.
        """
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

        # Base — naive baseline: sum of original (un-normalised) sub-criteria
        # values with no weighting applied.  This serves as a simple additive
        # composite that requires zero methodological choices beyond the raw
        # data, making it the most transparent possible baseline.
        try:
            base_scores = df_orig.sum(axis=1)    # raw sum in original units
            base_ranks  = base_scores.rank(ascending=False, method='min').astype(int)
            results['Base'] = {
                'scores': base_scores, 'ranks': base_ranks, 'higher_better': True
            }
        except Exception as e:
            logger.warning(f"    Base failed: {e}")

        return results

    # ==================================================================
    # Weight derivation
    # ==================================================================

    def _derive_hierarchical_weights(
        self,
        subcriteria_weights: Dict[str, float],
        hierarchy: HierarchyMapping,
        ctx=None,                            # YearContext | None
        criterion_weights_override: Optional[Dict[str, float]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        Derive criterion-level and within-group sub-criteria weights.

        When ``criterion_weights_override`` is provided (supplied externally
        by ``HybridWeightingCalculator`` Level 2), those values are used
        directly for criterion-level weights.  The within-group normalization
        is always derived from the year-active ``subcriteria_weights``.

        Criterion weight  = Σ(active SC weights in group)   [or override]
        Within-group weight = SC weight / group_total

        Parameters
        ----------
        subcriteria_weights : dict
            {SC_code: weight} — year-filtered global SC weights.
        hierarchy : HierarchyMapping
        ctx : YearContext or None
        criterion_weights_override : dict, optional
            Pre-computed criterion weights from Level 2 MC ensemble.
            When provided, summation-based derivation is skipped for the
            criterion level, but within-group normalisation still uses
            ``subcriteria_weights``.

        Returns
        -------
        criterion_weights : {C01: w, …}   (normalised to sum 1)
        group_weights     : {C01: {SC01: w_local, …}, …}
        """
        criterion_weights: Dict[str, float] = {}
        group_weights:     Dict[str, Dict[str, float]] = {}

        for crit_id in hierarchy.all_criteria:
            if ctx is not None:
                subcrit_list = ctx.criterion_subcriteria.get(crit_id, [])
            else:
                subcrit_list = hierarchy.criteria_to_subcriteria[crit_id]

            group_w = sum(subcriteria_weights.get(sc, 0.0) for sc in subcrit_list)

            # Criterion weight: use override when available
            if criterion_weights_override is not None:
                criterion_weights[crit_id] = criterion_weights_override.get(
                    crit_id, group_w
                )
            else:
                criterion_weights[crit_id] = group_w

            # Within-group normalisation always from subcriteria_weights
            local: Dict[str, float] = {}
            for sc in subcrit_list:
                w_sc = subcriteria_weights.get(sc, 0.0)
                local[sc] = (
                    w_sc / group_w
                    if group_w > 0
                    else 1.0 / max(len(subcrit_list), 1)
                )
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
    # Normalization helpers
    # ==================================================================

    @staticmethod
    def _minmax_normalize(
        df: pd.DataFrame,
        cost_criteria: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Min-max normalize to [0, 1].  Cost criteria are inverted.

        In the primary path, ``df`` is guaranteed NaN-free:
        :meth:`PanelData.get_criterion_matrix` uses ``YearContext`` to return
        only observed province-SC pairs (complete-case matrix).  No NaN
        imputation is performed here — any residual NaN cells are preserved
        so that the absence of data is visible downstream rather than masked
        by a synthetic neutral score.

        The degenerate case (constant or all-NaN column) sets the full column
        to 0.5 as an undefined-normalisation fallback; this is not imputation
        — it indicates that CRITIC / MCDM has no discriminating information
        for that sub-criterion.
        """
        result = df.copy().astype(float)
        cost_criteria = cost_criteria or []

        for col in result.columns:
            col_min = result[col].min()          # pandas skips NaN by default
            col_max = result[col].max()
            rng = col_max - col_min

            if pd.isna(rng) or rng < 1e-12:
                result[col] = 0.5  # constant or all-NaN column (degenerate — not imputation)
            elif col in cost_criteria:
                result[col] = (col_max - result[col]) / rng
            else:
                result[col] = (result[col] - col_min) / rng

        # NaN cells are preserved; callers must supply a NaN-free matrix
        # (guaranteed by get_criterion_matrix / YearContext in the primary path).
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
