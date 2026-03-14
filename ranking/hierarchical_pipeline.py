# -*- coding: utf-8 -*-
"""
Hierarchical Ranking Pipeline
==============================

Two-stage ranking system that runs 5 traditional MCDM methods within each
criterion group, then aggregates results using Evidential Reasoning.

Architecture
------------
Stage 1 — Within-Criterion Ranking
    For each criterion C_k (k = 1…8):
        • Extract subcriteria data for C_k.
        • Build crisp decision matrix → run 5 traditional methods.
        • Normalize all 5 method scores to [0, 1].

Stage 2 — Global Aggregation via Evidential Reasoning
    • Convert method scores to belief distributions (5 grades).
    • Stage 1 ER: combine 6 methods per criterion.
    • Stage 2 ER: combine 8 criterion beliefs with criterion weights.
    • Final ranking from average utility of fused belief.

References
----------
[1] Yang, J.B. & Xu, D.L. (2002). Evidential Reasoning algorithm.
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
from .evidential_reasoning import (
    HierarchicalEvidentialReasoning, HierarchicalERResult,
    BeliefDistribution, EvidentialReasoningEngine,
)

logger = logging.getLogger('ml_mcdm')


# =========================================================================
# Result container
# =========================================================================

@dataclass
class HierarchicalRankingResult:
    """Container for the result of :class:`HierarchicalRankingPipeline`.

    Attributes
    ----------
    er_result : HierarchicalERResult or None
        Full ER aggregation output (rankings, beliefs, uncertainty).
        ``None`` when ``use_evidential_reasoning=False``.
    criterion_method_scores : dict
        {criterion: {method: pd.Series}} — raw normalized scores.
    criterion_method_ranks : dict
        {criterion: {method: pd.Series}} — per-method ranks.
    criterion_weights_used : dict
        {criterion: float} — weights fed to Stage 2.
    subcriteria_weights_used : dict
        {criterion: {subcrit: float}} — subcriteria weights within each group.
    methods_used : list
        Names of all methods (MCDM × 5 + Base).
    target_year : int
        Year for which ranking was computed.
    final_scores_direct : pd.Series or None
        When ER is disabled, holds the criterion-weighted mean-of-method
        composite score. ``None`` when ER is enabled.
    final_ranking_direct : pd.Series or None
        When ER is disabled, holds the rank order from ``final_scores_direct``.
        ``None`` when ER is enabled.
    """

    er_result: Optional['HierarchicalERResult']
    criterion_method_scores: Dict[str, Dict[str, 'pd.Series']]
    criterion_method_ranks: Dict[str, Dict[str, 'pd.Series']]
    criterion_weights_used: Dict[str, float]
    subcriteria_weights_used: Dict[str, Dict[str, float]]
    methods_used: List[str]
    target_year: int
    # Optional fields — populated only in non-ER mode
    final_scores_direct: Optional['pd.Series'] = None
    final_ranking_direct: Optional['pd.Series'] = None

    # ------------------------------------------------------------------
    # Convenience delegation — transparently handles ER-on and ER-off modes
    # ------------------------------------------------------------------

    @property
    def final_ranking(self) -> Optional['pd.Series']:
        if self.er_result is not None:
            return self.er_result.final_ranking
        return self.final_ranking_direct

    @property
    def final_scores(self) -> Optional['pd.Series']:
        if self.er_result is not None:
            return self.er_result.final_scores
        return self.final_scores_direct

    @property
    def kendall_w(self) -> Optional[float]:
        if self.er_result is not None:
            return self.er_result.kendall_w
        return float('nan')

    def top_n(self, n: int = 10) -> Optional['pd.DataFrame']:
        if self.er_result is not None:
            return self.er_result.top_n(n)
        # Non-ER fallback
        if self.final_scores_direct is None or self.final_ranking_direct is None:
            return None
        import pandas as _pd
        df = _pd.DataFrame({
            'Score': self.final_scores_direct,
            'Rank': self.final_ranking_direct,
        })
        return df.nsmallest(n, 'Rank')

    def summary(self) -> str:
        if self.er_result is not None:
            return self.er_result.summary()
        n_alts = len(self.final_scores_direct) if self.final_scores_direct is not None else 0
        return (
            f"Hierarchical ranking (ER disabled) — "
            f"{n_alts} alternatives, "
            f"{len(self.criterion_method_scores)} criteria, "
            f"year {self.target_year}."
        )


# =========================================================================
# Pipeline
# =========================================================================

class HierarchicalRankingPipeline:
    """
    Orchestrates two-stage hierarchical ranking
    (5 traditional MCDM methods + Base baseline + Evidential Reasoning).

    The five MCDM methods are TOPSIS, VIKOR, PROMETHEE, COPRAS, and EDAS.
    ``Base`` is a standalone naive baseline that sums the original (raw,
    un-normalised) sub-criteria values directly, with no weighting applied.
    Base is stored in ``criterion_method_scores`` for comparison purposes
    but is **not** fed into the ER aggregation — ER uses only the 5 MCDM
    methods.

    Parameters
    ----------
    n_grades : int
        Number of ER evaluation grades (default 5).
    method_weight_scheme : str
        How to weight method contributions in Stage 1 ER:
        ``'equal'`` or ``'rank_performance'``.
    cost_criteria : list, optional
        Subcriteria codes where lower values are preferred.
    use_evidential_reasoning : bool
        When True (default), execute two-stage ER aggregation and produce a
        composite ranking.  When False, Stage 1 MCDM scores are computed but
        Stage 2 ER is skipped; no composite ranking is produced (``er_result``
        and all derived properties are ``None``).
    """

    # All methods stored in results (includes standalone Base baseline)
    TRADITIONAL_METHODS = ['TOPSIS', 'VIKOR', 'PROMETHEE', 'COPRAS', 'EDAS', 'Base']
    ALL_METHODS = TRADITIONAL_METHODS
    # Only these 5 are fed into ER aggregation
    ER_METHODS = ['TOPSIS', 'VIKOR', 'PROMETHEE', 'COPRAS', 'EDAS']

    def __init__(
        self,
        n_grades: int = 5,
        method_weight_scheme: str = 'equal',
        cost_criteria: Optional[List[str]] = None,
        use_evidential_reasoning: bool = True,
    ):
        self.n_grades = n_grades
        self.method_weight_scheme = method_weight_scheme
        self.cost_criteria = cost_criteria or []
        self.use_evidential_reasoning = use_evidential_reasoning

        # Initialise ER aggregator (only used when ER is enabled)
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

        # ------------------------------------------------------------------
        # Stage 2: Ranking aggregation
        # ------------------------------------------------------------------
        if self.use_evidential_reasoning:
            # ER aggregation (5 MCDM methods only — Base excluded)
            # Build a filtered view of method scores that excludes the Base
            # baseline so ER aggregation is not influenced by the raw sum.
            er_method_scores: Dict[str, Dict[str, pd.Series]] = {
                crit_id: {
                    m: s for m, s in crit_data.items() if m in self.ER_METHODS
                }
                for crit_id, crit_data in all_method_scores.items()
            }
            logger.info("  Running Evidential Reasoning aggregation (5 MCDM methods)...")
            er_result = self.er_aggregator.aggregate(
                method_scores=er_method_scores,
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
                methods_used=self.ALL_METHODS,
                target_year=target_year,
            )
        else:
            # ER disabled — Stage 2 is entirely skipped.
            # No composite ranking is produced; the 5 MCDM methods and Base
            # remain as separate independent outputs in criterion_method_scores.
            logger.info(
                "  ER disabled — Stage 2 skipped. "
                "Individual method scores preserved (no composite ranking)."
            )
            return HierarchicalRankingResult(
                er_result=None,
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
    ) -> Optional['HierarchicalERResult']:
        """
        Lightweight re-ranking using precomputed MCDM scores.

        When ER is enabled, skips ALL 5 MCDM method computations and re-runs
        only the ER aggregation step with new criterion weights derived from
        *subcriteria_weights*.  Yields ~50-100x speedup over :meth:`rank` for
        sensitivity-analysis weight perturbations.

        When ER is disabled, returns None (callers must check).
        """
        if not self.use_evidential_reasoning:
            return None

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

        return self.er_aggregator.aggregate(
            method_scores      = precomputed_scores,
            criterion_weights  = crit_w,
            alternatives       = alternatives,
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
        Run 5 traditional MCDM methods + Base baseline on a **clean**
        criterion-level decision matrix.

        The matrix is guaranteed NaN-free by the caller (``rank()`` uses
        :meth:`PanelData.get_criterion_matrix` which applies dynamic exclusion
        via ``YearContext``).  No internal NaN filtering or province
        back-filling is performed here.

        Returns
        -------
        scores  : {method_name: pd.Series}  (MCDM scores normalised [0, 1]; Base in raw units)
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
        """Run 5 traditional MCDM methods + Base baseline. Returns raw scores + ranks.

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
