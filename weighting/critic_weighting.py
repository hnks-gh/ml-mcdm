# -*- coding: utf-8 -*-
"""
Hierarchical CRITIC Weighting Engine
====================================

Orchestrates a two-level objective weighting process for panel data:
1.  Level 1: Local sub-criteria weights within each criterion group.
2.  Level 2: Criterion weights across composite criterion scores.

The resulting global weights are the product of these two levels, 
ensuring a mathematically grounded and balanced weighting scheme that 
respects the hierarchical structure of the governance model.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging

from .base import WeightResult
from .critic import CRITICWeightCalculator
from .normalization import global_min_max_normalize
from analysis.critic_temporal_stability import WindowedTemporalStabilityAnalyzer
from analysis.critic_sensitivity_analysis import CRITICSensitivityAnalyzer

logger = logging.getLogger(__name__)


class CRITICWeightingCalculator:
    """
    Two-level deterministic CRITIC weighting system.

    Implements a hierarchical approach where sub-criteria are first 
    weighted within their respective functional groups, and groups are 
    subsequently weighted based on their information content across the 
    entire panel.
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self._critic_calc = CRITICWeightCalculator(epsilon=config.epsilon)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(
        self,
        panel_df: pd.DataFrame,
        criteria_groups: Dict[str, List[str]],
        entity_col: str = "Province",
        time_col: str = "Year",
        run_temporal_stability: bool = True,
        run_sensitivity_analysis: bool = True,
        weight_all_years: Dict[int, Dict[str, float]] = None,
    ) -> WeightResult:
        """
        Execute the hierarchical weighting process on panel data.

        Coordinates Level 1 (local) and Level 2 (global) weighting steps, 
        handling structural gaps in the panel through year-regime analysis.

        Parameters
        ----------
        panel_df : pd.DataFrame
            The long-format panel data.
        criteria_groups : Dict[str, List[str]]
            Definition of which sub-criteria belong to which criterion.
        entity_col : str
            The column name for entities (provinces).
        time_col : str
            The column name for the temporal dimension (years).
        run_temporal_stability : bool
            Whether to perform stability metrics across windows.
        run_sensitivity_analysis : bool
            Whether to perform perturbation audits.
        weight_all_years : Optional[Dict[int, Dict[str, float]]]
            Historical weights for comparative diagnostics.

        Returns
        -------
        WeightResult
            The final result containing global sub-criteria weights and 
            hierarchical diagnostics.
        """
        eps = self.config.epsilon
        panel_df = panel_df.copy()

        # ── NaN guard: keep only SCs with ≥1 non-NaN observation ────────
        active_groups: Dict[str, List[str]] = {}
        for crit_id, sc_cols in criteria_groups.items():
            active = [
                sc for sc in sc_cols
                if sc in panel_df.columns and panel_df[sc].notna().any()
            ]
            if active:
                active_groups[crit_id] = active
        criteria_groups = active_groups

        all_sc_cols = [sc for scs in criteria_groups.values() for sc in scs]
        criterion_ids = list(criteria_groups.keys())

        # ── Year-aware row filtering ─────────────────────────────────────
        # Preserve rows from years with structural SC gaps (e.g. SC71-SC83
        # absent 2011-2017) instead of a naïve global notna().all() drop.
        if time_col in panel_df.columns:
            valid_rows = pd.Series(False, index=panel_df.index)
            for _yr, _yr_grp in panel_df.groupby(time_col):
                _yr_act = [sc for sc in all_sc_cols if _yr_grp[sc].notna().any()]
                if _yr_act:
                    _yr_valid = _yr_grp[_yr_act].notna().all(axis=1)
                    valid_rows.loc[_yr_grp.index] |= _yr_valid
        else:
            valid_rows = panel_df[all_sc_cols].notna().all(axis=1)
        panel_df = panel_df[valid_rows].copy().reset_index(drop=True)

        n_obs = len(panel_df)
        n_provinces = int(panel_df[entity_col].nunique()) if entity_col in panel_df.columns else 1
        n_years = int(panel_df[time_col].nunique()) if time_col in panel_df.columns else 1

        logger.info(
            "CRITICWeightingCalculator: %d obs, %d sub-crit across %d groups",
            n_obs, len(all_sc_cols), len(criterion_ids),
        )

        X_raw_all = panel_df[all_sc_cols].values.astype(np.float64)

        # ── Level 1: CRITIC per criterion group ─────────────────────────
        local_weights: Dict[str, Dict[str, float]] = {}
        level1_diagnostics: Dict[str, dict] = {}

        for crit_id, sc_cols_k in criteria_groups.items():
            X_k = panel_df[sc_cols_k].values.astype(np.float64)
            logger.info("  Level 1: %s (%d SCs)", crit_id, len(sc_cols_k))

            # ── F-01: complete-case exclusion per group ──────────────────
            # Criterion groups with structural year-gaps (e.g. SC71-SC73
            # absent 2011-2017) produce NaN rows in X_k even after the
            # year-aware row-filter above, because those years ARE kept
            # for other groups whose SCs were fully observed that year.
            # Dropping these rows rather than imputing is the statistically
            # correct approach: column-mean imputation would attenuate σ_j
            # and inflate r_{jk}, producing biased CRITIC weights.
            group_valid = ~np.isnan(X_k).any(axis=1)
            n_dropped_k = int((~group_valid).sum())
            if n_dropped_k > 0:
                logger.info(
                    "  Level 1 %s: %d row(s) excluded "
                    "(structural year-gap — complete-case analysis).",
                    crit_id, n_dropped_k,
                )
            X_k_cc = X_k[group_valid]

            if len(X_k_cc) < 2:
                logger.warning(
                    "Level 1 CRITIC %s: fewer than 2 complete-case rows "
                    "(%d available) — using equal local weights.",
                    crit_id, len(X_k_cc),
                )
                lw = {sc: 1.0 / len(sc_cols_k) for sc in sc_cols_k}
            else:
                X_k_norm = global_min_max_normalize(X_k_cc, epsilon=eps)
                df_k = pd.DataFrame(X_k_norm, columns=sc_cols_k)
                try:
                    c_k = self._critic_calc.calculate(df_k)
                    lw = {sc: float(c_k.weights.get(sc, eps)) for sc in sc_cols_k}
                except Exception as _exc:
                    logger.warning(
                        "Level 1 CRITIC failed for %s, using equal weights: %s",
                        crit_id, _exc,
                    )
                    lw = {sc: 1.0 / len(sc_cols_k) for sc in sc_cols_k}

            # Normalise within group
            lw_tot = sum(lw.values())
            if lw_tot > 0:
                lw = {sc: w / lw_tot for sc, w in lw.items()}

            local_weights[crit_id] = lw
            level1_diagnostics[crit_id] = {"local_sc_weights": lw}

        # ── Level 2: criterion composite matrix (renormalized partial weights) ──
        # Z[i, k] = partial-available composite score for criterion k, row i.
        #
        # F-02 revision — replacing the naive dot-product (X_raw @ u_k):
        #   The dot-product propagates NaN from any single absent SC, so
        #   Z[i, k] = NaN for every year with even one SC gap.  In this panel
        #   the only years with all 29 SCs present are 2019-2020, reducing Level
        #   2 to ≈126 observations — far too few to estimate 8 × 8 correlations.
        #
        # Correct approach — renormalized weighted average over available SCs:
        #
        #              Σ_{j ∈ avail_k(i)}  local_w_k[j] · X_raw[i, j]
        #   Z[i, k] = ─────────────────────────────────────────────────
        #              Σ_{j ∈ avail_k(i)}  local_w_k[j]
        #
        #   avail_k(i) = SCs of group k that are non-NaN for row i.
        #   Z[i, k] = NaN only when ALL SCs of group k are absent for row i.
        #
        # Effect on this 14-year panel:
        #   • 2011-2017: SC24 absent → C02 uses SC21-SC23 renormalized (valid).
        #                SC71-73 absent → C07 = NaN (entire group absent).
        #                SC81-83 absent → C08 = NaN (entire group absent).
        #   • 2018     : SC83 absent → C08 uses SC81+SC82 renormalized (valid).
        #   • 2021-2024: SC52 absent → C05 uses SC51+SC53+SC54 renormalized (valid).
        #   ⟹ Z[i, k] = NaN only for rows 2011-2017 in columns C07 and C08.
        n_crit = len(criterion_ids)
        Z = np.full((n_obs, n_crit), np.nan, dtype=np.float64)
        for k_idx, (crit_id, sc_cols_k) in enumerate(criteria_groups.items()):
            col_idx_k = [all_sc_cols.index(sc) for sc in sc_cols_k]
            u_k       = np.array(
                [local_weights[crit_id][sc] for sc in sc_cols_k], dtype=np.float64
            )
            X_k_raw  = X_raw_all[:, col_idx_k]                     # (n_obs, n_sc_k)
            nan_mask = np.isnan(X_k_raw)                            # True = SC absent
            # Per-row effective weight: zero out absent SCs
            W_eff    = np.where(nan_mask, 0.0, u_k[np.newaxis, :]) # (n_obs, n_sc_k)
            W_sum    = W_eff.sum(axis=1)                            # (n_obs,)
            valid_k  = W_sum > 0                                    # ≥1 SC available
            X_filled = np.where(nan_mask, 0.0, X_k_raw)
            Z[valid_k, k_idx] = (
                (X_filled[valid_k, :] * W_eff[valid_k, :]).sum(axis=1)
                / W_sum[valid_k]
            )
            # Z[~valid_k, k_idx] stays NaN: entire criterion absent for these rows

        # ── F-02 (revised): year-regime-aware Level 2 CRITIC ─────────────────
        # After the renormalized Z, Z[i, k] = NaN only when ALL SCs of criterion
        # k are entirely absent for row i (e.g. C07/C08 in 2011-2017).  Groups
        # of rows that share the same NaN pattern form distinct "year regimes"
        # — years where the same set of criteria is available.
        #
        # Algorithm:
        #   1. Detect regimes by encoding each row's NaN pattern as a bitmask.
        #   2. For each regime: run Level 2 CRITIC on its active criteria;
        #      normalise regime criterion-weights to sum to 1.
        #   3. Aggregate via observation-count-weighted average:
        #        global_cw[k] = Σ_r { cw_r[k] · n_r } / Σ_r n_r
        #      Criteria absent from regime r contribute cw_r[k] = 0.
        #   4. Renormalise global criterion-weights to Σ = 1.
        #
        # Result for this panel (~882 obs total):
        #   Regime 2011-2017 (~441 obs): active = {C01…C06}  → 6 criterion weights
        #   Regime 2018-2024 (~441 obs): active = {C01…C08}  → 8 criterion weights
        #   Global weights are the obs-weighted blend of both regimes.
        Z_nan = np.isnan(Z)                                         # (n_obs, n_crit)
        # Encode each row's NaN pattern as a bitmask (integer)
        pattern_vec = (
            Z_nan @ (2 ** np.arange(n_crit, dtype=np.int64))
        ).astype(np.int64)
        unique_pats, regime_sizes = np.unique(pattern_vec, return_counts=True)

        logger.info(
            "  Level 2: %d year-regime(s), %d criteria, %d total obs",
            len(unique_pats), n_crit, n_obs,
        )

        regime_records: List[dict] = []   # for aggregation and diagnostics

        for pat_code, n_reg in zip(unique_pats.tolist(), regime_sizes.tolist()):
            row_mask  = pattern_vec == pat_code
            Z_reg     = Z[row_mask, :]                              # (n_reg, n_crit)
            avail_col = ~Z_nan[row_mask, :].any(axis=0)             # (n_crit,) bool
            avail_ids = [cid for cid, a in zip(criterion_ids, avail_col) if a]
            n_avail   = len(avail_ids)

            if n_avail == 0:
                logger.warning(
                    "  Level 2 regime (pat=%d, n=%d): no active criteria — skipped.",
                    pat_code, n_reg,
                )
                continue

            logger.info(
                "  Level 2 regime (pat=%d, n=%d): active=[%s]",
                pat_code, n_reg, ", ".join(avail_ids),
            )
            Z_avail = Z_reg[:, avail_col]                           # (n_reg, n_avail)

            if n_avail < 2 or n_reg < 2:
                cw_reg = {cid: 1.0 / n_avail for cid in avail_ids}
                logger.info(
                    "    → equal weights (n_crit=%d or n_obs=%d < 2).",
                    n_avail, n_reg,
                )
            else:
                Z_norm_reg = global_min_max_normalize(Z_avail, epsilon=eps)
                df_Z_reg   = pd.DataFrame(Z_norm_reg, columns=avail_ids)
                try:
                    c_reg  = self._critic_calc.calculate(df_Z_reg)
                    cw_reg = {
                        cid: float(c_reg.weights.get(cid, eps))
                        for cid in avail_ids
                    }
                except Exception as _exc:
                    logger.warning(
                        "    Level 2 CRITIC failed (pat=%d): %s — equal weights.",
                        pat_code, _exc,
                    )
                    cw_reg = {cid: 1.0 / n_avail for cid in avail_ids}

            # Normalise within regime so the regime sums to 1
            cw_sum = sum(cw_reg.values())
            if cw_sum > 0:
                cw_reg = {cid: w / cw_sum for cid, w in cw_reg.items()}

            regime_records.append({
                "pattern_code":      pat_code,
                "n_obs":             n_reg,
                "active_criteria":   avail_ids,
                "criterion_weights": cw_reg,
            })

        # Observation-count-weighted average across regimes → global criterion weights
        criterion_weights_acc: Dict[str, float] = {cid: 0.0 for cid in criterion_ids}
        total_reg_obs = sum(r["n_obs"] for r in regime_records)
        for rec in regime_records:
            for cid in criterion_ids:
                criterion_weights_acc[cid] += (
                    rec["criterion_weights"].get(cid, 0.0) * rec["n_obs"]
                )

        cw_agg_sum = sum(criterion_weights_acc.values())
        if cw_agg_sum > 0:
            criterion_weights: Dict[str, float] = {
                cid: w / cw_agg_sum for cid, w in criterion_weights_acc.items()
            }
        else:
            logger.warning("Level 2: aggregated weights sum to 0 — using equal weights.")
            criterion_weights = {cid: 1.0 / n_crit for cid in criterion_ids}

        logger.info(
            "  Criterion weights (aggregated, %d regime(s), %d obs): %s",
            len(regime_records), total_reg_obs,
            ", ".join(f"{k}={v:.3f}" for k, v in sorted(criterion_weights.items())),
        )

        # ── Global SC weights ────────────────────────────────────────────
        global_sc_weights: Dict[str, float] = {}
        for crit_id, sc_cols_k in criteria_groups.items():
            v_k = criterion_weights.get(crit_id, 0.0)
            for sc in sc_cols_k:
                global_sc_weights[sc] = local_weights[crit_id][sc] * v_k

        gw_total = sum(global_sc_weights.values())
        if gw_total > 0:
            global_sc_weights = {sc: w / gw_total for sc, w in global_sc_weights.items()}

        details: dict = {
            "level1": level1_diagnostics,
            "level2": {
                "criterion_weights": criterion_weights,
                # Per-regime diagnostics: one entry per detected year-regime.
                # Each entry: {pattern_code, n_obs, active_criteria, criterion_weights}
                "regimes": regime_records,
            },
            # Flat aliases for downstream consumers (pipeline, csv_writer, plots)
            "global_sc_weights":        global_sc_weights,
            "critic_sc_weights":        global_sc_weights,        # alias
            "critic_criterion_weights": criterion_weights,         # alias
            "n_observations":    n_obs,
            "n_criteria_groups": len(criterion_ids),
            "n_subcriteria":     len(all_sc_cols),
            "n_provinces":       n_provinces,
            "n_years":           n_years,
        }

        # Initialize result
        result = WeightResult(
            weights=global_sc_weights,
            method="critic_weighting",
            details=details,
        )

        # Compute temporal stability analysis if enabled and weight history provided
        if run_temporal_stability and weight_all_years is not None:
            try:
                temporal_analyzer = WindowedTemporalStabilityAnalyzer()
                result.temporal_stability = temporal_analyzer.analyze(weight_all_years)
                logger.info("Temporal stability analysis completed")
            except Exception as e:
                logger.warning(f"Temporal stability analysis failed: {e}")

        # Compute sensitivity analysis if enabled and weight history provided
        if run_sensitivity_analysis and weight_all_years is not None:
            try:
                sensitivity_analyzer = CRITICSensitivityAnalyzer()
                result.sensitivity_analysis = sensitivity_analyzer.analyze(weight_all_years)
                logger.info("Sensitivity analysis completed")
            except Exception as e:
                logger.warning(f"Sensitivity analysis failed: {e}")

        return result
