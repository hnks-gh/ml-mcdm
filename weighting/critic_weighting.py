# -*- coding: utf-8 -*-
"""
CRITIC Weighting for Panel MCDM Data
=====================================

Primary class
-------------
``CRITICWeightingCalculator``
    Two-level deterministic CRITIC weighting.
    Level 1 — local SC weights per criterion group (sum to 1 within group).
    Level 2 — criterion weights over composite matrix (sum to 1 globally).
    Global  — global_w[SC_j] = local_w[SC_j | C_k] × criterion_w[C_k].

No Monte Carlo, no tuning, no Beta blending — fully deterministic.
Temporal stability analysis is delegated to ``analysis/``.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
import logging

from .base import WeightResult
from .critic import CRITICWeightCalculator
from .normalization import global_min_max_normalize

logger = logging.getLogger(__name__)


class CRITICWeightingCalculator:
    """
    Two-level deterministic CRITIC weighting for panel MCDM data.

    Level 1  : CRITIC weights per criterion group → local SC weights
               (sum to 1 within each group).
    Level 2  : CRITIC weights over criterion composite matrix → criterion
               weights (sum to 1 globally).
    Global   : global_w[SC_j] = local_w[SC_j | C_k] × criterion_w[C_k],
               re-normalised to sum to 1.

    Parameters
    ----------
    config : WeightingConfig
        Configuration object.  Only ``config.epsilon`` is used.
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
    ) -> WeightResult:
        """
        Run two-level deterministic CRITIC weighting.

        Parameters
        ----------
        panel_df : pd.DataFrame
            Long-format panel (entity_col, time_col, SC columns).
            Pre-cleaned by caller; further NaN guards applied internally.
        criteria_groups : dict
            ``{criterion_id: [sc_col1, …]}`` — criterion groups.
        entity_col : str, default 'Province'
        time_col   : str, default 'Year'

        Returns
        -------
        WeightResult
            weights = {sc: float} — global SC weights summing to 1.
            method  = 'critic_weighting'
            details = diagnostics dict (see Notes).

        Notes
        -----
        Details dict schema::

            level1: {crit_id: {local_sc_weights: {sc: float}}}
            level2: {criterion_weights: {crit_id: float}}
            global_sc_weights:       {sc: float}
            critic_sc_weights:       {sc: float}   # alias of global_sc_weights
            critic_criterion_weights:{crit_id: float} # alias of level2 weights
            n_observations:    int
            n_criteria_groups: int
            n_subcriteria:     int
            n_provinces:       int
            n_years:           int
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

            X_k_norm = global_min_max_normalize(X_k, epsilon=eps)
            if np.isnan(X_k_norm).any():
                _cm = np.nanmean(X_k_norm, axis=0)
                _cm = np.where(np.isnan(_cm), eps, _cm)
                _nr, _nc = np.where(np.isnan(X_k_norm))
                X_k_norm[_nr, _nc] = _cm[_nc]

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

        # ── Level 2: criterion composite matrix, CRITIC on it ───────────
        # Z[i, k] = Σ_j  local_w[C_k][sc_j] × X_raw[i, j]  for j ∈ SC_k
        Z = np.zeros((n_obs, len(criterion_ids)), dtype=np.float64)
        for k_idx, (crit_id, sc_cols_k) in enumerate(criteria_groups.items()):
            col_idx = [all_sc_cols.index(sc) for sc in sc_cols_k]
            u_k = np.array([local_weights[crit_id][sc] for sc in sc_cols_k])
            Z[:, k_idx] = X_raw_all[:, col_idx] @ u_k

        Z_norm = global_min_max_normalize(Z, epsilon=eps)
        if np.isnan(Z_norm).any():
            _cm = np.nanmean(Z_norm, axis=0)
            _cm = np.where(np.isnan(_cm), eps, _cm)
            _nr, _nc = np.where(np.isnan(Z_norm))
            Z_norm[_nr, _nc] = _cm[_nc]

        df_Z = pd.DataFrame(Z_norm, columns=criterion_ids)
        logger.info("  Level 2: %d criteria", len(criterion_ids))
        try:
            c_L2 = self._critic_calc.calculate(df_Z)
            criterion_weights: Dict[str, float] = {
                cid: float(c_L2.weights.get(cid, eps)) for cid in criterion_ids
            }
        except Exception as _exc:
            logger.warning("Level 2 CRITIC failed, using equal weights: %s", _exc)
            criterion_weights = {cid: 1.0 / len(criterion_ids) for cid in criterion_ids}

        # Normalise criterion weights
        cw_tot = sum(criterion_weights.values())
        if cw_tot > 0:
            criterion_weights = {cid: w / cw_tot for cid, w in criterion_weights.items()}

        logger.info(
            "  Criterion weights: %s",
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
            "level2": {"criterion_weights": criterion_weights},
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

        return WeightResult(
            weights=global_sc_weights,
            method="critic_weighting",
            details=details,
        )
