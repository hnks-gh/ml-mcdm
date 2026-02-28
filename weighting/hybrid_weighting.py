# -*- coding: utf-8 -*-
"""
Hybrid Weighting for Panel MCDM Data
=====================================

Primary class
-------------
``HybridWeightingCalculator``
    Two-level hierarchical MC ensemble: Entropy + CRITIC blend.
    Level 1 — local SC weights per criterion group (sum to 1 within group).
    Level 2 — criterion weights over composite matrix (sum to 1 globally).
    Global  — global_w[SC_j] = local_w[SC_j | C_k] × criterion_w[C_k].
"""

import dataclasses
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import logging

from .base import WeightResult
from .entropy import EntropyWeightCalculator
from .critic import CRITICWeightCalculator
from .normalization import global_min_max_normalize
from .validation import temporal_stability_verification

logger = logging.getLogger(__name__)


# ============================================================
# Internal result containers
# ============================================================

@dataclass
class _MCEnsembleResult:
    """Internal result from a single _run_mc_ensemble call."""
    mean_weights:       Dict[str, float]
    std_weights:        Dict[str, float]
    ci_lower:           Dict[str, float]
    ci_upper:           Dict[str, float]
    cv_weights:         Dict[str, float]
    avg_kendall_tau:    float
    avg_spearman_rho:   float
    kendall_w:          float
    top_k_rank_var:     float
    province_mean_rank: Optional[Dict[str, float]]
    province_std_rank:  Optional[Dict[str, float]]
    province_prob_top1: Optional[Dict[str, float]]
    province_prob_topk: Optional[Dict[str, float]]
    rank_win_matrix:    Optional[Dict[str, Dict[str, float]]]
    converged_at:       Optional[int]
    n_completed:        int
    quality_flag:       str   # 'ok' | 'low_convergence'


# ============================================================
# HybridWeightingCalculator
# ============================================================

class HybridWeightingCalculator:
    """
    Two-level hierarchical hybrid weighting via Probabilistic MC Ensemble.

    Combines Shannon Entropy and CRITIC through a Beta-distributed blending
    parameter, validated against perturbation stability across Monte Carlo
    simulations.

    Level 1  : per-criterion local SC weights (sum to 1 within each group).
    Level 2  : criterion weights over composite matrix (sum to 1 globally).
    Global   : global_w[SC_j] = local_w[SC_j | C_k] × criterion_w[C_k].

    Parameters
    ----------
    config : WeightingConfig
        All hyperparameters for the MC ensemble (see WeightingConfig docstring).
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self._rng   = np.random.RandomState(config.seed)
        self._entropy_calc = EntropyWeightCalculator(epsilon=config.epsilon)
        self._critic_calc  = CRITICWeightCalculator(epsilon=config.epsilon)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(
        self,
        panel_df: pd.DataFrame,
        criteria_groups: Dict[str, List[str]],
        entity_col: str = "Province",
        time_col:   str = "Year",
    ) -> WeightResult:
        """
        Run full two-level hierarchical MC ensemble weighting.

        Parameters
        ----------
        panel_df : pd.DataFrame
            Long-format panel (entity_col, time_col, SC columns).
            Pre-cleaned by caller; further NaN guards applied internally.
        criteria_groups : dict
            ``{criterion_id: [sc_col1, …]}`` — 8 criterion groups.
        entity_col : str, default 'Province'
        time_col   : str, default 'Year'

        Returns
        -------
        WeightResult
            weights = {sc: float} — 29 global SC weights summing to 1.
            method  = 'hybrid_weighting'
            details = full diagnostics (Section 9 schema).
        """
        cfg = self.config
        eps = cfg.epsilon
        rng = self._rng
        panel_df = panel_df.copy()

        # ── Internal NaN guard ──────────────────────────────────────────
        # Filter criteria_groups to only include SCs with ≥1 non-NaN value
        active_groups: Dict[str, List[str]] = {}
        for crit_id, sc_cols in criteria_groups.items():
            active = [
                sc for sc in sc_cols
                if sc in panel_df.columns and panel_df[sc].notna().any()
            ]
            if active:
                active_groups[crit_id] = active
        criteria_groups = active_groups

        # Drop rows where any *year-active* SC is NaN — year-aware so that
        # rows from years with structural Type-1 gaps (SC71-SC83 absent in
        # 2011-2017, SC52 absent in 2021-2024) are not discarded.  For each
        # calendar year in the panel, only SCs that have ≥1 valid observation
        # in *that* year are required to be non-NaN; the absent SCs produce
        # NaN cells that the downstream Entropy/CRITIC column-mean imputation
        # guards handle cleanly.
        all_sc_cols   = [sc for scs in criteria_groups.values() for sc in scs]
        criterion_ids = list(criteria_groups.keys())
        if time_col in panel_df.columns:
            valid_rows = pd.Series(False, index=panel_df.index)
            for _yr, _yr_grp in panel_df.groupby(time_col):
                _yr_act  = [sc for sc in all_sc_cols if _yr_grp[sc].notna().any()]
                if _yr_act:
                    _yr_valid = _yr_grp[_yr_act].notna().all(axis=1)
                    valid_rows.loc[_yr_grp.index] |= _yr_valid
        else:
            valid_rows = panel_df[all_sc_cols].notna().all(axis=1)
        panel_df = panel_df[valid_rows].copy().reset_index(drop=True)

        logger.info(
            "HybridWeightingCalculator: %d obs, %d sub-crit across %d groups",
            len(panel_df), len(all_sc_cols), len(criterion_ids),
        )

        # ── STEP 0: Data Preparation ────────────────────────────────────
        province_blocks = self._build_province_blocks(panel_df, entity_col)
        B               = len(province_blocks)
        province_order  = sorted(province_blocks.keys())
        n_years = panel_df[time_col].nunique() if time_col in panel_df.columns else 1
        logger.info("  Block bootstrap: %d provinces × %d years", B, n_years)

        X_raw_all = panel_df[all_sc_cols].values.astype(np.float64)

        # ── STEP 1: Baseline Weights (for tuning signal) ────────────────
        X_all_norm  = global_min_max_normalize(X_raw_all, epsilon=eps)
        df_all      = pd.DataFrame(X_all_norm, columns=all_sc_cols)
        try:
            e_base = self._entropy_calc.calculate(df_all)
            c_base = self._critic_calc.calculate(df_all)
            W_E_base = np.array([e_base.weights.get(sc, eps) for sc in all_sc_cols])
            W_C_base = np.array([c_base.weights.get(sc, eps) for sc in all_sc_cols])
        except Exception:
            n_sc = len(all_sc_cols)
            W_E_base = W_C_base = np.ones(n_sc) / n_sc

        W_base = (W_E_base + W_C_base) / 2.0
        W_base /= W_base.sum()
        r_base = self._saw_province_ranking(X_all_norm, W_base, province_blocks, province_order)

        # ── STEP 2: Hyperparameter Tuning ───────────────────────────────
        if cfg.perform_tuning:
            logger.info("  Hyperparameter tuning (64-point coarse grid) ...")
            theta = self._tune_hyperparameters(
                panel_df, criteria_groups, province_blocks, province_order, r_base, rng
            )
        else:
            theta = (cfg.beta_a, cfg.beta_b, cfg.noise_sigma_scale)

        alpha_a, alpha_b, sigma_scale = theta
        logger.info(
            "  θ* = (α_a=%.3f, α_b=%.3f, σ_scale=%.4f)",
            alpha_a, alpha_b, sigma_scale,
        )

        # ── STEP 3: Level 1 — Per-Criterion MC Ensemble ─────────────────
        local_weights:      Dict[str, Dict[str, float]] = {}   # {Ck: {sc: w}}
        level1_diagnostics: Dict[str, dict]             = {}

        for crit_id, sc_cols_k in criteria_groups.items():
            X_k = panel_df[sc_cols_k].values.astype(np.float64)
            logger.info("  Level 1: %s (%d SCs) ...", crit_id, len(sc_cols_k))
            result_k = self._run_mc_ensemble(
                X=X_k,
                province_blocks=province_blocks,
                province_order=province_order,
                col_names=sc_cols_k,
                theta=theta,
                rng=rng,
                config=cfg,
                province_labels=None,
            )
            local_weights[crit_id] = result_k.mean_weights
            diag_k = {
                "n_simulations_completed": result_k.n_completed,
                "converged_at":            result_k.converged_at,
                "mean_weights":            result_k.mean_weights,
                "std_weights":             result_k.std_weights,
                "ci_lower_2_5":            result_k.ci_lower,
                "ci_upper_97_5":           result_k.ci_upper,
                "cv_weights":              result_k.cv_weights,
                "avg_kendall_tau":         result_k.avg_kendall_tau,
                "avg_spearman_rho":        result_k.avg_spearman_rho,
                "kendall_w":               result_k.kendall_w,
                "top_k_rank_var":          result_k.top_k_rank_var,
            }
            if result_k.quality_flag != "ok":
                diag_k["quality_flag"] = result_k.quality_flag
            level1_diagnostics[crit_id] = {
                "local_sc_weights": result_k.mean_weights,
                "mc_diagnostics":   diag_k,
            }

        # ── STEP 4: Build Criterion Composite Matrix ─────────────────────
        # Z[i, k] = Σ_j  local_w[Ck][sc_j]  *  X_raw[i, j]  for j ∈ SC_k
        Z = np.zeros((len(panel_df), len(criterion_ids)), dtype=np.float64)
        for k_idx, (crit_id, sc_cols_k) in enumerate(criteria_groups.items()):
            col_idx = [all_sc_cols.index(sc) for sc in sc_cols_k]
            u_k     = np.array([local_weights[crit_id][sc] for sc in sc_cols_k])
            Z[:, k_idx] = X_raw_all[:, col_idx] @ u_k

        # ── STEP 5: Level 2 — Criterion MC Ensemble ──────────────────────
        logger.info("  Level 2: %d criteria ...", len(criterion_ids))
        result_L2 = self._run_mc_ensemble(
            X=Z,
            province_blocks=province_blocks,
            province_order=province_order,
            col_names=criterion_ids,
            theta=theta,
            rng=rng,
            config=cfg,
            province_labels=province_order,
        )
        criterion_weights = result_L2.mean_weights
        logger.info(
            "  Level 2 Kendall τ=%.4f, W=%.4f",
            result_L2.avg_kendall_tau, result_L2.kendall_w,
        )

        level2_diagnostics: dict = {
            "criterion_weights": criterion_weights,
            "mc_diagnostics": {
                "n_simulations_completed": result_L2.n_completed,
                "converged_at":            result_L2.converged_at,
                "mean_weights":            result_L2.mean_weights,
                "std_weights":             result_L2.std_weights,
                "ci_lower_2_5":            result_L2.ci_lower,
                "ci_upper_97_5":           result_L2.ci_upper,
                "cv_weights":              result_L2.cv_weights,
                "avg_kendall_tau":         result_L2.avg_kendall_tau,
                "avg_spearman_rho":        result_L2.avg_spearman_rho,
                "kendall_w":               result_L2.kendall_w,
                "top_k_rank_var":          result_L2.top_k_rank_var,
                "province_mean_rank":      result_L2.province_mean_rank,
                "province_std_rank":       result_L2.province_std_rank,
                "province_prob_top1":      result_L2.province_prob_top1,
                "province_prob_topK":      result_L2.province_prob_topk,
                **(
                    {"rank_win_matrix": result_L2.rank_win_matrix}
                    if result_L2.rank_win_matrix is not None else {}
                ),
            },
        }

        # ── STEP 6: Global SC Weights ────────────────────────────────────
        global_sc_weights: Dict[str, float] = {}
        for crit_id, sc_cols_k in criteria_groups.items():
            v_k = criterion_weights.get(crit_id, 0.0)
            for sc in sc_cols_k:
                global_sc_weights[sc] = local_weights[crit_id][sc] * v_k
        # Re-normalise (floating-point guard; by construction ≈ 1.0)
        gw_total = sum(global_sc_weights.values())
        if gw_total > 0:
            global_sc_weights = {sc: w / gw_total for sc, w in global_sc_weights.items()}

        # ── STEP 7: Temporal Stability Verification ───────────────────────
        stability: dict = {
            "cosine_similarity":   None,
            "pearson_correlation": None,
            "is_stable":           None,
            "split_point":         None,
            "note":                "not computed",
        }
        if time_col in panel_df.columns and panel_df[time_col].nunique() >= 2:
            try:
                # Build a no-tuning config that reuses the tuned θ*
                no_tune_cfg = dataclasses.replace(
                    cfg,
                    perform_tuning      = False,
                    mc_n_simulations    = 200,
                    beta_a              = alpha_a,
                    beta_b              = alpha_b,
                    noise_sigma_scale   = sigma_scale,
                )
                _stab_calc = HybridWeightingCalculator(config=no_tune_cfg)

                def _stability_callback(df_half: pd.DataFrame) -> np.ndarray:
                    try:
                        r_h = _stab_calc.calculate(
                            df_half, criteria_groups, entity_col, time_col
                        )
                        return np.array(
                            [r_h.weights.get(sc, 0.0) for sc in all_sc_cols]
                        )
                    except Exception:
                        return np.full(len(all_sc_cols), 1.0 / len(all_sc_cols))

                stab_result = temporal_stability_verification(
                    panel_df         = panel_df,
                    weight_calculator= _stability_callback,
                    entity_col       = entity_col,
                    time_col         = time_col,
                    criteria_cols    = all_sc_cols,
                    threshold        = cfg.stability_threshold,
                )
                stability = {
                    "cosine_similarity":   stab_result.cosine_similarity,
                    "pearson_correlation": stab_result.correlation,
                    "is_stable":           stab_result.is_stable,
                    "split_point":         stab_result.split_point,
                    "note":                "split-half temporal verification",
                }
            except Exception as _e:
                stability["note"] = f"stability check failed: {_e}"
                logger.warning("Temporal stability check failed: %s", _e)
        else:
            stability["note"] = (
                f"'{time_col}' not found or only one period — skipped"
            )

        # ── STEP 8: Build details dict & return ──────────────────────────
        tuning_best_score = float(result_L2.avg_kendall_tau)

        details: dict = {
            "level1":           level1_diagnostics,
            "level2":           level2_diagnostics,
            "global_sc_weights": global_sc_weights,
            "hyperparameters": {
                "beta_a":            alpha_a,
                "beta_b":            alpha_b,
                "noise_sigma_scale": sigma_scale,
                "boot_fraction":     cfg.boot_fraction,
                "tuning_performed":  cfg.perform_tuning,
                "tuning_objective":  cfg.tuning_objective,
                "tuning_grid_size":  64,
                "tuning_best_score": tuning_best_score,
            },
            "stability":         stability,
            "n_observations":    len(panel_df),
            "n_criteria_groups": len(criterion_ids),
            "n_subcriteria":     len(all_sc_cols),
            "n_provinces":       B,
            "n_years":           n_years,
        }

        return WeightResult(
            weights = global_sc_weights,
            method  = "hybrid_weighting",
            details = details,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_province_blocks(
        self, panel_df: pd.DataFrame, entity_col: str
    ) -> Dict[str, np.ndarray]:
        """Map each province to its 0-based row positions in panel_df."""
        entities     = panel_df[entity_col].values
        unique_provs = pd.unique(entities)
        blocks: Dict[str, np.ndarray] = {}
        for prov in unique_provs:
            blocks[prov] = np.where(entities == prov)[0]
        if len(blocks) < 10:
            logger.warning(
                "Only %d distinct provinces — block bootstrap may be unreliable; "
                "using row-level resampling fallback inside _run_mc_ensemble.",
                len(blocks),
            )
        return blocks

    def _saw_province_ranking(
        self,
        X_norm:          np.ndarray,
        weights:         np.ndarray,
        province_blocks: Dict[str, np.ndarray],
        province_order:  List[str],
    ) -> np.ndarray:
        """
        Rank B provinces by per-province-mean SAW score (1 = best).

        Applies weights to the original normalised matrix (not bootstrap),
        averages each province's rows, then ranks descending.
        """
        B = len(province_order)
        scores = np.zeros(B)
        for i, prov in enumerate(province_order):
            rows = province_blocks[prov]
            if rows.size > 0:
                scores[i] = X_norm[rows, :].mean(axis=0) @ weights
        order = np.argsort(-scores)
        ranks = np.empty(B, dtype=np.int32)
        ranks[order] = np.arange(1, B + 1)
        return ranks

    @staticmethod
    def _compute_kendall_w(rank_matrix: np.ndarray, m: int, N: int) -> float:
        """
        Kendall's W concordance coefficient.

        Parameters
        ----------
        rank_matrix : (N, m) int array  — per-simulation province ranks
        m           : int               — number of provinces
        N           : int               — number of simulations
        """
        if N < 2 or m < 2:
            return 0.0
        row_sums   = rank_matrix.sum(axis=0)           # shape (m,)
        grand_mean = N * (m + 1) / 2.0
        ss         = float(np.sum((row_sums - grand_mean) ** 2))
        denom      = N ** 2 * (m ** 3 - m) / 12.0
        if denom <= 0:
            return 0.0
        return float(np.clip(12.0 * ss / (N ** 2 * (m ** 3 - m)), 0.0, 1.0))

    def _run_mc_ensemble(
        self,
        X:               np.ndarray,
        province_blocks: Dict[str, np.ndarray],
        province_order:  List[str],
        col_names:       List[str],
        theta:           Tuple[float, float, float],
        rng:             np.random.RandomState,
        config:          Any,
        province_labels: Optional[List[str]] = None,
    ) -> _MCEnsembleResult:
        """
        Run the core MC ensemble perturbation-blend loop.

        Parameters
        ----------
        X               : (m_obs, n_cols) raw matrix (Level 1 sub-matrix or Level 2 composite)
        province_blocks : {province: row_positions}
        province_order  : sorted list of province names (defines column order in rank_samples)
        col_names       : names of the n_cols columns (SC codes or criterion IDs)
        theta           : (alpha_a, alpha_b, sigma_scale) blending/noise hyperparameters
        rng             : seeded RandomState
        config          : WeightingConfig
        province_labels : if not None, enables per-province rank distribution in result

        Returns
        -------
        _MCEnsembleResult
        """
        from scipy.stats import kendalltau, spearmanr

        alpha_a, alpha_b, sigma_scale = theta
        N   = config.mc_n_simulations
        eps = config.epsilon
        K   = config.top_k_stability
        B   = len(province_order)
        n_c = len(col_names)

        use_block_bootstrap = (B >= 10)

        # ── Normalise baseline X ──────────────────────────────────────
        X_norm = global_min_max_normalize(X, epsilon=eps)
        if np.isnan(X_norm).any():
            cm = np.nanmean(X_norm, axis=0)
            cm = np.where(np.isnan(cm), eps, cm)
            nr, nc = np.where(np.isnan(X_norm))
            X_norm[nr, nc] = cm[nc]

        # Per-column std of X_norm (for noise calibration)
        col_stds = np.std(X_norm, axis=0)
        col_stds = np.where(col_stds < eps, eps, col_stds)

        # ── Baseline weights + ranking ────────────────────────────────
        df_base = pd.DataFrame(X_norm, columns=col_names)
        try:
            w_e_base = np.array(
                [self._entropy_calc.calculate(df_base).weights.get(c, eps) for c in col_names]
            )
            w_c_base = np.array(
                [self._critic_calc.calculate(df_base).weights.get(c, eps) for c in col_names]
            )
            w_base = (w_e_base + w_c_base) / 2.0
            w_base /= w_base.sum()
        except Exception:
            w_base = np.ones(n_c) / n_c

        r_base = self._saw_province_ranking(X_norm, w_base, province_blocks, province_order)

        # Pre-compute province index → block array list for fast bootstrap
        block_arrays = [province_blocks[p] for p in province_order]

        # ── Storage ──────────────────────────────────────────────────
        weight_samples = np.zeros((N, n_c))
        rank_samples   = np.zeros((N, B), dtype=np.int32)
        kendall_taus:  List[float] = []
        spearman_rhos: List[float] = []
        failed_count   = 0
        n_completed    = 0
        converged_at   = None

        # Convergence state
        conv_check_every = max(10, N // 20)
        conv_min_iters   = max(30, int(N * config.conv_min_iters_fraction))
        prev_mean_conv   = None
        consecutive_passes = 0

        for _s in range(N):
            try:
                # ── Perturbation — Component 1: block bootstrap ───────
                if use_block_bootstrap:
                    sel = rng.randint(0, B, size=B)
                    boot_rows = np.concatenate([block_arrays[i] for i in sel])
                else:
                    # Dirichlet fallback for <10 provinces
                    d = rng.dirichlet(np.ones(len(X)))
                    boot_rows = rng.choice(len(X), size=len(X), replace=True,
                                           p=d / d.sum())
                X_boot = X[boot_rows, :]

                # ── Perturbation — Component 2: log-normal noise ──────
                noise  = rng.randn(*X_boot.shape) * (sigma_scale * col_stds)
                X_pert = X_boot * np.exp(noise)
                X_pert = np.maximum(X_pert, eps)

                # ── Normalise perturbed matrix ─────────────────────────
                X_pn = global_min_max_normalize(X_pert, epsilon=eps)
                if np.isnan(X_pn).any():
                    cm = np.nanmean(X_pn, axis=0)
                    cm = np.where(np.isnan(cm), eps, cm)
                    nr, nc = np.where(np.isnan(X_pn))
                    X_pn[nr, nc] = cm[nc]

                # ── Entropy + CRITIC on perturbed matrix ───────────────
                df_p = pd.DataFrame(X_pn, columns=col_names)
                e_res = self._entropy_calc.calculate(df_p)
                c_res = self._critic_calc.calculate(df_p)
                w_e = np.array([e_res.weights.get(c, eps) for c in col_names])
                w_c = np.array([c_res.weights.get(c, eps) for c in col_names])

                # ── Beta blend (primary; multiplicative fallback) ──────
                beta  = rng.beta(alpha_a, alpha_b)
                w_raw = beta * w_e + (1.0 - beta) * w_c
                s_raw = w_raw.sum()
                if s_raw > 0 and np.isfinite(s_raw) and np.all(w_raw >= 0):
                    w_sim = w_raw / s_raw
                else:
                    # Multiplicative fallback
                    w_prod = w_e * w_c
                    sp     = w_prod.sum()
                    w_sim  = w_prod / sp if sp > 0 else np.ones(n_c) / n_c

                # ── SAW surrogate ranking (original X_norm) ────────────
                r_sim = self._saw_province_ranking(
                    X_norm, w_sim, province_blocks, province_order
                )

                # ── Rank correlations ──────────────────────────────────
                tau, _ = kendalltau(r_base, r_sim)
                rho, _ = spearmanr(r_base, r_sim)

                weight_samples[n_completed] = w_sim
                rank_samples[n_completed]   = r_sim
                kendall_taus.append(float(tau)  if np.isfinite(tau)  else 0.0)
                spearman_rhos.append(float(rho) if np.isfinite(rho)  else 0.0)
                n_completed += 1

            except Exception:
                failed_count += 1
                continue

            # ── Convergence check ──────────────────────────────────────
            if (n_completed >= conv_min_iters
                    and n_completed % conv_check_every == 0
                    and converged_at is None):
                cur_mean = weight_samples[:n_completed].mean(axis=0)
                if prev_mean_conv is not None:
                    delta = np.max(np.abs(cur_mean - prev_mean_conv))
                    if delta < config.convergence_tolerance:
                        consecutive_passes += 1
                        if consecutive_passes >= 2:
                            converged_at = n_completed
                            break
                    else:
                        consecutive_passes = 0
                prev_mean_conv = cur_mean

        # ── Quality flag ──────────────────────────────────────────────
        quality_flag = "ok"
        if n_completed < int(0.80 * N):
            quality_flag = "low_convergence"
            logger.warning(
                "MC ensemble quality: %d/%d simulations succeeded", n_completed, N
            )

        # ── Compute diagnostics from completed samples ─────────────────
        ws = weight_samples[:n_completed]           # (n_completed, n_c)
        rs = rank_samples[:n_completed]             # (n_completed, B)

        mean_w = ws.mean(axis=0) if n_completed > 0 else np.ones(n_c) / n_c
        std_w  = ws.std(axis=0, ddof=1) if n_completed > 1 else np.zeros(n_c)
        ci_lo  = np.percentile(ws, 2.5,  axis=0) if n_completed > 0 else mean_w
        ci_hi  = np.percentile(ws, 97.5, axis=0) if n_completed > 0 else mean_w
        cv_w   = np.where(mean_w > eps, std_w / mean_w, 0.0)

        kendall_w_stat = self._compute_kendall_w(rs, B, n_completed)

        # Top-K rank variance
        if n_completed > 1:
            mean_rk_per_p = rs.mean(axis=0)
            top_k_ix      = np.argsort(mean_rk_per_p)[:K]
            top_k_var     = float(np.mean([np.var(rs[:, i]) for i in top_k_ix]))
        else:
            top_k_var = 0.0

        # Build dicts
        mean_w_d = {c: float(mean_w[j]) for j, c in enumerate(col_names)}
        std_w_d  = {c: float(std_w[j])  for j, c in enumerate(col_names)}
        ci_lo_d  = {c: float(ci_lo[j])  for j, c in enumerate(col_names)}
        ci_hi_d  = {c: float(ci_hi[j])  for j, c in enumerate(col_names)}
        cv_d     = {c: float(cv_w[j])   for j, c in enumerate(col_names)}

        # ── Province rank distribution (Level 2 only) ──────────────────
        province_mean_rank = province_std_rank = None
        province_prob_top1 = province_prob_topk = None
        rank_win_matrix    = None

        if province_labels is not None and n_completed > 0:
            province_mean_rank = {
                p: float(rs[:, i].mean()) for i, p in enumerate(province_labels)
            }
            province_std_rank = {
                p: float(rs[:, i].std()) for i, p in enumerate(province_labels)
            }
            province_prob_top1 = {
                p: float((rs[:, i] == 1).mean()) for i, p in enumerate(province_labels)
            }
            province_prob_topk = {
                p: float((rs[:, i] <= K).mean()) for i, p in enumerate(province_labels)
            }
            # Pairwise win probability matrix as nested dict
            rwm = np.zeros((B, B))
            for i in range(B):
                for j in range(B):
                    if i == j:
                        rwm[i, j] = 0.5
                    else:
                        rwm[i, j] = float((rs[:, i] < rs[:, j]).mean())
            rank_win_matrix = {
                province_labels[i]: {
                    province_labels[j]: float(rwm[i, j]) for j in range(B)
                }
                for i in range(B)
            }

        return _MCEnsembleResult(
            mean_weights       = mean_w_d,
            std_weights        = std_w_d,
            ci_lower           = ci_lo_d,
            ci_upper           = ci_hi_d,
            cv_weights         = cv_d,
            avg_kendall_tau    = float(np.mean(kendall_taus))  if kendall_taus  else 0.0,
            avg_spearman_rho   = float(np.mean(spearman_rhos)) if spearman_rhos else 0.0,
            kendall_w          = kendall_w_stat,
            top_k_rank_var     = top_k_var,
            province_mean_rank = province_mean_rank,
            province_std_rank  = province_std_rank,
            province_prob_top1 = province_prob_top1,
            province_prob_topk = province_prob_topk,
            rank_win_matrix    = rank_win_matrix,
            converged_at       = converged_at,
            n_completed        = n_completed,
            quality_flag       = quality_flag,
        )

    def _tune_hyperparameters(
        self,
        panel_df:        pd.DataFrame,
        criteria_groups: Dict[str, List[str]],
        province_blocks: Dict[str, np.ndarray],
        province_order:  List[str],
        r_base:          np.ndarray,
        rng:             np.random.RandomState,
    ) -> Tuple[float, float, float]:
        """
        64-point coarse grid search + optional Bayesian GP refinement.

        Tunes (α_a, α_b, σ_scale) to maximise AvgKendall τ_b on the Level 2
        criterion composite matrix using N_tune simulations per grid point.
        """
        cfg = self.config
        all_sc_cols   = [sc for scs in criteria_groups.values() for sc in scs]
        criterion_ids = list(criteria_groups.keys())
        X_raw_all     = panel_df[all_sc_cols].values.astype(np.float64)

        # Build composite matrix with equal local weights (fast approximation)
        Z_tune = np.zeros((len(panel_df), len(criterion_ids)))
        for k_idx, (crit_id, sc_cols_k) in enumerate(criteria_groups.items()):
            col_idx = [all_sc_cols.index(sc) for sc in sc_cols_k]
            Z_tune[:, k_idx] = X_raw_all[:, col_idx].mean(axis=1)

        # Tuning config: fewer simulations, no re-tuning
        tune_cfg = dataclasses.replace(
            cfg,
            mc_n_simulations = cfg.mc_n_tuning_simulations,
            perform_tuning   = False,
        )

        grid_a = [0.5, 1.0, 2.0, 4.0]
        grid_b = [0.5, 1.0, 2.0, 4.0]
        grid_s = [0.01, 0.03, 0.06, 0.10]

        best_score = -np.inf
        best_theta = (cfg.beta_a, cfg.beta_b, cfg.noise_sigma_scale)
        top5: List[Tuple[float, Tuple[float, float, float]]] = []

        for a_a in grid_a:
            for a_b in grid_b:
                if a_a + a_b > 8.0:   # prune extreme concentration
                    continue
                for s_sc in grid_s:
                    try:
                        r_t = self._run_mc_ensemble(
                            X               = Z_tune,
                            province_blocks = province_blocks,
                            province_order  = province_order,
                            col_names       = criterion_ids,
                            theta           = (a_a, a_b, s_sc),
                            rng             = rng,
                            config          = tune_cfg,
                            province_labels = None,
                        )
                        score = r_t.avg_kendall_tau
                        if score > best_score:
                            best_score = score
                            best_theta = (a_a, a_b, s_sc)
                        top5.append((score, (a_a, a_b, s_sc)))
                        top5.sort(key=lambda x: x[0], reverse=True)
                        top5 = top5[:5]
                    except Exception:
                        continue

        logger.info(
            "  Grid best θ=(%s, %s, %s), AvgKendall=%.4f",
            *best_theta, best_score,
        )

        # ── Optional Bayesian GP refinement ───────────────────────────
        if cfg.use_bayesian_tuning:
            try:
                from skopt import gp_minimize  # type: ignore[import]
                from skopt.space import Real   # type: ignore[import]

                space = [
                    Real(0.5, 5.0,  name="alpha_a"),
                    Real(0.5, 5.0,  name="alpha_b"),
                    Real(0.005, 0.15, name="sigma_scale"),
                ]

                def _objective(params: List[float]) -> float:
                    a_a, a_b, s_sc = params
                    if a_a + a_b > 8.0:
                        return 0.0
                    try:
                        r_t = self._run_mc_ensemble(
                            X               = Z_tune,
                            province_blocks = province_blocks,
                            province_order  = province_order,
                            col_names       = criterion_ids,
                            theta           = (a_a, a_b, s_sc),
                            rng             = rng,
                            config          = tune_cfg,
                            province_labels = None,
                        )
                        return -r_t.avg_kendall_tau   # minimise negated
                    except Exception:
                        return 0.0

                bo_result = gp_minimize(
                    _objective, space,
                    n_calls      = 20,
                    random_state = cfg.seed if cfg.seed is not None else 42,
                )
                if -bo_result.fun > best_score:
                    best_theta = tuple(bo_result.x)
                    best_score = -bo_result.fun
                    logger.info(
                        "  Bayesian refined θ=(%s), AvgKendall=%.4f",
                        best_theta, best_score,
                    )
            except ImportError:
                logger.warning(
                    "scikit-optimize not available; Bayesian tuning skipped."
                )

        return best_theta  # type: ignore[return-value]

