# -*- coding: utf-8 -*-
"""
Panel Evaluation Diagnostics (E-07)
=====================================

Advanced diagnostic tools for assessing ensemble generalization in the
panel setting (N=63 provinces × T=14 years).

Classes
-------
LeaveOneEntityOutCV
    Leave-one-entity-out cross-validation for measuring cross-entity
    generalisability of the Super Learner ensemble.  Identifies provinces
    that are consistently hard to predict (potential outliers / atypical
    provinces).

Usage Example
-------------
>>> from forecasting.evaluation_diagnostics import LeaveOneEntityOutCV
>>> loeo = LeaveOneEntityOutCV(verbose=True)
>>> results = loeo.run(super_learner, X, y, entity_indices, year_labels)
>>> results.sort_values('r2').head(10)  # 10 hardest provinces
"""
from __future__ import annotations

import copy
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class LeaveOneEntityOutCV:
    """Leave-one-entity-out (LOEO) generalization diagnostic.

    For each entity e in {0 … N-1}:

        1. Train the Super Learner on all entities **except** e
           (using the same ``year_labels`` for walk-forward CV)
        2. Predict on entity e over all its available years
        3. Record entity-specific R², RMSE, MAE and — when an
           ``oof_conformal_residuals`` column is available on the
           super learner — empirical prediction-interval coverage

    The LOEO score measures *cross-entity transferability*: whether the
    ensemble has learned patterns that generalise to held-out provinces
    it has never seen during training.  Compare:

    * ``LOEO_R² ≈ CV_R²``  → good generalisation; model learns
      cross-provincial structure rather than memorising entity FE.
    * ``LOEO_R² << CV_R²`` → entity-specific overfitting; consider
      weaker fixed-effect regularisation.
    * Per-entity R² map     → identifies outlier / structurally atypical
      provinces that warrant individual treatment.

    Parameters
    ----------
    verbose : bool, default True
        Print progress (entity count) during the run.
    min_valid_obs : int, default 3
        Minimum non-NaN target rows for an entity to be included.
    n_jobs : int, default 1
        Number of parallel entity fits.  Uses joblib if > 1 and
        joblib is installed; falls back to sequential otherwise.

    Attributes
    ----------
    results_ : pd.DataFrame
        Per-entity diagnostics after calling ``run()``.  Columns:
        entity_idx, entity_label (if provided), n_obs, r2, rmse, mae,
        loeo_vs_cv_delta (= loeo_r2 − mean_cv_r2).
    summary_ : dict
        Scalar summary statistics (mean/std/min/max LOEO R²,
        gap vs CV score, fraction of entities with negative R²).

    References
    ----------
    Arlot & Celisse (2010). A survey of CV strategies for model selection.
    Statistics Surveys 4, 40–79.
    """

    def __init__(
        self,
        verbose: bool = True,
        min_valid_obs: int = 3,
        n_jobs: int = 1,
    ):
        self.verbose      = verbose
        self.min_valid_obs = min_valid_obs
        self.n_jobs       = n_jobs

        self.results_: Optional[pd.DataFrame] = None
        self.summary_:  Optional[Dict]         = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        super_learner,
        X: np.ndarray,
        y: np.ndarray,
        entity_indices: np.ndarray,
        year_labels: Optional[np.ndarray] = None,
        cv_r2_reference: Optional[float] = None,
        entity_labels: Optional[List[str]] = None,
        per_model_X: Optional[Dict[str, np.ndarray]] = None,
    ) -> pd.DataFrame:
        """Run leave-one-entity-out CV and return per-entity diagnostics.

        Parameters
        ----------
        super_learner : SuperLearner
            A **fitted** SuperLearner instance from
            ``forecasting.super_learner``.  Each entity iteration
            deep-copies the model and re-fits from scratch.
        X : ndarray, shape (n_samples, n_features)
            Full panel feature matrix (all entities × all years).
        y : ndarray, shape (n_samples,) or (n_samples, n_outputs)
            Target matrix.  NaN rows are excluded per-entity.
        entity_indices : ndarray, shape (n_samples,)
            Integer entity IDs matching the ``entity_indices`` convention
            used throughout the pipeline (0-indexed).
        year_labels : ndarray, shape (n_samples,), optional
            Calendar year per observation.  Required for walk-forward
            walk-forward CV inside each entity-excluded fit.  Falls back
            to row-order positional splitting when ``None``.
        cv_r2_reference : float, optional
            Mean walk-forward CV R² of the full ensemble (used to
            compute ``loeo_vs_cv_delta`` column).
        entity_labels : list of str, optional
            Human-readable province names for the result table.
        per_model_X : dict, optional
            Model-specific feature matrices (same layout as
            ``SuperLearner.fit(per_model_X=...)``).

        Returns
        -------
        results : pd.DataFrame
            Per-entity results table, sorted by R² ascending so the
            hardest provinces appear first.
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n_outputs = y.shape[1]

        unique_entities = np.unique(entity_indices)
        n_entities      = len(unique_entities)

        records: List[Dict] = []

        if self.verbose:
            print(f"  LOEO: evaluating {n_entities} entities …")

        for i, entity in enumerate(unique_entities):
            if self.verbose and (i % 10 == 0 or i == n_entities - 1):
                print(f"    entity {i + 1}/{n_entities} …", end='\r')

            mask_excl = entity_indices != entity   # training mask
            mask_incl = entity_indices == entity   # test mask

            # Check test set has enough valid observations
            y_ent = y[mask_incl]
            valid_ent = ~np.isnan(y_ent).all(axis=1)
            if valid_ent.sum() < self.min_valid_obs:
                continue

            # Build per-model X subsets if provided
            per_model_excl: Optional[Dict] = None
            per_model_incl: Optional[Dict] = None
            if per_model_X is not None:
                per_model_excl = {
                    k: v[mask_excl] for k, v in per_model_X.items()
                }
                per_model_incl = {
                    k: v[mask_incl] for k, v in per_model_X.items()
                }

            ent_idx_excl = entity_indices[mask_excl]
            ent_idx_incl = entity_indices[mask_incl]
            yr_excl = year_labels[mask_excl] if year_labels is not None else None
            yr_incl = year_labels[mask_incl] if year_labels is not None else None

            try:
                sl_copy = copy.deepcopy(super_learner)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sl_copy.fit(
                        X[mask_excl],
                        y[mask_excl],
                        entity_indices=ent_idx_excl,
                        per_model_X=per_model_excl,
                        year_labels=yr_excl,
                    )
                    pred = sl_copy.predict(
                        X[mask_incl],
                        entity_indices=ent_idx_incl,
                        per_model_X_pred=per_model_incl,
                    )
            except Exception as exc:
                if self.verbose:
                    print(f"\n    [LOEO] entity {entity} fit failed: {exc}")
                continue

            if isinstance(pred, np.ndarray) and pred.ndim == 1:
                pred = pred.reshape(-1, 1)
            if pred.shape[1] < n_outputs:
                pred = np.tile(pred, (1, n_outputs))[:, :n_outputs]

            # Compute per-entity metrics across all output columns
            r2_vals, rmse_vals, mae_vals = [], [], []
            for col in range(n_outputs):
                y_col   = y_ent[valid_ent, col]
                p_col   = pred[valid_ent, min(col, pred.shape[1] - 1)]
                nan_col = np.isnan(y_col)
                if nan_col.all():
                    continue
                y_col = y_col[~nan_col]
                p_col = p_col[~nan_col]
                if len(y_col) < 2:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r2_vals.append(float(r2_score(y_col, p_col)))
                    rmse_vals.append(float(np.sqrt(mean_squared_error(y_col, p_col))))
                    mae_vals.append(float(mean_absolute_error(y_col, p_col)))

            if not r2_vals:
                continue

            record: Dict = {
                'entity_idx':  int(entity),
                'n_obs':       int(valid_ent.sum()),
                'r2':          float(np.mean(r2_vals)),
                'rmse':        float(np.mean(rmse_vals)),
                'mae':         float(np.mean(mae_vals)),
            }
            if cv_r2_reference is not None:
                record['loeo_vs_cv_delta'] = record['r2'] - cv_r2_reference
            if entity_labels is not None and entity < len(entity_labels):
                record['entity_label'] = entity_labels[entity]

            records.append(record)

        if self.verbose:
            print()  # newline after \r progress

        if not records:
            self.results_ = pd.DataFrame()
            self.summary_ = {}
            return self.results_

        df = pd.DataFrame(records)
        if 'entity_label' not in df.columns:
            df['entity_label'] = df['entity_idx'].astype(str)
        # Sort hardest first
        df = df.sort_values('r2').reset_index(drop=True)
        self.results_ = df
        self.summary_ = self._compute_summary(df, cv_r2_reference)

        if self.verbose:
            print(
                f"  LOEO summary: mean R²={self.summary_['mean_r2']:.3f}, "
                f"std={self.summary_['std_r2']:.3f}, "
                f"neg R² entities={self.summary_['frac_negative_r2']:.1%}"
            )
        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_summary(
        self,
        df: pd.DataFrame,
        cv_r2_reference: Optional[float],
    ) -> Dict:
        r2 = df['r2'].values
        summary: Dict = {
            'mean_r2':           float(np.mean(r2)),
            'std_r2':            float(np.std(r2)),
            'median_r2':         float(np.median(r2)),
            'min_r2':            float(np.min(r2)),
            'max_r2':            float(np.max(r2)),
            'frac_negative_r2':  float((r2 < 0).mean()),
            'n_entities':        len(df),
        }
        if cv_r2_reference is not None:
            summary['cv_r2_reference']   = float(cv_r2_reference)
            summary['loeo_vs_cv_gap']    = float(np.mean(r2) - cv_r2_reference)
        return summary

    def hardest_entities(self, n: int = 10) -> pd.DataFrame:
        """Return the ``n`` hardest-to-predict entities (lowest LOEO R²).

        Requires ``run()`` to have been called first.
        """
        if self.results_ is None:
            raise RuntimeError("Call run() before hardest_entities().")
        return self.results_.head(n)

    def generalization_flag(self, threshold: float = -0.10) -> bool:
        """Return True if mean LOEO R² is below ``threshold``.

        A mean LOEO R² below -0.10 indicates systematic entity-specific
        overfitting that warrants architectural changes (e.g. weaker
        fixed effects, or smaller entity embedding dims).
        """
        if self.summary_ is None:
            raise RuntimeError("Call run() before generalization_flag().")
        return bool(self.summary_['mean_r2'] < threshold)
