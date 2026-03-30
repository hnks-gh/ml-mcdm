"""
Incremental Ensemble Updater (E-10).

This module provides the `IncrementalEnsembleUpdater` class, which performs 
lightweight model updates to incorporate new observations (e.g., the most 
recent year) into a pre-trained ensemble. This allows the system to 
exploit 100% of available data for final predictions without re-running 
the full multi-fold training pipeline.

Strategies
----------
- **Gradient Continuation**: For CatBoost models, resumes training from the 
  existing booster for a small number of iterations with a reduced 
  learning rate.
- **Full Retrain**: For linear and kernel models, performs a quick refit on 
  the complete historical dataset (if provided) or just the new observations.
- **Meta-Weight Blending**: Recalibrates ensemble weights using a 
  secondary Non-Negative Least Squares (NNLS) fit on the new data, 
  blended with historical meta-weights to maintain stability.

Key Parameters
--------------
- `gamma`: Controls the balance between historical weights and new 
  calibration signal (default 0.3).
- `catboost_lr_factor`: Scales the learning rate for continuation to 
  prevent overfitting to the newest year.

References
----------
- Ljung & Söderström (1983). "Theory and Practice of Recursive 
  Identification." MIT Press.
"""
from __future__ import annotations

import copy
import warnings
from typing import Dict, Optional

import numpy as np
from scipy.optimize import nnls


class IncrementalEnsembleUpdater:
    """Online update of a fitted SuperLearner ensemble with new observations.

    Primary use-case: update a 2012–2023 trained ensemble with 2024 data
    prior to generating the 2025 forecast, exploiting 100% of available
    history without re-running the full multi-fold training pipeline.

    Parameters
    ----------
    strategy : {'auto', 'full_retrain'}
        Update strategy per base model (see module docstring).
    gamma : float
        Blending weight for meta-weight calibration (default 0.3).
        Higher γ → more weight on new calibration signal.
    catboost_extra_iter : int
        Additional CatBoost iterations during gradient continuation (E-10B).
    catboost_lr_factor : float
        Learning-rate multiplier for gradient continuation (< 1 to prevent
        over-fitting to the new year, default 0.5).
    min_rls_obs : int
        Minimum observations for RLS fallback threshold.
    verbose : bool
    """

    def __init__(
        self,
        strategy: str = 'auto',
        gamma: float = 0.3,
        catboost_extra_iter: int = 50,
        catboost_lr_factor: float = 0.5,
        verbose: bool = True,
    ):
        """
        Initialize the incremental ensemble updater.

        Parameters
        ----------
        strategy : {'auto', 'full_retrain'}, default='auto'
            The update strategy to use for base models. 'auto' selects 
            per-model optimal strategies.
        gamma : float, default=0.3
            Blending weight for meta-weight calibration. Higher values put 
            more weight on the most recent data.
        catboost_extra_iter : int, default=50
            Number of additional boosting iterations for CatBoost continuation.
        catboost_lr_factor : float, default=0.5
            Scaling factor for the learning rate during continuation.
        verbose : bool, default=True
            Whether to print progress information.
        """
        self.strategy            = strategy
        self.gamma               = float(np.clip(gamma, 0.0, 1.0))
        self.catboost_extra_iter = catboost_extra_iter
        self.catboost_lr_factor  = catboost_lr_factor
        self.verbose             = verbose

        # Diagnostics populated after update()
        self.updated_models_:   list = []
        self.update_strategy_per_model_: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        ensemble,
        X_new: np.ndarray,
        y_new: np.ndarray,
        entity_indices_new: Optional[np.ndarray] = None,
        year_label_new: Optional[int] = None,
        X_all: Optional[np.ndarray] = None,
        y_all: Optional[np.ndarray] = None,
        entity_indices_all: Optional[np.ndarray] = None,
        per_model_X_new: Optional[Dict[str, np.ndarray]] = None,
        per_model_X_all: Optional[Dict[str, np.ndarray]] = None,
    ):
        """Update base models and meta-weights with new observations.

        Parameters
        ----------
        ensemble : SuperLearner (fitted)
            The fitted ``SuperLearner`` instance to update in-place.
            A *deep copy* is made so the original is never mutated.
        X_new : ndarray, shape (n_new, n_features)
            Features for the new year (e.g., 2024).
        y_new : ndarray, shape (n_new,) or (n_new, n_outputs)
            Targets for the new year.
        entity_indices_new : ndarray, shape (n_new,), optional
        year_label_new : int, optional
            Calendar year of the new observations (for logging only).
        X_all : ndarray, shape (n_all, n_features), optional
            Full historical feature matrix (2012–2024) for ``full_retrain``.
            When ``None``, models that cannot be incrementally updated are
            updated on ``X_new`` only.
        y_all : ndarray, shape (n_all, n_outputs), optional
            Full historical targets (2012–2024).
        entity_indices_all : ndarray, optional
        per_model_X_new : dict, optional
            Model-specific feature matrices for new year (mirrors
            ``per_model_X`` in ``SuperLearner.fit``).
        per_model_X_all : dict, optional
            Model-specific feature matrices for all years.

        Returns
        -------
        updated_ensemble : SuperLearner
            Deep copy of ``ensemble`` with updated base models and
            blended meta-weights.  Original ``ensemble`` is unchanged.
        """
        if y_new.ndim == 1:
            y_new = y_new.reshape(-1, 1)
        if y_all is not None and y_all.ndim == 1:
            y_all = y_all.reshape(-1, 1)

        # Work on a deep copy
        sl = copy.deepcopy(ensemble)

        yr_str = f" ({year_label_new})" if year_label_new is not None else ""
        if self.verbose:
            print(f"  IncrementalUpdater: updating ensemble with {len(X_new)}"
                  f" new observations{yr_str}...")

        # Step 1: Update each base model
        for name, model in sl._fitted_base_models.items():
            strat = self._model_strategy(name)
            self.update_strategy_per_model_[name] = strat

            # Select per-model feature matrices
            X_m_new = (per_model_X_new[name] if per_model_X_new
                       and name in per_model_X_new else X_new)
            X_m_all = (per_model_X_all[name] if per_model_X_all
                       and name in per_model_X_all
                       else X_all)

            try:
                if strat == 'catboost_continuation':
                    self._catboost_continuation(model, X_m_new, y_new,
                                                X_m_all, y_all)
                else:  # 'full_retrain' or fallback
                    self._full_retrain(model, X_m_new, y_new,
                                       entity_indices_new,
                                       X_m_all, y_all, entity_indices_all)
                self.updated_models_.append(name)
            except Exception as exc:
                if self.verbose:
                    print(f"    [{name}] update failed ({exc}), skipping.")

        # Step 2: Update meta-weights via calibration blend
        self._update_meta_weights(sl, X_new, y_new, entity_indices_new,
                                   per_model_X_new)

        if self.verbose:
            print(f"  Updated: {self.updated_models_}")

        return sl

    # ------------------------------------------------------------------
    # Per-model update strategies
    # ------------------------------------------------------------------

    def _model_strategy(self, name: str) -> str:
        """Map model name to update strategy when strategy='auto'."""
        if self.strategy == 'full_retrain':
            return 'full_retrain'
        # 'auto' routing
        name_lo = name.lower()
        if 'catboost' in name_lo or 'gradientboost' in name_lo:
            return 'catboost_continuation'
        return 'full_retrain'     # BayesianRidge, QuantileRF, others

    def _catboost_continuation(
        self,
        model,
        X_new: np.ndarray,
        y_new: np.ndarray,
        X_all: Optional[np.ndarray],
        y_all: Optional[np.ndarray],
    ) -> None:
        """CatBoost gradient continuation from existing booster.

        Uses CatBoost's ``init_model`` parameter to resume training for
        ``catboost_extra_iter`` additional rounds at a reduced learning rate.
        Trains on the new data only (or full history when X_all is provided)
        so the booster sees a weight-adjusted loss that incorporates the
        current residuals.

        Falls back to full retrain if CatBoost API is unavailable.
        """
        cb_model = getattr(model, 'model', None)
        if cb_model is None or not hasattr(cb_model, 'fit'):
            self._full_retrain(model, X_new, y_new, None, X_all, y_all, None)
            return

        # Determine training data: prefer full history
        X_fit  = np.vstack([X_all, X_new])   if X_all is not None else X_new
        y_fit  = np.vstack([y_all, y_new])   if y_all is not None else y_new
        if y_fit.ndim > 1 and y_fit.shape[1] == 1:
            y_fit = y_fit.ravel()

        try:
            # Import CatBoostRegressor to create continuation model
            from catboost import CatBoostRegressor as _CBR
            n_out = y_fit.shape[1] if y_fit.ndim > 1 else 1
            loss  = 'MultiRMSE' if n_out > 1 else 'RMSE'
            cont_model = _CBR(
                iterations      = self.catboost_extra_iter,
                learning_rate   = max(
                    cb_model.get_params().get('learning_rate', 0.05)
                    * self.catboost_lr_factor, 1e-4
                ),
                depth           = cb_model.get_params().get('depth', 6),
                l2_leaf_reg     = cb_model.get_params().get('l2_leaf_reg', 3.0),
                loss_function   = loss,
                bootstrap_type  = 'Bernoulli',
                subsample       = cb_model.get_params().get('subsample', 0.8),
                boosting_type   = 'Plain',
                random_seed     = 42,
                verbose         = 0,
                allow_writing_files = False,
            )
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                cont_model.fit(X_fit, y_fit, init_model=cb_model)
            model.model = cont_model
            model._fitted = True
        except Exception:
            # Fallback: full retrain
            self._full_retrain(model, X_new, y_new, None, X_all, y_all, None)

    def _full_retrain(
        self,
        model,
        X_new:          np.ndarray,
        y_new:          np.ndarray,
        entity_indices: Optional[np.ndarray],
        X_all:          Optional[np.ndarray],
        y_all:          Optional[np.ndarray],
        entity_all:     Optional[np.ndarray],
    ) -> None:
        """Full retrain on either X_all (preferred) or X_new (fallback)."""
        if X_all is not None and y_all is not None:
            X_fit = X_all
            y_fit = y_all
            ent   = entity_all
        else:
            X_fit = X_new
            y_fit = y_new
            ent   = entity_indices

        import inspect
        sig = inspect.signature(model.fit)
        try:
            if 'entity_indices' in sig.parameters and ent is not None:
                model.fit(X_fit, y_fit, entity_indices=ent)
            else:
                model.fit(X_fit, y_fit)
        except Exception:
            pass  # leave model unchanged

    # ------------------------------------------------------------------
    # Meta-weight update
    # ------------------------------------------------------------------

    def _update_meta_weights(
        self,
        sl,
        X_new:          np.ndarray,
        y_new:          np.ndarray,
        entity_indices: Optional[np.ndarray],
        per_model_X_new: Optional[Dict[str, np.ndarray]],
    ) -> None:
        """Blend existing meta-weights with new-year calibration weights.

        Secondary calibration step (distinct from SuperLearner's Ridge meta-learner):
        1. Collects predictions from all updated base models on X_new.
        2. Fits NNLS on new-year predictions → new-year calibration weights w_new.
        3. Blends: w_final = (1−γ)·w_prev + γ·w_new, normalised to sum=1.
        """
        if y_new.ndim == 1:
            y_new = y_new.reshape(-1, 1)

        n_models = len(sl._fitted_base_models)
        model_names = list(sl._fitted_base_models.keys())
        n_outputs = sl._n_outputs

        # Collect new-year predictions
        preds = []
        active_names = []
        for name, model in sl._fitted_base_models.items():
            X_m = (per_model_X_new[name] if per_model_X_new
                   and name in per_model_X_new else X_new)
            try:
                import inspect
                sig = inspect.signature(model.predict)
                if 'entity_indices' in sig.parameters and entity_indices is not None:
                    p = model.predict(X_m, entity_indices=entity_indices)
                else:
                    p = model.predict(X_m)
                if isinstance(p, np.ndarray):
                    if p.ndim == 1:
                        p = p.reshape(-1, 1)
                    preds.append(p)
                    active_names.append(name)
            except Exception:
                pass

        if len(preds) < 2:
            return  # not enough models to recalibrate

        # Per-output NNLS calibration → w_new
        all_calib_coefs = []
        for out_col in range(n_outputs):
            preds_stack = np.column_stack([
                p[:, min(out_col, p.shape[1] - 1)] for p in preds
            ])   # (n_new, n_active)
            y_col = y_new[:, out_col]
            valid = ~np.isnan(preds_stack).any(axis=1) & ~np.isnan(y_col)
            if valid.sum() < 2:
                # Not enough data: equal weights
                all_calib_coefs.append(
                    np.ones(len(active_names)) / len(active_names)
                )
                continue
            try:
                coefs, _ = nnls(preds_stack[valid], y_col[valid])
            except Exception:
                coefs = np.ones(len(active_names)) / len(active_names)
            s = coefs.sum()
            all_calib_coefs.append(coefs / s if s > 1e-12 else coefs)

        w_new_active = np.mean(all_calib_coefs, axis=0)  # (n_active,)

        # Map back to full model vector
        w_new_full = np.zeros(n_models)
        for i, name in enumerate(active_names):
            if name in model_names:
                w_new_full[model_names.index(name)] = w_new_active[i]
        s = w_new_full.sum()
        if s < 1e-12:
            return  # calibration failed; keep existing weights
        w_new_full /= s

        # Blend with existing weights
        w_prev  = np.array([sl._meta_weights.get(n, 0.0) for n in model_names])
        w_blend = (1.0 - self.gamma) * w_prev + self.gamma * w_new_full
        w_blend = np.maximum(w_blend, 0.0)
        s_blend = w_blend.sum()
        if s_blend > 1e-12:
            w_blend /= s_blend
            sl._meta_weights = dict(zip(model_names, w_blend))

        if self.verbose:
            print("  Meta-weights post-update:",
                  {n: round(v, 4) for n, v in sl._meta_weights.items()})
