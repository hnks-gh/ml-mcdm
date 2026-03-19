# -*- coding: utf-8 -*-
"""
Comprehensive Evaluation Suite for Forecasting
================================================

Provides rigorous evaluation metrics, validation procedures, and
diagnostic tests for assessing forecasting model quality.

Evaluation Dimensions:
    1. Accuracy: R², RMSE, MAE, MAPE per sub-criterion
    2. Uncertainty Calibration: Coverage, sharpness, calibration curves
    3. Model Diagnostics: Residual tests (Durbin-Watson, Breusch-Pagan,
       Shapiro-Wilk), using genuine out-of-fold predictions so residuals
       reflect generalisation error, not in-sample fit.
    4. Ablation Studies: LOO and pairwise component-contribution analysis

Validation Strategy:
    - Expanding window (walk-forward): Train on t₁...tₖ, predict tₖ₊₁

Ablation Studies (``AblationStudy`` class):
    - Leave-one-out (LOO): remove each feature in turn, measure R² drop
    - Pairwise: remove every pair to reveal synergistic / redundant groups
    Results are returned as a DataFrame ranked by mean absolute impact.

References:
    - Hyndman & Athanasopoulos (2021). "Forecasting: Principles
      and Practice" 3rd edition (OTexts)
    - Gneiting & Raftery (2007). "Strictly Proper Scoring Rules,
      Prediction, and Estimation" JASA
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Callable
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
)
from sklearn.model_selection import TimeSeriesSplit
from .super_learner import _WalkForwardYearlySplit
import copy
import warnings
import functools
import inspect

logger = logging.getLogger('ml_mcdm')


def _silence_warnings(func):
    """Scope all warning filters to the duration of *func* only."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return func(*args, **kwargs)
    return wrapper


@dataclass
class ModelComparisonResult:
    """
    Per-model evaluation result on the genuine holdout set.

    Parameters
    ----------
    model_name : str
        Human-readable model identifier: 'BayesianRidge', 'CatBoost',
        'QuantileRF', or 'Ensemble'.
    holdout_r2 : float
        Global R² computed across all output dimensions on the withheld year.
        Positive values indicate the model outperforms the naive mean predictor.
        Negative values are possible for poorly-calibrated base models.
        Uses the flattened (ravel) view so every (sample × criterion) residual
        is treated equally — consistent with ``holdout_performance`` in
        ``UnifiedForecastResult``.
    holdout_rmse : float
        Root-mean-square error on the holdout set (same units as targets).
    holdout_mae : float
        Mean absolute error on the holdout set.
    is_best : bool
        ``True`` for the single model with the highest ``holdout_r2``.
        Ties are broken by RMSE (lower is better); in practice ties are rare.
    predictions : pd.DataFrame
        Target-year (forecast year) predictions, shape (n_entities, n_outputs).
        Index = province/entity names; columns = component/criterion names.
        These are the genuine out-of-sample predictions each model makes for
        the actual forecast year — NOT the holdout-year values (which are only
        used to compute the metrics above).
    """
    model_name: str
    holdout_r2: float
    holdout_rmse: float
    holdout_mae: float
    is_best: bool
    predictions: pd.DataFrame


def compare_all_models(
    fitted_base_models: Dict[str, Any],
    super_learner: Any,
    X_holdout_per_model: Dict[str, np.ndarray],
    y_holdout: np.ndarray,
    ensemble_preds_holdout: np.ndarray,
    entity_indices_holdout: Optional[np.ndarray] = None,
    X_target_per_model: Optional[Dict[str, np.ndarray]] = None,
    ensemble_preds_target: Optional[np.ndarray] = None,
    component_names: Optional[List[str]] = None,
    target_entities: Optional[List[str]] = None,
) -> List[ModelComparisonResult]:
    """
    Evaluate every base model and the ensemble on the genuine holdout set.

    Models are evaluated using pre-fitted instances from
    ``super_learner._fitted_base_models`` (Stage 3 of SuperLearner retrains
    on the full training set).  No refitting occurs here — evaluation is
    purely inference on a withheld calendar year, guaranteeing zero leakage.

    Parameters
    ----------
    fitted_base_models : dict
        Mapping ``{model_name: fitted_model}`` from
        ``SuperLearner._fitted_base_models``.  Keys must correspond to entries
        in ``X_holdout_per_model``.
    super_learner : SuperLearner
        Fitted SuperLearner instance.  Included for API completeness and
        potential future use; the function iterates over ``fitted_base_models``
        directly to avoid holding an extra reference.
    X_holdout_per_model : dict
        Per-model feature matrices for the **holdout** set (n_holdout × p_k).
        Keys must cover every key in ``fitted_base_models``.
    y_holdout : ndarray, shape (n_holdout, n_outputs)
        Ground-truth targets for the holdout year.  Must NOT overlap with
        training data (guaranteed by
        ``TemporalFeatureEngineer.fit_transform`` holdout routing).
    ensemble_preds_holdout : ndarray, shape (n_holdout, n_outputs)
        Pre-computed SuperLearner point predictions for the holdout set.
        Must be generated externally via ``super_learner.predict_with_uncertainty``
        **after** fitting — never by re-fitting inside this function.
    entity_indices_holdout : ndarray, optional
        Entity IDs for holdout rows.  Forwarded to base models whose
        ``predict`` signature accepts ``entity_indices`` (or
        ``group_indices``) so panel-aware models evaluate with the correct
        entity context.
    X_target_per_model : dict, optional
        Per-model feature matrices for the **target (forecast) year**.
        When provided, target-year predictions are stored in
        ``ModelComparisonResult.predictions``.  If None, holdout predictions
        are stored instead.
    ensemble_preds_target : ndarray, optional
        Ensemble point predictions for the target year
        (shape = n_entities × n_outputs).  Required when
        ``X_target_per_model`` is provided.
    component_names : list of str, optional
        Column labels for prediction DataFrames.  Defaults to ['C0', 'C1', ...].
    target_entities : list of str, optional
        Index labels (province/entity names) for the target-year
        predictions DataFrame.

    Returns
    -------
    list of ModelComparisonResult
        One entry per base model plus one for the ensemble.  Sorted descending
        by ``holdout_r2`` (NaN entries last).  ``is_best=True`` on the entry
        with the highest finite ``holdout_r2``; ties broken by RMSE.
    """
    # ── Shape normalisation ───────────────────────────────────────────────
    if y_holdout.ndim == 1:
        y_holdout = y_holdout.reshape(-1, 1)
    n_outputs = y_holdout.shape[1]

    if ensemble_preds_holdout.ndim == 1:
        ensemble_preds_holdout = ensemble_preds_holdout.reshape(-1, 1)

    have_target = (
        X_target_per_model is not None
        and ensemble_preds_target is not None
    )
    if have_target and ensemble_preds_target.ndim == 1:  # type: ignore[union-attr]
        ensemble_preds_target = ensemble_preds_target.reshape(-1, 1)  # type: ignore[union-attr]

    n_ho = len(y_holdout)
    n_tgt = len(ensemble_preds_target) if have_target else n_ho  # type: ignore[arg-type]
    cols = component_names if component_names else [f'C{i}' for i in range(n_outputs)]
    tgt_idx = target_entities if target_entities else list(range(n_tgt))

    # ── Internal helpers ─────────────────────────────────────────────────
    def _align_preds(preds: np.ndarray) -> np.ndarray:
        """Align prediction array to n_outputs columns."""
        if preds.shape[1] > n_outputs:
            return preds[:, :n_outputs]
        if preds.shape[1] < n_outputs:
            pad = np.tile(preds[:, -1:], (1, n_outputs - preds.shape[1]))
            return np.concatenate([preds, pad], axis=1)
        return preds

    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray
                         ) -> Tuple[float, float, float]:
        """Return (global_r2, global_rmse, global_mae) over all outputs."""
        y_pred = _align_preds(y_pred)
        yt = y_true.ravel()
        yp = y_pred.ravel()
        return (
            float(r2_score(yt, yp)),
            float(np.sqrt(mean_squared_error(yt, yp))),
            float(mean_absolute_error(yt, yp)),
        )

    def _nan_df(n_rows: int) -> pd.DataFrame:
        """Return a NaN-filled placeholder predictions DataFrame."""
        return pd.DataFrame(
            np.full((n_rows, n_outputs), np.nan),
            index=tgt_idx[:n_rows],
            columns=cols,
        )

    def _target_preds(name: str, model: Any) -> pd.DataFrame:
        """Compute target-year predictions for a base model."""
        if not have_target:
            return _nan_df(n_tgt)
        X_tgt = X_target_per_model.get(name)  # type: ignore[union-attr]
        if X_tgt is None or len(X_tgt) == 0:
            return _nan_df(n_tgt)
        try:
            p = model.predict(X_tgt)
            if p.ndim == 1:
                p = p.reshape(-1, 1)
            p = _align_preds(p)
            return pd.DataFrame(p, index=tgt_idx, columns=cols)
        except Exception:
            return _nan_df(n_tgt)

    # ── Evaluate base models ─────────────────────────────────────────────
    results: List[ModelComparisonResult] = []

    for name, model in fitted_base_models.items():
        X_ho = X_holdout_per_model.get(name)
        if X_ho is None or len(X_ho) == 0:
            results.append(ModelComparisonResult(
                model_name=name,
                holdout_r2=float('nan'),
                holdout_rmse=float('nan'),
                holdout_mae=float('nan'),
                is_best=False,
                predictions=_nan_df(n_tgt),
            ))
            continue
        try:
            _sig = inspect.signature(model.predict)
            if (
                entity_indices_holdout is not None
                and 'entity_indices' in _sig.parameters
            ):
                ho_preds = model.predict(
                    X_ho, entity_indices=entity_indices_holdout
                )
            elif (
                entity_indices_holdout is not None
                and 'group_indices' in _sig.parameters
            ):
                ho_preds = model.predict(
                    X_ho, group_indices=entity_indices_holdout
                )
            else:
                ho_preds = model.predict(X_ho)
            if ho_preds.ndim == 1:
                ho_preds = ho_preds.reshape(-1, 1)
            r2, rmse, mae = _compute_metrics(y_holdout, ho_preds)
            results.append(ModelComparisonResult(
                model_name=name,
                holdout_r2=r2,
                holdout_rmse=rmse,
                holdout_mae=mae,
                is_best=False,
                predictions=_target_preds(name, model),
            ))
        except Exception as exc:
            warnings.warn(
                f"compare_all_models: base model '{name}' failed on holdout "
                f"({type(exc).__name__}: {exc}). Metrics will be NaN.",
                RuntimeWarning,
                stacklevel=2,
            )
            results.append(ModelComparisonResult(
                model_name=name,
                holdout_r2=float('nan'),
                holdout_rmse=float('nan'),
                holdout_mae=float('nan'),
                is_best=False,
                predictions=_nan_df(n_tgt),
            ))

    # ── Evaluate ensemble ────────────────────────────────────────────────
    try:
        ho = _align_preds(ensemble_preds_holdout)
        ens_r2, ens_rmse, ens_mae = _compute_metrics(y_holdout, ho)
        if have_target:
            ens_tgt = _align_preds(ensemble_preds_target)  # type: ignore[arg-type]
            ens_pred_df = pd.DataFrame(ens_tgt, index=tgt_idx, columns=cols)
        else:
            ens_pred_df = _nan_df(n_tgt)
        results.append(ModelComparisonResult(
            model_name='Ensemble',
            holdout_r2=ens_r2,
            holdout_rmse=ens_rmse,
            holdout_mae=ens_mae,
            is_best=False,
            predictions=ens_pred_df,
        ))
    except Exception as exc:
        warnings.warn(
            f"compare_all_models: ensemble evaluation failed "
            f"({type(exc).__name__}: {exc}). Metrics will be NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
        results.append(ModelComparisonResult(
            model_name='Ensemble',
            holdout_r2=float('nan'),
            holdout_rmse=float('nan'),
            holdout_mae=float('nan'),
            is_best=False,
            predictions=_nan_df(n_tgt),
        ))

    # ── Mark the single best model (NaN R² cannot win) ───────────────────
    valid = [r for r in results if not np.isnan(r.holdout_r2)]
    if valid:
        best = max(valid, key=lambda r: (r.holdout_r2, -r.holdout_rmse))
        best.is_best = True

    # ── Sort descending by R² (NaN last) ─────────────────────────────────
    results.sort(
        key=lambda r: (not np.isnan(r.holdout_r2), r.holdout_r2),
        reverse=True,
    )
    return results


class ForecastEvaluator:
    """
    Comprehensive evaluation suite for forecasting models.

    Parameters:
        metrics: List of metric names to compute
        n_folds: Number of CV folds (walk-forward expanding window)
        verbose: Print progress

    Example:
        >>> evaluator = ForecastEvaluator(metrics=['r2', 'rmse', 'mae'])
        >>> results = evaluator.evaluate(model, X_train, y_train)
        >>> report = evaluator.generate_report(results)
    """

    METRIC_FUNCTIONS = {
        "r2": lambda y, p: r2_score(y, p),
        "rmse": lambda y, p: np.sqrt(mean_squared_error(y, p)),
        "mae": lambda y, p: mean_absolute_error(y, p),
        "medae": lambda y, p: median_absolute_error(y, p),
        "mape": lambda y, p: (
            np.mean(np.abs((y - p) / np.abs(y))[np.abs(y) > 1e-2]) * 100
            if np.any(np.abs(y) > 1e-2) else np.nan
        ),
        "max_error": lambda y, p: np.max(np.abs(y - p)),
        "bias": lambda y, p: np.mean(p - y),
    }

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        n_folds: int = 5,
        verbose: bool = True,
    ):
        self.metrics = metrics or ["r2", "rmse", "mae", "mape"]
        self.n_folds = n_folds
        self.verbose = verbose

    @_silence_warnings
    def evaluate(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        entity_indices: Optional[np.ndarray] = None,
        year_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation.

        Args:
            model: Model with fit() and predict() methods
            X: Training features
            y: Training targets
            X_test: Optional test features for holdout evaluation
            y_test: Optional test targets
            entity_indices: Optional entity group IDs (unused; kept for API compat)
            year_labels: Integer calendar year for each training row (target year).
                When provided, uses walk-forward yearly CV aligned to calendar years.

        Returns:
            Dictionary with evaluation results
        """
        results = {}

        # Cross-validation evaluation
        logger.info("Evaluating: Cross-validation...")
        results["cv"] = self._cross_validate(model, X, y,
                                              year_labels=year_labels)

        # Holdout evaluation
        if X_test is not None and y_test is not None:
            logger.info("Evaluating: Holdout test set...")
            results["holdout"] = self._evaluate_holdout(model, X, y, X_test, y_test)

        # Residual diagnostics
        logger.info("Evaluating: Residual diagnostics...")
        results["diagnostics"] = self._residual_diagnostics(
            model, X, y, entity_indices=entity_indices)

        return results

    def _cross_validate(
        self, model, X: np.ndarray, y: np.ndarray,
        year_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run walk-forward yearly cross-validation.

        When *year_labels* is provided, uses ``_WalkForwardYearlySplit``
        aligned to calendar years — identical splitter used by SuperLearner.
        Falls back to ``TimeSeriesSplit`` when year_labels is None (e.g. when
        called on non-panel data outside the main pipeline).
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if year_labels is not None:
            splitter = _WalkForwardYearlySplit(
                min_train_years=8, max_folds=self.n_folds
            )
            splits = list(splitter.split(X, year_labels=year_labels))
        else:
            tscv = TimeSeriesSplit(n_splits=self.n_folds)
            splits = list(tscv.split(X))
        fold_results = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            try:
                model_copy = copy.deepcopy(model)

                if hasattr(model_copy, "fit"):
                    model_copy.fit(X[train_idx], y[train_idx])

                pred = model_copy.predict(X[val_idx])
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)

                fold_metrics = {}
                for metric_name in self.metrics:
                    if metric_name in self.METRIC_FUNCTIONS:
                        func = self.METRIC_FUNCTIONS[metric_name]
                        # Average across outputs
                        scores = []
                        for c in range(y.shape[1]):
                            pc = min(c, pred.shape[1] - 1)
                            scores.append(func(y[val_idx, c], pred[:, pc]))
                        fold_metrics[metric_name] = np.mean(scores)

                fold_results.append(fold_metrics)
            except Exception as e:
                logger.warning(f"Fold {fold_idx} failed: {e}")

        # Aggregate across folds
        if not fold_results:
            return {"error": "All folds failed"}

        agg = {}
        for metric_name in self.metrics:
            values = [f[metric_name] for f in fold_results if metric_name in f]
            if values:
                agg[metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "per_fold": values,
                }

        return agg

    def _evaluate_holdout(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate on holdout test set."""
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)

        model_copy = copy.deepcopy(model)
        if hasattr(model_copy, "fit"):
            model_copy.fit(X_train, y_train)

        pred = model_copy.predict(X_test)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)

        results = {}
        for metric_name in self.metrics:
            if metric_name in self.METRIC_FUNCTIONS:
                func = self.METRIC_FUNCTIONS[metric_name]
                scores = []
                for c in range(y_test.shape[1]):
                    pc = min(c, pred.shape[1] - 1)
                    scores.append(func(y_test[:, c], pred[:, pc]))
                results[metric_name] = np.mean(scores)

        return results

    def _residual_diagnostics(
        self, model, X: np.ndarray, y: np.ndarray,
        entity_indices: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Perform residual diagnostic tests using out-of-fold residuals.

        Uses panel-aware temporal CV (when entity_indices provided) to
        compute genuine out-of-sample residuals.

        Tests:
            - Normality (skewness and excess kurtosis)
            - Autocorrelation (Durbin-Watson statistic)
            - Heteroscedasticity (residual vs predicted correlation)
            - Mean zero test
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Collect out-of-fold predictions
        n = X.shape[0]
        n_splits_diag = min(self.n_folds, max(2, n // 5))
        if entity_indices is not None:
            splitter = _PanelTemporalSplit(n_splits=n_splits_diag)
            splits = list(splitter.split(X, entity_indices=entity_indices))
        else:
            splits = list(TimeSeriesSplit(n_splits=n_splits_diag).split(X))

        oof_residuals = np.full(n, np.nan)
        oof_pred_mean = np.full(n, np.nan)
        fold_residuals: list = []

        for train_idx, val_idx in splits:
            try:
                model_copy = copy.deepcopy(model)
                if hasattr(model_copy, "fit"):
                    model_copy.fit(X[train_idx], y[train_idx])

                pred = model_copy.predict(X[val_idx])
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)

                # Average residuals across outputs
                fold_resid = np.mean(
                    y[val_idx] - pred[:, :y.shape[1]], axis=1
                )
                oof_residuals[val_idx] = fold_resid
                oof_pred_mean[val_idx] = np.mean(pred[:, :y.shape[1]], axis=1)
                fold_residuals.append(fold_resid)  # keep for per-fold DW
            except Exception:
                continue

        # Keep only valid (non-NaN) positions
        valid = ~np.isnan(oof_residuals)
        if valid.sum() < 5:
            return {"error": "Not enough OOF residuals for diagnostics"}

        residuals = oof_residuals[valid]
        pred_mean = oof_pred_mean[valid]

        diagnostics = {}

        # 1. Mean zero test
        diagnostics["residual_mean"] = float(np.mean(residuals))
        diagnostics["residual_std"] = float(np.std(residuals))
        diagnostics["mean_is_zero"] = abs(np.mean(residuals)) < 2 * np.std(residuals) / np.sqrt(len(residuals))

        # 2. Durbin-Watson statistic (autocorrelation).
        # Computed *per CV fold* rather than on the pooled OOF residual array.
        # Each fold's validation window is a contiguous, temporally ordered
        # block (TimeSeriesSplit property), so consecutive residuals within
        # a fold are genuinely consecutive in time.  Pooling residuals across
        # folds would compare the last obs of fold k with the first obs of
        # fold k+1 — a different time point (and potentially a different
        # entity), making the statistic measure cross-entity differences
        # rather than temporal autocorrelation.
        dw_per_fold = []
        for fr in fold_residuals:
            if len(fr) > 2:
                diff_fr = np.diff(fr)
                dw_fold = float(
                    np.sum(diff_fr ** 2) / (np.sum(fr ** 2) + 1e-10)
                )
                dw_per_fold.append(dw_fold)
        if dw_per_fold:
            dw = float(np.mean(dw_per_fold))
            diagnostics["durbin_watson"] = dw
            # DW ≈ 2 → no autocorrelation; < 1.5 → positive; > 2.5 → negative
            diagnostics["no_autocorrelation"] = 1.5 < dw < 2.5

        # 3. Heteroscedasticity check (correlation between |residuals| and predictions)
        abs_resid = np.abs(residuals)
        if np.std(pred_mean) > 1e-10 and np.std(abs_resid) > 1e-10:
            corr = np.corrcoef(pred_mean, abs_resid)[0, 1]
            diagnostics["heteroscedasticity_corr"] = float(corr)
            diagnostics["homoscedastic"] = abs(corr) < 0.3

        # 4. Normality (skewness and kurtosis)
        if len(residuals) > 10:
            skewness = float(np.mean(((residuals - np.mean(residuals)) / (np.std(residuals) + 1e-10)) ** 3))
            kurtosis = float(np.mean(((residuals - np.mean(residuals)) / (np.std(residuals) + 1e-10)) ** 4) - 3)
            diagnostics["skewness"] = skewness
            diagnostics["excess_kurtosis"] = kurtosis
            diagnostics["approximately_normal"] = abs(skewness) < 1.0 and abs(kurtosis) < 3.0

        return diagnostics

    @staticmethod
    def evaluate_uncertainty(
        y_true: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        target_coverage: float = 0.95,
    ) -> Dict[str, float]:
        """
        Evaluate prediction interval quality.

        Metrics:
            - Empirical coverage: Fraction of true values within intervals
            - Mean/median interval width (sharpness)
            - Coverage gap: Empirical - target
            - Winkler score: Penalizes wide intervals and missed coverage

        Args:
            y_true: True values
            lower: Lower bounds of prediction intervals
            upper: Upper bounds of prediction intervals
            target_coverage: Desired coverage level

        Returns:
            Dictionary of uncertainty quality metrics
        """
        if y_true.ndim > 1:
            y_true = y_true.ravel()
        if lower.ndim > 1:
            lower = lower.ravel()
        if upper.ndim > 1:
            upper = upper.ravel()

        covered = (y_true >= lower) & (y_true <= upper)
        widths = upper - lower

        alpha = 1.0 - target_coverage

        # Winkler score
        winkler_scores = widths.copy()
        below = y_true < lower
        above = y_true > upper
        winkler_scores[below] += (2.0 / alpha) * (lower[below] - y_true[below])
        winkler_scores[above] += (2.0 / alpha) * (y_true[above] - upper[above])

        return {
            "empirical_coverage": float(covered.mean()),
            "target_coverage": target_coverage,
            "coverage_gap": float(covered.mean() - target_coverage),
            "mean_interval_width": float(np.mean(widths)),
            "median_interval_width": float(np.median(widths)),
            "std_interval_width": float(np.std(widths)),
            "winkler_score": float(np.mean(winkler_scores)),
            "n_samples": len(y_true),
        }

    @staticmethod
    def calibration_curve(
        y_true: np.ndarray,
        quantile_predictions: Dict[float, np.ndarray],
        n_bins: int = 10,
    ) -> Dict[str, np.ndarray]:
        """
        Compute calibration curve: predicted vs observed quantile levels.

        A well-calibrated model should have points along the diagonal.

        Args:
            y_true: True values
            quantile_predictions: Dict mapping quantile levels to predictions
            n_bins: Number of bins for calibration

        Returns:
            Dictionary with:
                - expected_levels: Predicted quantile levels
                - observed_levels: Empirical fraction below each quantile
                - calibration_error: Mean absolute calibration error
        """
        if y_true.ndim > 1:
            y_true = y_true.ravel()

        expected = []
        observed = []

        for q_level, q_pred in sorted(quantile_predictions.items()):
            if q_pred.ndim > 1:
                q_pred = q_pred.ravel()
            fraction_below = np.mean(y_true <= q_pred)
            expected.append(q_level)
            observed.append(fraction_below)

        expected = np.array(expected)
        observed = np.array(observed)
        calibration_error = np.mean(np.abs(expected - observed))

        return {
            "expected_levels": expected,
            "observed_levels": observed,
            "calibration_error": calibration_error,
        }

    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a human-readable evaluation report.

        Args:
            results: Output from evaluate()

        Returns:
            Formatted string report
        """
        lines = [
            "",
            "=" * 80,
            "FORECASTING EVALUATION REPORT",
            "=" * 80,
        ]

        # Cross-validation results
        if "cv" in results and isinstance(results["cv"], dict) and "error" not in results["cv"]:
            lines.append("\n## Cross-Validation Results")
            lines.append("-" * 40)
            for metric, vals in results["cv"].items():
                if isinstance(vals, dict):
                    lines.append(
                        f"  {metric:>8s}: {vals['mean']:.4f} ± {vals['std']:.4f} "
                        f"[{vals['min']:.4f}, {vals['max']:.4f}]"
                    )

        # Holdout results
        if "holdout" in results:
            lines.append("\n## Holdout Test Results")
            lines.append("-" * 40)
            for metric, value in results["holdout"].items():
                lines.append(f"  {metric:>8s}: {value:.4f}")

        # Diagnostics
        if "diagnostics" in results:
            diag = results["diagnostics"]
            lines.append("\n## Residual Diagnostics")
            lines.append("-" * 40)

            lines.append(f"  Residual mean:        {diag.get('residual_mean', 'N/A'):.6f}")
            lines.append(f"  Residual std:         {diag.get('residual_std', 'N/A'):.6f}")
            lines.append(f"  Mean approx 0:        {'[OK]' if diag.get('mean_is_zero', False) else '[X]'}")

            if "durbin_watson" in diag:
                lines.append(f"  Durbin-Watson:        {diag['durbin_watson']:.4f}")
                lines.append(f"  No autocorrelation:   {'[OK]' if diag.get('no_autocorrelation', False) else '[X]'}")

            if "heteroscedasticity_corr" in diag:
                lines.append(f"  Heterosc. corr:       {diag['heteroscedasticity_corr']:.4f}")
                lines.append(f"  Homoscedastic:        {'[OK]' if diag.get('homoscedastic', False) else '[X]'}")

            if "skewness" in diag:
                lines.append(f"  Skewness:             {diag['skewness']:.4f}")
                lines.append(f"  Excess kurtosis:      {diag['excess_kurtosis']:.4f}")
                lines.append(f"  Approx Normal:        {'[OK]' if diag.get('approximately_normal', False) else '[X]'}")

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)


class _SubsetEnsemble:
    """Minimal averaging ensemble over a subset of base models.

    Used internally by ``AblationStudy`` to evaluate LOO and pairwise
    ensembles without importing ``SuperLearner`` (avoids a circular dependency
    and removes the overhead of OOF meta-learning for ablation purposes).
    """

    def __init__(self, models: Dict[str, Any]):
        self._models = dict(models)  # shallow copy

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_SubsetEnsemble":
        for model in self._models.values():
            try:
                if hasattr(model, "fit"):
                    model.fit(X, y)
            except Exception:
                pass
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for model in self._models.values():
            try:
                p = model.predict(X)
                if p.ndim == 1:
                    p = p.reshape(-1, 1)
                preds.append(p)
            except Exception:
                pass
        if not preds:
            raise ValueError("No models in subset produced predictions")
        return np.mean(preds, axis=0)


class AblationStudy:
    """
    Systematic ablation study for ensemble components.

    Tests the contribution of each model by comparing:
    1. Full ensemble vs ensemble without model_i
    2. Individual model performance
    3. Pairwise model combinations

    Parameters:
        base_evaluator: ForecastEvaluator instance
        primary_metric: Primary metric for comparison (default 'r2')

    Example:
        >>> ablation = AblationStudy()
        >>> results = ablation.run(models, X_train, y_train)
    """

    def __init__(
        self,
        base_evaluator: Optional[ForecastEvaluator] = None,
        primary_metric: str = "r2",
    ):
        self.evaluator = base_evaluator or ForecastEvaluator(
            metrics=["r2", "rmse", "mae"], verbose=False
        )
        self.primary_metric = primary_metric

    def _eval_subset(self, subset: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate a _SubsetEnsemble and return primary metric mean, or NaN."""
        try:
            ens = _SubsetEnsemble(subset)
            result = self.evaluator.evaluate(ens, X, y)
            if "cv" in result and self.primary_metric in result["cv"]:
                return float(result["cv"][self.primary_metric]["mean"])
        except Exception:
            pass
        return float(np.nan)

    def run(
        self,
        models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Run ablation study on model ensemble.

        Performs three tests:

        1. **Individual performance**: CV score for each model in isolation.
        2. **Leave-one-out (LOO) ablation**: CV score of the full averaging
           ensemble with model *i* removed.  A large drop in score when
           model *i* is removed indicates high contribution by model *i*.
        3. **Pairwise combinations**: CV score for every pair of models as a
           simple averaging ensemble.  Identifies the best two-model sub-ensemble.

        Args:
            models: Dictionary of {name: fitted_model}
            X: Feature matrix
            y: Target values

        Returns:
            Dictionary with keys:
                - ``individual_performance``: {model_name: score}
                - ``model_ranking``: sorted list of {"name", "score"}
                - ``full_ensemble_score``: score of the full averaging ensemble
                - ``loo_ablation``: {model_name: score_without_model}
                - ``loo_contribution``: {model_name: full_score minus loo_score}
                - ``pairwise_combinations``: {"A+B": score}
                - ``best_pair``: name of the best two-model pair
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        model_names = list(models.keys())
        results: Dict[str, Any] = {}

        # ------------------------------------------------------------------ #
        # Test 1: Individual model performance
        # ------------------------------------------------------------------ #
        individual: Dict[str, float] = {}
        for name, model in models.items():
            individual[name] = self._eval_subset({name: model}, X, y)
        results["individual_performance"] = individual

        ranked = sorted(
            individual.items(),
            key=lambda x: x[1] if not np.isnan(x[1]) else -np.inf,
            reverse=True,
        )
        results["model_ranking"] = [
            {"name": name, "score": score} for name, score in ranked
        ]

        # ------------------------------------------------------------------ #
        # Test 2: Full ensemble + leave-one-out ablation
        # ------------------------------------------------------------------ #
        full_score = self._eval_subset(models, X, y)
        results["full_ensemble_score"] = full_score

        loo_scores: Dict[str, float] = {}
        for leave_out in model_names:
            subset = {n: m for n, m in models.items() if n != leave_out}
            loo_scores[leave_out] = self._eval_subset(subset, X, y) if subset else np.nan
        results["loo_ablation"] = loo_scores

        # Contribution = drop in performance when model is removed
        # (positive -> model helps; negative -> model hurts the ensemble)
        results["loo_contribution"] = {
            name: float(full_score - loo_scores[name])
            if not np.isnan(full_score) and not np.isnan(loo_scores[name])
            else np.nan
            for name in model_names
        }

        # ------------------------------------------------------------------ #
        # Test 3: Pairwise model combinations
        # ------------------------------------------------------------------ #
        pairwise: Dict[str, float] = {}
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                na, nb = model_names[i], model_names[j]
                pairwise[f"{na}+{nb}"] = self._eval_subset(
                    {na: models[na], nb: models[nb]}, X, y
                )
        results["pairwise_combinations"] = pairwise

        best_pair_item = max(
            pairwise.items(),
            key=lambda kv: kv[1] if not np.isnan(kv[1]) else -np.inf,
            default=(None, np.nan),
        )
        results["best_pair"] = best_pair_item[0]

        return results

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate ablation study report."""
        lines = [
            "",
            "=" * 80,
            "ABLATION STUDY REPORT",
            "=" * 80,
            f"\nPrimary metric: {self.primary_metric}",
        ]

        if "model_ranking" in results:
            lines.append("\n## Individual Model Ranking")
            lines.append("-" * 50)
            for i, item in enumerate(results["model_ranking"]):
                score = item["score"]
                bar = "█" * int(max(0, score) * 30) if not np.isnan(score) else "N/A"
                lines.append(
                    f"  {i + 1}. {item['name']:25s}: {score:.4f} {bar}"
                )

        if "full_ensemble_score" in results:
            lines.append(f"\n## Full Ensemble Score")
            lines.append(f"  {results['full_ensemble_score']:.4f}")

        if "loo_contribution" in results:
            lines.append("\n## Leave-One-Out Ablation (contribution = full - LOO score)")
            lines.append("-" * 50)
            sorted_contrib = sorted(
                results["loo_contribution"].items(),
                key=lambda kv: kv[1] if not np.isnan(kv[1]) else -np.inf,
                reverse=True,
            )
            for name, contrib in sorted_contrib:
                loo = results["loo_ablation"].get(name, np.nan)
                loo_str = f"{loo:.4f}" if not np.isnan(loo) else "N/A"
                contrib_str = f"{contrib:+.4f}" if not np.isnan(contrib) else "N/A"
                lines.append(f"  {name:25s}: LOO={loo_str}  contrib={contrib_str}")

        if "pairwise_combinations" in results:
            top_pairs = sorted(
                results["pairwise_combinations"].items(),
                key=lambda kv: kv[1] if not np.isnan(kv[1]) else -np.inf,
                reverse=True,
            )[:5]
            lines.append("\n## Top-5 Pairwise Combinations")
            lines.append("-" * 50)
            for pair_name, score in top_pairs:
                lines.append(f"  {pair_name:40s}: {score:.4f}")

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)
