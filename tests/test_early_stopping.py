# -*- coding: utf-8 -*-
"""
Phase 2: Tree Model Stabilization — Test Suite
===============================================

Validates all Phase 2 fixes:
    2.1 — Chronological early stopping for CatBoost
  2.4 — QuantileRF stabilization (n_estimators fix, adaptive leaf, no scaler)
  2.5 — cv_min_train_years=5 gives more OOF data

Each test documents the exact bug or feature it guards against so future
changes immediately surface regressions.

Run with:
    pytest tests/test_early_stopping.py -v
"""

import numpy as np
import pytest
import warnings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.RandomState(0)


@pytest.fixture
def medium_dataset(rng):
    """n=300 dataset with a weak linear signal (simulates governance panel)."""
    n, p, k = 300, 10, 3
    X = rng.randn(n, p)
    coefs = rng.randn(p, k) * 0.5
    # Weak signal: y = X @ coefs + large noise (mimics R² ≈ 0.15–0.25)
    y = X @ coefs + rng.randn(n, k) * 2.0
    return X, y


@pytest.fixture
def tiny_dataset(rng):
    """n=25 dataset — smaller than the early stopping threshold (40)."""
    n, p, k = 25, 5, 2
    X = rng.randn(n, p)
    y = rng.randn(n, k)
    return X, y


@pytest.fixture
def overfitting_dataset(rng):
    """Dataset where overfitting is detectable: large noise dominates signal."""
    n, p = 200, 8
    X = rng.randn(n, p)
    # Pure noise targets — no learnable signal
    y = rng.randn(n)
    return X, y


# ---------------------------------------------------------------------------
# 2.1.A — CatBoost Early Stopping
# ---------------------------------------------------------------------------

class TestCatBoostEarlyStopping:
    """Phase 2.1.A: CatBoost chronological early stopping."""

    def test_early_stopping_activates_on_medium_fold(self, medium_dataset):
        """
        With n=300 and large noise, early stopping should halt before max
        iterations.  ``_best_iteration_`` should be < eff_iter (adaptive=150).

        Root cause guarded: before Phase 2.1, CatBoost trained all
        200–300 iterations even on tiny validation folds, memorising noise.
        """
        from forecasting.catboost_forecaster import CatBoostForecaster

        X, y = medium_dataset
        # Use a high max to guarantee ES fires before the limit
        model = CatBoostForecaster(
            iterations=300, learning_rate=0.05,
            early_stopping_rounds=10, validation_fraction=0.20,
            random_state=42,
        )
        model.fit(X, y)

        # Best iteration must be recorded
        assert model._best_iteration_ is not None, (
            "CatBoost early stopping did not record _best_iteration_. "
            "Possible: ES disabled or n_train < 40."
        )
        # Best iteration should be strictly less than max iterations
        # The adaptive table sets eff_iter=150 for n=300 (200–399 bucket).
        # With pure noise + ES rounds=10, the model should stop well before 150.
        assert model._best_iteration_ < 150, (
            f"CatBoost best_iteration={model._best_iteration_} expected < 150. "
            "Early stopping may not be functioning correctly."
        )

    def test_early_stopping_bypass_on_tiny_fold(self, tiny_dataset):
        """
        For n=25 (< 40 threshold), early stopping is suppressed.
        Model must still fit and predict without errors.

        Root cause guarded: an unchecked chronological split on n=25 with
        validation_fraction=0.20 would leave only 20 training samples,
        causing CatBoost to fail with 'not enough data'.
        """
        from forecasting.catboost_forecaster import CatBoostForecaster

        X, y = tiny_dataset
        model = CatBoostForecaster(
            iterations=50, early_stopping_rounds=10,
            validation_fraction=0.20, random_state=42,
        )
        # Must not raise
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == y.shape, (
            f"CatBoost predict shape {pred.shape} != y.shape {y.shape}."
        )

    def test_early_stopping_disabled_when_rounds_zero(self, medium_dataset):
        """
        Setting early_stopping_rounds=0 must disable ES entirely.
        _best_iteration_ should remain None.
        """
        from forecasting.catboost_forecaster import CatBoostForecaster

        X, y = medium_dataset
        model = CatBoostForecaster(
            iterations=50,
            early_stopping_rounds=0,  # explicit disable
            random_state=42,
        )
        model.fit(X, y)
        # ES disabled → _best_iteration_ stays None
        assert model._best_iteration_ is None, (
            "CatBoost set _best_iteration_ even with early_stopping_rounds=0."
        )

    def test_catboost_predict_shape_multioutput(self, medium_dataset):
        """CatBoost with ES still returns correct (n_test, n_outputs) shape."""
        from forecasting.catboost_forecaster import CatBoostForecaster

        X, y = medium_dataset
        model = CatBoostForecaster(
            iterations=50, early_stopping_rounds=10, random_state=42,
        )
        model.fit(X, y)
        pred = model.predict(X[:10])
        assert pred.shape == (10, y.shape[1])

    def test_catboost_feature_importance_after_es(self, medium_dataset):
        """Feature importance vector must have length n_features after ES fit."""
        from forecasting.catboost_forecaster import CatBoostForecaster

        X, y = medium_dataset
        model = CatBoostForecaster(
            iterations=50, early_stopping_rounds=10, random_state=42,
        )
        model.fit(X, y)
        imp = model.get_feature_importance()
        assert len(imp) == X.shape[1], (
            f"Feature importance length {len(imp)} != n_features {X.shape[1]}."
        )
        assert np.all(imp >= 0), "Feature importances contain negative values."


# ---------------------------------------------------------------------------
# 2.1 Helper: ChronologicalSplit
# ---------------------------------------------------------------------------

class TestChronologicalSplit:
    """Unit tests for the _chronological_es_split helper."""

    def test_split_sizes_correct(self, rng):
        """Split produces correct train/val row counts."""
        from forecasting.catboost_forecaster import _chronological_es_split

        n, p, k = 100, 5, 2
        X = rng.randn(n, p)
        y = rng.randn(n, k)
        result = _chronological_es_split(X, y, validation_fraction=0.20,
                                          sample_weight=None)
        assert result is not None
        X_tr, y_tr, sw_tr, X_va, y_va = result
        assert X_tr.shape[0] + X_va.shape[0] == n
        assert X_va.shape[0] == max(10, int(round(n * 0.20)))

    def test_split_returns_none_for_tiny_dataset(self, rng):
        """Datasets with < 40 training rows after split get None (bypass ES)."""
        from forecasting.catboost_forecaster import _chronological_es_split

        n, p, k = 20, 3, 2
        X = rng.randn(n, p)
        y = rng.randn(n, k)
        result = _chronological_es_split(X, y, validation_fraction=0.20,
                                          sample_weight=None,
                                          min_n_train=30, min_n_val=10)
        # With n=20 and min_n_train=30: n_tr = 16 < 30 → should return None
        assert result is None, (
            "Expected None for tiny dataset but got a split. "
            "This means ES would run with too little training data."
        )

    def test_split_preserves_chronological_order(self, rng):
        """Last n_val rows go to validation — chronological ordering preserved."""
        from forecasting.catboost_forecaster import _chronological_es_split

        n, p, k = 100, 3, 1
        X = np.arange(n).reshape(-1, 1).repeat(p, axis=1).astype(float)
        y = np.arange(n).reshape(-1, 1).astype(float)
        result = _chronological_es_split(X, y, validation_fraction=0.20,
                                          sample_weight=None)
        assert result is not None
        X_tr, y_tr, _, X_va, y_va = result
        # Training rows should have smaller indices than validation rows
        assert X_tr[-1, 0] < X_va[0, 0], (
            "Chronological split broken: last training row >= first validation row. "
            "Random split may have been applied instead."
        )


# ---------------------------------------------------------------------------
# 2.4 — QuantileRF Stabilization
# ---------------------------------------------------------------------------

class TestQuantileRFStabilization:
    """Phase 2.4: QuantileRF n_estimators fix, adaptive leaf, no scaler."""

    def test_no_scaler_applied(self, medium_dataset):
        """
        Phase 2.4: RobustScaler must NOT be applied (scale-invariant tree model).

        Root cause guarded: the old implementation applied RobustScaler.fit_transform()
        to X before each fit, creating train-test distribution mismatch when the
        prediction set is out-of-distribution.  Tree models are scale-invariant.
        """
        from forecasting.quantile_forest import QuantileRandomForestForecaster

        X, y = medium_dataset
        model = QuantileRandomForestForecaster(n_estimators=20, random_state=42)
        model.fit(X, y)
        # scaler_ should be None (removed in Phase 2.4)
        assert model.scaler_ is None, (
            "QuantileRF still applies RobustScaler (scaler_ is not None). "
            "Tree models are scale-invariant; scaling adds unnecessary compute."
        )

    def test_adaptive_min_leaf_scales_with_n(self):
        """
        _compute_effective_min_leaf() returns larger values for smaller n_train.

        Root cause guarded: on small CV folds (n≈150), min_samples_leaf=3
        allows micro-leaves that memorise individual samples, causing negative
        CV R² analogous to overfitting (root cause of QRF CV R²=−0.088).
        """
        from forecasting.quantile_forest import QuantileRandomForestForecaster

        model = QuantileRandomForestForecaster(n_estimators=20, random_state=42)
        # Auto-scaling only when using default min_samples_leaf==3
        leaf_large = model._compute_effective_min_leaf(500)  # default: 3
        leaf_med   = model._compute_effective_min_leaf(300)  # should be ≥3
        leaf_small = model._compute_effective_min_leaf(100)  # should be ≥5

        assert leaf_small > leaf_large, (
            f"Small-n effective_min_leaf ({leaf_small}) should exceed "
            f"large-n ({leaf_large}). Adaptive scaling is broken."
        )
        assert leaf_small >= 5, (
            f"min_samples_leaf={leaf_small} for n=100 should be ≥ 5."
        )
        assert leaf_med >= 3

    def test_adaptive_leaf_honours_explicit_override(self):
        """
        Auto-scaling must NOT fire when min_samples_leaf is set explicitly
        to a non-default value.  Explicit overrides are always respected.
        """
        from forecasting.quantile_forest import QuantileRandomForestForecaster

        model = QuantileRandomForestForecaster(
            n_estimators=20, min_samples_leaf=10, random_state=42,
        )
        # With explicit min_samples_leaf=10 (≠ default 3), all n_train should
        # return exactly 10.
        for n in [50, 200, 800]:
            leaf = model._compute_effective_min_leaf(n)
            assert leaf == 10, (
                f"Explicit min_samples_leaf=10 was overridden to {leaf} "
                f"for n_train={n}. Auto-scaling must not override explicit values."
            )

    def test_fit_predict_no_scaler(self, medium_dataset):
        """fit() → predict() round-trip works without scaler."""
        from forecasting.quantile_forest import QuantileRandomForestForecaster

        X, y = medium_dataset
        model = QuantileRandomForestForecaster(n_estimators=20, random_state=42)
        model.fit(X, y)
        pred = model.predict(X[:10])
        assert pred.shape == (10, y.shape[1])
        assert not np.isnan(pred).any()

    def test_oob_score_available(self, medium_dataset):
        """OOB score must be finite after fit (confirms n_estimators is adequate)."""
        from forecasting.quantile_forest import QuantileRandomForestForecaster

        X, y = medium_dataset
        model = QuantileRandomForestForecaster(n_estimators=30, random_state=42)
        model.fit(X, y)
        oob = model.oob_score
        assert np.isfinite(oob), f"OOB score is non-finite: {oob}"


# ---------------------------------------------------------------------------
# 2.4 — _create_models() QRF n_estimators bug fix
# ---------------------------------------------------------------------------

class TestCreateModelsQRFBugFix:
    """
    Phase 2.4: _create_models() previously hardcoded QuantileRF n_estimators=100,
    ignoring ForecastConfig.qrf_n_estimators=300 and the class default of 200.
    """

    def test_qrf_receives_config_n_estimators(self):
        """
        When a ForecastConfig with qrf_n_estimators=300 is passed to
        UnifiedForecaster, _create_models() must produce a QuantileRF
        with n_estimators=300 (not the old hardcoded 100).

        Root cause guarded: before Phase 2.4, QuantileRF.n_estimators was
        always 100 regardless of config, causing systematic underfitting of
        QRF quantile estimates and contributing to CV R²=−0.088.
        """
        from forecasting.unified import UnifiedForecaster
        from config import ForecastConfig

        cfg = ForecastConfig(qrf_n_estimators=300)
        forecaster = UnifiedForecaster(config=cfg)
        models = forecaster._create_models()

        qrf = models.get('QuantileRF')
        assert qrf is not None, "_create_models() did not create 'QuantileRF'."
        assert qrf.n_estimators == 300, (
            f"QuantileRF.n_estimators={qrf.n_estimators}, expected 300. "
            "The n_estimators=100 hardcode bug was not fixed."
        )

    def test_qrf_default_config_gives_300(self):
        """Default ForecastConfig gives qrf_n_estimators=300."""
        from forecasting.unified import UnifiedForecaster
        from config import ForecastConfig

        forecaster = UnifiedForecaster(config=ForecastConfig())
        models = forecaster._create_models()
        qrf = models['QuantileRF']
        assert qrf.n_estimators == 300, (
            f"Default ForecastConfig gives qrf.n_estimators={qrf.n_estimators}, "
            f"expected 300 (from qrf_n_estimators default in ForecastConfig)."
        )

    def test_gb_models_receive_early_stopping_config(self):
        """
        CatBoost model created by _create_models() must have
        early_stopping_rounds=20 and validation_fraction=0.20 (Phase 2.1 config).
        """
        from forecasting.unified import UnifiedForecaster
        from config import ForecastConfig

        cfg = ForecastConfig(
            gb_early_stopping_rounds=20,
            gb_validation_fraction=0.20,
        )
        forecaster = UnifiedForecaster(config=cfg)
        models = forecaster._create_models()

        cb = models['CatBoost']
        assert cb.early_stopping_rounds == 20, (
            f"CatBoost.early_stopping_rounds={cb.early_stopping_rounds}, expected 20."
        )
        assert abs(cb.validation_fraction - 0.20) < 1e-9


# ---------------------------------------------------------------------------
# 2.5 — cv_min_train_years=5 gives more OOF data
# ---------------------------------------------------------------------------

class TestOOFExpansion:
    """
    Phase 2.5: cv_min_train_years reduced from 8 → 5 produces more OOF rows.
    """

    def test_cv_min_train_years_default_is_5(self):
        """
        ForecastConfig.cv_min_train_years default must be 5 after Phase 2.5.

        Root cause guarded: default of 8 produced only 5 validation folds
        (315 OOF rows / 8 criteria ≈ 40 rows each), which is insufficient
        for stable NNLS meta-learner weight estimation.
        """
        from config import ForecastConfig

        cfg = ForecastConfig()
        assert cfg.cv_min_train_years == 5, (
            f"cv_min_train_years={cfg.cv_min_train_years}, expected 5. "
            "Phase 2.5 reduced the default from 8 to 5 to produce 7 folds "
            "(441 OOF rows) instead of 5 folds (315 rows)."
        )

    def test_fewer_min_train_years_gives_more_folds(self):
        """
        A SuperLearner with min_train_years=5 generates more CV folds than
        one with min_train_years=8, given the same 13-year panel (2012–2024).

        Validates that the cv_min_train_years reduction actually increases the
        number of validation folds and therefore the OOF row count.
        """
        from forecasting.super_learner import SuperLearner

        # Simulate 13 year-labels (corresponding to 2012–2024 target years)
        n_years = 13
        n_entities = 10
        n_samples = n_years * n_entities
        n_features = 5
        rng = np.random.RandomState(7)

        X = rng.randn(n_samples, n_features)
        y = rng.randn(n_samples, 2)
        # Year labels: each entity has one row per year
        year_labels = np.tile(np.arange(2012, 2012 + n_years), n_entities)
        
        # Test directly on the PanelWalkForwardCV splitter to avoid
        # import dependency issues with base models.
        from forecasting.super_learner import PanelWalkForwardCV

        # With 13 years (2012-2024):
        # - min_train_years=5 means the first validation year is 5 years after the start (i.e. 2017)
        #   Folds: 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024 (8 folds)
        #   Folds: 2017... (8 folds, so set max_folds > 8)
        # - min_train_years=8 means the first validation year is 8 years after the start (i.e. 2020)
        #   Folds: 2020... (5 folds)
        splits_5 = list(PanelWalkForwardCV(min_train_years=5, max_folds=10).split(X, year_labels))
        splits_8 = list(PanelWalkForwardCV(min_train_years=8, max_folds=10).split(X, year_labels))

        assert len(splits_5) > len(splits_8), (
            f"min_train_years=5 gives {len(splits_5)} folds, "
            f"but min_train_years=8 gives {len(splits_8)} folds. "
            "Expected more folds with min_train_years=5."
        )

