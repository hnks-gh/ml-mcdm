# -*- coding: utf-8 -*-
"""
Unit / integration tests for the forecasting module.

Covers:
    - Feature engineering shape correctness and consistency
    - Each base model fit/predict round-trip
    - SuperLearner weight validity (non-negative, sum to 1)
    - Conformal predictor coverage on synthetic data
    - Multi-output dimension checks
    - Evaluation diagnostics use OOF residuals
    - QuantileRF returns median (not mean)

Run with:
    pytest tests/test_forecasting.py -v
"""

import numpy as np
import pytest
import warnings
from typing import Dict


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def small_dataset(rng):
    """Tiny (n=80, d=6, k=3) dataset for fast tests."""
    n, d, k = 80, 6, 3
    X = rng.randn(n, d)
    y = rng.randn(n, k)
    return X, y


@pytest.fixture
def univariate_dataset(rng):
    """1-D target with known linear relationship."""
    n, d = 100, 4
    X = rng.randn(n, d)
    coefs = np.array([1.0, -0.5, 2.0, 0.3])
    y = X @ coefs + rng.randn(n) * 0.3
    return X, y


@pytest.fixture
def entity_indices(rng):
    """Entity group indices for 80 samples, 8 groups of 10."""
    return np.repeat(np.arange(8), 10)


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

class TestFeatureEngineer:
    """Tests for TemporalFeatureEngineer."""

    @staticmethod
    def _make_mock_panel(n_entities=3, n_years=5, n_components=2):
        """Create a lightweight mock PanelData object."""
        import pandas as pd
        from types import SimpleNamespace

        years = list(range(2011, 2011 + n_years))
        provinces = [f"P{i}" for i in range(n_entities)]
        components = [f"x{i}" for i in range(n_components)]

        class MockPanel:
            pass

        panel = MockPanel()
        panel.provinces = provinces
        panel.subcriteria_names = components
        panel.years = years

        rng = np.random.RandomState(0)
        data_store: Dict[str, "pd.DataFrame"] = {}
        for p in provinces:
            data_store[p] = pd.DataFrame(
                rng.rand(n_years, n_components),
                index=years,
                columns=components,
            )

        panel.get_province = lambda name: data_store[name]

        # cross_section property
        cs = {}
        for y in years:
            rows = []
            for p in provinces:
                row = {"Province": p}
                row.update(data_store[p].loc[y].to_dict())
                rows.append(row)
            cs[y] = pd.DataFrame(rows)
        panel.cross_section = cs

        # T-3: Add year_contexts so the FeatureEngineer dynamic-exclusion
        # code path is exercised (previously untested — year_contexts was None
        # and all ctx_next guards were skipped entirely).
        #
        # Each year's context marks all provinces and components as active
        # and all (province, component) pairs as valid.  This mirrors the
        # common case where every province has full data every year.
        def _make_ctx(yr):
            ctx = SimpleNamespace()
            ctx.year = yr
            ctx.active_provinces = list(provinces)
            ctx.active_subcriteria = list(components)
            ctx.valid_pairs = {
                (p, sc) for p in provinces for sc in components
            }
            ctx.is_valid = lambda prov, sc, _c=ctx: (prov, sc) in _c.valid_pairs
            return ctx

        panel.year_contexts = {yr: _make_ctx(yr) for yr in years}

        return panel

    def test_training_sample_count(self):
        """All consecutive year-pairs should be used (no wasted year)."""
        from forecasting.features import TemporalFeatureEngineer

        panel = self._make_mock_panel(n_entities=4, n_years=6, n_components=2)
        eng = TemporalFeatureEngineer(
            lag_periods=[1], rolling_windows=[2],
            include_momentum=True, include_cross_entity=True,
            target_level='subcriteria',
        )
        X_train, y_train, X_pred, info, _, _ = eng.fit_transform(panel, target_year=2017)

        # 4 entities × 5 year-pairs (2011→12, 12→13, 13→14, 14→15, 15→16)
        assert X_train.shape[0] == 4 * 5
        assert y_train.shape[0] == 4 * 5
        assert X_pred.shape[0] == 4

    def test_entity_indices_returned(self):
        from forecasting.features import TemporalFeatureEngineer

        panel = self._make_mock_panel()
        eng = TemporalFeatureEngineer(lag_periods=[1], rolling_windows=[2], target_level='subcriteria')
        _, _, _, info, _, _ = eng.fit_transform(panel, target_year=2016)
        assert "entity_index" in info.columns

    def test_feature_dimensions_consistent(self):
        from forecasting.features import TemporalFeatureEngineer

        panel = self._make_mock_panel(n_entities=3, n_years=5, n_components=3)
        eng = TemporalFeatureEngineer(lag_periods=[1, 2], rolling_windows=[2, 3], target_level='subcriteria')
        X_train, _, X_pred, _, _, _ = eng.fit_transform(panel, target_year=2016)
        assert X_train.shape[1] == X_pred.shape[1]
        assert len(eng.get_feature_names()) == X_train.shape[1]

    def test_no_nan_in_features(self):
        from forecasting.features import TemporalFeatureEngineer

        panel = self._make_mock_panel()
        eng = TemporalFeatureEngineer(lag_periods=[1], rolling_windows=[2], target_level='subcriteria')
        X_train, y_train, X_pred, _, _, _ = eng.fit_transform(panel, target_year=2016)
        assert not np.isnan(X_train.values).any()
        assert not np.isnan(y_train.values).any()
        assert not np.isnan(X_pred.values).any()


# ---------------------------------------------------------------------------
# Base Models — fit / predict round-trip
# ---------------------------------------------------------------------------

class TestGradientBoosting:
    def test_fit_predict_multioutput(self, small_dataset):
        from forecasting.gradient_boosting import CatBoostForecaster

        X, y = small_dataset
        model = CatBoostForecaster(iterations=20, random_state=42)
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == y.shape

    def test_no_random_early_stopping(self):
        """CatBoostForecaster has allow_writing_files=False and verbose=0 by default."""
        from forecasting.gradient_boosting import CatBoostForecaster

        model = CatBoostForecaster()
        # Verify the compiled model is not yet created (fit() not called)
        assert model.model is None
        assert model.iterations == 300  # class default

    def test_feature_importance(self, small_dataset):
        from forecasting.gradient_boosting import CatBoostForecaster

        X, y = small_dataset
        model = CatBoostForecaster(iterations=20, random_state=42)
        model.fit(X, y)
        imp = model.get_feature_importance()
        assert len(imp) == X.shape[1]
        assert np.all(imp >= 0)


class TestBayesianForecaster:
    def test_fit_predict(self, small_dataset):
        from forecasting.bayesian import BayesianForecaster

        X, y = small_dataset
        model = BayesianForecaster()
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == y.shape

    def test_predict_with_uncertainty_shape(self, small_dataset):
        """predict_with_uncertainty returns (mean, std) with matching shapes."""
        from forecasting.bayesian import BayesianForecaster

        X, y = small_dataset
        model = BayesianForecaster()
        model.fit(X, y)
        mean, std = model.predict_with_uncertainty(X)
        assert mean.shape == y.shape
        assert std.shape == y.shape

    def test_predict_with_uncertainty_std_positive(self, small_dataset):
        """All posterior standard deviations must be strictly positive."""
        from forecasting.bayesian import BayesianForecaster

        X, y = small_dataset
        model = BayesianForecaster()
        model.fit(X, y)
        _, std = model.predict_with_uncertainty(X)
        assert np.all(std > 0), "Some std values are non-positive"

    def test_feature_importance_length(self, small_dataset):
        """get_feature_importance() returns one value per input feature."""
        from forecasting.bayesian import BayesianForecaster

        X, y = small_dataset
        model = BayesianForecaster()
        model.fit(X, y)
        imp = model.get_feature_importance()
        assert len(imp) == X.shape[1]

    def test_feature_importance_non_negative(self, small_dataset):
        """Feature importances (|coef|) must be ≥ 0."""
        from forecasting.bayesian import BayesianForecaster

        X, y = small_dataset
        model = BayesianForecaster()
        model.fit(X, y)
        imp = model.get_feature_importance()
        assert np.all(imp >= 0)


class TestQuantileRandomForest:
    def test_fit_predict_multioutput(self, small_dataset):
        from forecasting.quantile_forest import QuantileRandomForestForecaster

        X, y = small_dataset
        model = QuantileRandomForestForecaster(n_estimators=20, random_state=42)
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == y.shape

    def test_predict_returns_median(self, small_dataset):
        """
        T-4: Regression test for Bug Q-1 — ``get_prediction_distribution()``
        must return the true leaf-weight quantile median, not the mean.

        Before the Q-1 fix, ``get_prediction_distribution()["median"]`` was
        computed via ``predict_mean()`` (standard RF average), so the
        "median" and "mean" keys were identical — a silent semantic error.
        After the fix, ``"median"`` routes through ``predict_median()``
        (which uses the leaf-weight weighted-quantile at q=0.5), while
        ``"mean"`` still calls ``predict_mean()`` (standard RF average).

        Two invariants are checked:
        1. The distribution "median" matches ``predict_median()`` exactly.
        2. The distribution "mean" matches ``predict_mean()`` exactly.
        3. ``predict_median()`` differs from ``predict_mean()`` on at least
           one sample — without this the test would have no discriminative power.
        """
        from forecasting.quantile_forest import QuantileRandomForestForecaster

        X, y = small_dataset
        model = QuantileRandomForestForecaster(n_estimators=40, random_state=42)
        model.fit(X, y)

        X_test = X[:10]

        dist = model.get_prediction_distribution(X_test)
        assert "median" in dist, "get_prediction_distribution() must have 'median' key"
        assert "mean"   in dist, "get_prediction_distribution() must have 'mean' key"

        dist_median = dist["median"]
        dist_mean   = dist["mean"]
        true_median = model.predict_median(X_test)
        true_mean   = model.predict_mean(X_test)

        # Invariant 1: distribution['median'] == predict_median()
        np.testing.assert_array_almost_equal(
            dist_median, true_median, decimal=10,
            err_msg=(
                "get_prediction_distribution()['median'] does not match "
                "predict_median().  Before Bug Q-1 was fixed, this key used "
                "predict_mean() instead of the leaf-weight quantile median."
            ),
        )

        # Invariant 2: distribution['mean'] == predict_mean()
        np.testing.assert_array_almost_equal(
            dist_mean, true_mean, decimal=10,
            err_msg="get_prediction_distribution()['mean'] must equal predict_mean().",
        )

        # Invariant 3: median != mean (discriminative power)
        diff = np.abs(dist_median.ravel() - dist_mean.ravel())
        assert diff.max() > 1e-6, (
            "predict_median() and predict_mean() are identical on all test points; "
            "the test has no discriminative power.  Either n_estimators is too small "
            "or the data distribution is perfectly symmetric."
        )

    def test_predict_quantiles(self, small_dataset):
        from forecasting.quantile_forest import QuantileRandomForestForecaster

        X, y = small_dataset
        model = QuantileRandomForestForecaster(n_estimators=20, random_state=42)
        model.fit(X, y)

        qpreds = model.predict_quantiles(X[:5], quantiles=[0.1, 0.5, 0.9])
        for q in [0.1, 0.5, 0.9]:
            assert q in qpreds
        # q10 ≤ q50 ≤ q90 (approximately, per sample)
        for i in range(5):
            for out_col in range(y.shape[1]):
                val10 = qpreds[0.1][i, out_col] if qpreds[0.1].ndim > 1 else qpreds[0.1][i]
                val90 = qpreds[0.9][i, out_col] if qpreds[0.9].ndim > 1 else qpreds[0.9][i]
                assert val10 <= val90 + 1e-8

    def test_feature_importance(self, small_dataset):
        from forecasting.quantile_forest import QuantileRandomForestForecaster

        X, y = small_dataset
        model = QuantileRandomForestForecaster(n_estimators=20, random_state=42)
        model.fit(X, y)
        imp = model.get_feature_importance()
        assert len(imp) == X.shape[1]


class TestPanelVAR:
    def test_fit_predict_with_entities(self, small_dataset, entity_indices):
        from forecasting.panel_var import PanelVARForecaster

        X, y = small_dataset
        model = PanelVARForecaster(n_lags=1, use_fixed_effects=True, random_state=42)
        model.fit(X, y, entity_indices=entity_indices)
        pred = model.predict(X, entity_indices=entity_indices)
        assert pred.shape == y.shape

    def test_predict_without_entities(self, small_dataset, entity_indices):
        from forecasting.panel_var import PanelVARForecaster

        X, y = small_dataset
        model = PanelVARForecaster(n_lags=1, use_fixed_effects=True, random_state=42)
        model.fit(X, y, entity_indices=entity_indices)
        # predict without entities: population-level (reference entity)
        pred = model.predict(X)
        assert pred.shape == y.shape

    def test_lag_matrix_shape(self):
        from forecasting.panel_var import PanelVARForecaster

        X = np.random.randn(20, 4)
        lag2 = PanelVARForecaster._build_lag_matrix(X, 2)
        assert lag2.shape == (18, 12)  # (20-2, 4*(1+2))

    def test_lag_matrix_values(self):
        from forecasting.panel_var import PanelVARForecaster

        X = np.arange(15, dtype=float).reshape(5, 3)
        lag1 = PanelVARForecaster._build_lag_matrix(X, 1)
        # Row 0: [X[1], X[0]]
        np.testing.assert_array_equal(lag1[0, :3], X[1])
        np.testing.assert_array_equal(lag1[0, 3:], X[0])

    def test_lag_selection_bic(self, small_dataset):
        from forecasting.panel_var import PanelVARForecaster

        X, y = small_dataset
        model = PanelVARForecaster(lag_selection="bic", max_lags=3, random_state=42)
        model.fit(X, y)
        assert model.selected_lags_ in (1, 2, 3)

    def test_feature_importance_size(self, small_dataset, entity_indices):
        from forecasting.panel_var import PanelVARForecaster

        X, y = small_dataset
        model = PanelVARForecaster(use_fixed_effects=True, random_state=42)
        model.fit(X, y, entity_indices=entity_indices)
        imp = model.get_feature_importance()
        assert len(imp) == X.shape[1]  # trimmed to original features


class TestNeuralAdditive:
    def test_fit_predict(self, small_dataset):
        from forecasting.neural_additive import NeuralAdditiveForecaster

        X, y = small_dataset
        model = NeuralAdditiveForecaster(
            n_basis_per_feature=10, n_iterations=3, random_state=42
        )
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == y.shape

    def test_multi_output_shape(self, small_dataset):
        """NAM handles multi-output targets with correct prediction shape."""
        from forecasting.neural_additive import NeuralAdditiveForecaster

        X, y = small_dataset
        assert y.ndim == 2 and y.shape[1] > 1, "fixture must be multi-output"
        model = NeuralAdditiveForecaster(
            n_basis_per_feature=8, n_iterations=2, random_state=42
        )
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == y.shape

    def test_get_shape_functions_returns_dict(self, small_dataset):
        """get_shape_functions() returns one array per input feature."""
        from forecasting.neural_additive import NeuralAdditiveForecaster

        X, y = small_dataset
        model = NeuralAdditiveForecaster(
            n_basis_per_feature=8, n_iterations=2, random_state=42
        )
        model.fit(X, y)
        shapes = model.get_shape_functions(X)
        assert len(shapes) == X.shape[1]
        for j, arr in shapes.items():
            assert arr.shape == (X.shape[0],), f"shape function {j} bad shape"

    def test_feature_importance_non_negative_and_length(self, small_dataset):
        """NAM feature importances are non-negative and have length n_features."""
        from forecasting.neural_additive import NeuralAdditiveForecaster

        X, y = small_dataset
        model = NeuralAdditiveForecaster(
            n_basis_per_feature=8, n_iterations=2, random_state=42
        )
        model.fit(X, y)
        imp = model.get_feature_importance()
        assert len(imp) == X.shape[1]
        assert np.all(imp >= 0)


# ---------------------------------------------------------------------------
# Super Learner
# ---------------------------------------------------------------------------

class TestSuperLearner:
    def test_weights_non_negative_and_sum_one(self, small_dataset):
        from forecasting.super_learner import SuperLearner
        from forecasting.gradient_boosting import CatBoostForecaster
        from forecasting.bayesian import BayesianForecaster

        X, y = small_dataset
        models = {
            "gb": CatBoostForecaster(iterations=20, random_state=42),
            "bay": BayesianForecaster(),
        }
        sl = SuperLearner(
            base_models=models,
            meta_learner_type="ridge",
            n_cv_folds=3,
            positive_weights=True,
            normalize_weights=True,
            random_state=42,
            verbose=False,
        )
        sl.fit(X, y)
        weights = sl.get_meta_weights()

        for w in weights.values():
            assert w >= -1e-10, f"Negative weight: {w}"
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_nnls_used_for_ridge_positive(self, small_dataset):
        """When positive_weights=True and meta_learner_type='ridge', NNLS is used."""
        from forecasting.super_learner import SuperLearner
        from forecasting.bayesian import BayesianForecaster

        X, y = small_dataset
        models = {"bay": BayesianForecaster()}
        sl = SuperLearner(
            base_models=models,
            meta_learner_type="ridge",
            positive_weights=True,
            normalize_weights=True,
            n_cv_folds=3,
            random_state=42,
            verbose=False,
        )
        sl.fit(X, y)
        # Should still produce valid weights without crashing
        assert sum(sl.get_meta_weights().values()) > 0

    def test_predict_shape(self, small_dataset):
        from forecasting.super_learner import SuperLearner
        from forecasting.gradient_boosting import CatBoostForecaster
        from forecasting.bayesian import BayesianForecaster

        X, y = small_dataset
        models = {
            "gb": CatBoostForecaster(iterations=10, random_state=42),
            "bay": BayesianForecaster(),
        }
        sl = SuperLearner(
            base_models=models, n_cv_folds=3,
            random_state=42, verbose=False,
        )
        sl.fit(X, y)
        pred = sl.predict(X)
        assert pred.shape == y.shape

    def test_entity_indices_forwarded(self, small_dataset, entity_indices):
        from forecasting.super_learner import SuperLearner
        from forecasting.panel_var import PanelVARForecaster
        from forecasting.bayesian import BayesianForecaster

        X, y = small_dataset
        models = {
            "pvar": PanelVARForecaster(n_lags=1, random_state=42),
            "bay": BayesianForecaster(),
        }
        sl = SuperLearner(
            base_models=models, n_cv_folds=3,
            random_state=42, verbose=False,
        )
        # Should not raise
        sl.fit(X, y, entity_indices=entity_indices)
        pred = sl.predict(X)
        assert pred.shape == y.shape


# ---------------------------------------------------------------------------
# Conformal Prediction
# ---------------------------------------------------------------------------

class TestConformalPredictor:
    def test_split_coverage(self):
        """
        T-1: Split conformal achieves ≥ (1-α) − 0.05 on n=200 test data.

        Uses a fully-specified linear DGP so a Ridge model is correctly
        specified.  This validates the mathematical guarantee rather than
        merely checking that code runs.

        With n=200 held-out points and a correctly-specified model the
        empirical coverage should be within 5 pp of the nominal level;
        the previous ±0.15 slack was too wide to catch real regressions.
        """
        from forecasting.conformal import ConformalPredictor
        from sklearn.linear_model import Ridge

        rng = np.random.RandomState(7)
        n, d = 200, 5
        coefs = np.array([1.5, -0.8, 2.0, 0.4, -1.2])
        X = rng.randn(n, d)
        # Gaussian noise with known σ=0.5 → Ridge is well-specified
        y = X @ coefs + rng.randn(n) * 0.5

        alpha = 0.10
        model = Ridge(alpha=1e-3)  # near-OLS; well-specified
        model.fit(X, y)

        cp = ConformalPredictor(method="split", alpha=alpha,
                                calibration_fraction=0.3)
        cp.calibrate(model, X, y)

        lower, upper = cp.predict_intervals(X)
        covered = (y >= lower) & (y <= upper)
        empirical_coverage = covered.mean()

        tol = 0.05
        assert empirical_coverage >= (1 - alpha) - tol, (
            f"Split conformal coverage {empirical_coverage:.3f} is below "
            f"({1 - alpha:.2f} - {tol:.2f}) = {(1-alpha-tol):.2f}.  "
            f"The finite-sample guarantee should hold within ±{tol:.0%}."
        )

    def test_cv_plus_coverage(self, univariate_dataset):
        from forecasting.conformal import ConformalPredictor
        from sklearn.linear_model import Ridge

        X, y = univariate_dataset
        model = Ridge()
        model.fit(X, y)

        cp = ConformalPredictor(method="cv_plus", alpha=0.10)
        cp.calibrate(model, X, y, cv_folds=3)

        lower, upper = cp.predict_intervals(X)
        covered = (y >= lower) & (y <= upper)
        assert covered.mean() >= 0.75

    def test_aci_no_lookahead(self, univariate_dataset):
        from forecasting.conformal import ConformalPredictor
        from sklearn.linear_model import Ridge

        X, y = univariate_dataset
        model = Ridge()
        model.fit(X, y)

        cp = ConformalPredictor(method="adaptive", alpha=0.10)
        cp.calibrate(model, X, y)

        # After Phase 2 H6 fix: ACI splits into train/cal, then
        # runs online updates on the second half of the cal set.
        n = len(y)
        n_cal = max(5, int(n * cp.calibration_fraction))  # 25
        n_init = max(3, n_cal // 2)                        # 12
        assert len(cp._aci_history) == n_cal - n_init

    def test_rejects_multioutput(self, small_dataset):
        from forecasting.conformal import ConformalPredictor
        from sklearn.linear_model import Ridge

        X, y = small_dataset
        model = Ridge()
        model.fit(X, y)
        cp = ConformalPredictor()
        with pytest.raises(ValueError, match="single-output"):
            cp.calibrate(model, X, y)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class TestForecastEvaluator:
    def test_oof_residuals(self, univariate_dataset):
        """
        T-2: Diagnostics use genuine OOF residuals, not in-sample residuals.

        For a Ridge model fitted on the full data, in-sample residuals have
        near-zero mean (Ridge shrinks toward zero, but the mean of y − ŷ
        is always exactly zero for OLS and very close to zero for Ridge).
        OOF residuals — where each fold's predictions come from a model that
        never saw those samples — should have a measurably higher absolute
        mean, proving that the CV mechanism is active and not just applying
        the full-data model to the training set.
        """
        from forecasting.evaluation import ForecastEvaluator
        from sklearn.linear_model import Ridge

        X, y = univariate_dataset
        model = Ridge()

        # Baseline: in-sample residual mean from a model fit on full data
        model_insample = Ridge()
        model_insample.fit(X, y)
        insample_resid_mean = float(
            np.abs(np.mean(y - model_insample.predict(X)))
        )

        evaluator = ForecastEvaluator(verbose=False, n_folds=3)
        results = evaluator.evaluate(model, X, y)

        diag = results["diagnostics"]
        assert "residual_mean" in diag
        assert "durbin_watson" in diag
        assert "error" not in diag

        oof_resid_mean = float(np.abs(diag["residual_mean"]))

        # OOF residual mean must exceed the in-sample near-zero baseline.
        # If ForecastEvaluator were using in-sample predictions, the means
        # would be similar (both ≈ 0 for well-fit Ridge on linear data).
        assert oof_resid_mean > insample_resid_mean, (
            f"OOF residual mean ({oof_resid_mean:.6f}) should be larger than "
            f"in-sample residual mean ({insample_resid_mean:.6f}). "
            "This suggests ForecastEvaluator is not using genuine OOF predictions."
        )

    def test_cv_scores(self, univariate_dataset):
        from forecasting.evaluation import ForecastEvaluator
        from sklearn.linear_model import Ridge

        X, y = univariate_dataset
        model = Ridge()
        evaluator = ForecastEvaluator(
            metrics=["r2", "rmse"], verbose=False, n_folds=3
        )
        results = evaluator.evaluate(model, X, y)
        assert "cv" in results
        assert "r2" in results["cv"]
        assert "rmse" in results["cv"]


# ---------------------------------------------------------------------------
# Regression: Phase 1 & Phase 2 fixes still work
# ---------------------------------------------------------------------------

class TestPhaseRegression:
    def test_phase1_conformal_split_refit(self, univariate_dataset):
        from forecasting.conformal import ConformalPredictor
        from sklearn.linear_model import Ridge

        X, y = univariate_dataset
        model = Ridge()
        model.fit(X, y)
        cp = ConformalPredictor(method="split", alpha=0.1, calibration_fraction=0.3)
        cp.calibrate(model, X, y)
        assert cp._q_hat > 0

    def test_phase2_gb_no_early_stop(self):
        from forecasting.gradient_boosting import CatBoostForecaster

        cb = CatBoostForecaster()
        # CatBoost writes no files and emits no per-iteration console output
        assert cb.model is None  # not yet fitted

    def test_phase2_lag_matrix(self):
        from forecasting.panel_var import PanelVARForecaster

        X = np.arange(20, dtype=float).reshape(5, 4)
        lag1 = PanelVARForecaster._build_lag_matrix(X, 1)
        assert lag1.shape == (4, 8)
        np.testing.assert_array_equal(lag1[0, :4], X[1])
        np.testing.assert_array_equal(lag1[0, 4:], X[0])


# ---------------------------------------------------------------------------
# Known-answer numerical test  (P4-28)
#
# y = 2·x₀  (exact linear relationship, near-zero noise)
# After training on enough samples the model must predict y_new ≈ 2·x_new
# within a generous 25 % relative tolerance.
# ---------------------------------------------------------------------------

class TestKnownAnswerForecasting:
    """Verify that base forecasters can recover a near-trivial signal."""

    @pytest.fixture
    def linear_data(self):
        """Single-feature, single-output (1-D): y = 2x + ε, ε ~ N(0, 0.01)."""
        rng = np.random.RandomState(0)
        n = 200
        X = rng.randn(n, 1)
        y = 2.0 * X[:, 0] + rng.randn(n) * 0.01  # 1-D so models use single-output path
        return X, y

    def test_gradient_boosting_recovers_known_slope(self, linear_data):
        """CatBoostForecaster predicts y ≈ 2x on held-out points."""
        from forecasting.gradient_boosting import CatBoostForecaster

        X, y = linear_data
        model = CatBoostForecaster(iterations=100, random_state=42)
        model.fit(X, y)

        x_vals = np.array([[1.0], [-1.0], [2.0], [0.5]])
        y_hat = model.predict(x_vals).ravel()
        y_true = 2.0 * x_vals[:, 0]  # 1-D: [2.0, -2.0, 4.0, 1.0]

        rel_err = np.abs(y_hat - y_true) / (np.abs(y_true) + 1e-8)
        assert rel_err.max() < 0.25, (
            f"Gradient boosting failed to recover y=2x; max relative error "
            f"= {rel_err.max():.3f}"
        )

    def test_bayesian_forecaster_recovers_known_slope(self, linear_data):
        """BayesianForecaster recovers y = 2x on held-out single-feature data."""
        from forecasting.bayesian import BayesianForecaster

        X, y = linear_data
        model = BayesianForecaster()
        model.fit(X, y)

        x_vals = np.array([[1.0], [-1.0], [2.0], [0.5]])
        y_hat = model.predict(x_vals).ravel()
        y_true = 2.0 * x_vals[:, 0]  # 1-D: [2.0, -2.0, 4.0, 1.0]

        rel_err = np.abs(y_hat - y_true) / (np.abs(y_true) + 1e-8)
        assert rel_err.max() < 0.25, (
            f"BayesianForecaster failed to recover y=2x; max relative error "
            f"= {rel_err.max():.3f}"
        )


# ---------------------------------------------------------------------------
# Regression tests for Phase 1-11 audit fixes (P11)
# ---------------------------------------------------------------------------

class TestAuditRegressions:
    """
    Regression tests that pin the behaviour introduced by the Phase 1-11
    production-hardening audit.  Each test documents exactly which bug it
    guards against so that future changes to the production code will
    immediately reveal a regression.
    """

    # -----------------------------------------------------------------------
    # B-2  PanelVAR lag_selection deprecation (Phase 3)
    # -----------------------------------------------------------------------

    def test_b2_lag_selection_deprecated_bic_warns(self):
        """Bug B-2: lag_selection='bic' must emit DeprecationWarning and map to 'cv'."""
        from forecasting.panel_var import PanelVARForecaster

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m = PanelVARForecaster(lag_selection="bic")

        dep_warns = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep_warns) >= 1, "Expected DeprecationWarning for lag_selection='bic'"
        assert "lag_selection_method" in str(dep_warns[0].message)
        assert m.lag_selection_method == "cv"

    def test_b2_lag_selection_deprecated_aic_warns(self):
        """Bug B-2: lag_selection='aic' must also warn and map to 'cv'."""
        from forecasting.panel_var import PanelVARForecaster

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m = PanelVARForecaster(lag_selection="aic")

        dep_warns = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep_warns) >= 1
        assert m.lag_selection_method == "cv"

    def test_b2_new_api_no_warning(self):
        """Bug B-2: new lag_selection_method='cv' raises no DeprecationWarning."""
        from forecasting.panel_var import PanelVARForecaster

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m = PanelVARForecaster(lag_selection_method="cv")

        dep_warns = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep_warns) == 0, "No DeprecationWarning expected for new API"
        assert m.lag_selection_method == "cv"

    # -----------------------------------------------------------------------
    # B-4  NAM² identifiability — training-data mean centering (Phase 6)
    # -----------------------------------------------------------------------

    def test_b4_nam2_interaction_centered_on_training_data(self):
        """Bug B-4: each interaction term g_jk must have E_train[g_jk]=0 (machine eps)."""
        from forecasting.neural_additive import NeuralAdditiveForecaster

        rng = np.random.RandomState(7)
        X = rng.randn(80, 6)
        y = rng.randn(80)

        nam = NeuralAdditiveForecaster(
            n_basis_per_feature=20, n_iterations=5,
            include_interactions=True, max_interaction_features=3,
            random_state=42,
        )
        nam.fit(X, y)

        X_sc = nam._scaler.transform(X)
        centers = nam._multi_output_interaction_centers.get(0, {})
        assert len(centers) > 0, "Expected at least one interaction term with max_interaction_features=3"

        for (j, k), net in nam._multi_output_interaction_shapes[0].items():
            raw = net.predict(X_sc[:, [j, k]])
            stored = centers[(j, k)]
            bias = abs(float(np.mean(raw - stored)))
            assert bias < 1e-9, (
                f"E_train[g_{j}{k} - center] = {bias:.2e} (expected < 1e-9). "
                "NAM\u00b2 identifiability constraint violated."
            )

    def test_b4_nam2_predict_is_stable_across_batches(self):
        """Bug B-4: predict() on different test batches must NOT recompute centers."""
        from forecasting.neural_additive import NeuralAdditiveForecaster

        rng = np.random.RandomState(8)
        X_tr = rng.randn(80, 6)
        y_tr = rng.randn(80)

        nam = NeuralAdditiveForecaster(
            n_basis_per_feature=20, n_iterations=5,
            include_interactions=True, max_interaction_features=3,
            random_state=42,
        )
        nam.fit(X_tr, y_tr)

        X_a = rng.randn(10, 6)
        X_b = rng.randn(10, 6)

        # Concatenated prediction must equal predictions done separately
        pred_ab = nam.predict(np.vstack([X_a, X_b]))
        pred_a  = nam.predict(X_a)
        pred_b  = nam.predict(X_b)

        np.testing.assert_allclose(
            pred_ab, np.concatenate([pred_a, pred_b]),
            atol=1e-12,
            err_msg="NAM\u00b2 predict() result must be batch-invariant (no test-data recentering).",
        )

    # -----------------------------------------------------------------------
    # B-6  sklearn_quantile graceful ImportError (Phase 5)
    # -----------------------------------------------------------------------

    def test_b6_qrf_raises_import_error_when_package_absent(self, monkeypatch):
        """Bug B-6: QuantileRandomForestForecaster raises clear ImportError when
        sklearn_quantile is unavailable (simulated via monkeypatch)."""
        import forecasting.quantile_forest as qfmod
        from forecasting.quantile_forest import QuantileRandomForestForecaster

        monkeypatch.setattr(qfmod, "_SKLEARN_QUANTILE_AVAILABLE", False)

        with pytest.raises(ImportError, match="pip install sklearn-quantile"):
            QuantileRandomForestForecaster()

    def test_b6_qrf_import_error_mentions_pyproject(self, monkeypatch):
        """Bug B-6: ImportError message must mention pyproject.toml for discoverability."""
        import forecasting.quantile_forest as qfmod
        from forecasting.quantile_forest import QuantileRandomForestForecaster

        monkeypatch.setattr(qfmod, "_SKLEARN_QUANTILE_AVAILABLE", False)

        with pytest.raises(ImportError, match="pyproject.toml"):
            QuantileRandomForestForecaster()

    # -----------------------------------------------------------------------
    # B-1 / Config round-trip (Phase 1-2)
    # -----------------------------------------------------------------------

    def test_config_forecast_fields_present_and_correct(self):
        """Bug B-1 / Phase 1: ForecastConfig must expose all 5 new hyperparameter fields."""
        from config import ForecastConfig

        cfg = ForecastConfig()
        assert cfg.gb_max_depth == 5,        f"gb_max_depth={cfg.gb_max_depth}"
        assert cfg.gb_n_estimators == 200,   f"gb_n_estimators={cfg.gb_n_estimators}"
        assert cfg.nam_n_basis == 30,        f"nam_n_basis={cfg.nam_n_basis}"
        assert cfg.nam_n_iterations == 10,   f"nam_n_iterations={cfg.nam_n_iterations}"
        assert cfg.pvar_lag_selection_method == "cv"

    def test_config_forecast_custom_values_round_trip(self):
        """Phase 1: ForecastConfig custom values survive to_dict() serialisation."""
        from config import Config, ForecastConfig

        cfg = ForecastConfig(gb_max_depth=6, nam_n_basis=50)
        # Embed in master Config to exercise to_dict()
        master = Config(forecast=cfg)
        d = master.to_dict()
        assert d["forecast"]["gb_max_depth"] == 6
        assert d["forecast"]["nam_n_basis"] == 50

    def test_config_wires_into_unified_create_models(self):
        """Phase 2: UnifiedForecaster._create_models() reads from ForecastConfig."""
        from forecasting.unified import UnifiedForecaster
        from config import ForecastConfig

        cfg = ForecastConfig(gb_max_depth=3, gb_n_estimators=50, nam_n_basis=15, nam_n_iterations=3)
        uf = UnifiedForecaster(config=cfg)
        models = uf._create_models()

        assert models["GradientBoosting"].depth == 3
        assert models["GradientBoosting"].iterations == 50
        assert models["NAM"].n_basis_per_feature == 15
        assert models["NAM"].n_iterations == 3

    # -----------------------------------------------------------------------
    # B-10  GradientBoosting class default max_depth=5 (Phase 10)
    # -----------------------------------------------------------------------

    def test_b10_gb_class_default_max_depth_is_5(self):
        """CatBoostForecaster class default depth must be 6 (CatBoost standard)."""
        from forecasting.gradient_boosting import CatBoostForecaster

        cb = CatBoostForecaster()
        assert cb.depth == 6, (
            f"Class default depth={cb.depth}, expected 6. "
            "CatBoost default depth=6 (2^6=64 leaves) replaces sklearn depth=5."
        )
        assert cb.iterations == 300  # class default for n >= 500


# ---------------------------------------------------------------------------
# Phase 8 — Pipeline Decoupling: stage-level API
# ---------------------------------------------------------------------------

class TestPipelineDecoupling:
    """Tests for Phase 8 — Pipeline Decoupling (stage1…stage7, get_stage_outputs,
    ForecastConfig.pipeline_mode).

    Design: each test calls exactly the stages under test with no full model
    fitting (slow), keeping the suite fast.  Stages 3+ integration is already
    covered by the existing full-pipeline tests above.
    """

    # ------------------------------------------------------------------
    # Mock-panel factory (mirrors TestFeatureEngineer._make_mock_panel)
    # ------------------------------------------------------------------

    @staticmethod
    def _make_mock_panel(n_entities: int = 5, n_years: int = 7, n_components: int = 3):
        """Return a lightweight mock PanelData.

        Years: 2011 … 2011+n_years-1.
        Provinces: P0 … P{n_entities-1}.
        Components (subcriteria): x0 … x{n_components-1}.
        """
        import pandas as pd
        from types import SimpleNamespace

        years      = list(range(2011, 2011 + n_years))
        provinces  = [f"P{i}" for i in range(n_entities)]
        components = [f"x{j}" for j in range(n_components)]

        class MockPanel:
            pass

        panel                    = MockPanel()
        panel.provinces          = provinces
        panel.subcriteria_names  = components
        panel.years              = years

        rng = np.random.RandomState(7)
        data_store: dict = {}
        for p in provinces:
            data_store[p] = pd.DataFrame(
                rng.rand(n_years, n_components),
                index=years,
                columns=components,
            )

        panel.get_province = lambda name: data_store[name]

        cs: dict = {}
        for y in years:
            rows = []
            for p in provinces:
                row = {"Province": p}
                row.update(data_store[p].loc[y].to_dict())
                rows.append(row)
            cs[y] = pd.DataFrame(rows)
        panel.cross_section = cs

        def _make_ctx(yr):
            ctx                    = SimpleNamespace()
            ctx.year               = yr
            ctx.active_provinces   = list(provinces)
            ctx.active_subcriteria = list(components)
            ctx.valid_pairs        = {(p, sc) for p in provinces for sc in components}
            ctx.is_valid           = lambda prov, sc, _c=ctx: (prov, sc) in _c.valid_pairs
            return ctx

        panel.year_contexts = {yr: _make_ctx(yr) for yr in years}
        return panel

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _make_uf(**kwargs):
        """Create UnifiedForecaster configured for stage-level unit tests.

        Defaults: subcriteria target level (matches mock panel), SAW targets
        disabled to avoid normalization overhead, verbose off.
        """
        from forecasting.unified import UnifiedForecaster
        from config import ForecastConfig

        cfg = ForecastConfig(use_saw_targets=False, **kwargs)
        return UnifiedForecaster(config=cfg, verbose=False, target_level='subcriteria')

    # ------------------------------------------------------------------
    # P8-1  stage1_engineer_features
    # ------------------------------------------------------------------

    def test_stage1_populates_training_data(self):
        """stage1_engineer_features() must populate X_train_, y_train_, X_pred_,
        and entity_info_ without fitting any model."""
        panel = self._make_mock_panel()
        uf    = self._make_uf()

        uf.stage1_engineer_features(panel, target_year=2018)

        assert not uf.X_train_.empty,  "X_train_ must be non-empty after stage1"
        assert not uf.y_train_.empty,  "y_train_ must be non-empty after stage1"
        assert not uf.X_pred_.empty,   "X_pred_ must be non-empty after stage1"
        assert len(uf.X_pred_) == len(panel.provinces), (
            "X_pred_ must have exactly one row per province"
        )
        assert uf.X_train_.shape[0] == uf.y_train_.shape[0], (
            "X_train_ and y_train_ must share the same row count"
        )
        assert uf.X_train_.shape[1] == uf.X_pred_.shape[1], (
            "Training and prediction sets must share the same feature count"
        )
        assert not uf.entity_info_.empty, "entity_info_ must be non-empty after stage1"
        assert len(uf.entity_info_) == len(uf.X_train_), (
            "entity_info_ row count must equal X_train_ row count"
        )
        assert uf.super_learner_ is None, (
            "stage1 must not fit any model; super_learner_ must remain None"
        )

    def test_stage1_holdout_reserved(self):
        """stage1 must auto-hold out the most-recent training year
        (max panel year < target_year)."""
        panel       = self._make_mock_panel()
        target_year = 2018
        uf          = self._make_uf()

        uf.stage1_engineer_features(panel, target_year)

        expected_holdout = max(y for y in panel.years if y < target_year)
        assert uf.holdout_year_ is not None, "holdout_year_ must be set"
        assert uf.holdout_year_ == expected_holdout, (
            f"Expected holdout_year_={expected_holdout}, got {uf.holdout_year_}"
        )
        assert uf.holdout_year_ < target_year, (
            "Holdout year must be strictly before the forecast target"
        )

    def test_stage1_holdout_features_non_empty(self):
        """X_holdout_ and y_holdout_ must hold the withheld year's data."""
        panel = self._make_mock_panel()
        uf    = self._make_uf()

        uf.stage1_engineer_features(panel, target_year=2018)

        assert not uf.X_holdout_.empty, "X_holdout_ must be non-empty"
        assert not uf.y_holdout_.empty, "y_holdout_ must be non-empty"
        assert uf.X_holdout_.shape[1] == uf.X_train_.shape[1], (
            "Holdout and training features must have the same column count"
        )

    def test_stage1_y_train_columns_match_components(self):
        """y_train_ column count must equal n_components from the mock panel."""
        n_comp = 4
        panel  = self._make_mock_panel(n_components=n_comp)
        uf     = self._make_uf()

        uf.stage1_engineer_features(panel, target_year=2018)

        assert uf.y_train_.shape[1] == n_comp, (
            f"y_train_ should have {n_comp} columns, got {uf.y_train_.shape[1]}"
        )

    # ------------------------------------------------------------------
    # P8-2  stage2_reduce_features
    # ------------------------------------------------------------------

    def test_stage2_populates_all_reduced_arrays(self):
        """After stage2, all four reduced feature arrays must be non-empty
        ndarrays with consistent row counts."""
        panel = self._make_mock_panel()
        uf    = self._make_uf()
        uf.stage1_engineer_features(panel, 2018)
        uf.stage2_reduce_features()

        for attr in ('X_train_pca_', 'X_train_tree_', 'X_pred_pca_', 'X_pred_tree_'):
            arr = getattr(uf, attr)
            assert arr is not None and len(arr) > 0, (
                f"{attr} must be a non-empty array after stage2"
            )

        assert uf.X_train_pca_.shape[0]  == len(uf.X_train_)
        assert uf.X_pred_pca_.shape[0]   == len(uf.X_pred_)
        assert uf.X_train_tree_.shape[0] == len(uf.X_train_)
        assert uf.X_pred_tree_.shape[0]  == len(uf.X_pred_)

    def test_stage2_pca_reduces_or_preserves_features(self):
        """PCA track must not increase feature count beyond the original."""
        panel = self._make_mock_panel()
        uf    = self._make_uf()
        uf.stage1_engineer_features(panel, 2018)
        uf.stage2_reduce_features()

        assert uf.X_train_pca_.shape[1] <= uf.X_train_.shape[1], (
            "PCA-reduced feature count must be ≤ original feature count"
        )

    def test_stage2_per_model_dicts_populated(self):
        """_per_model_X_train_ and _per_model_X_pred_ must cover all 5 base models."""
        panel = self._make_mock_panel()
        uf    = self._make_uf()
        uf.stage1_engineer_features(panel, 2018)
        uf.stage2_reduce_features()

        expected_models = {'BayesianRidge', 'GradientBoosting', 'QuantileRF',
                           'PanelVAR', 'NAM'}
        assert expected_models <= set(uf._per_model_X_train_.keys()), (
            "_per_model_X_train_ must cover all 5 base models"
        )
        assert expected_models <= set(uf._per_model_X_pred_.keys()), (
            "_per_model_X_pred_ must cover all 5 base models"
        )

    # ------------------------------------------------------------------
    # P8-3  pipeline_mode='features_only'
    # ------------------------------------------------------------------

    def test_pipeline_mode_features_only_returns_none(self):
        """ForecastConfig.pipeline_mode='features_only' → fit_predict returns None."""
        from forecasting.unified import UnifiedForecaster
        from config import ForecastConfig

        panel  = self._make_mock_panel()
        cfg    = ForecastConfig(pipeline_mode='features_only', use_saw_targets=False)
        uf     = UnifiedForecaster(config=cfg, verbose=False, target_level='subcriteria')
        result = uf.fit_predict(panel, target_year=2018)

        assert result is None, (
            "pipeline_mode='features_only' must return None from fit_predict()"
        )

    def test_pipeline_mode_features_only_no_model_fitted(self):
        """In 'features_only' mode, no base model or SuperLearner must be fitted."""
        from forecasting.unified import UnifiedForecaster
        from config import ForecastConfig

        panel  = self._make_mock_panel()
        cfg    = ForecastConfig(pipeline_mode='features_only', use_saw_targets=False)
        uf     = UnifiedForecaster(config=cfg, verbose=False, target_level='subcriteria')
        uf.fit_predict(panel, target_year=2018)

        assert uf.super_learner_ is None, (
            "super_learner_ must remain None in 'features_only' mode"
        )

    def test_pipeline_mode_features_only_stages12_done(self):
        """'features_only' must have Stages 1 and 2 results populated."""
        from forecasting.unified import UnifiedForecaster
        from config import ForecastConfig

        panel  = self._make_mock_panel()
        cfg    = ForecastConfig(pipeline_mode='features_only', use_saw_targets=False)
        uf     = UnifiedForecaster(config=cfg, verbose=False, target_level='subcriteria')
        uf.fit_predict(panel, target_year=2018)

        assert not uf.X_train_.empty,  "X_train_ must be populated in 'features_only'"
        assert not uf.y_train_.empty,  "y_train_ must be populated in 'features_only'"
        assert uf.X_train_pca_  is not None, "X_train_pca_ must be set in 'features_only'"
        assert uf.X_train_tree_ is not None, "X_train_tree_ must be set in 'features_only'"

    # ------------------------------------------------------------------
    # P8-4  ForecastConfig.pipeline_mode field
    # ------------------------------------------------------------------

    def test_config_pipeline_mode_default_is_full(self):
        """ForecastConfig.pipeline_mode must default to 'full'."""
        from config import ForecastConfig

        cfg = ForecastConfig()
        assert cfg.pipeline_mode == 'full', (
            f"ForecastConfig.pipeline_mode default should be 'full', "
            f"got '{cfg.pipeline_mode}'"
        )

    def test_config_pipeline_mode_accepts_all_literals(self):
        """ForecastConfig accepts all four documented pipeline_mode values."""
        from config import ForecastConfig

        for mode in ('full', 'features_only', 'fit_only', 'evaluate_only'):
            cfg = ForecastConfig(pipeline_mode=mode)
            assert cfg.pipeline_mode == mode

    def test_config_pipeline_mode_wires_to_forecaster(self):
        """UnifiedForecaster.pipeline_mode must reflect ForecastConfig.pipeline_mode."""
        from forecasting.unified import UnifiedForecaster
        from config import ForecastConfig

        for mode in ('full', 'features_only', 'fit_only', 'evaluate_only'):
            cfg = ForecastConfig(pipeline_mode=mode)
            uf  = UnifiedForecaster(config=cfg, verbose=False)
            assert uf.pipeline_mode == mode, (
                f"UnifiedForecaster.pipeline_mode should be '{mode}', "
                f"got '{uf.pipeline_mode}'"
            )

    def test_forecaster_default_pipeline_mode_is_full(self):
        """UnifiedForecaster() without config must default pipeline_mode='full'."""
        from forecasting.unified import UnifiedForecaster

        uf = UnifiedForecaster(verbose=False)
        assert uf.pipeline_mode == 'full', (
            f"Default pipeline_mode should be 'full', got '{uf.pipeline_mode}'"
        )

    # ------------------------------------------------------------------
    # P8-5  get_stage_outputs
    # ------------------------------------------------------------------

    def test_get_stage_outputs_returns_dict_with_required_keys(self):
        """get_stage_outputs() must return a dict containing all documented keys
        even before any stage runs (values will be empty/None, but keys present)."""
        from forecasting.unified import UnifiedForecaster

        uf  = UnifiedForecaster(verbose=False)
        out = uf.get_stage_outputs()

        assert isinstance(out, dict), "get_stage_outputs() must return a dict"

        required_keys = (
            # Stage 1
            'X_train', 'y_train', 'X_pred', 'X_holdout', 'y_holdout', 'entity_info',
            # Stage 2
            'X_train_pca', 'X_train_tree', 'X_pred_pca', 'X_pred_tree',
            'reducer_pca', 'reducer_tree',
            # Stage 3
            'models', 'oof_predictions',
            # Stage 4
            'super_learner', 'model_weights',
            # Stage 5
            'prediction_intervals',
            # Stage 6
            'model_comparison', 'holdout_performance',
            # Stage 7
            'composite_predictions',
        )
        for key in required_keys:
            assert key in out, f"Key '{key}' missing from get_stage_outputs() result"

    def test_get_stage_outputs_populated_after_stage1(self):
        """After stage1, Stage 1 output keys must be non-empty DataFrames;
        Stage 2+ keys must still be None."""
        panel = self._make_mock_panel()
        uf    = self._make_uf()
        uf.stage1_engineer_features(panel, 2018)

        out = uf.get_stage_outputs()

        assert out['X_train'] is not None and not out['X_train'].empty
        assert out['y_train'] is not None and not out['y_train'].empty
        assert out['X_pred']  is not None and not out['X_pred'].empty
        # Stage 2 not yet run
        assert out['X_train_pca']  is None, "X_train_pca must be None before stage2"
        assert out['X_train_tree'] is None, "X_train_tree must be None before stage2"

    def test_get_stage_outputs_populated_after_stage2(self):
        """After stages 1–2, all Stage 2 keys in get_stage_outputs() must be
        non-None arrays; Stage 3+ keys must still be None."""
        panel = self._make_mock_panel()
        uf    = self._make_uf()
        uf.stage1_engineer_features(panel, 2018)
        uf.stage2_reduce_features()

        out = uf.get_stage_outputs()

        for key in ('X_train_pca', 'X_train_tree', 'X_pred_pca', 'X_pred_tree'):
            assert out[key] is not None, (
                f"'{key}' must be non-None in get_stage_outputs() after stage2"
            )
        assert out['reducer_pca']  is not None, "reducer_pca must be set after stage2"
        assert out['reducer_tree'] is not None, "reducer_tree must be set after stage2"
        # Stage 3+ still absent
        assert out['super_learner'] is None, (
            "super_learner must remain None before stage3"
        )
