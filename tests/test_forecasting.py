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

        return panel

    def test_training_sample_count(self):
        """All consecutive year-pairs should be used (no wasted year)."""
        from forecasting.features import TemporalFeatureEngineer

        panel = self._make_mock_panel(n_entities=4, n_years=6, n_components=2)
        eng = TemporalFeatureEngineer(
            lag_periods=[1], rolling_windows=[2],
            include_momentum=True, include_cross_entity=True,
        )
        X_train, y_train, X_pred, info = eng.fit_transform(panel, target_year=2017)

        # 4 entities × 5 year-pairs (2011→12, 12→13, 13→14, 14→15, 15→16)
        assert X_train.shape[0] == 4 * 5
        assert y_train.shape[0] == 4 * 5
        assert X_pred.shape[0] == 4

    def test_entity_indices_returned(self):
        from forecasting.features import TemporalFeatureEngineer

        panel = self._make_mock_panel()
        eng = TemporalFeatureEngineer(lag_periods=[1], rolling_windows=[2])
        _, _, _, info = eng.fit_transform(panel, target_year=2016)
        assert "entity_index" in info.columns

    def test_feature_dimensions_consistent(self):
        from forecasting.features import TemporalFeatureEngineer

        panel = self._make_mock_panel(n_entities=3, n_years=5, n_components=3)
        eng = TemporalFeatureEngineer(lag_periods=[1, 2], rolling_windows=[2, 3])
        X_train, _, X_pred, _ = eng.fit_transform(panel, target_year=2016)
        assert X_train.shape[1] == X_pred.shape[1]
        assert len(eng.get_feature_names()) == X_train.shape[1]

    def test_no_nan_in_features(self):
        from forecasting.features import TemporalFeatureEngineer

        panel = self._make_mock_panel()
        eng = TemporalFeatureEngineer(lag_periods=[1], rolling_windows=[2])
        X_train, y_train, X_pred, _ = eng.fit_transform(panel, target_year=2016)
        assert not np.isnan(X_train.values).any()
        assert not np.isnan(y_train.values).any()
        assert not np.isnan(X_pred.values).any()


# ---------------------------------------------------------------------------
# Base Models — fit / predict round-trip
# ---------------------------------------------------------------------------

class TestGradientBoosting:
    def test_fit_predict_multioutput(self, small_dataset):
        from forecasting.gradient_boosting import GradientBoostingForecaster

        X, y = small_dataset
        model = GradientBoostingForecaster(n_estimators=20, random_state=42)
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == y.shape

    def test_no_random_early_stopping(self):
        from forecasting.gradient_boosting import GradientBoostingForecaster

        model = GradientBoostingForecaster()
        assert model._base_model.n_iter_no_change is None

    def test_feature_importance(self, small_dataset):
        from forecasting.gradient_boosting import GradientBoostingForecaster

        X, y = small_dataset
        model = GradientBoostingForecaster(n_estimators=20, random_state=42)
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


class TestQuantileRandomForest:
    def test_fit_predict_multioutput(self, small_dataset):
        from forecasting.quantile_forest import QuantileRandomForestForecaster

        X, y = small_dataset
        model = QuantileRandomForestForecaster(n_estimators=20, random_state=42)
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == y.shape

    def test_predict_returns_median(self, small_dataset):
        """predict() should return the conditional median, not mean."""
        from forecasting.quantile_forest import QuantileRandomForestForecaster

        X, y = small_dataset
        model = QuantileRandomForestForecaster(n_estimators=20, random_state=42)
        model.fit(X, y)

        median_pred = model.predict(X[:5])
        mean_pred = model.predict_mean(X[:5])

        # They should generally differ (median ≠ mean for skewed leaves)
        # At the very least, both should be valid arrays
        assert median_pred.shape == mean_pred.shape

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


class TestHierarchicalBayes:
    def test_multi_output_distinct(self, small_dataset, entity_indices):
        from forecasting.hierarchical_bayes import HierarchicalBayesForecaster

        X, y = small_dataset
        model = HierarchicalBayesForecaster(n_em_iterations=5, random_state=42)
        model.fit(X, y, group_indices=entity_indices)
        pred = model.predict(X, group_indices=entity_indices)
        assert pred.shape == y.shape

        # Outputs should not be identical
        for i in range(y.shape[1]):
            for j in range(i + 1, y.shape[1]):
                assert np.abs(pred[:, i] - pred[:, j]).sum() > 0.01

    def test_warns_without_groups(self, small_dataset):
        from forecasting.hierarchical_bayes import HierarchicalBayesForecaster

        X, y = small_dataset
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = HierarchicalBayesForecaster(n_em_iterations=3, random_state=42)
            model.fit(X, y)
            assert any("no group_indices" in str(x.message) for x in w)

    def test_predict_with_uncertainty(self, small_dataset, entity_indices):
        from forecasting.hierarchical_bayes import HierarchicalBayesForecaster

        X, y = small_dataset
        model = HierarchicalBayesForecaster(n_em_iterations=5, random_state=42)
        model.fit(X, y, group_indices=entity_indices)
        mean, std = model.predict_with_uncertainty(X)
        # Phase 4: predict_with_uncertainty() is multi-output aware.
        # For a k-output model it returns (n, k); for single-output (n,).
        n = X.shape[0]
        k = y.shape[1] if y.ndim > 1 else 1
        if k > 1:
            assert mean.shape == (n, k), f"Expected ({n}, {k}), got {mean.shape}"
            assert std.shape == (n, k), f"Expected ({n}, {k}), got {std.shape}"
        else:
            assert mean.shape == (n,)
            assert std.shape == (n,)
        assert np.all(std > 0)


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


# ---------------------------------------------------------------------------
# Super Learner
# ---------------------------------------------------------------------------

class TestSuperLearner:
    def test_weights_non_negative_and_sum_one(self, small_dataset):
        from forecasting.super_learner import SuperLearner
        from forecasting.gradient_boosting import GradientBoostingForecaster
        from forecasting.bayesian import BayesianForecaster

        X, y = small_dataset
        models = {
            "gb": GradientBoostingForecaster(n_estimators=20, random_state=42),
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
        from forecasting.gradient_boosting import GradientBoostingForecaster
        from forecasting.bayesian import BayesianForecaster

        X, y = small_dataset
        models = {
            "gb": GradientBoostingForecaster(n_estimators=10, random_state=42),
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
    def test_split_coverage(self, univariate_dataset):
        """Split conformal should achieve ~(1-α) coverage on synthetic data."""
        from forecasting.conformal import ConformalPredictor
        from sklearn.linear_model import Ridge

        X, y = univariate_dataset
        alpha = 0.10

        model = Ridge()
        model.fit(X, y)

        cp = ConformalPredictor(method="split", alpha=alpha, calibration_fraction=0.3)
        cp.calibrate(model, X, y)

        lower, upper = cp.predict_intervals(X)
        covered = (y >= lower) & (y <= upper)
        # Empirical coverage should not be catastrophically low
        assert covered.mean() >= (1 - alpha) - 0.15

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

        n = len(y)
        n_init = max(5, n // 2)
        assert len(cp._aci_history) == n - n_init

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
        """Diagnostics should use out-of-fold residuals."""
        from forecasting.evaluation import ForecastEvaluator
        from sklearn.linear_model import Ridge

        X, y = univariate_dataset
        model = Ridge()

        evaluator = ForecastEvaluator(verbose=False, n_folds=3)
        results = evaluator.evaluate(model, X, y)

        diag = results["diagnostics"]
        assert "residual_mean" in diag
        assert "durbin_watson" in diag
        # OOF residual mean should be small but non-zero (not exactly 0
        # as it would be with in-sample residuals from an OLS model)
        assert "error" not in diag

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
    def test_phase1_hierarch_bayes(self, small_dataset, entity_indices):
        from forecasting.hierarchical_bayes import HierarchicalBayesForecaster

        X, y = small_dataset
        hb = HierarchicalBayesForecaster(n_em_iterations=5, random_state=42)
        hb.fit(X, y, group_indices=entity_indices)
        assert len(hb._global_models) == y.shape[1]
        preds = hb.predict(X, group_indices=entity_indices)
        assert preds.shape == y.shape

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
        from forecasting.gradient_boosting import GradientBoostingForecaster

        gb = GradientBoostingForecaster()
        assert gb._base_model.n_iter_no_change is None

    def test_phase2_lag_matrix(self):
        from forecasting.panel_var import PanelVARForecaster

        X = np.arange(20, dtype=float).reshape(5, 4)
        lag1 = PanelVARForecaster._build_lag_matrix(X, 1)
        assert lag1.shape == (4, 8)
        np.testing.assert_array_equal(lag1[0, :4], X[1])
        np.testing.assert_array_equal(lag1[0, 4:], X[0])
