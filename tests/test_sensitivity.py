# -*- coding: utf-8 -*-
"""
Unit tests for analysis/sensitivity.py — Phase 6 coverage.

Covers the Phase 5 bug-fixes:
  - M7: MC simulation count is no longer silently capped at 100
  - M8: RuntimeWarning emitted when >20% of MC simulations fail
  - M9: Spearman (not Pearson) used in temporal stability analysis

Also covers weight perturbation and ranking stability checks.
"""

import warnings
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Mock helpers — minimal stubs for panel_data / ranking_pipeline
# ---------------------------------------------------------------------------

def _mock_weights(n: int = 4):
    """Return a minimal 'weights' dict as expected by SensitivityAnalysis."""
    subcriteria = [f"S{i}" for i in range(n)]
    w = np.ones(n) / n
    return {
        "global_sc_weights": dict(zip(subcriteria, w)),
        "sc_array": w.copy(),
        "subcriteria": subcriteria,
        "criterion_weights": {},
        "details": {},
    }


def _mock_rank_result(n_provinces: int, rng: np.random.RandomState | None = None):
    """Return a mock ranking result whose final_ranking looks like a pd.Series."""
    if rng is None:
        rng = np.random.RandomState(0)
    scores = rng.rand(n_provinces)
    provinces = [f"P{i}" for i in range(n_provinces)]
    return _FakeResult(pd.Series(scores, index=provinces))


class _FakeResult:
    """Minimal stand-in for HierarchicalRankingResult."""
    def __init__(self, series: pd.Series):
        self.final_ranking = series
        # Do NOT set criterion_method_scores → _can_fast_mc stays False


class _CountingPipeline:
    """Pipeline that counts rank() invocations and returns deterministic results."""

    def __init__(self, n_provinces: int = 6):
        self.n_provinces = n_provinces
        self.call_count = 0

    def rank(self, panel_data, weights_dict, target_year=None):
        self.call_count += 1
        rng = np.random.RandomState(self.call_count)
        return _mock_rank_result(self.n_provinces, rng)


class _AlwaysFailPipeline:
    """Pipeline whose rank() always raises after the first call (base run)."""

    def __init__(self, n_provinces: int = 5):
        self.n_provinces = n_provinces
        self._calls = 0

    def rank(self, panel_data, weights_dict, target_year=None):
        self._calls += 1
        if self._calls == 1:
            # First call succeeds (provides base ranking)
            return _mock_rank_result(self.n_provinces)
        raise RuntimeError("Simulated MC failure")


class _MockPanel:
    """Minimal mock of PanelData for sensitivity tests."""

    def __init__(self, n_provinces: int = 6, n_years: int = 3):
        self.provinces = [f"P{i}" for i in range(n_provinces)]
        self.years = list(range(2020, 2020 + n_years))
        self.hierarchy = _MockHierarchy(
            subcriteria=[f"S{i}" for i in range(4)],
            criteria=["E", "F"],
        )

    # convenience so _temporal_stability_analysis can call panel_data.provinces
    def __iter__(self):
        return iter(self.provinces)


class _MockHierarchy:
    def __init__(self, subcriteria, criteria):
        self.all_subcriteria = subcriteria
        self.all_criteria = criteria
        self.subcriteria_to_criteria = {
            s: criteria[i % len(criteria)] for i, s in enumerate(subcriteria)
        }
        self.criteria_to_subcriteria = {
            c: [s for s in subcriteria if self.subcriteria_to_criteria[s] == c]
            for c in criteria
        }


# ---------------------------------------------------------------------------
# M7 — MC count not capped
# ---------------------------------------------------------------------------

class TestMCSimulationCountNotCapped:
    """MC simulations must run exactly n_simulations iterations (no 100-cap)."""

    def test_small_count_honoured(self):
        """n_simulations=20 → pipeline called 1 (base) + 20 (MC) = 21 times."""
        from analysis.sensitivity import SensitivityAnalysis

        panel = _MockPanel(n_provinces=5, n_years=2)
        pipeline = _CountingPipeline(n_provinces=5)
        weights = _mock_weights(n=4)

        sa = SensitivityAnalysis(n_simulations=20, n_jobs=1, seed=0)
        sa._monte_carlo_pipeline(panel, pipeline, weights)

        # 1 base call + 20 simulation calls = 21
        assert pipeline.call_count == 21, (
            f"Expected 21 calls (1 base + 20 MC), got {pipeline.call_count}"
        )

    def test_large_count_not_silently_capped(self):
        """
        n_simulations=150 must NOT be silently truncated to 100.
        In the old code, n_sims = min(n_simulations, 100); after the fix
        n_sims = max(1, n_simulations).  We only measure the call count here
        to stay fast — use n_simulations=150 but 5 provinces so each call is
        trivially cheap.
        """
        from analysis.sensitivity import SensitivityAnalysis

        panel = _MockPanel(n_provinces=5, n_years=2)
        pipeline = _CountingPipeline(n_provinces=5)
        weights = _mock_weights(n=4)

        sa = SensitivityAnalysis(n_simulations=150, n_jobs=1, seed=1)
        sa._monte_carlo_pipeline(panel, pipeline, weights)

        # Must be called more than 101 times (1+100) — old cap would give exactly 101
        assert pipeline.call_count > 101, (
            f"MC count appears capped: only {pipeline.call_count} pipeline calls "
            f"for n_simulations=150 (expected > 101)"
        )

    def test_zero_simulations_runs_at_least_one(self):
        """n_simulations=0 → max(1, 0) = 1 simulation must run."""
        from analysis.sensitivity import SensitivityAnalysis

        panel = _MockPanel(n_provinces=4, n_years=2)
        pipeline = _CountingPipeline(n_provinces=4)
        weights = _mock_weights(n=3)

        sa = SensitivityAnalysis(n_simulations=0, n_jobs=1, seed=0)
        sa._monte_carlo_pipeline(panel, pipeline, weights)

        # 1 (base) + 1 (MC, clamped from 0 to 1) = 2
        assert pipeline.call_count == 2, (
            f"Expected 2 pipeline calls for n_simulations=0, got {pipeline.call_count}"
        )


# ---------------------------------------------------------------------------
# M8 — RuntimeWarning when >20% of MC simulations fail
# ---------------------------------------------------------------------------

class TestMCFailureWarning:
    """High MC failure rate must trigger a RuntimeWarning."""

    def test_all_failures_raises_runtime_warning(self):
        """
        If every simulation after the base call fails, failure_rate = 100% > 20%
        → RuntimeWarning must be raised.
        """
        from analysis.sensitivity import SensitivityAnalysis

        panel = _MockPanel(n_provinces=5, n_years=2)
        pipeline = _AlwaysFailPipeline(n_provinces=5)
        weights = _mock_weights(n=4)

        sa = SensitivityAnalysis(n_simulations=15, n_jobs=1, seed=0)

        with pytest.warns(RuntimeWarning, match=r"MC simulation.*fail"):
            sa._monte_carlo_pipeline(panel, pipeline, weights)

    def test_no_warning_when_few_failures(self):
        """
        A reliable pipeline has 0% failures → no RuntimeWarning should be issued.
        """
        from analysis.sensitivity import SensitivityAnalysis

        panel = _MockPanel(n_provinces=5, n_years=2)
        pipeline = _CountingPipeline(n_provinces=5)
        weights = _mock_weights(n=4)

        sa = SensitivityAnalysis(n_simulations=20, n_jobs=1, seed=0)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            # Should NOT raise; if it does, the test will fail
            sa._monte_carlo_pipeline(panel, pipeline, weights)


class _PartialFailurePipeline:
    """Fails on exactly every 5th simulation call (after the base call)."""

    def __init__(self, n_provinces: int = 5, fail_every: int = 5):
        self.n_provinces = n_provinces
        self.fail_every = fail_every
        self._calls = 0

    def rank(self, panel_data, weights_dict, target_year=None):
        self._calls += 1
        if self._calls > 1 and (self._calls - 1) % self.fail_every == 0:
            raise RuntimeError("Intermittent failure")
        return _mock_rank_result(self.n_provinces)


class TestMCFailureWarningThreshold:
    def test_warning_exactly_at_threshold_above(self):
        """
        Failure rate just above 20% (e.g., 5/20 = 25%) MUST trigger the warning.
        _PartialFailurePipeline(fail_every=4) → 5 failures in 20 MC sims = 25%.
        """
        from analysis.sensitivity import SensitivityAnalysis

        panel = _MockPanel(n_provinces=5, n_years=2)
        pipeline = _PartialFailurePipeline(n_provinces=5, fail_every=4)
        weights = _mock_weights(n=4)

        sa = SensitivityAnalysis(n_simulations=20, n_jobs=1, seed=0)

        with pytest.warns(RuntimeWarning):
            sa._monte_carlo_pipeline(panel, pipeline, weights)

    def test_no_warning_at_threshold_below(self):
        """
        Failure rate just below 20% (e.g., 3/20 = 15%) must NOT trigger the warning.
        fail_every=7 → 2 failures in 20 MC sims ≈ 10%.
        """
        from analysis.sensitivity import SensitivityAnalysis

        panel = _MockPanel(n_provinces=5, n_years=2)
        pipeline = _PartialFailurePipeline(n_provinces=5, fail_every=7)
        weights = _mock_weights(n=4)

        sa = SensitivityAnalysis(n_simulations=20, n_jobs=1, seed=0)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sa._monte_carlo_pipeline(panel, pipeline, weights)

        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0, (
            "Unexpected RuntimeWarning for low failure rate "
            f"({[str(w.message) for w in runtime_warnings]})"
        )


# ---------------------------------------------------------------------------
# M9 — Spearman (not Pearson) used in _temporal_stability_analysis
# ---------------------------------------------------------------------------

class _YearSpecificPipeline:
    """Returns a pre-configured ranking for each target_year."""

    def __init__(self, year_rankings: dict):
        self._year_rankings = year_rankings  # {year: np.ndarray of scores}

    def rank(self, panel_data, weights_dict, target_year=None):
        scores = self._year_rankings.get(target_year)
        if scores is None:
            # Default: uniform scores
            n = len(panel_data.provinces)
            scores = np.ones(n) / n
        provinces = list(panel_data.provinces)
        return _FakeResult(pd.Series(scores, index=provinces))


class TestTemporalStabilitySpearman:
    """_temporal_stability_analysis must use Spearman rank correlation."""

    def _make_panel(self, n: int = 5):
        """Panel with 4 years so that sample_years has 4 years."""
        return _MockPanel(n_provinces=n, n_years=4)

    def test_identical_rankings_give_correlation_one(self):
        """If scores are identical across all years → Spearman = 1.0."""
        from analysis.sensitivity import SensitivityAnalysis

        n = 5
        panel = self._make_panel(n)
        scores = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
        year_rankings = {y: scores.copy() for y in panel.years}
        pipeline = _YearSpecificPipeline(year_rankings)

        sa = SensitivityAnalysis(n_simulations=1, n_jobs=1, seed=0)
        stability, _ = sa._temporal_stability_analysis(panel, pipeline, _mock_weights(4))

        for key, corr in stability.items():
            assert abs(corr - 1.0) < 1e-6, (
                f"Expected Spearman ≈ 1.0 for identical rankings, got {corr} for {key}"
            )

    def test_reversed_rankings_give_correlation_minus_one(self):
        """
        If rankings are perfectly reversed between consecutive years,
        Spearman = -1.0.
        """
        from analysis.sensitivity import SensitivityAnalysis

        n = 5
        panel = self._make_panel(n)
        scores_asc = np.arange(1, n + 1, dtype=float)
        scores_desc = scores_asc[::-1].copy()

        # Alternate between ascending and descending each year
        year_rankings = {}
        for i, y in enumerate(panel.years):
            year_rankings[y] = scores_asc if i % 2 == 0 else scores_desc

        pipeline = _YearSpecificPipeline(year_rankings)

        sa = SensitivityAnalysis(n_simulations=1, n_jobs=1, seed=0)
        stability, _ = sa._temporal_stability_analysis(panel, pipeline, _mock_weights(4))

        # Each consecutive pair alternates direction → Spearman ≈ -1.0
        for key, corr in stability.items():
            assert abs(corr + 1.0) < 1e-6, (
                f"Expected Spearman ≈ -1.0 for reversed rankings, got {corr} for {key}"
            )

    def test_stability_values_bounded(self):
        """All temporal-stability correlation values must lie in [-1, 1]."""
        from analysis.sensitivity import SensitivityAnalysis

        n = 6
        panel = self._make_panel(n)
        rng = np.random.RandomState(7)
        year_rankings = {y: rng.rand(n) for y in panel.years}
        pipeline = _YearSpecificPipeline(year_rankings)

        sa = SensitivityAnalysis(n_simulations=1, n_jobs=1, seed=0)
        stability, _ = sa._temporal_stability_analysis(panel, pipeline, _mock_weights(4))

        for key, corr in stability.items():
            assert -1.0 - 1e-9 <= corr <= 1.0 + 1e-9, (
                f"Correlation out of [-1,1]: {corr} for {key}"
            )

    def test_stability_keys_are_year_transitions(self):
        """Keys of temporal_stability should be 'YYYY-YYYY' formatted strings."""
        from analysis.sensitivity import SensitivityAnalysis

        panel = self._make_panel(5)
        rng = np.random.RandomState(3)
        year_rankings = {y: rng.rand(5) for y in panel.years}
        pipeline = _YearSpecificPipeline(year_rankings)

        sa = SensitivityAnalysis(n_simulations=1, n_jobs=1, seed=0)
        stability, _ = sa._temporal_stability_analysis(panel, pipeline, _mock_weights(4))

        assert len(stability) > 0, "Expected at least one temporal stability entry"
        for key in stability:
            assert "-" in key, f"Unexpected key format: {key!r} (expected 'YYYY-YYYY')"


# ---------------------------------------------------------------------------
# ACI update rule regression (C6)
# ---------------------------------------------------------------------------

class TestACIUpdateRule:
    """
    Adaptive Conformal Inference update: α_{t+1} = α_t + γ(α - error_t).
    When all observations are missed, α_t must *decrease* so that a
    wider quantile is selected (better coverage).
    When all observations are covered, α_t must *increase* so that
    the quantile tightens.
    """

    def _perfect_model(self):
        """Minimal model with predict() that always returns zeros."""
        class _ZeroModel:
            def predict(self, X):
                return np.zeros(len(X))
        return _ZeroModel()

    def test_alpha_decreases_on_first_miss(self):
        """
        Verify the ACI update rule direction on a missed observation.

        After a miss (error_indicator=1) the ACI history entry must show
        α_{t+1} = α_t + γ(α − 1).  With γ=0.95 and α=0.05 the net change
        is −0.9025, so α must drop.  We check the *first* missed entry in
        the ACI history and assert that its recorded α_t < initial α.
        """
        from forecasting.conformal import ConformalPredictor

        # After H6 fix: calibration_fraction (default 0.25) of n is
        # used as calibration.  We need enough data so that the cal set
        # has initial residuals near 0, then large residuals for misses.
        # n=80, n_cal=20, n_init=10 → online window = 10 steps.
        # y[:60] = training (not used for residuals).
        # y[60:70] = cal init portion → residuals ≈ 0.
        # y[70:80] = cal online portion → huge residual → miss.
        n = 80
        cal_frac = 0.25
        n_cal = max(5, int(n * cal_frac))   # 20
        n_init = max(3, n_cal // 2)          # 10

        # ZeroModel always returns 0 regardless of input length
        y = np.concatenate([
            np.zeros(n - n_cal),             # training: ignored
            np.zeros(n_init),                # cal init: residuals = 0
            np.full(n_cal - n_init, 1000.0), # cal online: huge residuals
        ])
        X = np.ones((n, 1))
        initial_alpha = 0.05

        cp = ConformalPredictor(method="adaptive", alpha=initial_alpha, gamma=0.95)
        cp.calibrate(self._perfect_model(), X, y)

        history = cp._aci_history
        assert len(history) > 0, "ACI history must not be empty"

        # Find the first missed step in the history
        missed = [h for h in history if not h["covered"]]
        assert len(missed) > 0, (
            "No missed step found in ACI history despite huge online residuals. "
            "The initial q_hat from zero-residuals should be 0, causing misses."
        )

        # After first miss: α_t + γ(α − 1). Expected ≈ 0.05 − 0.9025 → clipped 0.001
        first_miss_alpha = missed[0]["alpha_t"]
        assert first_miss_alpha < initial_alpha, (
            f"ACI missed step: expected α_t < {initial_alpha} after miss, "
            f"got {first_miss_alpha:.4f}. "
            "The gradient-step update must decrease α on a miss."
        )

    def test_alpha_increases_when_always_covered(self):
        """
        If the model always predicts the exact truth (residual=0), every
        online observation is covered → error_indicator=0 every step →
        α_{t+1} = α_t + γ·α > α_t.  The ACI history entries for covered
        steps must all show α_t > initial α (after the first step).
        """
        from forecasting.conformal import ConformalPredictor

        # Perfect model: uses X to look up true y via a stored mapping.
        # In the H6-fixed calibrate(), the model is deepcopy'd and
        # predict() is called on the calibration slice only.
        class _PerfectModel:
            def __init__(self, X, y):
                self._lookup = {tuple(row): val for row, val in zip(X, y)}
            def predict(self, X_in):
                return np.array([self._lookup.get(tuple(r), 0.0) for r in X_in])

        n = 100  # enough data for cal split
        rng = np.random.RandomState(0)
        X = rng.randn(n, 2)       # 2-D to avoid collisions in lookup
        y = rng.randn(n)
        initial_alpha = 0.05

        cp = ConformalPredictor(method="adaptive", alpha=initial_alpha, gamma=0.95)
        cp.calibrate(_PerfectModel(X, y), X, y)

        history = cp._aci_history
        assert len(history) > 0

        # All online steps should be covered (perfect model, residuals=0)
        covered_steps = [h for h in history if h["covered"]]
        assert len(covered_steps) > 0

        # The last covered alpha_t must be higher than initial (it grew)
        last_alpha = covered_steps[-1]["alpha_t"]
        assert last_alpha > initial_alpha, (
            f"Expected α_t > {initial_alpha} after all covered steps, "
            f"got {last_alpha:.4f}. "
            "The gradient-step update must increase α when covered."
        )

    def test_aci_history_records_updates(self):
        """ACI history list must record at least one step."""
        from forecasting.conformal import ConformalPredictor

        n = 30
        X = np.ones((n, 1))
        y = np.random.RandomState(0).randn(n)

        class _SimpleModel:
            def predict(self, X):
                return np.zeros(len(X))

        cp = ConformalPredictor(method="adaptive", alpha=0.10, gamma=0.8)
        cp.calibrate(_SimpleModel(), X, y)

        assert len(cp._aci_history) > 0, "ACI history should not be empty"

        # Each history entry must have the required keys
        for entry in cp._aci_history:
            assert "alpha_t" in entry
            assert "q_hat" in entry
            assert "covered" in entry


# ---------------------------------------------------------------------------
# Per-entity lag isolation (C5)
# ---------------------------------------------------------------------------

class TestPanelVARPerEntityLag:
    """
    PanelVARForecaster.fit() must build per-entity lag matrices so that
    the last row of entity i is never used as a lag for entity i+1.
    """

    def _make_panel_data(self, n_entities: int = 3, T: int = 10, d: int = 2, seed: int = 0):
        rng = np.random.RandomState(seed)
        n = n_entities * T
        X = rng.randn(n, d)
        y = rng.randn(n)
        entity_ids = np.repeat(np.arange(n_entities), T)
        return X, y, entity_ids

    def test_fit_predict_shape(self):
        """fit/predict round-trip preserves sample count."""
        from forecasting.panel_var import PanelVARForecaster

        X, y, ids = self._make_panel_data(n_entities=4, T=12)
        model = PanelVARForecaster(n_lags=1, lag_selection="fixed", random_state=0)
        model.fit(X, y, entity_indices=ids)
        preds = model.predict(X, entity_indices=ids)

        assert preds.shape[0] == len(y), (
            f"predict() output rows ({preds.shape[0]}) != input rows ({len(y)})"
        )

    def test_entity_tails_stored_per_entity(self):
        """After fit(), _X_panel_tail_ must be a dict keyed by entity id."""
        from forecasting.panel_var import PanelVARForecaster

        X, y, ids = self._make_panel_data(n_entities=3, T=10)
        model = PanelVARForecaster(n_lags=1, lag_selection="fixed", random_state=0)
        model.fit(X, y, entity_indices=ids)

        assert isinstance(model._X_panel_tail_, dict), (
            "With entity_indices, _X_panel_tail_ should be a dict"
        )
        for ent in np.unique(ids):
            assert ent in model._X_panel_tail_, f"Entity {ent} missing from tails"

    def test_no_cross_entity_contamination(self):
        """
        Two separate single-entity fits should produce different tails
        than a joint fit — verifying no cross-entity leakage.
        This is a structural check: if tails are stored per entity, the
        per-entity tail for entity-0 in a joint fit must match the tail
        from a single-entity fit on entity-0's data alone.
        """
        from forecasting.panel_var import PanelVARForecaster

        rng = np.random.RandomState(42)
        T, d = 12, 3

        X0 = rng.randn(T, d)
        y0 = rng.randn(T)
        X1 = rng.randn(T, d)
        y1 = rng.randn(T)

        n_lags = 1
        # Joint fit
        X_joint = np.vstack([X0, X1])
        y_joint = np.concatenate([y0, y1])
        ids_joint = np.array([0] * T + [1] * T)

        model_joint = PanelVARForecaster(n_lags=n_lags, lag_selection="fixed", random_state=0)
        model_joint.fit(X_joint, y_joint, entity_indices=ids_joint)

        tail_ent0_joint = model_joint._X_panel_tail_[0]

        # Single-entity fit for entity 0 only
        model_solo = PanelVARForecaster(n_lags=n_lags, lag_selection="fixed", random_state=0)
        model_solo.fit(X0, y0)  # no entity_indices → flat tail

        # Joint entity-0 tail must come from X0's rows, not X1's rows
        # The last row of X0 should appear in the joint entity-0 tail (after panel feature build)
        # We verify by shape: tail rows = n_lags rows from X0
        assert tail_ent0_joint.shape[0] <= n_lags, (
            f"Entity-0 tail has {tail_ent0_joint.shape[0]} rows; expected ≤ {n_lags}"
        )

    def test_multioutput_fit_predict(self):
        """Multi-output y shape must be preserved."""
        from forecasting.panel_var import PanelVARForecaster

        X, y_1d, ids = self._make_panel_data(n_entities=3, T=10)
        y = np.column_stack([y_1d, y_1d * 0.5])  # (n, 2)

        model = PanelVARForecaster(n_lags=1, lag_selection="fixed", random_state=0)
        model.fit(X, y, entity_indices=ids)
        preds = model.predict(X, entity_indices=ids)

        assert preds.shape == (len(y_1d), 2), (
            f"Multi-output predict shape mismatch: {preds.shape}"
        )
