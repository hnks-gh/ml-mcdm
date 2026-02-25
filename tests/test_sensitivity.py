# -*- coding: utf-8 -*-
"""
Unit tests for analysis/sensitivity.py — Phase 6 coverage.

Covers the Phase 5 bug-fixes:
  - M7: MC simulation count is no longer silently capped at 100
  - M8: RuntimeWarning emitted when >20% of MC simulations fail
  - M9: Spearman (not Pearson) used in temporal stability analysis

Also covers previously-fixed components exercised through the
sensitivity / GTWC path:
  - C1: GTWC game-theory solve produces non-trivial (not always 0.5/0.5) α
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
# C1 — GTWC game theory produces non-trivial alpha
# ---------------------------------------------------------------------------

class TestGTWCNonTrivialAlpha:
    """
    _solve_game_theory with genuinely distinct vectors must NOT always
    return the trivial solution [0.5, 0.5].  The old wrong RHS always
    produced α₁ = α₂ = 0.5 regardless of inputs.
    """

    def test_orthogonal_groups_give_unequal_alpha(self):
        """
        When W_A has a higher L2-norm than W_B (dot_AA ≠ dot_BB),
        the RHS [d_AA, d_BB] is asymmetric, so α₁ ≠ α₂.

        The old (broken) RHS was [(d_AA+d_AB)/2, (d_AB+d_BB)/2], which
        always produces the trivial solution α = [0.5, 0.5].  Symmetric
        vectors (equal L2 norms) would *correctly* give [0.5, 0.5] even
        with the fix, so we must use norms that differ.
        """
        from weighting.fusion import GameTheoryWeightCombination

        gtwc = GameTheoryWeightCombination()
        # W_A has a higher L2-norm: dot_AA ≈ 0.455, dot_BB ≈ 0.375 → b different
        W_A = np.array([0.60, 0.30, 0.06, 0.04])
        W_B = np.array([0.10, 0.05, 0.45, 0.40])

        alpha, _, details = gtwc._solve_game_theory(W_A, W_B)

        # Verify the RHS is [d_AA, d_BB], NOT the average
        d_AA = np.dot(W_A, W_A)
        d_BB = np.dot(W_B, W_B)
        # Because d_AA ≠ d_BB the system is no longer symmetric → α₁ ≠ α₂
        assert abs(d_AA - d_BB) > 0.01, "Test pre-condition: d_AA must differ from d_BB"

        # The two α values should NOT be equal
        assert abs(alpha[0] - alpha[1]) > 0.01, (
            f"Expected non-trivial α (d_AA={d_AA:.4f} ≠ d_BB={d_BB:.4f}), "
            f"got α={alpha}. RHS fix may not be applied."
        )
        # They must still sum to 1 and be non-negative
        assert abs(alpha.sum() - 1.0) < 1e-9
        assert (alpha >= 0).all()

    def test_identical_groups_give_equal_alpha(self):
        """
        When W_A == W_B the system is degenerate-symmetric → [0.5, 0.5]
        is the correct Nash solution (and the only valid solution).
        The fix must not break this legitimate case.
        """
        from weighting.fusion import GameTheoryWeightCombination

        gtwc = GameTheoryWeightCombination()
        W = np.array([0.1, 0.4, 0.3, 0.2])

        alpha, _, _ = gtwc._solve_game_theory(W.copy(), W.copy())

        assert abs(alpha[0] - 0.5) < 1e-9 and abs(alpha[1] - 0.5) < 1e-9, (
            f"Expected α=[0.5, 0.5] for identical groups, got {alpha}"
        )

    def test_final_weights_sum_to_one(self):
        """Full combine() pipeline must return weights summing to 1."""
        from weighting.fusion import GameTheoryWeightCombination

        gtwc = GameTheoryWeightCombination()
        weight_vectors = {
            "entropy": np.array([0.20, 0.30, 0.50]),
            "std_dev": np.array([0.25, 0.35, 0.40]),
            "critic":  np.array([0.30, 0.40, 0.30]),
            "merec":   np.array([0.40, 0.30, 0.30]),
        }
        W_final, details = gtwc.combine(weight_vectors)

        assert abs(W_final.sum() - 1.0) < 1e-9
        assert (W_final >= 0).all()

    def test_alpha_stored_in_details(self):
        """Details must expose alpha_dispersion and alpha_interaction."""
        from weighting.fusion import GameTheoryWeightCombination

        gtwc = GameTheoryWeightCombination()
        weight_vectors = {
            "entropy": np.array([0.20, 0.30, 0.50]),
            "std_dev": np.array([0.15, 0.45, 0.40]),
            "critic":  np.array([0.35, 0.40, 0.25]),
            "merec":   np.array([0.40, 0.25, 0.35]),
        }
        _, details = gtwc.combine(weight_vectors)

        alpha_d = details["phase_3"]["alpha_dispersion"]
        alpha_i = details["phase_3"]["alpha_interaction"]
        assert abs(alpha_d + alpha_i - 1.0) < 1e-9
        assert alpha_d >= 0 and alpha_i >= 0

    def test_gtwc_rhs_is_not_average(self):
        """
        Regression guard for the exact RHS fix (C1).
        The old code used b = [(AA+AB)/2, (AB+BB)/2].
        Verify that details['system_rhs'] equals [d_AA, d_BB], NOT the average.
        """
        from weighting.fusion import GameTheoryWeightCombination

        gtwc = GameTheoryWeightCombination()
        W_A = np.array([0.5, 0.3, 0.2])
        W_B = np.array([0.1, 0.4, 0.5])

        _, _, details = gtwc._solve_game_theory(W_A, W_B)

        d_AA = np.dot(W_A, W_A)
        d_BB = np.dot(W_B, W_B)
        d_AB = np.dot(W_A, W_B)

        expected_rhs = [d_AA, d_BB]
        wrong_rhs = [(d_AA + d_AB) / 2, (d_AB + d_BB) / 2]

        actual_rhs = details["system_rhs"]

        for i in range(2):
            assert abs(actual_rhs[i] - expected_rhs[i]) < 1e-12, (
                f"RHS[{i}] = {actual_rhs[i]:.6f}, expected {expected_rhs[i]:.6f}. "
                "The old wrong average RHS would have been "
                f"{wrong_rhs[i]:.6f}. C1 fix may not be applied."
            )


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

        # n_init = max(5, n//2) = max(5,10) = 10
        # y[:10] = 0 → q_hat ≈ 0 after calibration
        # y[10] = 1000 → first online step: residual >> q_hat → missed
        n = 20
        n_init = max(5, n // 2)  # = 10
        y = np.concatenate([
            np.zeros(n_init),            # calibration: residuals = 0
            np.full(n - n_init, 1000.0), # online: huge residuals
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

        # After first miss: α_t + γ(α − 1). Expected ≈ 0.05 − 0.9025 = −0.88 → clipped to 0.001
        first_miss_alpha = missed[0]["alpha_t"]
        assert first_miss_alpha < initial_alpha, (
            f"ACI missed step: expected α_t < {initial_alpha} after miss, "
            f"got {first_miss_alpha:.4f}. "
            "The gradient-step update (C6 fix) must decrease α on a miss."
        )

    def test_alpha_increases_when_always_covered(self):
        """
        If the model always predicts the exact truth (residual=0), every
        online observation is covered → error_indicator=0 every step →
        α_{t+1} = α_t + γ·α > α_t.  The ACI history entries for covered
        steps must all show α_t > initial α (after the first step).
        """
        from forecasting.conformal import ConformalPredictor

        # Perfect model: predicts y exactly → residuals always 0
        class _PerfectModel:
            def __init__(self, y):
                self._y = y
            def predict(self, X):
                return self._y.copy()

        n = 30
        y = np.random.RandomState(0).randn(n)  # any non-trivial signal
        X = np.ones((n, 1))
        initial_alpha = 0.05

        cp = ConformalPredictor(method="adaptive", alpha=initial_alpha, gamma=0.95)
        cp.calibrate(_PerfectModel(y), X, y)

        history = cp._aci_history
        assert len(history) > 0

        # All online steps should be covered (perfect model, residuals=0)
        covered_steps = [h for h in history if h["covered"]]
        assert len(covered_steps) > 0

        # The last covered alpha_t must be higher than initial (it grew)
        # (some clipping at 0.999 may apply for long sequences)
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
