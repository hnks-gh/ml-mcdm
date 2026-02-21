# -*- coding: utf-8 -*-
"""
ML-MCDM Pipeline Orchestrator
==============================

Seven-phase production pipeline:

  Phase 1  Data Loading
  Phase 2  Weight Calculation       (GTWC + Bayesian Bootstrap)
  Phase 3  Hierarchical Ranking     (12 MCDM + two-stage ER)
  Phase 4  ML Forecasting           (6 models + Super Learner + Conformal)
  Phase 5  Sensitivity Analysis     (Hierarchical multi-level robustness)
  Phase 6  Visualisation            (high-resolution figures)
  Phase 7  Result Export            (CSV / JSON / text report)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import warnings
import time
import json

warnings.filterwarnings('ignore')

# Internal imports (support both package and direct execution)
try:
    from .config import Config, get_default_config
    from .logger import setup_logger, ProgressLogger
    from .data_loader import DataLoader, PanelData
    from .mcdm.traditional import TOPSISCalculator
    from .ranking import HierarchicalRankingPipeline, HierarchicalRankingResult
    from .analysis import SensitivityAnalysis
    from .visualization import PanelVisualizer
    from .output_manager import OutputManager
except ImportError:
    from config import Config, get_default_config
    from logger import setup_logger, ProgressLogger
    from data_loader import DataLoader, PanelData
    from mcdm.traditional import TOPSISCalculator
    from ranking import HierarchicalRankingPipeline, HierarchicalRankingResult
    from analysis import SensitivityAnalysis
    from visualization import PanelVisualizer
    from output_manager import OutputManager


def _to_array(x) -> np.ndarray:
    """Convert Series / list / array to plain ndarray."""
    if x is None:
        return np.array([])
    if hasattr(x, 'values'):
        return x.values
    return np.asarray(x)


# =========================================================================
# Result container
# =========================================================================

@dataclass
class PipelineResult:
    """Container for all pipeline results."""
    # Data
    panel_data: PanelData
    decision_matrix: np.ndarray

    # Weights
    entropy_weights: np.ndarray
    critic_weights: np.ndarray
    merec_weights: np.ndarray
    std_dev_weights: np.ndarray
    fused_weights: np.ndarray
    weight_details: Dict[str, Any]

    # Hierarchical Ranking (IFS + ER)
    ranking_result: HierarchicalRankingResult

    # ML Forecasting (UnifiedForecaster)
    forecast_result: Optional[Any] = None

    # Analysis
    sensitivity_result: Any = None

    # Meta
    execution_time: float = 0.0
    config: Optional[Config] = None

    # ---- convenience accessors ----

    def get_final_ranking_df(self) -> pd.DataFrame:
        """Sorted DataFrame with province, rank, score, Kendall's W."""
        return pd.DataFrame({
            'Province': self.ranking_result.final_ranking.index,
            'ER_Score': self.ranking_result.final_scores.values,
            'ER_Rank': self.ranking_result.final_ranking.values,
        }).sort_values('ER_Rank').reset_index(drop=True)


# =========================================================================
# Pipeline
# =========================================================================

class MLMCDMPipeline:
    """
    Production-grade ML-MCDM pipeline for panel data analysis.

    Integrates
    ----------
    * Robust Global Hybrid Weighting (Entropy + CRITIC + MEREC + SD)
    * 12 MCDM methods per criterion group:
        Traditional : TOPSIS, VIKOR, PROMETHEE, COPRAS, EDAS, SAW
        IFS         : IFS-TOPSIS, IFS-VIKOR, IFS-PROMETHEE, IFS-COPRAS,
                      IFS-EDAS, IFS-SAW
    * Two-stage Evidential Reasoning aggregation (Yang & Xu, 2002)
    * ML Forecasting: 6-model ensemble + Super Learner (optional)
    * Hierarchical sensitivity analysis
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_default_config()

        # Output directories
        self._setup_output_directory()

        # Logger
        debug_file = Path(self.config.output_dir) / 'logs' / 'debug.log'
        self.logger = setup_logger('ml_mcdm', debug_file=debug_file)

        # Output manager (production-ready result persistence)
        self.output_manager = OutputManager(base_output_dir=self.config.output_dir)

        # Visualiser & output helpers
        self.visualizer = PanelVisualizer(
            output_dir=str(Path(self.config.output_dir) / 'figures'),
            dpi=self.config.visualization.dpi,
        )

    # -----------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------

    def _setup_output_directory(self) -> None:
        out = Path(self.config.output_dir)
        for sub in ('figures', 'results', 'reports', 'logs'):
            (out / sub).mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------

    def run(self, data_path: Optional[str] = None) -> PipelineResult:
        """Execute full analysis pipeline and return results."""
        start_time = time.time()

        self.logger.info("=" * 60)
        self.logger.info("ML-MCDM PANEL DATA ANALYSIS PIPELINE")
        self.logger.info("State-of-the-Art: IFS + ER + Forecasting")
        self.logger.info("=" * 60)

        # Phase 1: Data Loading
        with ProgressLogger(self.logger, "Phase 1: Data Loading"):
            panel_data = self._load_data(data_path)

        # Phase 2: Weight Calculation
        with ProgressLogger(self.logger, "Phase 2: Weight Calculation (GTWC)"):
            weights = self._calculate_weights(panel_data)

        # Phase 3: Hierarchical Ranking (12 MCDM methods + ER)
        with ProgressLogger(self.logger, "Phase 3: Hierarchical Ranking (IFS + ER)"):
            ranking_result = self._run_hierarchical_ranking(panel_data, weights)

        # Phase 4: ML Forecasting (6 models + Super Learner + Conformal)
        forecast_result = None
        with ProgressLogger(self.logger, "Phase 4: ML Forecasting (SOTA Ensemble)"):
            try:
                forecast_result = self._run_forecasting(panel_data)
            except Exception as e:
                self.logger.warning(f"Forecasting skipped: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())

        # Phase 5: Sensitivity Analysis
        analysis_results: Dict[str, Any] = {'sensitivity': None}
        with ProgressLogger(self.logger, "Phase 5: Sensitivity Analysis"):
            try:
                analysis_results = self._run_analysis(
                    panel_data, ranking_result, weights, forecast_result)
            except (Exception, KeyboardInterrupt) as e:
                self.logger.warning(f"Sensitivity analysis skipped: {e}")

        execution_time = time.time() - start_time

        # Phase 6: Visualisations
        with ProgressLogger(self.logger, "Phase 6: Generating Figures"):
            try:
                self._generate_all_visualizations(
                    panel_data, weights, ranking_result,
                    analysis_results, forecast_result=forecast_result,
                )
            except Exception as e:
                self.logger.warning(f"Visualisation failed: {e}")

        # Phase 7: Save Results
        with ProgressLogger(self.logger, "Phase 7: Saving Results"):
            try:
                self._save_all_results(
                    panel_data, weights, ranking_result, forecast_result,
                    analysis_results, execution_time,
                )
            except Exception as e:
                self.logger.warning(f"Result saving failed: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())

        self.logger.info("=" * 60)
        self.logger.info(f"Pipeline completed in {execution_time:.2f}s")
        self.logger.info(f"Outputs → {self.config.output_dir}")
        self.logger.info("=" * 60)

        # Build result object
        subcriteria = weights['subcriteria']
        latest_year = max(panel_data.years)
        decision_matrix = panel_data.subcriteria_cross_section[latest_year][subcriteria].values

        return PipelineResult(
            panel_data=panel_data,
            decision_matrix=decision_matrix,
            entropy_weights=weights['entropy'],
            critic_weights=weights['critic'],
            merec_weights=weights['merec'],
            std_dev_weights=weights['std_dev'],
            fused_weights=weights['fused'],
            weight_details=weights.get('details', {}),
            ranking_result=ranking_result,
            forecast_result=forecast_result,
            sensitivity_result=analysis_results.get('sensitivity'),
            execution_time=execution_time,
            config=self.config,
        )

    # -----------------------------------------------------------------
    # Phase 1: Data Loading
    # -----------------------------------------------------------------

    def _load_data(self, data_path: Optional[str]) -> PanelData:
        loader = DataLoader(self.config)
        panel_data = loader.load()
        self.logger.info(
            f"Loaded: {len(panel_data.provinces)} provinces, "
            f"{len(panel_data.years)} years, "
            f"{panel_data.n_subcriteria} subcriteria, "
            f"{panel_data.n_criteria} criteria"
        )
        return panel_data

    # -----------------------------------------------------------------
    # Phase 2: Weight Calculation
    # -----------------------------------------------------------------

    def _calculate_weights(self, panel_data: PanelData) -> Dict[str, Any]:
        """Run GTWC hybrid weighting on the full panel."""
        from .weighting import RobustGlobalWeighting

        subcriteria = panel_data.hierarchy.all_subcriteria
        panel_df = panel_data.subcriteria_long.copy()

        self.logger.info("GTWC weighting pipeline")
        self.logger.info(
            f"  Panel: {len(panel_df)} obs "
            f"({len(panel_data.years)} yr x "
            f"{len(panel_data.provinces)} prov x "
            f"{len(subcriteria)} sub)"
        )

        calc = RobustGlobalWeighting(
            bootstrap_iterations=self.config.weighting.bootstrap_iterations,
            stability_threshold=self.config.weighting.stability_threshold,
            epsilon=self.config.weighting.epsilon,
            seed=self.config.random.seed,
        )

        result = calc.calculate(
            panel_df,
            entity_col='Province',
            time_col='Year',
            criteria_cols=subcriteria,
        )

        # Extract individual method weights
        indiv = result.details["individual_weights"]
        entropy_w = np.array([indiv["entropy"][c] for c in subcriteria])
        critic_w = np.array([indiv["critic"][c] for c in subcriteria])
        merec_w = np.array([indiv["merec"][c] for c in subcriteria])
        stddev_w = np.array([indiv["std_dev"][c] for c in subcriteria])
        fused_w = np.array([result.weights[c] for c in subcriteria])

        self.logger.info(f"  Entropy  : [{entropy_w.min():.4f}, {entropy_w.max():.4f}]")
        self.logger.info(f"  CRITIC   : [{critic_w.min():.4f}, {critic_w.max():.4f}]")
        self.logger.info(f"  MEREC    : [{merec_w.min():.4f}, {merec_w.max():.4f}]")
        self.logger.info(f"  StdDev   : [{stddev_w.min():.4f}, {stddev_w.max():.4f}]")
        self.logger.info(f"  Fused    : [{fused_w.min():.4f}, {fused_w.max():.4f}]")

        # Reliability & stability
        fusion_info = result.details.get("fusion", {})
        rel = fusion_info.get("reliability_scores", {})
        if rel:
            self.logger.info("  Reliability: " +
                             ", ".join(f"{k}={v:.4f}" for k, v in rel.items()))

        stab = result.details.get("stability", {})
        if stab:
            self.logger.info(
                f"  Stability: cos={stab.get('cosine_similarity', 0):.4f}, "
                f"pearson={stab.get('pearson_correlation', 0):.4f}, "
                f"stable={stab.get('is_stable', 'N/A')}"
            )

        boot = result.details.get("bootstrap", {})
        if boot:
            mean_std = np.mean([boot["std_weights"][c] for c in subcriteria])
            self.logger.info(
                f"  Bootstrap ({boot.get('iterations', '?')} iters): "
                f"mean σ_w = {mean_std:.6f}"
            )

        fused_dict = {sc: result.weights[sc] for sc in subcriteria}

        return {
            'entropy': entropy_w,
            'critic': critic_w,
            'merec': merec_w,
            'std_dev': stddev_w,
            'fused': fused_w,
            'fused_dict': fused_dict,
            'subcriteria': subcriteria,
            'details': result.details,
        }

    # -----------------------------------------------------------------
    # Phase 3: Hierarchical Ranking
    # -----------------------------------------------------------------

    def _run_hierarchical_ranking(
        self,
        panel_data: PanelData,
        weights: Dict[str, Any],
    ) -> HierarchicalRankingResult:
        pipeline = HierarchicalRankingPipeline(
            n_grades=self.config.ifs.n_grades,
            method_weight_scheme=self.config.er.method_weight_scheme,
            ifs_spread_factor=self.config.ifs.spread_factor,
        )
        return pipeline.rank(
            panel_data=panel_data,
            subcriteria_weights=weights['fused_dict'],
        )

    # -----------------------------------------------------------------
    # Phase 4: ML Forecasting (State-of-the-Art Ensemble)
    # -----------------------------------------------------------------

    def _run_forecasting(
        self,
        panel_data: PanelData,
    ) -> Any:
        """
        Run state-of-the-art forecasting using UnifiedForecaster.
        
        Architecture:
        - 6 diverse base models (GB, BayesianRidge, QuantileRF, PanelVAR, HierarchBayes, NAM)
        - Super Learner meta-ensemble (automatic optimal weighting)
        - Conformal Prediction (distribution-free 95% intervals)
        """
        if not self.config.forecast.enabled:
            self.logger.info("Forecasting disabled in config")
            return None
        
        from forecasting import UnifiedForecaster
        
        # Determine target year
        target_year = self.config.forecast.target_year
        if target_year is None:
            target_year = max(panel_data.years) + 1
        
        self.logger.info(f"Target year: {target_year}")
        self.logger.info(f"Base models: 6 (GB, BayesianRidge, QuantileRF, PanelVAR, HierarchBayes, NAM)")
        self.logger.info(f"Meta-learner: Super Learner (Ridge)")
        self.logger.info(f"Calibration: Conformal {self.config.forecast.conformal_method} (α={self.config.forecast.conformal_alpha})")
        
        forecaster = UnifiedForecaster(
            conformal_method=self.config.forecast.conformal_method,
            conformal_alpha=self.config.forecast.conformal_alpha,
            cv_folds=self.config.forecast.cv_folds,
            random_state=self.config.forecast.random_state,
            verbose=self.config.forecast.verbose,
        )
        
        result = forecaster.fit_predict(panel_data, target_year=target_year)
        
        # Log results
        if hasattr(result, 'model_weights'):
            self.logger.info("Super Learner weights:")
            for model, weight in sorted(result.model_weights.items(), key=lambda x: x[1], reverse=True):
                self.logger.info(f"  {model:20s}: {weight:.4f}")
        
        if hasattr(result, 'cv_metrics'):
            cv = result.cv_metrics
            self.logger.info(f"CV Performance: R²={cv.get('r2', 0):.4f}, RMSE={cv.get('rmse', 0):.4f}")
        
        return result

    # -----------------------------------------------------------------
    # Phase 5: Enhanced Sensitivity Analysis
    # -----------------------------------------------------------------

    def _run_analysis(
        self,
        panel_data: PanelData,
        ranking_result: HierarchicalRankingResult,
        weights: Dict[str, Any],
        forecast_result: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Run hierarchical sensitivity analysis on the full pipeline.
        
        Tests robustness at multiple levels:
        - Subcriteria weight perturbations
        - Criteria weight perturbations  
        - Temporal stability across years
        - IFS uncertainty sensitivity
        - Forecast robustness (if available)
        """
        self.logger.info("Running hierarchical sensitivity analysis")
        
        # Create ranking pipeline instance for re-running with perturbed weights
        ranking_pipeline = HierarchicalRankingPipeline(
            n_grades=self.config.ifs.n_grades,
            method_weight_scheme=self.config.er.method_weight_scheme,
            ifs_spread_factor=self.config.ifs.spread_factor,
        )
        
        # Run sensitivity analysis
        analyzer = SensitivityAnalysis(
            n_simulations=self.config.validation.n_simulations,
            perturbation_range=0.15,  # ±15% weight perturbation
            ifs_perturbation=0.10,     # ±10% IFS uncertainty
            seed=self.config.random.seed
        )
        
        sens_result = analyzer.analyze_full_pipeline(
            panel_data=panel_data,
            ranking_pipeline=ranking_pipeline,
            weights=weights,
            ranking_result=ranking_result,
            forecast_result=forecast_result
        )
        
        # Validate production-ready hierarchical structure
        self._validate_sensitivity_result(sens_result)
        
        self.logger.info(
            f"Sensitivity analysis: robustness = {sens_result.overall_robustness:.4f}"
        )
        self.logger.info(
            f"  Criteria sensitivity: {len(sens_result.criteria_sensitivity)} criteria analyzed"
        )
        self.logger.info(
            f"  Subcriteria sensitivity: {len(sens_result.subcriteria_sensitivity)} subcriteria analyzed"
        )
        self.logger.info(
            f"  Temporal stability: {len(sens_result.temporal_stability)} year pairs"
        )
        self.logger.info(
            f"  Top-5 stability: {sens_result.top_n_stability.get(5, 0):.1%}"
        )
        self.logger.info(
            f"  IFS sensitivity: μ={sens_result.ifs_membership_sensitivity:.4f}, "
            f"ν={sens_result.ifs_nonmembership_sensitivity:.4f}"
        )
        
        return {'sensitivity': sens_result}
    
    def _validate_sensitivity_result(self, sens_result: Any) -> None:
        """
        Validate that sensitivity result has all required hierarchical attributes.
        Ensures production-ready structure - no legacy fallbacks allowed.
        """
        required_attrs = {
            'subcriteria_sensitivity': 'Subcriteria-level weight sensitivity',
            'criteria_sensitivity': 'Criteria-level weight sensitivity',
            'temporal_stability': 'Temporal stability across years',
            'top_n_stability': 'Top-N ranking stability',
            'rank_stability': 'Province-level rank stability',
            'ifs_membership_sensitivity': 'IFS membership sensitivity',
            'ifs_nonmembership_sensitivity': 'IFS non-membership sensitivity',
            'overall_robustness': 'Overall robustness score',
        }
        
        missing = []
        empty = []
        
        for attr, description in required_attrs.items():
            if not hasattr(sens_result, attr):
                missing.append(f"{attr} ({description})")
            else:
                value = getattr(sens_result, attr)
                # Check if dict/list attributes are empty
                if isinstance(value, (dict, list)) and len(value) == 0:
                    empty.append(f"{attr} ({description})")
        
        if missing:
            raise ValueError(
                f"Sensitivity result missing required attributes:\n  " +
                "\n  ".join(missing) +
                "\n\nEnsure SensitivityAnalysis.analyze_full_pipeline() returns "
                "complete hierarchical structure."
            )
        
        if empty:
            self.logger.warning(
                f"Sensitivity result has empty attributes:\n  " +
                "\n  ".join(empty)
            )

    # -----------------------------------------------------------------
    # Phase 6: Visualisations (Publication-Quality Suite)
    # -----------------------------------------------------------------

    def _generate_all_visualizations(
        self,
        panel_data: PanelData,
        weights: Dict[str, Any],
        ranking_result: HierarchicalRankingResult,
        analysis_results: Dict[str, Any],
        forecast_result: Optional[Any] = None,
    ) -> None:
        """Generate comprehensive publication-quality figure suite.
        
        Each figure is individually wrapped in try/except to ensure one
        failure does not prevent subsequent figures from being generated.
        """
        fig_count = 0
        fig_fail = 0

        def _safe_plot(label: str, fn, *args, **kwargs):
            """Call a plot function safely; log failures and continue."""
            nonlocal fig_count, fig_fail
            try:
                path = fn(*args, **kwargs)
                if path:
                    fig_count += 1
                    return path
            except Exception as exc:
                fig_fail += 1
                self.logger.debug(f"Figure '{label}' skipped: {exc}")
            return None

        scores = _to_array(ranking_result.final_scores)
        ranks = _to_array(ranking_result.final_ranking)
        provinces = panel_data.provinces
        subcriteria = weights['subcriteria']
        sens = analysis_results.get('sensitivity')

        # ── RANKING FIGURES ──────────────────────────────────────
        _safe_plot('fig01', self.visualizer.plot_final_ranking,
                   provinces, scores, ranks,
                   title='Hierarchical ER Final Ranking',
                   save_name='fig01_final_er_ranking.png')

        _safe_plot('fig02', self.visualizer.plot_final_ranking_summary,
                   provinces, scores, ranks,
                   title='ER Ranking Summary (Top & Bottom)',
                   save_name='fig02_ranking_summary.png')

        _safe_plot('fig03', self.visualizer.plot_score_distribution,
                   scores,
                   title='ER Score Distribution',
                   save_name='fig03_score_distribution.png')

        # ── WEIGHT FIGURES ───────────────────────────────────────
        w_dict = {
            'Entropy': weights['entropy'],
            'CRITIC': weights['critic'],
            'MEREC': weights['merec'],
            'Std Dev': weights['std_dev'],
            'Fused': weights['fused'],
        }
        _safe_plot('fig04', self.visualizer.plot_weights_comparison,
                   w_dict, subcriteria,
                   title='Subcriteria Weight Comparison',
                   save_name='fig04_weights_comparison.png')

        _safe_plot('fig05', self.visualizer.plot_weight_radar,
                   w_dict, subcriteria,
                   save_name='fig05_weight_radar.png')

        _safe_plot('fig06', self.visualizer.plot_weight_heatmap,
                   w_dict, subcriteria,
                   save_name='fig06_weight_heatmap.png')

        # ── METHOD AGREEMENT ─────────────────────────────────────
        all_method_ranks = {}
        for crit_id, method_ranks in ranking_result.criterion_method_ranks.items():
            for method, rank_series in method_ranks.items():
                col = f'{crit_id}_{method}'
                all_method_ranks[col] = (
                    rank_series.values if hasattr(rank_series, 'values')
                    else np.asarray(rank_series))

        if all_method_ranks:
            _safe_plot('fig07', self.visualizer.plot_method_agreement_matrix,
                       all_method_ranks,
                       save_name='fig07_method_agreement.png')
            _safe_plot('fig08', self.visualizer.plot_rank_parallel_coordinates,
                       all_method_ranks, provinces,
                       save_name='fig08_rank_parallel.png')

        # ── PER-CRITERION SCORES ─────────────────────────────────
        for ci, (crit_id, method_scores) in enumerate(
                ranking_result.criterion_method_scores.items()):
            _safe_plot(f'fig09_{crit_id}', self.visualizer.plot_criterion_scores,
                       method_scores, crit_id, top_n=20,
                       save_name=f'fig09_{crit_id}_scores.png')

        # ── SENSITIVITY FIGURES ──────────────────────────────────
        if sens is not None:
            if hasattr(sens, 'criteria_sensitivity') and sens.criteria_sensitivity:
                _safe_plot('fig10', self.visualizer.plot_sensitivity_tornado,
                           sens.criteria_sensitivity,
                           save_name='fig10_criteria_sensitivity_tornado.png')
                _safe_plot('fig11', self.visualizer.plot_sensitivity_analysis,
                           sens.criteria_sensitivity,
                           title='Criteria Weight Sensitivity',
                           save_name='fig11_criteria_sensitivity.png')

            if hasattr(sens, 'subcriteria_sensitivity') and sens.subcriteria_sensitivity:
                _safe_plot('fig12', self.visualizer.plot_subcriteria_sensitivity,
                           sens.subcriteria_sensitivity,
                           save_name='fig12_subcriteria_sensitivity.png')

            if hasattr(sens, 'top_n_stability') and sens.top_n_stability:
                _safe_plot('fig13', self.visualizer.plot_top_n_stability,
                           sens.top_n_stability,
                           save_name='fig13_top_n_stability.png')

            if hasattr(sens, 'temporal_stability') and sens.temporal_stability:
                _safe_plot('fig14', self.visualizer.plot_temporal_stability,
                           sens.temporal_stability,
                           save_name='fig14_temporal_stability.png')

            if hasattr(sens, 'rank_stability') and sens.rank_stability:
                _safe_plot('fig15', self.visualizer.plot_rank_volatility,
                           sens.rank_stability,
                           save_name='fig15_rank_volatility.png')

            if hasattr(sens, 'ifs_membership_sensitivity'):
                _safe_plot('fig16', self.visualizer.plot_ifs_sensitivity,
                           sens.ifs_membership_sensitivity,
                           getattr(sens, 'ifs_nonmembership_sensitivity', 0),
                           save_name='fig16_ifs_sensitivity.png')

            if hasattr(sens, 'overall_robustness'):
                _safe_plot('fig17', self.visualizer.plot_robustness_summary,
                           sens.overall_robustness,
                           getattr(sens, 'confidence_level', 0.95),
                           getattr(sens, 'criteria_sensitivity', {}),
                           getattr(sens, 'top_n_stability', {}),
                           getattr(sens, 'ifs_membership_sensitivity', 0),
                           getattr(sens, 'ifs_nonmembership_sensitivity', 0),
                           save_name='fig17_robustness_summary.png')

        # ── ER UNCERTAINTY ───────────────────────────────────────
        _safe_plot('fig18', self.visualizer.plot_er_uncertainty,
                   ranking_result.er_result.uncertainty, provinces,
                   save_name='fig18_er_uncertainty.png')

        # ── FORECAST FIGURES ─────────────────────────────────────
        if forecast_result is not None:
            if hasattr(forecast_result, 'training_info'):
                ti = forecast_result.training_info or {}
                actual = ti.get('y_test')
                predicted = ti.get('y_pred')
                if actual is not None and predicted is not None:
                    ent = ti.get('test_entities')
                    _safe_plot('fig19', self.visualizer.plot_forecast_scatter,
                               np.asarray(actual), np.asarray(predicted),
                               entity_names=ent,
                               save_name='fig19_forecast_scatter.png')
                    _safe_plot('fig20', self.visualizer.plot_forecast_residuals,
                               np.asarray(actual), np.asarray(predicted),
                               save_name='fig20_forecast_residuals.png')

            if hasattr(forecast_result, 'feature_importance') and forecast_result.feature_importance is not None:
                imp = forecast_result.feature_importance
                if hasattr(imp, 'to_dict'):
                    imp_dict = (imp['Importance'].to_dict() if 'Importance' in imp.columns
                                else imp.iloc[:, 0].to_dict())
                else:
                    imp_dict = imp
                _safe_plot('fig21', self.visualizer.plot_feature_importance,
                           imp_dict, save_name='fig21_feature_importance.png')

            if hasattr(forecast_result, 'model_contributions') and forecast_result.model_contributions:
                _safe_plot('fig22', self.visualizer.plot_model_weights_donut,
                           forecast_result.model_contributions,
                           save_name='fig22_model_weights.png')

            if hasattr(forecast_result, 'model_performance') and forecast_result.model_performance:
                _safe_plot('fig23', self.visualizer.plot_model_performance,
                           forecast_result.model_performance,
                           save_name='fig23_model_performance.png')

            if hasattr(forecast_result, 'cross_validation_scores') and forecast_result.cross_validation_scores:
                _safe_plot('fig24', self.visualizer.plot_cv_boxplots,
                           forecast_result.cross_validation_scores,
                           save_name='fig24_cv_boxplots.png')

            if (hasattr(forecast_result, 'prediction_intervals')
                    and forecast_result.prediction_intervals):
                preds = forecast_result.predictions
                intervals = forecast_result.prediction_intervals
                lower = intervals.get('lower')
                upper = intervals.get('upper')
                if lower is not None and upper is not None:
                    _safe_plot('fig25', self.visualizer.plot_prediction_intervals,
                               preds, lower, upper,
                               save_name='fig25_prediction_intervals.png')

            if hasattr(forecast_result, 'predictions') and forecast_result.predictions is not None:
                pred_df = forecast_result.predictions
                if len(pred_df.columns) > 0:
                    pred_col = pred_df.columns[0]
                    pred_scores = pred_df[pred_col].values
                    _safe_plot('fig26', self.visualizer.plot_rank_change_bubble,
                               provinces, scores, pred_scores,
                               prediction_year=getattr(
                                   forecast_result, 'target_year',
                                   max(panel_data.years) + 1),
                               save_name='fig26_rank_change_bubble.png')

        # ── EXECUTIVE DASHBOARD ──────────────────────────────────
        top10_idx = np.argsort(ranks)[:10]
        top10 = [(provinces[i], scores[i]) for i in top10_idx]
        kpis = {
            'Provinces': len(provinces),
            'Years': len(panel_data.years),
            'Subcriteria': panel_data.n_subcriteria,
            'MCDM Methods': len(ranking_result.methods_used),
        }
        rob_text = ''
        if sens and hasattr(sens, 'overall_robustness'):
            rob_text = (
                f'Overall Robustness : {sens.overall_robustness:.4f}\n'
                f'Confidence Level   : {getattr(sens, "confidence_level", 0.95):.0%}\n'
                f'IFS mu Sensitivity : {getattr(sens, "ifs_membership_sensitivity", 0):.4f}\n'
                f'IFS nu Sensitivity : {getattr(sens, "ifs_nonmembership_sensitivity", 0):.4f}\n'
            )
        _safe_plot('fig27', self.visualizer.plot_executive_dashboard, {
            'kpis': kpis,
            'top_10': top10,
            'fused_weights': weights['fused'],
            'subcriteria_names': subcriteria,
            'robustness_text': rob_text,
        }, save_name='fig27_executive_dashboard.png')

        self.logger.info(
            f"Publication-quality figures generated: {fig_count} "
            f"({fig_fail} skipped)"
        )

    # -----------------------------------------------------------------
    # Phase 7: Save Results (Production-Ready using OutputManager)
    # -----------------------------------------------------------------

    def _save_all_results(
        self,
        panel_data: PanelData,
        weights: Dict[str, Any],
        ranking_result: HierarchicalRankingResult,
        forecast_result: Optional[Any],
        analysis_results: Dict[str, Any],
        execution_time: float,
    ) -> None:
        """
        Save all results using OutputManager for production-ready persistence.
        
        Produces a complete set of CSV files, JSON metadata, and a
        comprehensive analysis report.
        """
        results_dir = Path(self.config.output_dir) / 'results'
        
        # 1. Subcriteria weights (Entropy, CRITIC, MEREC, StdDev, Fused + stats)
        subcriteria = weights['subcriteria']
        weights_dict = {
            'entropy': weights['entropy'],
            'critic': weights['critic'],
            'merec': weights['merec'],
            'std_dev': weights['std_dev'],
            'fused': weights['fused'],
        }
        self.output_manager.save_weights(weights_dict, subcriteria)
        self.logger.info("Saved: weights_analysis.csv")
        
        # 2. Final ER rankings (with tiers, z-scores, percentiles, uncertainty)
        self.output_manager.save_rankings(ranking_result, panel_data.provinces)
        self.logger.info("Saved: final_rankings.csv")
        
        # 3. MCDM scores per criterion group (scores + ranks + consensus)
        saved_scores = self.output_manager.save_mcdm_scores_by_criterion(
            ranking_result, panel_data.provinces
        )
        self.logger.info(f"Saved: mcdm_scores_*.csv ({len(saved_scores)} files)")
        
        # 4. Full rank comparison matrix (all methods × all criteria + stats)
        self.output_manager.save_rank_comparison(ranking_result, panel_data.provinces)
        self.logger.info("Saved: mcdm_rank_comparison.csv")
        
        # 5. Criterion weights used by ER Stage 2
        crit_w = ranking_result.criterion_weights_used
        pd.DataFrame([crit_w]).to_csv(
            results_dir / 'criterion_weights.csv', index=False
        )
        self.logger.info("Saved: criterion_weights.csv")
        
        # 6. ER uncertainty per province
        self.output_manager.save_er_uncertainty(ranking_result, panel_data.provinces)
        self.logger.info("Saved: prediction_uncertainty_er.csv")
        
        # 7. Data summary statistics (descriptive + skewness, kurtosis, CV)
        self.output_manager.save_data_summary(panel_data)
        self.logger.info("Saved: data_summary_statistics.csv")
        
        # 8. ML Forecasting results (predictions, intervals, models, features, CV)
        if forecast_result is not None:
            saved_forecast = self.output_manager.save_forecast_results(forecast_result)
            for key, path in saved_forecast.items():
                filename = Path(path).name
                self.logger.info(f"Saved: {filename}")
        
        # 9. Sensitivity Analysis (criteria, subcriteria, stability, IFS, temporal)
        if analysis_results.get('sensitivity'):
            saved_analysis = self.output_manager.save_analysis_results(analysis_results)
            for key, path in saved_analysis.items():
                filename = Path(path).name
                self.logger.info(f"Saved: {filename}")
        
        # 10. Execution summary (JSON)
        summary = {
            'execution_time_seconds': round(execution_time, 2),
            'n_provinces': len(panel_data.provinces),
            'n_years': len(panel_data.years),
            'years': panel_data.years,
            'n_subcriteria': panel_data.n_subcriteria,
            'n_criteria': panel_data.n_criteria,
            'n_mcdm_methods': len(ranking_result.methods_used),
            'methods_used': ranking_result.methods_used,
            'target_year': ranking_result.target_year,
            'kendall_w': ranking_result.kendall_w,
            'aggregation': 'Evidential Reasoning (Yang & Xu, 2002)',
            'fuzzy_extension': 'Intuitionistic Fuzzy Sets (Atanassov, 1986)',
        }
        with open(results_dir / 'execution_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        self.logger.info("Saved: execution_summary.json")
        
        # 11. Config snapshot
        try:
            self.config.save(results_dir / 'config_snapshot.json')
            self.logger.info("Saved: config_snapshot.json")
        except Exception:
            pass
        
        # 12. Comprehensive report (using OutputManager)
        try:
            figure_paths = self.visualizer.get_generated_figures()
            report = self.output_manager.build_comprehensive_report(
                panel_data=panel_data,
                weights=weights,
                ranking_result=ranking_result,
                forecast_result=forecast_result,
                analysis_results=analysis_results,
                execution_time=execution_time,
                figure_paths=figure_paths,
            )
            self.logger.info("Saved: report.txt (comprehensive analysis report)")
        except Exception as e:
            self.logger.warning(f"Report generation failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
        
        total_files = len(self.output_manager.get_saved_files())
        self.logger.info(f"Total output files: {total_files}")
        self.logger.info(f"All results saved to {results_dir}")

# Convenience function
# =========================================================================

def run_pipeline(
    data_path: Optional[str] = None,
    config: Optional[Config] = None,
) -> PipelineResult:
    """Run the full pipeline. Returns PipelineResult."""
    pipeline = MLMCDMPipeline(config)
    return pipeline.run(data_path)
