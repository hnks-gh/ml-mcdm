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
  Phase 6  Visualization            (high-resolution figures)
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
import traceback

warnings.filterwarnings('ignore')

# Internal imports (support both package and direct execution)
try:
    from .config import Config, get_default_config
    from .loggers import setup_logging
    from .loggers.context import PhaseMetrics
    from .data_loader import DataLoader, PanelData
    from .mcdm.traditional import TOPSISCalculator
    from .ranking import HierarchicalRankingPipeline, HierarchicalRankingResult
    from .analysis import SensitivityAnalysis
    from .visualization import VisualizationOrchestrator
    from .output import OutputOrchestrator
except ImportError:
    from config import Config, get_default_config
    from loggers import setup_logging
    from loggers.context import PhaseMetrics
    from data_loader import DataLoader, PanelData
    from mcdm.traditional import TOPSISCalculator
    from ranking import HierarchicalRankingPipeline, HierarchicalRankingResult
    from analysis import SensitivityAnalysis
    from visualization import VisualizationOrchestrator
    from output import OutputOrchestrator


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

        # Logging (console + debug JSON)
        self.console, self.debug_log = setup_logging(self.config.output_dir)
        # stdlib-compatible logger used by submodules
        import logging as _logging
        self.logger = _logging.getLogger('ml_mcdm')

        # Output orchestrator (CSV + Markdown report)
        self.output_orch = OutputOrchestrator(
            base_output_dir=self.config.output_dir,
        )

        # Visualization orchestrator
        self.visualizer = VisualizationOrchestrator(
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

        self.console.banner('ML-MCDM Panel Data Analysis Pipeline',
                            subtitle='IFS + ER + ML Forecasting')

        # Phase 1: Data Loading
        with self.console.phase('Data Loading') as ph:
            panel_data = self._load_data()
            ph.detail(f'{len(panel_data.provinces)} provinces, '
                      f'{len(panel_data.years)} years, '
                      f'{panel_data.n_subcriteria} subcriteria')

        # Phase 2: Weight Calculation
        with self.console.phase('Weight Calculation (GTWC)') as ph:
            weights = self._calculate_weights(panel_data)

        # Phase 3: Hierarchical Ranking (12 MCDM methods + ER)
        with self.console.phase('Hierarchical Ranking (IFS + ER)') as ph:
            ranking_result = self._run_hierarchical_ranking(panel_data, weights)

        # Phase 4: ML Forecasting (6 models + Super Learner + Conformal)
        forecast_result = None
        with self.console.phase('ML Forecasting (SOTA Ensemble)') as ph:
            try:
                forecast_result = self._run_forecasting(panel_data)
            except Exception as e:
                ph.warning(f'Forecasting skipped: {e}')
                self.logger.debug(traceback.format_exc())

        # Phase 5: Sensitivity Analysis & Validation
        analysis_results: Dict[str, Any] = {'sensitivity': None, 'validation': None}
        with self.console.phase('Sensitivity Analysis & Validation') as ph:
            try:
                analysis_results = self._run_analysis(
                    panel_data, ranking_result, weights, forecast_result)
            except Exception as e:
                ph.warning(f'Sensitivity analysis skipped: {e}')

            try:
                analysis_results['validation'] = self._run_validation(
                    panel_data, weights, ranking_result, forecast_result)
            except Exception as e:
                ph.warning(f'Validation skipped: {e}')

        execution_time = time.time() - start_time

        # Phase 6: Visualizations
        with self.console.phase('Generating Figures') as ph:
            try:
                fig_count = self.visualizer.generate_all(
                    panel_data=panel_data,
                    weights=weights,
                    ranking_result=ranking_result,
                    analysis_results=analysis_results,
                    forecast_result=forecast_result,
                )
                ph.metric('Figures', fig_count)
            except Exception as e:
                ph.warning(f'Visualization failed: {e}')

        # Phase 7: Save Results
        with self.console.phase('Saving Results') as ph:
            try:
                figure_paths = self.visualizer.get_generated_figures()
                self.output_orch.save_all(
                    panel_data=panel_data,
                    weights=weights,
                    ranking_result=ranking_result,
                    forecast_result=forecast_result,
                    analysis_results=analysis_results,
                    execution_time=execution_time,
                    figure_paths=figure_paths,
                    config=self.config,
                )
            except Exception as e:
                ph.warning(f'Result saving failed: {e}')
                self.logger.debug(traceback.format_exc())

        self.console.separator()
        self.console.info(f'Pipeline completed in {execution_time:.2f}s')
        self.console.info(f'Outputs → {self.config.output_dir}')
        self.console.separator()

        # Flush debug log
        self.debug_log.close()

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

    def _load_data(self) -> PanelData:
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
        from .weighting import HybridWeightingPipeline

        subcriteria = panel_data.hierarchy.all_subcriteria
        panel_df = panel_data.subcriteria_long.copy()

        self.logger.info("GTWC weighting pipeline")
        self.logger.info(
            f"  Panel: {len(panel_df)} obs "
            f"({len(panel_data.years)} yr x "
            f"{len(panel_data.provinces)} prov x "
            f"{len(subcriteria)} sub)"
        )

        calc = HybridWeightingPipeline(
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
        
        try:
            from .forecasting import UnifiedForecaster
        except ImportError:
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
        if hasattr(result, 'model_contributions'):
            self.logger.info("Super Learner weights:")
            for model, weight in sorted(result.model_contributions.items(), key=lambda x: x[1], reverse=True):
                self.logger.info(f"  {model:20s}: {weight:.4f}")
        
        if hasattr(result, 'cross_validation_scores'):
            import numpy as _np
            cv = result.cross_validation_scores
            all_r2 = [s for scores in cv.values() for s in scores]
            self.logger.info(f"CV Performance: Mean R²={_np.mean(all_r2):.4f} ± {_np.std(all_r2):.4f}")
        
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
    
    def _run_validation(
        self,
        panel_data: PanelData,
        weights: Dict[str, Any],
        ranking_result: HierarchicalRankingResult,
        forecast_result: Optional[Any] = None,
    ) -> Any:
        """Run production validation on the full pipeline."""
        from .analysis import Validator
        
        validator = Validator()
        val_result = validator.validate_full_pipeline(
            panel_data=panel_data,
            weights=weights,
            ranking_result=ranking_result,
            forecast_result=forecast_result,
        )
        
        self.logger.info(
            f"Validation: {'PASSED' if val_result.validation_passed else 'FAILED'} "
            f"(score={val_result.overall_validity:.4f})"
        )
        if val_result.validation_warnings:
            for w in val_result.validation_warnings:
                self.logger.warning(f"  {w}")
        
        return val_result

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
    # Phase 6 & 7 — now handled by VisualizationOrchestrator and
    #               OutputOrchestrator respectively.
    # -----------------------------------------------------------------

# Convenience function
# =========================================================================

def run_pipeline(
    data_path: Optional[str] = None,
    config: Optional[Config] = None,
) -> PipelineResult:
    """Run the full pipeline. Returns PipelineResult."""
    pipeline = MLMCDMPipeline(config)
    return pipeline.run(data_path)
