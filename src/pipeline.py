# -*- coding: utf-8 -*-
"""ML-MCDM pipeline orchestrator."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import warnings
import time
import shutil

warnings.filterwarnings('ignore')

# Internal imports
from .config import Config, get_default_config
from .logger import setup_logger, ProgressLogger
from .data_loader import DataLoader, PanelData
from .output_manager import OutputManager, to_array

from .mcdm import (
    RobustGlobalWeighting,
    TOPSISCalculator, DynamicTOPSIS, VIKORCalculator, MultiPeriodVIKOR,
    PROMETHEECalculator, COPRASCalculator, EDASCalculator,
)

from .mcdm.traditional.saw import SAWCalculator

from .ranking import HierarchicalRankingPipeline, HierarchicalRankingResult

from .ml import (
    RandomForestTS,
    TemporalFeatureEngineer
)

from .analysis import (
    SensitivityAnalysis, CrossValidator, BootstrapValidator
)

from .visualization import PanelVisualizer


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
    
    # Hierarchical Ranking (IFS + ER)
    ranking_result: HierarchicalRankingResult
    
    # ML Results
    rf_feature_importance: Dict[str, float]
    
    # Analysis
    sensitivity_result: Any
    
    # Future Predictions (Next Year - 2025)
    future_predictions: Optional[Dict[str, Any]] = None
    
    # Meta
    execution_time: float = 0.0
    config: Config = None
    
    def get_final_ranking_df(self) -> pd.DataFrame:
        """Get final aggregated ranking as DataFrame."""
        entities = self.panel_data.provinces
        return pd.DataFrame({
            'province': entities,
            'final_rank': self.ranking_result.final_ranking,
            'final_score': self.ranking_result.final_scores,
            'kendall_w': self.ranking_result.kendall_w,
        }).sort_values('final_rank')
    
    def get_future_ranking_df(self) -> Optional[pd.DataFrame]:
        """Get predicted future year ranking as DataFrame."""
        if self.future_predictions is None:
            return None
        
        entities = self.panel_data.provinces
        fp = self.future_predictions
        return pd.DataFrame({
            'province': entities,
            'predicted_topsis_score': fp['topsis_scores'],
            'predicted_topsis_rank': fp['topsis_rankings'],
            'predicted_vikor_q': fp['vikor']['Q'],
            'predicted_vikor_rank': fp['vikor']['rankings'],
            'prediction_year': fp['prediction_year']
        }).sort_values('predicted_topsis_rank')


class MLMCDMPipeline:
    """
    Production-grade ML-MCDM pipeline for panel data analysis.
    
    Integrates:
    - Robust Global Hybrid Weighting (Entropy + CRITIC + MEREC + SD)
    - 12 MCDM methods per criterion:
      * Traditional: TOPSIS, VIKOR, PROMETHEE, COPRAS, EDAS, SAW
      * IFS: IFS-TOPSIS, IFS-VIKOR, IFS-PROMETHEE, IFS-COPRAS, IFS-EDAS, IFS-SAW
    - Two-stage Evidential Reasoning aggregation (Yang & Xu, 2002)
    - ML forecasting (RF, GB, Bayesian, Neural)
    - Sensitivity analysis and validation
    
    Architecture
    ------------
    Stage 1: Within each criterion (8 groups), run 12 MCDM methods.
    Stage 2: Combine criterion-level beliefs via ER with criterion weights.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize pipeline.
        
        Parameters
        ----------
        config : Config, optional
            Pipeline configuration
        """
        self.config = config or get_default_config()
        
        # Setup clean output directory structure
        self._setup_output_directory()
        
        # Setup logging: INFO level console (simple text), DEBUG level to debug.log
        debug_file = Path(self.config.output_dir) / 'logs' / 'debug.log'
        self.logger = setup_logger('ml_mcdm', debug_file=debug_file)
        
        self.visualizer = PanelVisualizer(
            output_dir=str(Path(self.config.output_dir) / 'figures'),
            dpi=300  # High resolution
        )
        self.output_manager = OutputManager(self.config.output_dir)
    
    def _setup_output_directory(self) -> None:
        """Setup clean output directory structure."""
        output_dir = Path(self.config.output_dir)
        
        # Create required directories
        (output_dir / 'figures').mkdir(parents=True, exist_ok=True)
        (output_dir / 'results').mkdir(parents=True, exist_ok=True)
        (output_dir / 'reports').mkdir(parents=True, exist_ok=True)
        (output_dir / 'logs').mkdir(parents=True, exist_ok=True)
    
    def run(self, data_path: Optional[str] = None) -> PipelineResult:
        """
        Execute full analysis pipeline.
        
        Parameters
        ----------
        data_path : str, optional
            Path to panel data CSV
        
        Returns
        -------
        PipelineResult
            Comprehensive results object
        """
        start_time = time.time()
        
        self.logger.info("=" * 60)
        self.logger.info("ML-MCDM PANEL DATA ANALYSIS PIPELINE")
        self.logger.info("IFS + Evidential Reasoning Architecture")
        self.logger.info("=" * 60)
        
        # Phase 1: Data Loading
        with ProgressLogger(self.logger, "Phase 1: Data Loading"):
            panel_data = self._load_data(data_path)
        
        # Phase 2: Weight Calculation
        with ProgressLogger(self.logger, "Phase 2: Weight Calculation"):
            weights = self._calculate_weights(panel_data)
        
        # Phase 3: Hierarchical Ranking (12 MCDM methods + ER)
        with ProgressLogger(self.logger, "Phase 3: Hierarchical Ranking (IFS + ER)"):
            ranking_result = self._run_hierarchical_ranking(panel_data, weights)
        
        # Phase 4: ML Analysis
        with ProgressLogger(self.logger, "Phase 4: ML Analysis"):
            try:
                ml_results = self._run_ml(panel_data, ranking_result)
            except Exception as e:
                self.logger.warning(f"ML Analysis failed: {e}, continuing with empty results")
                ml_results = {
                    'panel_regression': None,
                    'rf_result': None,
                    'rf_importance': {}
                }
        
        # Phase 5: Advanced Analysis
        with ProgressLogger(self.logger, "Phase 5: Advanced Analysis"):
            analysis_results = self._run_analysis(panel_data, ranking_result, weights)
        
        # Phase 6: Future Year Prediction
        with ProgressLogger(self.logger, "Phase 6: Future Year Prediction"):
            try:
                future_predictions = self._run_future_prediction(panel_data, weights)
            except Exception as e:
                self.logger.warning(f"Future prediction failed: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())
                future_predictions = None
        
        execution_time = time.time() - start_time
        
        # Phase 7: Generate All Visualizations
        with ProgressLogger(self.logger, "Phase 7: Generating High-Resolution Figures"):
            try:
                self._generate_all_visualizations(
                    panel_data, weights, ranking_result,
                    analysis_results, ml_results=ml_results,
                    future_predictions=future_predictions,
                )
            except Exception as e:
                self.logger.warning(f"Visualization generation failed: {e}")
                self.logger.info("Continuing with result saving...")
        
        # Phase 8: Save All Results
        with ProgressLogger(self.logger, "Phase 8: Saving All Results"):
            try:
                self._save_all_results(
                    panel_data, weights, ranking_result, ml_results,
                    analysis_results, execution_time,
                    future_predictions=future_predictions,
                )
            except Exception as e:
                self.logger.warning(f"Result saving failed: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())
        
        self.logger.info("=" * 60)
        self.logger.info(f"Pipeline completed in {execution_time:.2f} seconds")
        self.logger.info(f"Outputs saved to: {self.config.output_dir}")
        self.logger.info("=" * 60)
        
        # Compile results
        return PipelineResult(
            panel_data=panel_data,
            decision_matrix=panel_data.subcriteria_cross_section[max(panel_data.years)].values,
            entropy_weights=weights['entropy'],
            critic_weights=weights['critic'],
            merec_weights=weights['merec'],
            std_dev_weights=weights['std_dev'],
            fused_weights=weights['fused'],
            ranking_result=ranking_result,
            rf_feature_importance=ml_results['rf_importance'],
            sensitivity_result=analysis_results['sensitivity'],
            future_predictions=future_predictions,
            execution_time=execution_time,
            config=self.config
        )
    
    def _load_data(self, data_path: Optional[str]) -> PanelData:
        """Load and prepare panel data."""
        loader = DataLoader(self.config)
        panel_data = loader.load()
        
        self.logger.info(f"Panel data loaded: {len(panel_data.provinces)} provinces, "
                        f"{len(panel_data.years)} years, "
                        f"{panel_data.n_subcriteria} subcriteria, "
                        f"{panel_data.n_criteria} criteria")
        
        return panel_data
    
    def _calculate_weights(self, panel_data: PanelData) -> Dict[str, np.ndarray]:
        """
        Calculate weights using the Robust Global Hybrid Weighting pipeline.
        
        Operates on the full panel (all entities × all time periods) with:
        1. Global Min-Max Normalization
        2. PCA Structural Decomposition & Residualization
        3. PCA-Residualized CRITIC Weights
        4. Global Entropy Weights
        5. PCA Loadings-based Weights
        6. KL-Divergence Fusion (geometric mean)
        7. Bayesian Bootstrap validation (Dirichlet-weighted)
        + Split-half stability verification
        """
        subcriteria = panel_data.hierarchy.all_subcriteria
        
        # Prepare full panel DataFrame from subcriteria long format
        panel_df = panel_data.subcriteria_long.copy()
        
        self.logger.info("Using ROBUST GLOBAL HYBRID weighting pipeline")
        self.logger.info(f"  Panel: {len(panel_df)} obs ({len(panel_data.years)} years × "
                        f"{len(panel_data.provinces)} provinces × "
                        f"{len(subcriteria)} subcriteria)")
        
        # Run the unified Robust Global Weighting
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
        
        # Extract individual weight vectors for diagnostics
        indiv = result.details["individual_weights"]
        entropy_weights = np.array([indiv["entropy"][c] for c in subcriteria])
        critic_weights = np.array([indiv["critic"][c] for c in subcriteria])
        merec_weights = np.array([indiv["merec"][c] for c in subcriteria])
        std_dev_weights = np.array([indiv["std_dev"][c] for c in subcriteria])
        fused_weights = np.array([result.weights[c] for c in subcriteria])
        
        # Log results
        self.logger.info(f"  Entropy weights range: [{entropy_weights.min():.4f}, "
                        f"{entropy_weights.max():.4f}]")
        self.logger.info(f"  CRITIC weights range: [{critic_weights.min():.4f}, "
                        f"{critic_weights.max():.4f}]")
        self.logger.info(f"  MEREC weights range: [{merec_weights.min():.4f}, "
                        f"{merec_weights.max():.4f}]")
        self.logger.info(f"  StdDev weights range: [{std_dev_weights.min():.4f}, "
                        f"{std_dev_weights.max():.4f}]")
        self.logger.info(f"  Final (fused+bootstrap) range: [{fused_weights.min():.4f}, "
                        f"{fused_weights.max():.4f}]")
        
        # Log fusion reliability scores
        fusion_info = result.details["fusion"]
        reliability_scores = fusion_info["reliability_scores"]
        self.logger.info(f"  Reliability scores: " +
                        ", ".join([f"{k}={v:.4f}" for k, v in reliability_scores.items()]))
        
        # Log stability
        stability = result.details["stability"]
        self.logger.info(f"  Stability: cosine={stability['cosine_similarity']:.4f}, "
                        f"spearman={stability['spearman_correlation']:.4f}, "
                        f"stable={stability['is_stable']}")
        
        # Log bootstrap stats
        boot = result.details["bootstrap"]
        mean_std = np.mean([boot["std_weights"][c] for c in subcriteria])
        self.logger.info(f"  Bootstrap ({boot['iterations']} iters): "
                        f"mean weight std = {mean_std:.6f}")
        
        # Build subcriteria weights dict for ranking pipeline
        fused_weights_dict = {sc: result.weights[sc] for sc in subcriteria}
        
        return {
            'entropy': entropy_weights,
            'critic': critic_weights,
            'merec': merec_weights,
            'std_dev': std_dev_weights,
            'fused': fused_weights,
            'fused_dict': fused_weights_dict,
            'subcriteria': subcriteria,
        }
    
    def _run_hierarchical_ranking(
        self,
        panel_data: PanelData,
        weights: Dict[str, Any],
    ) -> HierarchicalRankingResult:
        """
        Execute hierarchical ranking: 12 MCDM methods + Evidential Reasoning.
        
        Stage 1: For each criterion (8), run 6 traditional + 6 IFS methods
                 on the subcriteria belonging to that criterion.
        Stage 2: Aggregate criterion-level beliefs via ER.
        """
        ranking_pipeline = HierarchicalRankingPipeline(
            n_grades=5,
            method_weight_scheme='equal',
            ifs_spread_factor=self.config.ifs.spread_factor,
        )
        
        return ranking_pipeline.rank(
            panel_data=panel_data,
            subcriteria_weights=weights['fused_dict'],
        )
    
    def _run_ml(self, panel_data: PanelData, 
                ranking_result: HierarchicalRankingResult) -> Dict[str, Any]:
        """
        Run ML analysis — Random Forest for feature importance.
        
        Uses the ER final scores as target for RF training.
        """
        results = {
            'rf_result': None,
            'rf_importance': {}
        }
        
        subcriteria = panel_data.hierarchy.all_subcriteria
        
        # Prepare features from subcriteria long format
        try:
            features_df = panel_data.subcriteria_long.copy()
        except Exception as e:
            self.logger.warning(f"Feature preparation failed: {e}")
            return results
        
        # Random Forest with Time-Series CV
        try:
            rf = RandomForestTS(
                n_estimators=self.config.random_forest.n_estimators,
                max_depth=self.config.random_forest.max_depth,
                min_samples_split=self.config.random_forest.min_samples_split,
                min_samples_leaf=self.config.random_forest.min_samples_leaf,
                max_features=self.config.random_forest.max_features,
                n_splits=self.config.random_forest.n_splits
            )
            
            # Map ER scores to each province
            df_with_target = features_df.copy()
            er_score_map = ranking_result.final_scores.to_dict()
            df_with_target['target_score'] = df_with_target['Province'].map(er_score_map)
            
            # Filter valid observations
            df_rf = df_with_target.dropna(subset=['target_score'])
            
            unique_years = df_rf['Year'].nunique() if 'Year' in df_rf.columns else 0
            rf_splits = self.config.random_forest.n_splits
            
            if unique_years < rf_splits + 1 and unique_years >= 2:
                rf_splits = max(1, unique_years - 1)
                rf.n_splits = rf_splits
                self.logger.info(f"Adjusted RF splits to {rf_splits} for {unique_years} years")
            
            if len(df_rf) >= 10 and unique_years >= 2:
                rf_result = rf.fit_predict(df_rf, 'target_score', subcriteria)
                results['rf_result'] = rf_result
                results['rf_importance'] = rf_result.feature_importance.to_dict()
                cv_r2_mean = np.mean(rf_result.cv_scores['r2']) if rf_result.cv_scores.get('r2') else 0
                self.logger.info(f"Random Forest CV R²: {cv_r2_mean:.4f}")
            else:
                self.logger.warning(f"Skipping RF: insufficient data ({len(df_rf)} samples, {unique_years} years)")
        except Exception as e:
            self.logger.warning(f"Random Forest failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
        
        return results
    
    def _run_analysis(self, panel_data: PanelData,
                      ranking_result: HierarchicalRankingResult,
                      weights: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run validation analysis (sensitivity to weight perturbation).
        """
        results = {}
        subcriteria = panel_data.hierarchy.all_subcriteria
        latest_year = max(panel_data.years)
        matrix = panel_data.subcriteria_cross_section[latest_year][subcriteria].values
        
        # Sensitivity Analysis - validates ranking robustness
        def topsis_ranking_func(m, w):
            df = pd.DataFrame(m, columns=subcriteria)
            weights_dict = dict(zip(subcriteria, w))
            calc = TOPSISCalculator()
            result = calc.calculate(df, weights_dict)
            return result.ranks.values
        
        sensitivity = SensitivityAnalysis(
            n_simulations=self.config.validation.n_simulations
        )
        
        sens_result = sensitivity.analyze(
            matrix,
            weights['fused'],
            topsis_ranking_func,
            criteria_names=subcriteria,
            alternative_names=panel_data.provinces,
        )
        results['sensitivity'] = sens_result
        
        self.logger.info(f"Sensitivity: Overall robustness = {sens_result.overall_robustness:.4f}")
        
        return results
    
    def _run_future_prediction(self, panel_data: PanelData,
                               weights: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Predict subcriteria values and ranking for the next year.
        
        Uses all historical data to forecast next year's subcriteria values,
        then runs hierarchical ranking on the predicted data.
        """
        from .ml.forecasting import UnifiedForecaster, ForecastMode
        
        current_year = max(panel_data.years)
        prediction_year = current_year + 1
        subcriteria = panel_data.hierarchy.all_subcriteria
        
        self.logger.info(f"Forecasting year {prediction_year} using data from "
                        f"{min(panel_data.years)}-{current_year}")
        
        # Step 1: Forecast subcriteria values
        self.logger.info("Training ML models on all historical data...")
        
        forecaster = UnifiedForecaster(
            mode=ForecastMode.BALANCED,
            include_neural=self.config.neural.enabled,
            include_tree_ensemble=True,
            include_linear=True,
            cv_folds=min(3, len(panel_data.years) - 1),
            random_state=42,
            verbose=False
        )
        
        forecast_result = forecaster.fit_predict(
            panel_data,
            target_year=prediction_year
        )
        
        predicted_components = forecast_result.predictions
        self.logger.info(f"Predicted {len(subcriteria)} subcriteria for "
                        f"{len(panel_data.provinces)} provinces")
        
        # Step 2: Calculate TOPSIS on predicted data for reference
        w = weights['fused']
        weights_dict = {c: w[i] for i, c in enumerate(subcriteria)}
        
        if isinstance(predicted_components, pd.DataFrame):
            predicted_df = predicted_components.copy()
            available_cols = [c for c in subcriteria if c in predicted_df.columns]
            predicted_df = predicted_df[available_cols]
        else:
            predicted_df = pd.DataFrame(
                predicted_components,
                index=panel_data.provinces,
                columns=subcriteria
            )
        
        predicted_df = predicted_df.clip(0, 1)
        
        # TOPSIS on full predicted subcriteria for reference
        topsis = TOPSISCalculator(normalization=self.config.topsis.normalization.value)
        topsis_result = topsis.calculate(predicted_df, weights_dict)
        
        self.logger.info(f"Predicted TOPSIS: Top performer score = {topsis_result.scores.max():.4f}")
        
        # Step 3: VIKOR on predicted data
        vikor = VIKORCalculator(v=self.config.vikor.v)
        vikor_result = vikor.calculate(predicted_df, weights_dict)
        
        self.logger.info(f"Predicted VIKOR: Best alternative Q = {vikor_result.Q.min():.4f}")
        
        # Step 4: Build prediction intervals
        uncertainty_df = forecast_result.uncertainty
        
        future_results = {
            'prediction_year': prediction_year,
            'training_years': list(panel_data.years),
            'predicted_components': predicted_df,
            'prediction_uncertainty': uncertainty_df,
            'topsis_scores': topsis_result.scores.values,
            'topsis_rankings': topsis_result.ranks.values,
            'topsis_result': topsis_result,
            'vikor': {
                'Q': vikor_result.Q,
                'S': vikor_result.S,
                'R': vikor_result.R,
                'rankings': vikor_result.final_ranks
            },
            'vikor_result': vikor_result,
            'model_contributions': forecast_result.model_contributions,
            'forecast_summary': forecast_result.training_info,
        }
        
        # Log top 5 predicted rankings
        predicted_topsis_rankings = topsis_result.ranks.values
        predicted_topsis_scores = topsis_result.scores.values
        top_indices = np.argsort(predicted_topsis_rankings)[:5]
        self.logger.info(f"Predicted {prediction_year} Top 5:")
        for i, idx in enumerate(top_indices):
            entity = panel_data.provinces[idx]
            score = predicted_topsis_scores[idx]
            self.logger.info(f"  {i+1}. {entity}: {score:.4f}")
        
        return future_results

    def _save_all_results(
        self,
        panel_data: PanelData,
        weights: Dict[str, Any],
        ranking_result: HierarchicalRankingResult,
        ml_results: Dict[str, Any],
        analysis_results: Dict[str, Any],
        execution_time: float,
        future_predictions: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save all numerical results to CSV/JSON."""
        results_dir = Path(self.config.output_dir) / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Final rankings
        ranking_df = pd.DataFrame({
            'Province': ranking_result.final_ranking.index,
            'ER_Score': ranking_result.final_scores.values,
            'ER_Rank': ranking_result.final_ranking.values,
            'Kendall_W': ranking_result.kendall_w,
        }).sort_values('ER_Rank')
        ranking_df.to_csv(results_dir / 'final_rankings.csv', index=False)
        self.logger.info("Saved: final_rankings.csv")
        
        # 2. Criterion weights
        crit_w = ranking_result.criterion_weights_used
        pd.DataFrame([crit_w]).to_csv(results_dir / 'criterion_weights.csv', index=False)
        
        # 3. MCDM scores per criterion
        for crit_id, method_scores in ranking_result.criterion_method_scores.items():
            crit_df = pd.DataFrame(method_scores)
            crit_df.to_csv(results_dir / f'mcdm_scores_{crit_id}.csv')
        
        # 4. Uncertainty
        ranking_result.er_result.uncertainty.to_csv(
            results_dir / 'prediction_uncertainty_er.csv'
        )
        
        # 5. Weights analysis
        subcriteria = weights['subcriteria']
        weights_df = pd.DataFrame({
            'subcriteria': subcriteria,
            'entropy': weights['entropy'],
            'critic': weights['critic'],
            'merec': weights['merec'],
            'std_dev': weights['std_dev'],
            'fused': weights['fused'],
        })
        weights_df.to_csv(results_dir / 'weights_analysis.csv', index=False)
        self.logger.info("Saved: weights_analysis.csv")
        
        # 6. ML results
        if ml_results.get('rf_result'):
            rf = ml_results['rf_result']
            rf.feature_importance.to_csv(results_dir / 'feature_importance.csv')
        
        # 7. Sensitivity
        if analysis_results.get('sensitivity'):
            sens = analysis_results['sensitivity']
            if hasattr(sens, 'weight_sensitivity'):
                pd.DataFrame(sens.weight_sensitivity).to_csv(
                    results_dir / 'sensitivity_analysis.csv'
                )
        
        # 8. Execution summary
        import json
        summary = {
            'execution_time_seconds': execution_time,
            'n_provinces': len(panel_data.provinces),
            'n_years': len(panel_data.years),
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
        
        self.logger.info(f"All results saved to {results_dir}")

    def _generate_all_visualizations(
        self,
        panel_data: PanelData,
        weights: Dict[str, Any],
        ranking_result: HierarchicalRankingResult,
        analysis_results: Dict[str, Any],
        ml_results: Optional[Dict[str, Any]] = None,
        future_predictions: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Generate high-resolution visualisation charts."""
        figure_count = 0
        
        try:
            # ===== FINAL ER RANKING =====
            self.visualizer.plot_final_ranking_summary(
                panel_data.provinces,
                to_array(ranking_result.final_scores),
                to_array(ranking_result.final_ranking),
                title='Hierarchical ER Final Ranking',
                save_name='01_er_final_ranking.png'
            )
            figure_count += 1
            self.logger.info("Generated: 01_er_final_ranking.png")
            
            # ===== ER SCORE DISTRIBUTION =====
            self.visualizer.plot_score_distribution(
                to_array(ranking_result.final_scores),
                title='ER Score Distribution',
                save_name='02_er_score_distribution.png'
            )
            figure_count += 1
            self.logger.info("Generated: 02_er_score_distribution.png")
            
            # ===== WEIGHT ANALYSIS =====
            subcriteria = weights['subcriteria']
            self.visualizer.plot_weights_comparison(
                weights, subcriteria,
                title='Subcriteria Weights Comparison',
                save_name='03_weights_comparison.png'
            )
            figure_count += 1
            self.logger.info("Generated: 03_weights_comparison.png")
            
            # ===== SENSITIVITY =====
            if analysis_results.get('sensitivity'):
                self.visualizer.plot_sensitivity_analysis(
                    analysis_results['sensitivity'].weight_sensitivity,
                    title='Weight Sensitivity Analysis',
                    save_name='04_sensitivity_analysis.png'
                )
                figure_count += 1
                self.logger.info("Generated: 04_sensitivity_analysis.png")
            
            # ===== ML FEATURE IMPORTANCE =====
            if ml_results and ml_results.get('rf_importance'):
                self.visualizer.plot_feature_importance_single(
                    ml_results['rf_importance'],
                    title='Random Forest Feature Importance',
                    save_name='05_feature_importance.png'
                )
                figure_count += 1
                self.logger.info("Generated: 05_feature_importance.png")
            
            # ===== FUTURE PREDICTIONS =====
            if future_predictions:
                prediction_year = future_predictions.get('prediction_year', 2025)
                self.visualizer.plot_future_predictions(
                    panel_data.provinces,
                    to_array(ranking_result.final_scores),
                    to_array(future_predictions['topsis_scores']),
                    prediction_year,
                    title=f'Future Predictions ({prediction_year})',
                    save_name='06_future_predictions.png'
                )
                figure_count += 1
                self.logger.info("Generated: 06_future_predictions.png")
            
            self.logger.info(f"Total figures generated: {figure_count}")
            
        except Exception as e:
            self.logger.warning(f"Visualization error: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())


def run_pipeline(data_path: Optional[str] = None,
                config: Optional[Config] = None) -> PipelineResult:
    """
    Convenience function to run the full pipeline.
    
    Parameters
    ----------
    data_path : str, optional
        Path to panel data CSV
    config : Config, optional
        Pipeline configuration
    
    Returns
    -------
    PipelineResult
        Comprehensive results
    """
    pipeline = MLMCDMPipeline(config)
    return pipeline.run(data_path)
