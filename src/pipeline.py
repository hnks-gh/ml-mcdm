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
from .data_loader import PanelDataLoader, PanelData, TemporalFeatureEngineer
from .output_manager import OutputManager, to_array

from .mcdm import (
    EntropyWeightCalculator, CRITICWeightCalculator, EnsembleWeightCalculator,
    PCAWeightCalculator,
    TOPSISCalculator, DynamicTOPSIS, VIKORCalculator, MultiPeriodVIKOR,
    PROMETHEECalculator, COPRASCalculator, EDASCalculator,
    FuzzyTOPSIS, FuzzyVIKOR, FuzzyPROMETHEE, FuzzyCOPRAS, FuzzyEDAS
)

from .ml import (
    RandomForestTS  # For feature importance and time-series validation
)

from .ensemble import (
    StackingEnsemble, BordaCount, CopelandMethod, KemenyYoung, aggregate_rankings
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
    pca_weights: np.ndarray
    ensemble_weights: np.ndarray
    
    # MCDM Results (Current Year - 2024)
    topsis_scores: np.ndarray
    topsis_rankings: np.ndarray
    dynamic_topsis_scores: np.ndarray
    vikor_results: Dict
    fuzzy_topsis_scores: np.ndarray
    
    # ML Results
    rf_feature_importance: Dict[str, float]
    
    # Ensemble
    stacking_result: Any
    aggregated_ranking: Any
    
    # Analysis
    sensitivity_result: Any
    
    # Future Predictions (Next Year - 2025)
    future_predictions: Optional[Dict[str, Any]] = None
    
    # Meta
    execution_time: float = 0.0
    config: Config = None
    
    def get_final_ranking_df(self) -> pd.DataFrame:
        """Get final aggregated ranking as DataFrame."""
        entities = self.panel_data.entities
        return pd.DataFrame({
            'province': entities,
            'final_rank': self.aggregated_ranking.final_ranking,
            'final_score': self.aggregated_ranking.final_scores,
            'topsis_rank': self.topsis_rankings,
            'kendall_w': self.aggregated_ranking.kendall_w
        }).sort_values('final_rank')
    
    def get_future_ranking_df(self) -> Optional[pd.DataFrame]:
        """Get predicted future year ranking as DataFrame."""
        if self.future_predictions is None:
            return None
        
        entities = self.panel_data.entities
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
    Production-grade ML-MCDM pipeline for panel data analysis with multiple methods.
    
    Supports 10 MCDM methods (5 traditional + 5 fuzzy).
    
    Integrates:
    - Multiple weighting methods (Entropy, CRITIC, Ensemble)
    - 10 MCDM methods:
      * Traditional: TOPSIS, VIKOR, PROMETHEE, COPRAS, EDAS
      * Fuzzy: Fuzzy TOPSIS, Fuzzy VIKOR, Fuzzy PROMETHEE, Fuzzy COPRAS, Fuzzy EDAS
    - Advanced ML forecasting (Unified ensemble with RF, GB, Bayesian; Neural disabled by default)
    - Ensemble aggregation (Stacking, Borda Count, Copeland, Kemeny)
    - Comprehensive analysis (Convergence, Sensitivity, Validation)
    
    Outputs:
    - Professional high-resolution figures (300 DPI)
    - Complete numerical results in CSV format
    - Comprehensive analysis reports
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
        self.logger.info("=" * 60)
        
        # Phase 1: Data Loading
        with ProgressLogger(self.logger, "Phase 1: Data Loading"):
            panel_data = self._load_data(data_path)
        
        # Phase 2: Weight Calculation
        with ProgressLogger(self.logger, "Phase 2: Weight Calculation"):
            weights = self._calculate_weights(panel_data)
        
        # Phase 3: MCDM Analysis
        with ProgressLogger(self.logger, "Phase 3: MCDM Analysis"):
            mcdm_results = self._run_mcdm(panel_data, weights)
        
        # Phase 4: ML Analysis
        with ProgressLogger(self.logger, "Phase 4: ML Analysis"):
            try:
                ml_results = self._run_ml(panel_data, mcdm_results)
            except Exception as e:
                self.logger.warning(f"ML Analysis failed: {e}, continuing with empty results")
                ml_results = {
                    'panel_regression': None,
                    'rf_result': None,
                    'rf_importance': {}
                }
        
        # Phase 5: Ensemble Integration
        with ProgressLogger(self.logger, "Phase 5: Ensemble Integration"):
            ensemble_results = self._run_ensemble(mcdm_results, ml_results, panel_data)
        
        # Phase 6: Advanced Analysis
        with ProgressLogger(self.logger, "Phase 6: Advanced Analysis"):
            analysis_results = self._run_analysis(panel_data, mcdm_results, weights)
        
        # Phase 6.5: Future Year Prediction (2025)
        with ProgressLogger(self.logger, "Phase 6.5: Future Year Prediction"):
            try:
                future_predictions = self._run_future_prediction(panel_data, weights)
            except Exception as e:
                self.logger.warning(f"Future prediction failed: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())
                future_predictions = None
        
        execution_time = time.time() - start_time
        
        # Phase 7: Generate All Visualizations (high-resolution individual charts)
        with ProgressLogger(self.logger, "Phase 7: Generating High-Resolution Figures"):
            try:
                self._generate_all_visualizations(panel_data, weights, mcdm_results, 
                                                  ensemble_results, analysis_results,
                                                  ml_results=ml_results,
                                                  future_predictions=future_predictions)
            except Exception as e:
                self.logger.warning(f"Visualization generation failed: {e}")
                self.logger.info("Continuing with result saving...")
        
        # Get list of generated figures
        figure_paths = self.visualizer.get_generated_figures() if hasattr(self.visualizer, 'get_generated_figures') else []
        
        # Phase 8: Save All Results (comprehensive CSV/JSON outputs)
        with ProgressLogger(self.logger, "Phase 8: Saving All Results"):
            saved_files = self.output_manager.save_all_results(
                panel_data, weights, mcdm_results, ml_results,
                ensemble_results, analysis_results, execution_time,
                future_predictions=future_predictions,
                config=self.config,
                figure_paths=figure_paths
            )
            self.logger.info(f"Saved {len(saved_files['files'])} result categories")
        
        self.logger.info("=" * 60)
        self.logger.info(f"Pipeline completed in {execution_time:.2f} seconds")
        self.logger.info(f"Outputs saved to: {self.config.output_dir}")
        self.logger.info("=" * 60)
        
        # Compile results
        return PipelineResult(
            panel_data=panel_data,
            decision_matrix=panel_data.cross_section.values,
            entropy_weights=weights['entropy'],
            critic_weights=weights['critic'],
            pca_weights=weights['pca'],
            ensemble_weights=weights['ensemble'],
            topsis_scores=mcdm_results['topsis_scores'],
            topsis_rankings=mcdm_results['topsis_rankings'],
            dynamic_topsis_scores=mcdm_results['dynamic_topsis_scores'],
            vikor_results=mcdm_results['vikor'],
            fuzzy_topsis_scores=mcdm_results['fuzzy_scores'],
            rf_feature_importance=ml_results['rf_importance'],
            stacking_result=ensemble_results['stacking'],
            aggregated_ranking=ensemble_results['aggregated'],
            sensitivity_result=analysis_results['sensitivity'],
            future_predictions=future_predictions,
            execution_time=execution_time,
            config=self.config
        )
    
    def _load_data(self, data_path: Optional[str]) -> PanelData:
        """Load and prepare panel data."""
        loader = PanelDataLoader(self.config)
        
        if data_path:
            panel_data = loader.load(data_path)
        else:
            # Auto-generate if no data provided
            self.logger.info("No data path provided, generating synthetic panel data")
            panel_data = loader.generate_synthetic(
                n_provinces=self.config.panel.n_provinces,
                n_years=self.config.panel.n_years,
                n_components=self.config.panel.n_components
            )
        
        self.logger.info(f"Panel data loaded: {len(panel_data.provinces)} entities, "
                        f"{len(panel_data.years)} periods, "
                        f"{len(panel_data.components)} components")
        
        return panel_data
    
    def _calculate_weights(self, panel_data: PanelData) -> Dict[str, np.ndarray]:
        """Calculate weights using multiple methods (Entropy, CRITIC, PCA + Ensemble)."""
        # Get latest cross-section as DataFrame
        latest_year = max(panel_data.years)
        df = panel_data.cross_section[latest_year][panel_data.components]
        
        # Entropy weights
        entropy_calc = EntropyWeightCalculator()
        entropy_result = entropy_calc.calculate(df)
        
        # CRITIC weights
        critic_calc = CRITICWeightCalculator()
        critic_result = critic_calc.calculate(df)
        
        # PCA weights
        pca_calc = PCAWeightCalculator(
            variance_threshold=self.config.weighting.pca_variance_threshold
        )
        pca_result = pca_calc.calculate(df)
        
        # Ensemble weights (configurable strategy, defaults to integrated_hybrid)
        ensemble_calc = EnsembleWeightCalculator(
            methods=self.config.weighting.methods,
            aggregation=self.config.weighting.ensemble_strategy,
            pca_variance_threshold=self.config.weighting.pca_variance_threshold,
            bootstrap_samples=self.config.weighting.bootstrap_samples
        )
        ensemble_result = ensemble_calc.calculate(df)
        
        # Convert weight dicts to arrays
        components = panel_data.components
        entropy_weights = np.array([entropy_result.weights[c] for c in components])
        critic_weights = np.array([critic_result.weights[c] for c in components])
        pca_weights = np.array([pca_result.weights[c] for c in components])
        ensemble_weights = np.array([ensemble_result.weights[c] for c in components])
        
        self.logger.info(f"Entropy weights range: [{entropy_weights.min():.4f}, "
                        f"{entropy_weights.max():.4f}]")
        self.logger.info(f"CRITIC weights range: [{critic_weights.min():.4f}, "
                        f"{critic_weights.max():.4f}]")
        self.logger.info(f"PCA weights range: [{pca_weights.min():.4f}, "
                        f"{pca_weights.max():.4f}]")
        self.logger.info(f"Ensemble strategy: {self.config.weighting.ensemble_strategy}")
        
        # Log PCA details
        n_retained = pca_result.details.get('n_components_retained', '?')
        var_explained = pca_result.details.get('total_variance_explained', 0)
        self.logger.info(f"PCA: {n_retained} components retained, "
                        f"{var_explained:.1%} variance explained")
        
        return {
            'entropy': entropy_weights,
            'critic': critic_weights,
            'pca': pca_weights,
            'ensemble': ensemble_weights
        }
    
    def _run_mcdm(self, panel_data: PanelData, 
                  weights: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Run ALL 10 MCDM analysis methods.
        
        Traditional Methods (5):
            - TOPSIS: Distance to ideal solution
            - Dynamic TOPSIS: Panel-aware temporal TOPSIS
            - VIKOR: Compromise ranking
            - PROMETHEE: Outranking with preference flows
            - COPRAS: Complex proportional assessment
            - EDAS: Distance from average solution
        
        Fuzzy Methods (5):
            - Fuzzy TOPSIS, VIKOR, PROMETHEE, COPRAS, EDAS
            All use Triangular Fuzzy Numbers with temporal variance
        """
        results = {}
        
        # Use ensemble weights for main analysis
        w = weights['ensemble']
        # Get latest cross-section as DataFrame
        latest_year = max(panel_data.years)
        df = panel_data.cross_section[latest_year][panel_data.components]
        
        # Convert numpy weights to dict format
        weights_dict = {c: w[i] for i, c in enumerate(panel_data.components)}
        
        # ========== TRADITIONAL MCDM METHODS ==========
        self.logger.info("Running Traditional MCDM Methods...")
        
        # 1. Static TOPSIS
        topsis = TOPSISCalculator(normalization=self.config.topsis.normalization.value)
        topsis_result = topsis.calculate(df, weights_dict)
        results['topsis_scores'] = topsis_result.scores.values
        results['topsis_rankings'] = topsis_result.ranks.values
        results['topsis_result'] = topsis_result
        self.logger.info(f"  TOPSIS: Top performer score = {topsis_result.scores.max():.4f}")
        
        # 2. Dynamic TOPSIS (panel)
        dynamic_topsis = DynamicTOPSIS(
            temporal_discount=self.config.topsis.temporal_discount,
            trajectory_weight=self.config.topsis.trajectory_weight,
            stability_weight=self.config.topsis.stability_weight,
            normalization=self.config.topsis.normalization.value
        )
        dynamic_result = dynamic_topsis.calculate(panel_data, weights_dict)
        results['dynamic_topsis_scores'] = dynamic_result.scores.values
        results['dynamic_topsis'] = dynamic_result
        self.logger.info(f"  Dynamic TOPSIS: Top performer = {dynamic_result.scores.max():.4f}")
        
        # 3. VIKOR
        vikor = VIKORCalculator(v=self.config.vikor.v)
        vikor_result = vikor.calculate(df, weights_dict)
        results['vikor'] = {
            'Q': vikor_result.Q,
            'S': vikor_result.S,
            'R': vikor_result.R,
            'rankings': vikor_result.final_ranks
        }
        results['vikor_result'] = vikor_result
        self.logger.info(f"  VIKOR: Best alternative Q = {vikor_result.Q.min():.4f}")
        
        # 4. PROMETHEE
        promethee = PROMETHEECalculator(
            preference_function="vshape",
            preference_threshold=0.3,
            indifference_threshold=0.1
        )
        promethee_result = promethee.calculate(df, weights_dict)
        results['promethee'] = {
            'phi_positive': promethee_result.phi_positive,
            'phi_negative': promethee_result.phi_negative,
            'phi_net': promethee_result.phi_net,
            'rankings': promethee_result.ranks_promethee_ii
        }
        results['promethee_result'] = promethee_result
        self.logger.info(f"  PROMETHEE: Best phi_net = {promethee_result.phi_net.max():.4f}")
        
        # 5. COPRAS
        copras = COPRASCalculator()
        copras_result = copras.calculate(df, weights_dict)
        results['copras'] = {
            'Q': copras_result.Q,
            'S_plus': copras_result.S_plus,
            'S_minus': copras_result.S_minus,
            'utility_degree': copras_result.utility_degree,
            'rankings': copras_result.ranks
        }
        results['copras_result'] = copras_result
        self.logger.info(f"  COPRAS: Top utility = {copras_result.utility_degree.max():.1f}%")
        
        # 6. EDAS
        edas = EDASCalculator()
        edas_result = edas.calculate(df, weights_dict)
        results['edas'] = {
            'AS': edas_result.AS,
            'SP': edas_result.SP,
            'SN': edas_result.SN,
            'rankings': edas_result.ranks
        }
        results['edas_result'] = edas_result
        self.logger.info(f"  EDAS: Best AS = {edas_result.AS.max():.4f}")
        
        # ========== FUZZY MCDM METHODS ==========
        self.logger.info("Running Fuzzy MCDM Methods (with temporal uncertainty)...")
        
        # 7. Fuzzy TOPSIS
        fuzzy_topsis = FuzzyTOPSIS()
        fuzzy_topsis_result = fuzzy_topsis.calculate_from_panel(panel_data, weights_dict)
        results['fuzzy_topsis'] = {
            'scores': fuzzy_topsis_result.scores,
            'd_positive': fuzzy_topsis_result.d_positive,
            'd_negative': fuzzy_topsis_result.d_negative,
            'rankings': fuzzy_topsis_result.ranks
        }
        results['fuzzy_topsis_result'] = fuzzy_topsis_result
        # Keep backward compatibility
        results['fuzzy_scores'] = fuzzy_topsis_result.scores
        results['fuzzy_result'] = fuzzy_topsis_result
        self.logger.info(f"  Fuzzy TOPSIS: Top performer = {fuzzy_topsis_result.scores.max():.4f}")
        
        # 8. Fuzzy VIKOR
        fuzzy_vikor = FuzzyVIKOR(v=self.config.vikor.v)
        fuzzy_vikor_result = fuzzy_vikor.calculate_from_panel(panel_data, weights_dict)
        results['fuzzy_vikor'] = {
            'Q': fuzzy_vikor_result.Q,
            'S': fuzzy_vikor_result.S,
            'R': fuzzy_vikor_result.R,
            'rankings': fuzzy_vikor_result.ranks_Q
        }
        results['fuzzy_vikor_result'] = fuzzy_vikor_result
        self.logger.info(f"  Fuzzy VIKOR: Best Q = {fuzzy_vikor_result.Q.min():.4f}")
        
        # 9. Fuzzy PROMETHEE
        fuzzy_promethee = FuzzyPROMETHEE(preference_function='vshape')
        fuzzy_promethee_result = fuzzy_promethee.calculate_from_panel(panel_data, weights_dict)
        results['fuzzy_promethee'] = {
            'phi_positive': fuzzy_promethee_result.phi_positive,
            'phi_negative': fuzzy_promethee_result.phi_negative,
            'phi_net': fuzzy_promethee_result.phi_net,
            'rankings': fuzzy_promethee_result.ranks
        }
        results['fuzzy_promethee_result'] = fuzzy_promethee_result
        self.logger.info(f"  Fuzzy PROMETHEE: Best phi_net = {fuzzy_promethee_result.phi_net.max():.4f}")
        
        # 10. Fuzzy COPRAS
        fuzzy_copras = FuzzyCOPRAS()
        fuzzy_copras_result = fuzzy_copras.calculate_from_panel(panel_data, weights_dict)
        results['fuzzy_copras'] = {
            'Q': fuzzy_copras_result.Q,
            'utility_degree': fuzzy_copras_result.utility_degree,
            'rankings': fuzzy_copras_result.ranks
        }
        results['fuzzy_copras_result'] = fuzzy_copras_result
        self.logger.info(f"  Fuzzy COPRAS: Top utility = {fuzzy_copras_result.utility_degree.max():.1f}%")
        
        # 11. Fuzzy EDAS
        fuzzy_edas = FuzzyEDAS()
        fuzzy_edas_result = fuzzy_edas.calculate_from_panel(panel_data, weights_dict)
        results['fuzzy_edas'] = {
            'AS': fuzzy_edas_result.AS,
            'rankings': fuzzy_edas_result.ranks
        }
        results['fuzzy_edas_result'] = fuzzy_edas_result
        self.logger.info(f"  Fuzzy EDAS: Best AS = {fuzzy_edas_result.AS.max():.4f}")
        
        self.logger.info("Completed all 10 MCDM methods (5 traditional + 5 fuzzy)")
        
        return results
    
    def _run_ml(self, panel_data: PanelData, 
                mcdm_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run appropriate ML analysis methods.
        
        For Objective 1 (Current Year Index): Random Forest provides feature importance
        to understand which components drive sustainability rankings.
        
        For Objective 2 (Forecasting): The forecasting is handled separately in
        _run_future_prediction using AdvancedMLForecaster.
        """
        results = {
            'rf_result': None,
            'rf_importance': {}
        }
        
        # Prepare features with fallback
        try:
            feature_eng = TemporalFeatureEngineer()
            features_df = feature_eng.create_all_features(panel_data)
        except Exception as e:
            self.logger.warning(f"Feature engineering failed: {e}, using raw panel data")
            features_df = panel_data.long.copy()
        
        # Random Forest with Time-Series CV
        # Purpose: Feature importance analysis to identify which components
        # most strongly influence sustainability rankings
        try:
            rf = RandomForestTS(
                n_estimators=self.config.random_forest.n_estimators,
                max_depth=self.config.random_forest.max_depth,
                min_samples_split=self.config.random_forest.min_samples_split,
                min_samples_leaf=self.config.random_forest.min_samples_leaf,
                max_features=self.config.random_forest.max_features,
                n_splits=self.config.random_forest.n_splits
            )
            
            # Create target column in long-form data
            # Map TOPSIS scores to each province-year observation
            df_with_target = features_df.copy()
            topsis_map = dict(zip(panel_data.get_latest().index, 
                                  mcdm_results['topsis_scores']))
            df_with_target['topsis_score'] = df_with_target['Province'].map(topsis_map)
            
            # Filter valid observations
            df_rf = df_with_target.dropna(subset=['topsis_score'])
            
            # Check actual unique years in the filtered data
            unique_years = df_rf['Year'].nunique() if 'Year' in df_rf.columns else 0
            rf_splits = self.config.random_forest.n_splits
            
            # Adjust splits if not enough years
            if unique_years < rf_splits + 1 and unique_years >= 2:
                rf_splits = max(1, unique_years - 1)
                rf.n_splits = rf_splits
                self.logger.info(f"Adjusted RF splits to {rf_splits} for {unique_years} years")
            
            if len(df_rf) >= 10 and unique_years >= 2:
                rf_result = rf.fit_predict(df_rf, 'topsis_score', panel_data.components)
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
    
    def _run_ensemble(self, mcdm_results: Dict[str, Any],
                      ml_results: Dict[str, Any],
                      panel_data: PanelData) -> Dict[str, Any]:
        """
        Run ensemble methods using ALL 10 MCDM methods.
        
        Includes:
        - Stacking Ensemble (meta-learner combining scores)
        - Rank Aggregation with Borda, Copeland, and Kemeny-Young
        """
        results = {}
        n_alternatives = len(panel_data.entities)
        
        # Helper to convert scores to rankings (handle both Series and arrays)
        def scores_to_ranks(scores, higher_is_better=True):
            arr = to_array(scores)
            if higher_is_better:
                return n_alternatives - np.argsort(np.argsort(arr))
            else:
                return np.argsort(np.argsort(arr)) + 1
        
        # ========== 1. Stacking Ensemble for score prediction ==========
        # Use all available MCDM scores
        base_predictions = {
            'TOPSIS': to_array(mcdm_results['topsis_scores']),
            'Dynamic_TOPSIS': to_array(mcdm_results['dynamic_topsis_scores']),
            'VIKOR_Q': 1 - to_array(mcdm_results['vikor']['Q']),  # Invert so higher is better
            'PROMETHEE': to_array(mcdm_results['promethee']['phi_net']),
            'COPRAS': to_array(mcdm_results['copras']['Q']),
            'EDAS': to_array(mcdm_results['edas']['AS']),
            'Fuzzy_TOPSIS': to_array(mcdm_results['fuzzy_topsis']['scores']),
            'Fuzzy_VIKOR': 1 - to_array(mcdm_results['fuzzy_vikor']['Q']),  # Invert
            'Fuzzy_PROMETHEE': to_array(mcdm_results['fuzzy_promethee']['phi_net']),
            'Fuzzy_COPRAS': to_array(mcdm_results['fuzzy_copras']['Q']),
            'Fuzzy_EDAS': to_array(mcdm_results['fuzzy_edas']['AS'])
        }
        
        # Use TOPSIS as pseudo-target for meta-learning
        target = to_array(mcdm_results['topsis_scores'])
        
        stacking = StackingEnsemble(
            meta_learner=self.config.ensemble.meta_learner,
            alpha=self.config.ensemble.alpha
        )
        stacking_result = stacking.fit_predict(base_predictions, target)
        results['stacking'] = stacking_result
        
        self.logger.info(f"Stacking Meta-Model R²: {stacking_result.meta_model_r2:.4f}")
        
        # ========== 2. Rank Aggregation using ALL 10 MCDM methods ==========
        # Collect rankings from all methods
        rankings = {
            # Traditional methods
            'TOPSIS': to_array(mcdm_results['topsis_rankings']),
            'Dynamic_TOPSIS': scores_to_ranks(mcdm_results['dynamic_topsis_scores'], True),
            'VIKOR': to_array(mcdm_results['vikor']['rankings']),
            'PROMETHEE': to_array(mcdm_results['promethee']['rankings']),
            'COPRAS': to_array(mcdm_results['copras']['rankings']),
            'EDAS': to_array(mcdm_results['edas']['rankings']),
            # Fuzzy methods
            'Fuzzy_TOPSIS': to_array(mcdm_results['fuzzy_topsis']['rankings']),
            'Fuzzy_VIKOR': to_array(mcdm_results['fuzzy_vikor']['rankings']),
            'Fuzzy_PROMETHEE': to_array(mcdm_results['fuzzy_promethee']['rankings']),
            'Fuzzy_COPRAS': to_array(mcdm_results['fuzzy_copras']['rankings']),
            'Fuzzy_EDAS': to_array(mcdm_results['fuzzy_edas']['rankings'])
        }
        
        self.logger.info(f"Aggregating rankings from {len(rankings)} MCDM methods")
        
        # 2a. Borda Count aggregation
        borda = BordaCount()
        borda_result = borda.aggregate(rankings)
        results['borda'] = borda_result
        results['aggregated'] = borda_result  # Keep backward compatibility
        self.logger.info(f"  Borda Count - Kendall's W: {borda_result.kendall_w:.4f}")
        
        # 2b. Copeland aggregation
        copeland = CopelandMethod()
        copeland_result = copeland.aggregate(rankings)
        results['copeland'] = copeland_result
        self.logger.info(f"  Copeland - Kendall's W: {copeland_result.kendall_w:.4f}")
        
        # 2c. Kemeny-Young aggregation (optimal consensus ranking)
        # max_exact=6 means exact O(n!) solution for n<=6, approximation for larger
        # This balances accuracy vs computation time
        kemeny = KemenyYoung(max_exact=6)
        kemeny_result = kemeny.aggregate(rankings)
        results['kemeny'] = kemeny_result
        self.logger.info(f"  Kemeny-Young - Kendall's W: {kemeny_result.kendall_w:.4f}")
        
        return results
    
    def _run_analysis(self, panel_data: PanelData,
                      mcdm_results: Dict[str, Any],
                      weights: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Run validation analysis.
        
        Sensitivity Analysis tests robustness of rankings to weight perturbations,
        which is critical for validating MCDM results (Objective 1).
        """
        results = {}
        
        # Sensitivity Analysis - validates ranking robustness
        def topsis_ranking_func(matrix, w):
            # Convert numpy arrays back to DataFrame for TOPSIS
            df = pd.DataFrame(matrix, columns=panel_data.components)
            weights_dict = dict(zip(panel_data.components, w))
            calc = TOPSISCalculator()
            result = calc.calculate(df, weights_dict)
            return result.ranks.values
        
        sensitivity = SensitivityAnalysis(
            n_simulations=self.config.validation.n_simulations
        )
        
        sens_result = sensitivity.analyze(
            panel_data.get_latest()[panel_data.components].values,
            weights['ensemble'],
            topsis_ranking_func,
            criteria_names=panel_data.components,
            alternative_names=panel_data.entities
        )
        results['sensitivity'] = sens_result
        
        self.logger.info(f"Sensitivity: Overall robustness = {sens_result.overall_robustness:.4f}")
        
        return results
    
    def _run_future_prediction(self, panel_data: PanelData,
                               weights: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Predict component values and MCDM scores for the next year (2025).
        
        Uses all historical data (2020-2024) to forecast 2025 values,
        then calculates TOPSIS and VIKOR rankings for predicted data.
        
        Parameters
        ----------
        panel_data : PanelData
            Panel data with historical observations (2020-2024)
        weights : Dict[str, np.ndarray]
            Calculated criteria weights
        
        Returns
        -------
        Dict[str, Any]
            Predicted components, MCDM scores, and rankings for 2025
        """
        from .ml.forecasting import UnifiedForecaster, ForecastMode
        
        current_year = max(panel_data.years)
        prediction_year = current_year + 1
        
        self.logger.info(f"Forecasting year {prediction_year} using data from {min(panel_data.years)}-{current_year}")
        
        # Step 1: Forecast all components for 2025 using UnifiedForecaster
        # This uses an ensemble of: Gradient Boosting, Random Forest, Extra Trees,
        # Bayesian Ridge, and Huber regression with optimal weighting
        # Note: Neural networks disabled due to insufficient data for reliable training
        self.logger.info("Training ML models on all historical data...")
        
        forecaster = UnifiedForecaster(
            mode=ForecastMode.BALANCED,
            include_neural=self.config.neural.enabled,  # Disabled by default - insufficient data
            include_tree_ensemble=True,
            include_linear=True,
            cv_folds=min(3, len(panel_data.years) - 1),
            random_state=42,
            verbose=False
        )
        
        # Forecast all components
        forecast_result = forecaster.fit_predict(
            panel_data,
            target_year=prediction_year
        )
        
        # Get predicted component values for 2025
        predicted_components = forecast_result.predictions
        self.logger.info(f"Predicted {len(panel_data.components)} components for {len(panel_data.entities)} entities")
        
        # Step 2: Calculate MCDM scores on predicted 2025 data
        self.logger.info("Calculating MCDM scores for predicted 2025 data...")
        
        # Use ensemble weights for MCDM
        w = weights['ensemble']
        weights_dict = {c: w[i] for i, c in enumerate(panel_data.components)}
        
        # Ensure predicted_components is properly formatted
        if isinstance(predicted_components, pd.DataFrame):
            predicted_df = predicted_components.copy()
            # Ensure columns match component order
            predicted_df = predicted_df[panel_data.components]
        else:
            predicted_df = pd.DataFrame(
                predicted_components,
                index=panel_data.entities,
                columns=panel_data.components
            )
        
        # Clip predictions to valid range [0, 1]
        predicted_df = predicted_df.clip(0, 1)
        
        # Calculate TOPSIS for 2025 predictions
        topsis = TOPSISCalculator(normalization=self.config.topsis.normalization.value)
        topsis_result = topsis.calculate(predicted_df, weights_dict)
        
        predicted_topsis_scores = topsis_result.scores.values
        predicted_topsis_rankings = topsis_result.ranks.values
        
        self.logger.info(f"Predicted TOPSIS: Top performer score = {topsis_result.scores.max():.4f}")
        
        # Calculate VIKOR for 2025 predictions
        vikor = VIKORCalculator(v=self.config.vikor.v)
        vikor_result = vikor.calculate(predicted_df, weights_dict)
        
        predicted_vikor = {
            'Q': vikor_result.Q,
            'S': vikor_result.S,
            'R': vikor_result.R,
            'rankings': vikor_result.final_ranks
        }
        
        self.logger.info(f"Predicted VIKOR: Best alternative Q = {vikor_result.Q.min():.4f}")
        
        # Step 3: Compile future prediction results
        # Build prediction intervals DataFrame if available
        uncertainty_df = forecast_result.uncertainty
        
        future_results = {
            'prediction_year': prediction_year,
            'training_years': list(panel_data.years),
            'predicted_components': predicted_df,
            'prediction_uncertainty': uncertainty_df,
            'topsis_scores': predicted_topsis_scores,
            'topsis_rankings': predicted_topsis_rankings,
            'topsis_result': topsis_result,
            'vikor': predicted_vikor,
            'vikor_result': vikor_result,
            'model_contributions': forecast_result.model_contributions,
            'forecast_summary': forecast_result.training_info
        }
        
        # Log top 5 predicted rankings
        top_indices = np.argsort(predicted_topsis_rankings)[:5]
        self.logger.info(f"Predicted {prediction_year} Top 5:")
        for i, idx in enumerate(top_indices):
            entity = panel_data.entities[idx]
            score = predicted_topsis_scores[idx]
            self.logger.info(f"  {i+1}. {entity}: {score:.4f}")
        
        return future_results

    def _generate_visualizations(self, panel_data: PanelData,
                                mcdm_results: Dict[str, Any],
                                ensemble_results: Dict[str, Any],
                                analysis_results: Dict[str, Any],
                                ml_results: Optional[Dict[str, Any]] = None) -> None:
        """Generate all visualizations."""
        try:
            # Score evolution
            scores_df = panel_data.long.copy()
            scores_df['score'] = np.tile(
                mcdm_results['topsis_scores'],
                len(panel_data.time_periods)
            )[:len(scores_df)]
            
            self.visualizer.plot_score_evolution(
                scores_df, 'Province', 'Year', 'score',
                title='Sustainability Score Evolution'
            )
            
            # Method comparison
            rankings = {
                'TOPSIS': mcdm_results['topsis_rankings'],
                'VIKOR': mcdm_results['vikor']['rankings'],
            }
            self.visualizer.plot_method_comparison(
                rankings, panel_data.entities,
                title='MCDM Method Comparison'
            )
            
            # Weight sensitivity
            if analysis_results.get('sensitivity'):
                self.visualizer.plot_weight_sensitivity(
                    analysis_results['sensitivity'].weight_sensitivity
                )
            
            # Ensemble weights
            if ensemble_results.get('stacking'):
                method_weights = dict(zip(
                    ensemble_results['stacking'].base_model_predictions.keys(),
                    ensemble_results['stacking'].meta_model_weights
                ))
                self.visualizer.plot_ensemble_weights(method_weights)
            
            # ML VISUALIZATIONS (Random Forest only)
            if ml_results:
                self._generate_ml_visualizations(panel_data, mcdm_results, 
                                                 ml_results, ensemble_results)
            
            self.logger.info("Visualizations saved to outputs/figures/")
            
        except Exception as e:
            self.logger.warning(f"Visualization generation failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
    
    def _generate_ml_visualizations(self, panel_data: PanelData,
                                    mcdm_results: Dict[str, Any],
                                    ml_results: Dict[str, Any],
                                    ensemble_results: Dict[str, Any]) -> None:
        """Generate ML-specific visualizations (Random Forest only)."""
        try:
            # ===== Random Forest Feature Importance Detailed =====
            if ml_results.get('rf_result'):
                rf_result = ml_results['rf_result']
                
                self.visualizer.plot_rf_feature_importance_detailed(
                    feature_importance=rf_result.feature_importance.to_dict(),
                    title='Random Forest Feature Importance with Cumulative Contribution',
                    save_name='16_rf_feature_importance_detailed.png'
                )
                self.logger.info("Generated: 16_rf_feature_importance_detailed.png")
                
                # Random Forest CV Progression
                self.visualizer.plot_rf_cv_progression(
                    cv_scores=rf_result.cv_scores,
                    title='Random Forest Cross-Validation Score Progression',
                    save_name='17_rf_cv_progression.png'
                )
                self.logger.info("Generated: 17_rf_cv_progression.png")
                
                # RF Actual vs Predicted
                self.visualizer.plot_actual_vs_predicted(
                    actual=rf_result.test_actual.values,
                    predicted=rf_result.test_predictions.values,
                    model_name='Random Forest',
                    entity_names=list(rf_result.test_actual.index),
                    title='Random Forest Model: Actual vs Predicted Values',
                    save_name='18_rf_actual_vs_predicted.png'
                )
                self.logger.info("Generated: 18_rf_actual_vs_predicted.png")
                
                # RF Residual Analysis
                self.visualizer.plot_residual_analysis(
                    actual=rf_result.test_actual.values,
                    predicted=rf_result.test_predictions.values,
                    model_name='Random Forest',
                    title='Random Forest Model: Residual Distribution Analysis',
                    save_name='19_rf_residual_analysis.png'
                )
                self.logger.info("Generated: 19_rf_residual_analysis.png")
                
                # RF Rank Correlation Analysis
                self.visualizer.plot_rank_correlation_analysis(
                    actual=rf_result.test_actual.values,
                    predicted=rf_result.test_predictions.values,
                    entity_names=list(rf_result.test_actual.index),
                    model_name='Random Forest',
                    title='Random Forest Model: Rank Prediction Accuracy',
                    save_name='20_rf_rank_correlation.png'
                )
                self.logger.info("Generated: 20_rf_rank_correlation.png")
            
            # Ensemble Model Contribution
            if ensemble_results.get('stacking'):
                stacking = ensemble_results['stacking']
                base_preds = stacking.base_model_predictions
                actual = mcdm_results['topsis_scores']
                
                weights_dict = dict(zip(
                    base_preds.keys(),
                    stacking.meta_model_weights
                ))
                
                # Convert base_preds values to numpy arrays
                base_preds_np = {k: np.array(v) for k, v in base_preds.items()}
                
                self.visualizer.plot_ensemble_contribution_analysis(
                    base_predictions=base_preds_np,
                    weights=weights_dict,
                    actual=to_array(actual),
                    title='Ensemble Model: Base Model Contribution Analysis',
                    save_name='21_ensemble_contribution.png'
                )
                self.logger.info("Generated: 21_ensemble_contribution.png")
            
            # Model Performance Comparison (Random Forest)
            model_metrics = {}
            
            if ml_results.get('rf_result'):
                rf = ml_results['rf_result']
                model_metrics['Random Forest'] = {
                    'R²': rf.test_metrics.get('r2', 0),
                    'MAE': rf.test_metrics.get('mae', 0),
                    'RMSE': np.sqrt(rf.test_metrics.get('mse', 0)),
                    'Rank Corr': rf.rank_correlation
                }
            
            if model_metrics:
                self.visualizer.plot_model_comparison(
                    model_results=model_metrics,
                    metrics=['R²', 'MAE', 'Rank Corr'],
                    title='Random Forest Model Performance',
                    save_name='22_rf_model_performance.png'
                )
                self.logger.info("Generated: 22_rf_model_performance.png")
                
        except Exception as e:
            self.logger.warning(f"ML visualization generation failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
    
    def _generate_all_visualizations(self, panel_data: PanelData,
                                     weights: Dict[str, np.ndarray],
                                     mcdm_results: Dict[str, Any],
                                     ensemble_results: Dict[str, Any],
                                     analysis_results: Dict[str, Any],
                                     ml_results: Optional[Dict[str, Any]] = None,
                                     future_predictions: Optional[Dict[str, Any]] = None) -> None:
        """
        Generate all visualizations as individual high-resolution charts.
        
        This produces a complete set of professional-quality figures.
        Each PNG contains ONLY a single high-resolution chart.
        """
        figure_count = 0
        
        try:
            # ===== SCORE EVOLUTION CHARTS =====
            scores_df = panel_data.long.copy()
            scores_df['score'] = np.tile(
                mcdm_results['topsis_scores'],
                len(panel_data.time_periods)
            )[:len(scores_df)]
            
            # Top performers evolution
            self.visualizer.plot_score_evolution_top(
                scores_df, 'Province', 'Year', 'score', top_n=10,
                title='Top 10 Performers - Score Evolution',
                save_name='01_score_evolution_top.png'
            )
            figure_count += 1
            self.logger.info("Generated: 01_score_evolution_top.png")
            
            # Bottom performers evolution
            self.visualizer.plot_score_evolution_bottom(
                scores_df, 'Province', 'Year', 'score', bottom_n=10,
                title='Bottom 10 Performers - Score Evolution',
                save_name='02_score_evolution_bottom.png'
            )
            figure_count += 1
            self.logger.info("Generated: 02_score_evolution_bottom.png")
            
            # ===== WEIGHTS ANALYSIS =====
            self.visualizer.plot_weights_comparison(
                weights, panel_data.components,
                title='Criteria Weights Comparison (Entropy vs CRITIC vs PCA vs Ensemble)',
                save_name='03_weights_comparison.png'
            )
            figure_count += 1
            self.logger.info("Generated: 03_weights_comparison.png")
            
            # ===== MCDM RESULTS =====
            # TOPSIS scores ranking
            self.visualizer.plot_topsis_scores_bar(
                to_array(mcdm_results['topsis_scores']), panel_data.entities,
                title='TOPSIS Scores - Complete Ranking',
                save_name='04_topsis_scores.png'
            )
            figure_count += 1
            self.logger.info("Generated: 04_topsis_scores.png")
            
            # VIKOR analysis
            self.visualizer.plot_vikor_analysis(
                to_array(mcdm_results['vikor']['Q']),
                to_array(mcdm_results['vikor']['S']),
                to_array(mcdm_results['vikor']['R']),
                panel_data.entities,
                title='VIKOR Analysis (Q, S, R Values)',
                save_name='05_vikor_analysis.png'
            )
            figure_count += 1
            self.logger.info("Generated: 05_vikor_analysis.png")
            
            # Method agreement matrix
            rankings_dict = {
                'TOPSIS': to_array(mcdm_results['topsis_rankings']),
                'Dynamic TOPSIS': len(to_array(mcdm_results['dynamic_topsis_scores'])) - 
                                 np.argsort(np.argsort(to_array(mcdm_results['dynamic_topsis_scores']))),
                'VIKOR': to_array(mcdm_results['vikor']['rankings']),
                'Fuzzy TOPSIS': len(to_array(mcdm_results['fuzzy_scores'])) - 
                               np.argsort(np.argsort(to_array(mcdm_results['fuzzy_scores'])))
            }
            self.visualizer.plot_method_agreement_matrix(
                rankings_dict,
                title='MCDM Methods Ranking Agreement (Spearman Correlation)',
                save_name='06_method_agreement.png'
            )
            figure_count += 1
            self.logger.info("Generated: 06_method_agreement.png")
            
            # Score distribution
            self.visualizer.plot_score_distribution(
                to_array(mcdm_results['topsis_scores']),
                title='TOPSIS Score Distribution',
                save_name='07_score_distribution.png'
            )
            figure_count += 1
            self.logger.info("Generated: 07_score_distribution.png")
            
            # ===== ML FEATURE IMPORTANCE =====
            if ml_results and ml_results.get('rf_importance'):
                self.visualizer.plot_feature_importance_single(
                    ml_results['rf_importance'],
                    title='Random Forest Feature Importance',
                    save_name='08_feature_importance.png'
                )
                figure_count += 1
                self.logger.info("Generated: 08_feature_importance.png")
            
            # ===== SENSITIVITY ANALYSIS =====
            if analysis_results.get('sensitivity'):
                self.visualizer.plot_sensitivity_analysis(
                    analysis_results['sensitivity'].weight_sensitivity,
                    title='Criteria Weight Sensitivity Analysis',
                    save_name='09_sensitivity_analysis.png'
                )
                figure_count += 1
                self.logger.info("Generated: 09_sensitivity_analysis.png")
            
            # ===== FINAL RANKING =====
            if ensemble_results.get('aggregated'):
                agg = ensemble_results['aggregated']
                self.visualizer.plot_final_ranking_summary(
                    panel_data.entities, 
                    to_array(agg.final_scores), 
                    to_array(agg.final_ranking),
                    title='Final Aggregated Ranking',
                    save_name='10_final_ranking.png'
                )
                figure_count += 1
                self.logger.info("Generated: 10_final_ranking.png")
            
            # ===== ML MODEL VISUALIZATIONS =====
            if ml_results:
                self._generate_ml_visualizations(panel_data, mcdm_results, 
                                                 ml_results, ensemble_results)
                figure_count += 7  # RF visualizations (16-22)
            
            # ===== METHOD COMPARISON (Parallel Coordinates) =====
            self.visualizer.plot_method_comparison(
                rankings_dict, panel_data.entities,
                title='MCDM Methods Ranking Comparison',
                save_name='11_method_comparison.png'
            )
            figure_count += 1
            self.logger.info("Generated: 11_method_comparison.png")
            
            # ===== ENSEMBLE WEIGHTS =====
            if ensemble_results.get('stacking'):
                method_weights = dict(zip(
                    ensemble_results['stacking'].base_model_predictions.keys(),
                    ensemble_results['stacking'].meta_model_weights
                ))
                self.visualizer.plot_ensemble_weights(
                    method_weights,
                    title='Stacking Ensemble Model Weights',
                    save_name='12_ensemble_weights.png'
                )
                figure_count += 1
                self.logger.info("Generated: 12_ensemble_weights.png")
            
            # ===== FUTURE PREDICTIONS =====
            if future_predictions:
                prediction_year = future_predictions.get('prediction_year', 2025)
                self.visualizer.plot_future_predictions(
                    panel_data.entities,
                    to_array(mcdm_results['topsis_scores']),
                    to_array(future_predictions['topsis_scores']),
                    prediction_year,
                    title=f'Future Predictions Comparison ({prediction_year})',
                    save_name='13_future_predictions.png'
                )
                figure_count += 1
                self.logger.info("Generated: 13_future_predictions.png")
            
            self.logger.info(f"Total figures generated: {figure_count}")
            self.logger.info(f"All figures saved to: {self.visualizer.output_dir}")
            
        except Exception as e:
            self.logger.warning(f"Visualization generation error: {e}")
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
