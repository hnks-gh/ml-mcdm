# -*- coding: utf-8 -*-
"""
Output Manager Module
=====================

Professional output management for ML-MCDM analysis results.
Handles all file exports, report generation, and result organization.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json


class OutputManager:
    """
    Professional output manager for ML-MCDM analysis results.
    
    Organizes outputs into a clean, professional structure:
    - results/         : All numerical results (CSV, JSON)
    - figures/         : All visualization charts (PNG)
    - reports/         : Comprehensive analysis reports
    """
    
    def __init__(self, base_output_dir: str = 'outputs'):
        """
        Initialize output manager.
        
        Parameters
        ----------
        base_output_dir : str
            Base directory for all outputs
        """
        self.base_dir = Path(base_output_dir)
        self.results_dir = self.base_dir / 'results'
        self.figures_dir = self.base_dir / 'figures'
        self.reports_dir = self.base_dir / 'reports'
        
        # Create directory structure
        self._setup_directories()
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def _setup_directories(self) -> None:
        """Create clean directory structure."""
        # Create main directories
        for d in [self.results_dir, self.figures_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # NUMERICAL RESULTS EXPORT
    # =========================================================================
    
    def save_weights(self, weights: Dict[str, np.ndarray], 
                     component_names: List[str]) -> str:
        """
        Save all weight calculation results.
        
        Parameters
        ----------
        weights : Dict[str, np.ndarray]
            Dictionary of weight arrays by method
        component_names : List[str]
            Names of components/criteria
        
        Returns
        -------
        str
            Path to saved file
        """
        # Create comprehensive weights DataFrame
        df = pd.DataFrame({'Component': component_names})
        
        for method, w in weights.items():
            df[f'{method.title()}_Weight'] = w
        
        # Add statistics
        df['Mean_Weight'] = df[[c for c in df.columns if 'Weight' in c]].mean(axis=1)
        df['Std_Weight'] = df[[c for c in df.columns if 'Weight' in c]].std(axis=1)
        df['Rank'] = df['Mean_Weight'].rank(ascending=False).astype(int)
        
        # Sort by mean weight
        df = df.sort_values('Mean_Weight', ascending=False).reset_index(drop=True)
        
        save_path = self.results_dir / 'weights_analysis.csv'
        df.to_csv(save_path, index=False, float_format='%.6f')
        
        return str(save_path)
    
    def save_rankings(self, panel_data: Any, 
                      mcdm_results: Dict[str, Any],
                      ensemble_results: Dict[str, Any]) -> str:
        """
        Save comprehensive ranking results.
        
        Parameters
        ----------
        panel_data : PanelData
            Panel data object
        mcdm_results : Dict
            MCDM analysis results
        ensemble_results : Dict
            Ensemble aggregation results
        
        Returns
        -------
        str
            Path to saved file
        """
        entities = panel_data.entities
        
        # Helper function to convert to numpy array
        def to_array(x):
            if hasattr(x, 'values'):
                return x.values
            return np.array(x) if not isinstance(x, np.ndarray) else x
        
        # Build comprehensive ranking DataFrame
        df = pd.DataFrame({
            'Rank': range(1, len(entities) + 1),
            'Entity': entities,
            'TOPSIS_Score': to_array(mcdm_results['topsis_scores']),
            'TOPSIS_Rank': to_array(mcdm_results['topsis_rankings']),
            'Dynamic_TOPSIS_Score': to_array(mcdm_results['dynamic_topsis_scores']),
            'VIKOR_Q': to_array(mcdm_results['vikor']['Q']),
            'VIKOR_S': to_array(mcdm_results['vikor']['S']),
            'VIKOR_R': to_array(mcdm_results['vikor']['R']),
            'VIKOR_Rank': to_array(mcdm_results['vikor']['rankings']),
            'Fuzzy_TOPSIS_Score': to_array(mcdm_results['fuzzy_scores']),
        })
        
        # Add ensemble results
        if ensemble_results.get('aggregated'):
            agg = ensemble_results['aggregated']
            df['Final_Score'] = to_array(agg.final_scores)
            df['Final_Rank'] = to_array(agg.final_ranking)
            df['Kendall_W'] = agg.kendall_w
        
        # Sort by final rank or TOPSIS rank
        sort_col = 'Final_Rank' if 'Final_Rank' in df.columns else 'TOPSIS_Rank'
        df = df.sort_values(sort_col).reset_index(drop=True)
        df['Rank'] = range(1, len(df) + 1)
        
        save_path = self.results_dir / 'final_rankings.csv'
        df.to_csv(save_path, index=False, float_format='%.6f')
        
        return str(save_path)
    
    def save_mcdm_scores(self, panel_data: Any,
                         mcdm_results: Dict[str, Any]) -> str:
        """
        Save detailed MCDM scores for all methods.
        
        Parameters
        ----------
        panel_data : PanelData
            Panel data object
        mcdm_results : Dict
            MCDM analysis results
        
        Returns
        -------
        str
            Path to saved file
        """
        entities = panel_data.entities
        
        # Helper function to convert to numpy array
        def to_array(x):
            if hasattr(x, 'values'):
                return x.values
            return np.array(x) if not isinstance(x, np.ndarray) else x
        
        df = pd.DataFrame({
            'Entity': entities,
            'TOPSIS_Score': to_array(mcdm_results['topsis_scores']),
            'TOPSIS_Rank': to_array(mcdm_results['topsis_rankings']),
            'TOPSIS_Distance_Positive': to_array(mcdm_results['topsis_result'].distances_positive) 
                if hasattr(mcdm_results.get('topsis_result'), 'distances_positive') else np.nan,
            'TOPSIS_Distance_Negative': to_array(mcdm_results['topsis_result'].distances_negative)
                if hasattr(mcdm_results.get('topsis_result'), 'distances_negative') else np.nan,
            'Dynamic_TOPSIS_Score': to_array(mcdm_results['dynamic_topsis_scores']),
            'Fuzzy_TOPSIS_Score': to_array(mcdm_results['fuzzy_scores']),
            'VIKOR_Q': to_array(mcdm_results['vikor']['Q']),
            'VIKOR_S': to_array(mcdm_results['vikor']['S']),
            'VIKOR_R': to_array(mcdm_results['vikor']['R']),
            'VIKOR_Rank': to_array(mcdm_results['vikor']['rankings']),
        })
        
        # Sort by TOPSIS score
        df = df.sort_values('TOPSIS_Score', ascending=False).reset_index(drop=True)
        
        save_path = self.results_dir / 'mcdm_scores_detailed.csv'
        df.to_csv(save_path, index=False, float_format='%.6f')
        
        return str(save_path)
    
    def save_ml_results(self, ml_results: Dict[str, Any],
                        panel_data: Any) -> Dict[str, str]:
        """
        Save all ML analysis results.
        
        Parameters
        ----------
        ml_results : Dict
            ML analysis results
        panel_data : PanelData
            Panel data object
        
        Returns
        -------
        Dict[str, str]
            Paths to saved files
        """
        saved_files = {}
        
        # 1. Feature Importance
        if ml_results.get('rf_importance'):
            imp_df = pd.DataFrame([
                {'Feature': k, 'Importance': v, 
                 'Rank': i + 1}
                for i, (k, v) in enumerate(
                    sorted(ml_results['rf_importance'].items(), 
                           key=lambda x: x[1], reverse=True)
                )
            ])
            path = self.results_dir / 'feature_importance.csv'
            imp_df.to_csv(path, index=False, float_format='%.6f')
            saved_files['feature_importance'] = str(path)
        
        # 2. Cross-Validation Results
        if ml_results.get('rf_result'):
            rf = ml_results['rf_result']
            cv_df = pd.DataFrame(rf.cv_scores)
            cv_df['Fold'] = range(1, len(cv_df) + 1)
            cv_df = cv_df[['Fold'] + [c for c in cv_df.columns if c != 'Fold']]
            
            # Add summary statistics
            summary_row = {'Fold': 'Mean'}
            summary_row.update({k: np.mean(v) for k, v in rf.cv_scores.items()})
            cv_df = pd.concat([cv_df, pd.DataFrame([summary_row])], ignore_index=True)
            
            summary_row = {'Fold': 'Std'}
            summary_row.update({k: np.std(v) for k, v in rf.cv_scores.items()})
            cv_df = pd.concat([cv_df, pd.DataFrame([summary_row])], ignore_index=True)
            
            path = self.results_dir / 'cv_scores.csv'
            cv_df.to_csv(path, index=False, float_format='%.6f')
            saved_files['cv_scores'] = str(path)
            
            # Test metrics
            test_df = pd.DataFrame([rf.test_metrics])
            path = self.results_dir / 'rf_test_metrics.csv'
            test_df.to_csv(path, index=False, float_format='%.6f')
            saved_files['rf_test_metrics'] = str(path)
        
        # 3. LSTM Results
        if ml_results.get('lstm_result'):
            lstm = ml_results['lstm_result']
            
            # Training history
            history_df = pd.DataFrame({
                'Epoch': range(1, len(lstm.train_loss) + 1),
                'Train_Loss': lstm.train_loss,
                'Val_Loss': lstm.val_loss if lstm.val_loss else [np.nan] * len(lstm.train_loss)
            })
            path = self.results_dir / 'lstm_training_history.csv'
            history_df.to_csv(path, index=False, float_format='%.6f')
            saved_files['lstm_history'] = str(path)
            
            # Test metrics
            test_df = pd.DataFrame([lstm.test_metrics])
            path = self.results_dir / 'lstm_test_metrics.csv'
            test_df.to_csv(path, index=False, float_format='%.6f')
            saved_files['lstm_test_metrics'] = str(path)
        
        # 4. Panel Regression Results
        if ml_results.get('panel_regression'):
            pr = ml_results['panel_regression']
            if hasattr(pr, 'coefficients'):
                coef_df = pd.DataFrame([
                    {'Variable': k, 'Coefficient': v}
                    for k, v in pr.coefficients.items()
                ])
                path = self.results_dir / 'panel_regression_coefficients.csv'
                coef_df.to_csv(path, index=False, float_format='%.6f')
                saved_files['panel_regression'] = str(path)
        
        # 5. Rough Set Results
        if ml_results.get('rough_set'):
            rs = ml_results['rough_set']
            rs_df = pd.DataFrame({
                'Metric': ['Original_Attributes', 'Reduced_Attributes', 
                          'Reduction_Ratio', 'Dependency_Quality'],
                'Value': [rs.original_n_attributes, rs.reduced_n_attributes,
                         1 - rs.reduced_n_attributes / rs.original_n_attributes,
                         rs.quality if hasattr(rs, 'quality') else np.nan]
            })
            
            if hasattr(rs, 'reduct') and rs.reduct:
                rs_df = pd.concat([rs_df, pd.DataFrame({
                    'Metric': ['Selected_Features'],
                    'Value': [', '.join(rs.reduct)]
                })], ignore_index=True)
            
            path = self.results_dir / 'rough_set_reduction.csv'
            rs_df.to_csv(path, index=False)
            saved_files['rough_set'] = str(path)
        
        return saved_files
    
    def save_ensemble_results(self, ensemble_results: Dict[str, Any],
                              panel_data: Any) -> Dict[str, str]:
        """
        Save ensemble model results.
        
        Parameters
        ----------
        ensemble_results : Dict
            Ensemble analysis results
        panel_data : PanelData
            Panel data object
        
        Returns
        -------
        Dict[str, str]
            Paths to saved files
        """
        saved_files = {}
        
        # 1. Stacking Results
        if ensemble_results.get('stacking'):
            stacking = ensemble_results['stacking']
            
            # Model weights
            weights_df = pd.DataFrame({
                'Base_Model': list(stacking.base_model_predictions.keys()),
                'Weight': stacking.meta_model_weights
            })
            weights_df = weights_df.sort_values('Weight', ascending=False)
            path = self.results_dir / 'stacking_weights.csv'
            weights_df.to_csv(path, index=False, float_format='%.6f')
            saved_files['stacking_weights'] = str(path)
            
            # Performance metrics
            metrics_df = pd.DataFrame([{
                'Meta_Model_R2': stacking.meta_model_r2,
                'Meta_Model_Type': type(stacking.meta_model).__name__ 
                    if hasattr(stacking, 'meta_model') else 'Unknown'
            }])
            path = self.results_dir / 'stacking_performance.csv'
            metrics_df.to_csv(path, index=False, float_format='%.6f')
            saved_files['stacking_performance'] = str(path)
        
        # 2. Rank Aggregation Results
        if ensemble_results.get('aggregated'):
            agg = ensemble_results['aggregated']
            
            agg_df = pd.DataFrame({
                'Entity': panel_data.entities,
                'Final_Score': agg.final_scores,
                'Final_Rank': agg.final_ranking,
            })
            agg_df = agg_df.sort_values('Final_Rank')
            
            # Add Kendall's W as metadata
            agg_df['Kendall_W'] = agg.kendall_w
            
            path = self.results_dir / 'rank_aggregation.csv'
            agg_df.to_csv(path, index=False, float_format='%.6f')
            saved_files['rank_aggregation'] = str(path)
        
        return saved_files
    
    def save_analysis_results(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Save advanced analysis results.
        
        Parameters
        ----------
        analysis_results : Dict
            Analysis results (convergence, sensitivity)
        
        Returns
        -------
        Dict[str, str]
            Paths to saved files
        """
        saved_files = {}
        
        # 1. Convergence Analysis
        if analysis_results.get('convergence'):
            conv = analysis_results['convergence']
            
            # Sigma convergence by year
            sigma_df = pd.DataFrame({
                'Year': list(conv.sigma_by_year.keys()),
                'Coefficient_of_Variation': list(conv.sigma_by_year.values())
            })
            path = self.results_dir / 'sigma_convergence.csv'
            sigma_df.to_csv(path, index=False, float_format='%.6f')
            saved_files['sigma_convergence'] = str(path)
            
            # Beta convergence summary
            beta_df = pd.DataFrame([{
                'Beta_Coefficient': conv.beta_coefficient,
                'Half_Life_Years': conv.half_life,
                'Convergence_Type': 'Converging' if conv.beta_coefficient < 0 else 'Diverging',
                'Speed': abs(conv.beta_coefficient)
            }])
            path = self.results_dir / 'beta_convergence.csv'
            beta_df.to_csv(path, index=False, float_format='%.6f')
            saved_files['beta_convergence'] = str(path)
        
        # 2. Sensitivity Analysis
        if analysis_results.get('sensitivity'):
            sens = analysis_results['sensitivity']
            
            sens_df = pd.DataFrame([
                {'Criterion': k, 'Sensitivity_Index': v, 
                 'Rank': i + 1}
                for i, (k, v) in enumerate(
                    sorted(sens.weight_sensitivity.items(), 
                           key=lambda x: x[1], reverse=True)
                )
            ])
            path = self.results_dir / 'sensitivity_analysis.csv'
            sens_df.to_csv(path, index=False, float_format='%.6f')
            saved_files['sensitivity'] = str(path)
            
            # Overall robustness
            robust_df = pd.DataFrame([{
                'Overall_Robustness': sens.overall_robustness,
                'N_Simulations': sens.n_simulations if hasattr(sens, 'n_simulations') else 'N/A'
            }])
            path = self.results_dir / 'robustness_summary.csv'
            robust_df.to_csv(path, index=False, float_format='%.6f')
            saved_files['robustness'] = str(path)
        
        return saved_files
    
    def save_panel_data_summary(self, panel_data: Any) -> str:
        """
        Save panel data summary statistics.
        
        Parameters
        ----------
        panel_data : PanelData
            Panel data object
        
        Returns
        -------
        str
            Path to saved file
        """
        # Get latest cross-section
        latest = panel_data.get_latest()
        
        # Summary statistics
        summary = latest[panel_data.components].describe()
        summary = summary.T
        summary['Component'] = summary.index
        summary = summary[['Component'] + [c for c in summary.columns if c != 'Component']]
        
        path = self.results_dir / 'data_summary_statistics.csv'
        summary.to_csv(path, index=False, float_format='%.6f')
        
        return str(path)
    
    # =========================================================================
    # COMPREHENSIVE REPORT GENERATION
    # =========================================================================
    
    def generate_comprehensive_report(self, 
                                       panel_data: Any,
                                       weights: Dict[str, np.ndarray],
                                       mcdm_results: Dict[str, Any],
                                       ml_results: Dict[str, Any],
                                       ensemble_results: Dict[str, Any],
                                       analysis_results: Dict[str, Any],
                                       execution_time: float) -> str:
        """
        Generate comprehensive analysis report.
        
        Parameters
        ----------
        All analysis results and metadata
        
        Returns
        -------
        str
            Path to saved report
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ML-MCDM PANEL DATA ANALYSIS - COMPREHENSIVE REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"\nGenerated: {timestamp}")
        report_lines.append(f"Execution Time: {execution_time:.2f} seconds")
        
        # 1. DATA OVERVIEW
        report_lines.append("\n" + "=" * 80)
        report_lines.append("1. DATA OVERVIEW")
        report_lines.append("=" * 80)
        report_lines.append(f"  Entities (Provinces): {len(panel_data.entities)}")
        report_lines.append(f"  Time Periods: {len(panel_data.time_periods)} ({min(panel_data.years)}-{max(panel_data.years)})")
        report_lines.append(f"  Components/Criteria: {len(panel_data.components)}")
        report_lines.append(f"  Total Observations: {len(panel_data.entities) * len(panel_data.time_periods)}")
        
        # 2. WEIGHT ANALYSIS
        report_lines.append("\n" + "=" * 80)
        report_lines.append("2. WEIGHT ANALYSIS")
        report_lines.append("=" * 80)
        
        for method, w in weights.items():
            report_lines.append(f"\n  {method.upper()} Weights:")
            report_lines.append(f"    Range: [{w.min():.6f}, {w.max():.6f}]")
            report_lines.append(f"    Mean: {w.mean():.6f}")
            report_lines.append(f"    Std: {w.std():.6f}")
            
            # Top 5 weights
            top_idx = np.argsort(w)[::-1][:5]
            report_lines.append(f"    Top 5 Components:")
            for i, idx in enumerate(top_idx):
                report_lines.append(f"      {i+1}. {panel_data.components[idx]}: {w[idx]:.6f}")
        
        # 3. MCDM RESULTS
        report_lines.append("\n" + "=" * 80)
        report_lines.append("3. MCDM ANALYSIS RESULTS")
        report_lines.append("=" * 80)
        
        # TOPSIS
        report_lines.append("\n  3.1 TOPSIS Analysis")
        scores = mcdm_results['topsis_scores']
        report_lines.append(f"    Score Range: [{scores.min():.6f}, {scores.max():.6f}]")
        report_lines.append(f"    Score Mean: {scores.mean():.6f}")
        report_lines.append(f"    Score Std: {scores.std():.6f}")
        
        report_lines.append("\n    Top 10 Rankings:")
        top_idx = np.argsort(mcdm_results['topsis_rankings'])[:10]
        for i, idx in enumerate(top_idx):
            report_lines.append(f"      {i+1}. {panel_data.entities[idx]}: "
                              f"Score={scores[idx]:.6f}")
        
        # Dynamic TOPSIS
        report_lines.append("\n  3.2 Dynamic TOPSIS (Panel-aware)")
        d_scores = mcdm_results['dynamic_topsis_scores']
        report_lines.append(f"    Score Range: [{np.min(d_scores):.6f}, {np.max(d_scores):.6f}]")
        report_lines.append(f"    Score Mean: {np.mean(d_scores):.6f}")
        
        # VIKOR
        report_lines.append("\n  3.3 VIKOR Analysis")
        vikor = mcdm_results['vikor']
        # Convert to numpy arrays if they are pandas Series
        Q_vals = np.array(vikor['Q']) if hasattr(vikor['Q'], 'values') else vikor['Q']
        S_vals = np.array(vikor['S']) if hasattr(vikor['S'], 'values') else vikor['S']
        R_vals = np.array(vikor['R']) if hasattr(vikor['R'], 'values') else vikor['R']
        vikor_ranks = np.array(vikor['rankings']) if hasattr(vikor['rankings'], 'values') else vikor['rankings']
        
        report_lines.append(f"    Q Value Range: [{np.min(Q_vals):.6f}, {np.max(Q_vals):.6f}]")
        report_lines.append(f"    S Value Range: [{np.min(S_vals):.6f}, {np.max(S_vals):.6f}]")
        report_lines.append(f"    R Value Range: [{np.min(R_vals):.6f}, {np.max(R_vals):.6f}]")
        
        report_lines.append("\n    Top 10 VIKOR Rankings (lowest Q is best):")
        top_idx = np.argsort(vikor_ranks)[:10]
        for i, idx in enumerate(top_idx):
            report_lines.append(f"      {i+1}. {panel_data.entities[idx]}: "
                              f"Q={Q_vals[idx]:.6f}, S={S_vals[idx]:.6f}, R={R_vals[idx]:.6f}")
        
        # Fuzzy TOPSIS
        report_lines.append("\n  3.4 Fuzzy TOPSIS Analysis")
        f_scores = mcdm_results['fuzzy_scores']
        report_lines.append(f"    Score Range: [{np.min(f_scores):.6f}, {np.max(f_scores):.6f}]")
        
        # 4. ML RESULTS
        report_lines.append("\n" + "=" * 80)
        report_lines.append("4. MACHINE LEARNING RESULTS")
        report_lines.append("=" * 80)
        
        # Random Forest
        if ml_results.get('rf_result'):
            rf = ml_results['rf_result']
            report_lines.append("\n  4.1 Random Forest Time-Series CV")
            report_lines.append(f"    Test R²: {rf.test_metrics.get('r2', 0):.6f}")
            report_lines.append(f"    Test MAE: {rf.test_metrics.get('mae', 0):.6f}")
            report_lines.append(f"    Test RMSE: {np.sqrt(rf.test_metrics.get('mse', 0)):.6f}")
            report_lines.append(f"    Rank Correlation: {rf.rank_correlation:.6f}")
            
            report_lines.append("\n    Cross-Validation Summary:")
            for metric, values in rf.cv_scores.items():
                report_lines.append(f"      {metric}: {np.mean(values):.6f} ± {np.std(values):.6f}")
            
            report_lines.append("\n    Top 10 Feature Importances:")
            sorted_imp = sorted(ml_results['rf_importance'].items(), 
                               key=lambda x: x[1], reverse=True)[:10]
            for i, (feat, imp) in enumerate(sorted_imp):
                report_lines.append(f"      {i+1}. {feat}: {imp:.6f}")
        
        # LSTM
        if ml_results.get('lstm_result'):
            lstm = ml_results['lstm_result']
            report_lines.append("\n  4.2 LSTM Forecasting")
            report_lines.append(f"    Final Train Loss: {lstm.train_loss[-1]:.6f}")
            if lstm.val_loss:
                report_lines.append(f"    Final Val Loss: {lstm.val_loss[-1]:.6f}")
            report_lines.append(f"    Test MSE: {lstm.test_metrics.get('mse', 0):.6f}")
            report_lines.append(f"    Test MAE: {lstm.test_metrics.get('mae', 0):.6f}")
            report_lines.append(f"    Rank Correlation: {lstm.rank_correlation:.6f}")
        
        # Panel Regression
        if ml_results.get('panel_regression'):
            pr = ml_results['panel_regression']
            report_lines.append("\n  4.3 Panel Regression")
            report_lines.append(f"    R²: {pr.r_squared:.6f}")
            if hasattr(pr, 'coefficients'):
                report_lines.append("\n    Significant Coefficients:")
                sorted_coef = sorted(pr.coefficients.items(), 
                                    key=lambda x: abs(x[1]), reverse=True)[:10]
                for var, coef in sorted_coef:
                    report_lines.append(f"      {var}: {coef:.6f}")
        
        # Rough Set
        if ml_results.get('rough_set'):
            rs = ml_results['rough_set']
            report_lines.append("\n  4.4 Rough Set Feature Reduction")
            report_lines.append(f"    Original Attributes: {rs.original_n_attributes}")
            report_lines.append(f"    Reduced Attributes: {rs.reduced_n_attributes}")
            report_lines.append(f"    Reduction Ratio: {1 - rs.reduced_n_attributes/rs.original_n_attributes:.2%}")
            if hasattr(rs, 'reduct') and rs.reduct:
                report_lines.append(f"    Selected Features: {', '.join(rs.reduct)}")
        
        # 5. ENSEMBLE RESULTS
        report_lines.append("\n" + "=" * 80)
        report_lines.append("5. ENSEMBLE INTEGRATION RESULTS")
        report_lines.append("=" * 80)
        
        if ensemble_results.get('stacking'):
            stacking = ensemble_results['stacking']
            report_lines.append("\n  5.1 Stacking Ensemble")
            report_lines.append(f"    Meta-Model R²: {stacking.meta_model_r2:.6f}")
            report_lines.append("\n    Base Model Weights:")
            for model, weight in zip(stacking.base_model_predictions.keys(), 
                                    stacking.meta_model_weights):
                report_lines.append(f"      {model}: {weight:.6f}")
        
        if ensemble_results.get('aggregated'):
            agg = ensemble_results['aggregated']
            report_lines.append("\n  5.2 Rank Aggregation")
            report_lines.append(f"    Kendall's W (Agreement): {agg.kendall_w:.6f}")
            
            # Convert to numpy arrays
            final_ranking = np.array(agg.final_ranking) if hasattr(agg.final_ranking, 'values') else np.array(agg.final_ranking)
            final_scores = np.array(agg.final_scores) if hasattr(agg.final_scores, 'values') else np.array(agg.final_scores)
            
            report_lines.append("\n    Final Top 10 Rankings:")
            sorted_idx = np.argsort(final_ranking)[:10]
            for i, idx in enumerate(sorted_idx):
                report_lines.append(f"      {i+1}. {panel_data.entities[idx]}: "
                                  f"Score={final_scores[idx]:.6f}")
        
        # 6. ANALYSIS RESULTS
        report_lines.append("\n" + "=" * 80)
        report_lines.append("6. ADVANCED ANALYSIS RESULTS")
        report_lines.append("=" * 80)
        
        if analysis_results.get('convergence'):
            conv = analysis_results['convergence']
            report_lines.append("\n  6.1 Convergence Analysis")
            report_lines.append(f"    Beta Coefficient: {conv.beta_coefficient:.6f}")
            report_lines.append(f"    Half-Life: {conv.half_life:.2f} years")
            report_lines.append(f"    Status: {'CONVERGING' if conv.beta_coefficient < 0 else 'DIVERGING'}")
            
            report_lines.append("\n    Sigma Convergence by Year:")
            for year, sigma in conv.sigma_by_year.items():
                report_lines.append(f"      {year}: {sigma:.6f}")
        
        if analysis_results.get('sensitivity'):
            sens = analysis_results['sensitivity']
            report_lines.append("\n  6.2 Sensitivity Analysis")
            report_lines.append(f"    Overall Robustness: {sens.overall_robustness:.6f}")
            
            report_lines.append("\n    Criteria Sensitivity (Top 10):")
            sorted_sens = sorted(sens.weight_sensitivity.items(), 
                                key=lambda x: x[1], reverse=True)[:10]
            for i, (crit, val) in enumerate(sorted_sens):
                report_lines.append(f"      {i+1}. {crit}: {val:.6f}")
        
        # 7. CONCLUSIONS
        report_lines.append("\n" + "=" * 80)
        report_lines.append("7. KEY FINDINGS AND CONCLUSIONS")
        report_lines.append("=" * 80)
        
        # Top performer
        if ensemble_results.get('aggregated'):
            agg = ensemble_results['aggregated']
            # Convert to numpy arrays
            final_ranking = np.array(agg.final_ranking) if hasattr(agg.final_ranking, 'values') else np.array(agg.final_ranking)
            final_scores = np.array(agg.final_scores) if hasattr(agg.final_scores, 'values') else np.array(agg.final_scores)
            best_idx = np.argmin(final_ranking)
            report_lines.append(f"\n  • Top Performer: {panel_data.entities[best_idx]}")
            report_lines.append(f"    Final Score: {final_scores[best_idx]:.6f}")
        
        # Method agreement
        if ensemble_results.get('aggregated'):
            w = ensemble_results['aggregated'].kendall_w
            agreement = "Strong" if w > 0.7 else "Moderate" if w > 0.5 else "Weak"
            report_lines.append(f"\n  • Method Agreement: {agreement} (Kendall's W = {w:.4f})")
        
        # Convergence status
        if analysis_results.get('convergence'):
            conv = analysis_results['convergence']
            status = "converging" if conv.beta_coefficient < 0 else "diverging"
            report_lines.append(f"\n  • Regional Status: Provinces are {status}")
            report_lines.append(f"    Estimated half-life: {conv.half_life:.1f} years")
        
        # Best ML model
        if ml_results.get('rf_result') or ml_results.get('lstm_result'):
            best_r2 = 0
            best_model = "N/A"
            if ml_results.get('rf_result'):
                rf_r2 = ml_results['rf_result'].test_metrics.get('r2', 0)
                if rf_r2 > best_r2:
                    best_r2 = rf_r2
                    best_model = "Random Forest"
            report_lines.append(f"\n  • Best ML Model: {best_model} (R² = {best_r2:.4f})")
        
        report_lines.append("\n" + "=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        
        # Save report
        report_text = '\n'.join(report_lines)
        path = self.reports_dir / 'analysis_report.txt'
        with open(path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        return str(path)
    
    def save_all_results(self,
                         panel_data: Any,
                         weights: Dict[str, np.ndarray],
                         mcdm_results: Dict[str, Any],
                         ml_results: Dict[str, Any],
                         ensemble_results: Dict[str, Any],
                         analysis_results: Dict[str, Any],
                         execution_time: float) -> Dict[str, Any]:
        """
        Save all analysis results in organized structure with robust error handling.
        
        Returns
        -------
        Dict[str, Any]
            Summary of all saved files
        """
        saved_files = {
            'timestamp': datetime.now().isoformat(),
            'execution_time': execution_time,
            'files': {}
        }
        
        # Helper function for safe saving
        def safe_save(name: str, save_func, *args, **kwargs):
            try:
                result = save_func(*args, **kwargs)
                saved_files['files'][name] = result
                return result
            except Exception as e:
                saved_files['files'][name] = {'error': str(e)}
                return None
        
        # Save all numerical results with error handling
        safe_save('weights', self.save_weights, weights, panel_data.components)
        
        safe_save('rankings', self.save_rankings, panel_data, mcdm_results, ensemble_results)
        
        safe_save('mcdm_scores', self.save_mcdm_scores, panel_data, mcdm_results)
        
        # ML results (may be partially empty)
        if ml_results:
            safe_save('ml_results', self.save_ml_results, ml_results, panel_data)
        else:
            saved_files['files']['ml_results'] = {'status': 'skipped - no ML results'}
        
        # Ensemble results
        if ensemble_results:
            safe_save('ensemble_results', self.save_ensemble_results, ensemble_results, panel_data)
        else:
            saved_files['files']['ensemble_results'] = {'status': 'skipped - no ensemble results'}
        
        # Analysis results
        if analysis_results:
            safe_save('analysis_results', self.save_analysis_results, analysis_results)
        else:
            saved_files['files']['analysis_results'] = {'status': 'skipped - no analysis results'}
        
        safe_save('data_summary', self.save_panel_data_summary, panel_data)
        
        safe_save('report', self.generate_comprehensive_report,
                  panel_data, weights, mcdm_results, ml_results,
                  ensemble_results, analysis_results, execution_time)
        
        # Save manifest
        try:
            manifest_path = self.results_dir / 'output_manifest.json'
            with open(manifest_path, 'w') as f:
                json.dump(saved_files, f, indent=2, default=str)
        except Exception as e:
            saved_files['manifest_error'] = str(e)
        
        return saved_files


def create_output_manager(output_dir: str = 'outputs') -> OutputManager:
    """Factory function to create output manager."""
    return OutputManager(output_dir)
