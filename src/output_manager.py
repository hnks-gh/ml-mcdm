# -*- coding: utf-8 -*-
"""Output management for analysis results, reports, and file exports."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
import json


def to_array(x: Any) -> np.ndarray:
    """
    Convert various data types to numpy array.
    
    Parameters
    ----------
    x : Any
        Input data (pandas Series/DataFrame, list, or numpy array)
    
    Returns
    -------
    np.ndarray
        Converted numpy array
    """
    if x is None:
        return np.array([])
    if hasattr(x, 'values'):
        return x.values
    return np.array(x) if not isinstance(x, np.ndarray) else x


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
        
        # Get TOPSIS distances from result object (correct attribute names: d_positive, d_negative)
        topsis_result = mcdm_results.get('topsis_result')
        d_positive = to_array(topsis_result.d_positive) if topsis_result and hasattr(topsis_result, 'd_positive') else np.full(len(entities), np.nan)
        d_negative = to_array(topsis_result.d_negative) if topsis_result and hasattr(topsis_result, 'd_negative') else np.full(len(entities), np.nan)
        
        df = pd.DataFrame({
            'Entity': entities,
            'TOPSIS_Score': to_array(mcdm_results['topsis_scores']),
            'TOPSIS_Rank': to_array(mcdm_results['topsis_rankings']),
            'TOPSIS_Distance_Positive': d_positive,
            'TOPSIS_Distance_Negative': d_negative,
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
        
        # 2. Rank Aggregation Metadata (save as JSON to avoid redundancy with final_rankings.csv)
        if ensemble_results.get('aggregated'):
            agg = ensemble_results['aggregated']
            
            # Save aggregation metadata as JSON (Kendall's W and method info)
            agg_metadata = {
                'kendall_w': float(agg.kendall_w),
                'interpretation': 'Strong agreement' if agg.kendall_w > 0.7 else 
                                 'Moderate agreement' if agg.kendall_w > 0.5 else 'Weak agreement',
                'n_entities': len(panel_data.entities),
                'aggregation_method': agg.method if hasattr(agg, 'method') else 'borda',
                'top_entity': panel_data.entities[int(np.argmin(to_array(agg.final_ranking)))],
                'top_score': float(np.max(to_array(agg.final_scores)))
            }
            
            path = self.results_dir / 'aggregation_metadata.json'
            with open(path, 'w') as f:
                json.dump(agg_metadata, f, indent=2)
            saved_files['aggregation_metadata'] = str(path)
        
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
                                       execution_time: float,
                                       future_predictions: Optional[Dict[str, Any]] = None) -> str:
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
        
        # 7. FUTURE PREDICTIONS
        if future_predictions:
            report_lines.append("\n" + "=" * 80)
            report_lines.append("7. FUTURE YEAR PREDICTIONS")
            report_lines.append("=" * 80)
            
            pred_year = future_predictions.get('prediction_year', 'Next Year')
            training_years = future_predictions.get('training_years', [])
            
            report_lines.append(f"\n  Prediction Year: {pred_year}")
            report_lines.append(f"  Training Data: {min(training_years)}-{max(training_years)} ({len(training_years)} years)")
            
            pred_scores = to_array(future_predictions['topsis_scores'])
            pred_ranks = to_array(future_predictions['topsis_rankings'])
            
            report_lines.append(f"\n  7.1 Predicted TOPSIS Scores")
            report_lines.append(f"    Score Range: [{pred_scores.min():.6f}, {pred_scores.max():.6f}]")
            report_lines.append(f"    Score Mean: {pred_scores.mean():.6f}")
            report_lines.append(f"    Score Std: {pred_scores.std():.6f}")
            
            report_lines.append(f"\n    Predicted Top 10 Rankings for {pred_year}:")
            top_idx = np.argsort(pred_ranks)[:10]
            for i, idx in enumerate(top_idx):
                entity = panel_data.entities[idx]
                score = pred_scores[idx]
                report_lines.append(f"      {i+1}. {entity}: Score={score:.6f}")
            
            # VIKOR predictions
            if 'vikor' in future_predictions:
                vikor = future_predictions['vikor']
                report_lines.append(f"\n  7.2 Predicted VIKOR Analysis")
                report_lines.append(f"    Q Value Range: [{to_array(vikor['Q']).min():.6f}, {to_array(vikor['Q']).max():.6f}]")
                
                vikor_ranks = to_array(vikor['rankings'])
                top_vikor_idx = np.argsort(vikor_ranks)[:5]
                report_lines.append(f"\n    Predicted VIKOR Top 5 for {pred_year}:")
                for i, idx in enumerate(top_vikor_idx):
                    entity = panel_data.entities[idx]
                    q_val = to_array(vikor['Q'])[idx]
                    report_lines.append(f"      {i+1}. {entity}: Q={q_val:.6f}")
            
            # Model contributions
            model_contrib = future_predictions.get('model_contributions', {})
            if model_contrib:
                report_lines.append(f"\n  7.3 Forecast Model Contributions")
                sorted_models = sorted(model_contrib.items(), key=lambda x: x[1], reverse=True)
                for model, weight in sorted_models[:5]:
                    report_lines.append(f"    {model}: {weight:.4f}")
        
        # 8. CONCLUSIONS
        report_lines.append("\n" + "=" * 80)
        report_lines.append("8. KEY FINDINGS AND CONCLUSIONS" if future_predictions else "7. KEY FINDINGS AND CONCLUSIONS")
        report_lines.append("=" * 80)
        
        # Top performer
        if ensemble_results.get('aggregated'):
            agg = ensemble_results['aggregated']
            # Convert to numpy arrays
            final_ranking = np.array(agg.final_ranking) if hasattr(agg.final_ranking, 'values') else np.array(agg.final_ranking)
            final_scores = np.array(agg.final_scores) if hasattr(agg.final_scores, 'values') else np.array(agg.final_scores)
            best_idx = np.argmin(final_ranking)
            report_lines.append(f"\n  • Current Top Performer ({max(panel_data.years)}): {panel_data.entities[best_idx]}")
            report_lines.append(f"    Final Score: {final_scores[best_idx]:.6f}")
        
        # Predicted top performer
        if future_predictions:
            pred_scores = to_array(future_predictions['topsis_scores'])
            pred_year = future_predictions.get('prediction_year', 'Next Year')
            best_pred_idx = np.argmax(pred_scores)
            report_lines.append(f"\n  • Predicted Top Performer ({pred_year}): {panel_data.entities[best_pred_idx]}")
            report_lines.append(f"    Predicted Score: {pred_scores[best_pred_idx]:.6f}")
        
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
            if ml_results.get('lstm_result'):
                lstm = ml_results['lstm_result']
                # Calculate approximate R² for LSTM from MSE
                lstm_r2 = 1 - lstm.test_metrics.get('mse', 1) if lstm.test_metrics.get('mse', 1) < 1 else 0
                if lstm_r2 > best_r2:
                    best_r2 = lstm_r2
                    best_model = "LSTM"
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
    
    def save_future_predictions(self, future_predictions: Dict[str, Any],
                                panel_data: Any) -> Dict[str, str]:
        """
        Save predicted future year (2025) results.
        
        Parameters
        ----------
        future_predictions : Dict
            Future prediction results including predicted components and MCDM scores
        panel_data : PanelData
            Panel data object
        
        Returns
        -------
        Dict[str, str]
            Paths to saved files
        """
        saved_files = {}
        prediction_year = future_predictions.get('prediction_year', 2025)
        
        # 1. Save predicted rankings for future year
        entities = panel_data.entities
        
        rankings_df = pd.DataFrame({
            'Rank': range(1, len(entities) + 1),
            'Entity': entities,
            'Predicted_TOPSIS_Score': to_array(future_predictions['topsis_scores']),
            'Predicted_TOPSIS_Rank': to_array(future_predictions['topsis_rankings']),
            'Predicted_VIKOR_Q': to_array(future_predictions['vikor']['Q']),
            'Predicted_VIKOR_S': to_array(future_predictions['vikor']['S']),
            'Predicted_VIKOR_R': to_array(future_predictions['vikor']['R']),
            'Predicted_VIKOR_Rank': to_array(future_predictions['vikor']['rankings']),
            'Prediction_Year': prediction_year
        })
        
        # Sort by predicted TOPSIS rank
        rankings_df = rankings_df.sort_values('Predicted_TOPSIS_Rank').reset_index(drop=True)
        rankings_df['Rank'] = range(1, len(rankings_df) + 1)
        
        path = self.results_dir / f'predicted_rankings_{prediction_year}.csv'
        rankings_df.to_csv(path, index=False, float_format='%.6f')
        saved_files['predicted_rankings'] = str(path)
        
        # 2. Save predicted component values
        predicted_components = future_predictions.get('predicted_components')
        if predicted_components is not None:
            if isinstance(predicted_components, pd.DataFrame):
                comp_df = predicted_components.copy()
                comp_df.insert(0, 'Entity', comp_df.index)
                comp_df = comp_df.reset_index(drop=True)
            else:
                comp_df = pd.DataFrame(
                    predicted_components,
                    index=entities,
                    columns=panel_data.components
                )
                comp_df.insert(0, 'Entity', comp_df.index)
                comp_df = comp_df.reset_index(drop=True)
            
            comp_df['Prediction_Year'] = prediction_year
            
            path = self.results_dir / f'predicted_components_{prediction_year}.csv'
            comp_df.to_csv(path, index=False, float_format='%.6f')
            saved_files['predicted_components'] = str(path)
        
        # 3. Save prediction uncertainty if available
        uncertainty = future_predictions.get('prediction_uncertainty')
        if uncertainty is not None:
            if isinstance(uncertainty, pd.DataFrame):
                unc_df = uncertainty.copy()
                unc_df.insert(0, 'Entity', unc_df.index)
                unc_df = unc_df.reset_index(drop=True)
            else:
                unc_df = pd.DataFrame(
                    uncertainty,
                    index=entities,
                    columns=panel_data.components
                )
                unc_df.insert(0, 'Entity', unc_df.index)
                unc_df = unc_df.reset_index(drop=True)
            
            path = self.results_dir / f'prediction_uncertainty_{prediction_year}.csv'
            unc_df.to_csv(path, index=False, float_format='%.6f')
            saved_files['prediction_uncertainty'] = str(path)
        
        # 4. Save model contributions
        model_contributions = future_predictions.get('model_contributions', {})
        if model_contributions:
            contrib_df = pd.DataFrame([
                {'Model': k, 'Weight': v}
                for k, v in sorted(model_contributions.items(), 
                                   key=lambda x: x[1], reverse=True)
            ])
            path = self.results_dir / f'forecast_model_weights_{prediction_year}.csv'
            contrib_df.to_csv(path, index=False, float_format='%.6f')
            saved_files['model_weights'] = str(path)
        
        # 5. Save forecast summary
        summary = {
            'prediction_year': prediction_year,
            'training_years': future_predictions.get('training_years', []),
            'n_entities': len(entities),
            'n_components': len(panel_data.components),
            'top_predicted_entity': entities[np.argmax(to_array(future_predictions['topsis_scores']))],
            'top_predicted_score': float(np.max(to_array(future_predictions['topsis_scores']))),
            'mean_predicted_score': float(np.mean(to_array(future_predictions['topsis_scores']))),
            'model_contributions': model_contributions
        }
        
        path = self.results_dir / f'forecast_summary_{prediction_year}.json'
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        saved_files['forecast_summary'] = str(path)
        
        return saved_files
    
    def save_config_snapshot(self, config: Any) -> str:
        """
        Save configuration snapshot used for the analysis run.
        
        Parameters
        ----------
        config : Config
            Configuration object used for the analysis
        
        Returns
        -------
        str
            Path to saved config file
        """
        config_data = {
            'timestamp': datetime.now().isoformat(),
            'config': {}
        }
        
        # Extract config as dictionary if possible
        if hasattr(config, 'to_dict'):
            config_data['config'] = config.to_dict()
        elif hasattr(config, '__dict__'):
            # Convert dataclass-like objects
            def serialize(obj):
                if hasattr(obj, '__dict__'):
                    return {k: serialize(v) for k, v in obj.__dict__.items() 
                            if not k.startswith('_')}
                elif hasattr(obj, 'value'):  # Enum
                    return obj.value
                elif isinstance(obj, Path):
                    return str(obj)
                elif isinstance(obj, (list, tuple)):
                    return [serialize(i) for i in obj]
                elif isinstance(obj, dict):
                    return {k: serialize(v) for k, v in obj.items()}
                return obj
            config_data['config'] = serialize(config)
        
        path = self.results_dir / 'config_snapshot.json'
        with open(path, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
        
        return str(path)
    
    def save_figure_manifest(self, figure_paths: List[str]) -> str:
        """
        Save manifest of all generated figures with descriptions.
        
        Parameters
        ----------
        figure_paths : List[str]
            List of paths to generated figures
        
        Returns
        -------
        str
            Path to saved manifest
        """
        # Figure descriptions based on naming convention
        figure_descriptions = {
            # Core MCDM Analysis (01-15)
            '01_score_evolution_top': 'Score evolution over time for top 10 performers',
            '02_score_evolution_bottom': 'Score evolution over time for bottom 10 performers',
            '03_weights_comparison': 'Comparison of criteria weights (Entropy vs CRITIC vs Ensemble)',
            '04_topsis_scores': 'TOPSIS scores bar chart - complete ranking',
            '05_vikor_analysis': 'VIKOR analysis showing Q, S, R values',
            '06_method_agreement': 'MCDM methods ranking agreement (Spearman correlation matrix)',
            '07_score_distribution': 'TOPSIS score distribution histogram',
            '08_sigma_convergence': 'Sigma convergence analysis over time',
            '09_beta_convergence': 'Beta convergence analysis information',
            '10_feature_importance': 'Random Forest feature importance',
            '11_sensitivity_analysis': 'Criteria weight sensitivity analysis',
            '12_final_ranking': 'Final aggregated ranking summary',
            '13_method_comparison': 'MCDM methods ranking comparison (parallel coordinates)',
            '14_ensemble_weights': 'Stacking ensemble model weights',
            '15_future_predictions': 'Future predictions comparison (current vs predicted rankings)',
            
            # ML Process & Progress Visualizations (16-27) - Single Charts
            '16_lstm_training_curve': 'LSTM neural network training progress with loss curves and convergence analysis',
            '17_lstm_actual_vs_predicted': 'LSTM model actual vs predicted values scatter plot with regression analysis',
            '18_lstm_residual_analysis': 'LSTM model residual distribution with statistical bands',
            '19_rf_feature_importance_detailed': 'Random Forest feature importance with cumulative contribution analysis',
            '20_rf_cv_progression': 'Random Forest cross-validation score progression across folds',
            '21_rf_actual_vs_predicted': 'Random Forest model actual vs predicted values with confidence intervals',
            '22_rf_residual_analysis': 'Random Forest model residual analysis with trend detection',
            '23_rf_rank_correlation': 'Random Forest rank prediction accuracy with Spearman correlation',
            '24_model_convergence': 'Multi-metric model convergence behavior analysis',
            '25_ensemble_contribution': 'Ensemble model base model contribution with R² comparison',
            '26_lstm_rank_correlation': 'LSTM model rank prediction accuracy analysis',
            '27_ml_model_comparison': 'Machine learning model performance comparison across metrics',
            
            # Legacy figure names (for backwards compatibility)
            'rf_analysis': 'Random Forest analysis (feature importance, predictions)',
            'cv_results': 'Cross-validation performance results',
            'rf_prediction_analysis': 'Random Forest prediction vs actual analysis',
            'lstm_forecast': 'LSTM time-series forecast results',
            'lstm_training_progress': 'LSTM training loss progression',
            'model_comparison': 'ML model performance comparison',
            'ensemble_model_analysis': 'Stacking ensemble analysis',
            'ml_dashboard': 'Comprehensive ML-MCDM analysis dashboard'
        }
        
        manifest = {
            'timestamp': datetime.now().isoformat(),
            'total_figures': len(figure_paths),
            'output_directory': str(self.figures_dir),
            'figures': []
        }
        
        for path in figure_paths:
            path_obj = Path(path)
            filename = path_obj.stem
            
            # Find matching description
            description = 'No description available'
            for key, desc in figure_descriptions.items():
                if key in filename:
                    description = desc
                    break
            
            manifest['figures'].append({
                'filename': path_obj.name,
                'path': str(path),
                'description': description,
                'format': path_obj.suffix.lstrip('.').upper()
            })
        
        path = self.figures_dir / 'figure_manifest.json'
        with open(path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return str(path)
    
    def save_execution_summary(self, 
                               execution_time: float,
                               phase_timings: Optional[Dict[str, float]] = None) -> str:
        """
        Save machine-readable execution summary.
        
        Parameters
        ----------
        execution_time : float
            Total execution time in seconds
        phase_timings : Dict[str, float], optional
            Timing for each pipeline phase
        
        Returns
        -------
        str
            Path to saved summary
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_execution_time_seconds': execution_time,
            'total_execution_time_formatted': f"{execution_time:.2f}s ({execution_time/60:.2f}min)",
            'phase_timings': phase_timings or {},
            'output_directories': {
                'results': str(self.results_dir),
                'figures': str(self.figures_dir),
                'reports': str(self.reports_dir)
            }
        }
        
        path = self.results_dir / 'execution_summary.json'
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return str(path)
    
    def save_all_results(self,
                         panel_data: Any,
                         weights: Dict[str, np.ndarray],
                         mcdm_results: Dict[str, Any],
                         ml_results: Dict[str, Any],
                         ensemble_results: Dict[str, Any],
                         analysis_results: Dict[str, Any],
                         execution_time: float,
                         future_predictions: Optional[Dict[str, Any]] = None,
                         config: Optional[Any] = None,
                         figure_paths: Optional[List[str]] = None) -> Dict[str, Any]:
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
        
        # Future predictions (2025)
        if future_predictions:
            safe_save('future_predictions', self.save_future_predictions, 
                      future_predictions, panel_data)
        else:
            saved_files['files']['future_predictions'] = {'status': 'skipped - no future predictions'}
        
        safe_save('data_summary', self.save_panel_data_summary, panel_data)
        
        safe_save('report', self.generate_comprehensive_report,
                  panel_data, weights, mcdm_results, ml_results,
                  ensemble_results, analysis_results, execution_time,
                  future_predictions)
        
        # Save config snapshot if provided
        if config is not None:
            safe_save('config_snapshot', self.save_config_snapshot, config)
        
        # Save figure manifest if figure paths provided
        if figure_paths:
            safe_save('figure_manifest', self.save_figure_manifest, figure_paths)
        
        # Save execution summary
        safe_save('execution_summary', self.save_execution_summary, execution_time)
        
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
