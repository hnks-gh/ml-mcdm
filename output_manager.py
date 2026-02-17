# -*- coding: utf-8 -*-
"""
Output Management for ML-MCDM Analysis Results
================================================

Provides the ``OutputManager`` class for persisting analysis artefacts
(CSV, JSON, text report) into an organised directory structure::

    outputs/
    ├── results/   — numerical data  (CSV, JSON)
    ├── figures/   — visualisation charts  (PNG)
    └── reports/   — comprehensive text report

This module is *not* called by the pipeline directly; the pipeline does
its own inline saving via ``_save_all_results``.  ``OutputManager`` is
kept available for advanced / ad-hoc usage and backward compatibility.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json


# =========================================================================
# Utility
# =========================================================================

def to_array(x: Any) -> np.ndarray:
    """Convert Series / list / scalar to plain ndarray."""
    if x is None:
        return np.array([])
    if hasattr(x, 'values'):
        return x.values
    return np.asarray(x) if not isinstance(x, np.ndarray) else x


# =========================================================================
# OutputManager
# =========================================================================

class OutputManager:
    """
    Manages structured output to ``results/``, ``figures/``, ``reports/``.

    Updated for the IFS + Evidential Reasoning architecture.
    """

    def __init__(self, base_output_dir: str = 'outputs'):
        self.base_dir = Path(base_output_dir)
        self.results_dir = self.base_dir / 'results'
        self.figures_dir = self.base_dir / 'figures'
        self.reports_dir = self.base_dir / 'reports'
        self._setup_directories()
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def _setup_directories(self) -> None:
        for d in [self.results_dir, self.figures_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Weight export
    # -----------------------------------------------------------------

    def save_weights(
        self,
        weights: Dict[str, np.ndarray],
        subcriteria_names: List[str],
    ) -> str:
        """Save subcriteria weights from all four methods + fused."""
        df = pd.DataFrame({'Subcriteria': subcriteria_names})
        for method, w in weights.items():
            if isinstance(w, np.ndarray) and len(w) == len(subcriteria_names):
                df[method.title()] = w
        df = df.sort_values(
            df.columns[-1], ascending=False
        ).reset_index(drop=True)
        path = self.results_dir / 'weights_analysis.csv'
        df.to_csv(path, index=False, float_format='%.6f')
        return str(path)

    # -----------------------------------------------------------------
    # Ranking export
    # -----------------------------------------------------------------

    def save_rankings(
        self,
        ranking_result: Any,
        provinces: List[str],
    ) -> str:
        """Save final ER rankings to CSV."""
        df = pd.DataFrame({
            'Province': ranking_result.final_ranking.index,
            'ER_Score': ranking_result.final_scores.values,
            'ER_Rank': ranking_result.final_ranking.values,
        }).sort_values('ER_Rank').reset_index(drop=True)
        df.insert(0, 'Rank', range(1, len(df) + 1))
        path = self.results_dir / 'final_rankings.csv'
        df.to_csv(path, index=False, float_format='%.6f')
        return str(path)

    # -----------------------------------------------------------------
    # MCDM scores per criterion
    # -----------------------------------------------------------------

    def save_mcdm_scores_by_criterion(
        self,
        ranking_result: Any,
        provinces: List[str],
    ) -> Dict[str, str]:
        """Save per-criterion MCDM method scores."""
        saved = {}
        for crit_id, method_scores in ranking_result.criterion_method_scores.items():
            df = pd.DataFrame(method_scores)
            df.index = provinces
            df.index.name = 'Province'
            path = self.results_dir / f'mcdm_scores_{crit_id}.csv'
            df.to_csv(path, float_format='%.6f')
            saved[crit_id] = str(path)
        return saved

    # -----------------------------------------------------------------
    # Forecasting results
    # -----------------------------------------------------------------

    def save_forecast_results(
        self,
        forecast_result: Any,
    ) -> Dict[str, str]:
        """
        Save ML forecasting results from UnifiedForecaster.
        
        Saves:
        - Predictions with uncertainty intervals
        - Model weights from Super Learner
        - Feature importance aggregated across models
        - Cross-validation metrics
        - Model performance diagnostics
        """
        saved: Dict[str, str] = {}
        
        # 1. Predictions with intervals
        if hasattr(forecast_result, 'predictions'):
            pred_df = forecast_result.predictions.copy()
            if hasattr(forecast_result, 'lower_bound'):
                pred_df = pd.concat([
                    pred_df,
                    forecast_result.lower_bound.add_suffix('_lower'),
                    forecast_result.upper_bound.add_suffix('_upper')
                ], axis=1)
            path = self.results_dir / 'forecast_predictions.csv'
            pred_df.to_csv(path, float_format='%.6f')
            saved['predictions'] = str(path)
        
        # 2. Model weights from Super Learner
        if hasattr(forecast_result, 'model_weights'):
            weights_df = pd.DataFrame([
                {'Model': k, 'Weight': v}
                for k, v in sorted(
                    forecast_result.model_weights.items(),
                    key=lambda x: x[1], reverse=True
                )
            ])
            path = self.results_dir / 'forecast_model_weights.csv'
            weights_df.to_csv(path, index=False, float_format='%.6f')
            saved['model_weights'] = str(path)
        
        # 3. Feature importance
        if hasattr(forecast_result, 'feature_importance'):
            imp_df = forecast_result.feature_importance.copy()
            path = self.results_dir / 'forecast_feature_importance.csv'
            imp_df.to_csv(path, float_format='%.6f')
            saved['feature_importance'] = str(path)
        
        # 4. CV metrics
        if hasattr(forecast_result, 'cv_metrics'):
            cv_df = pd.DataFrame([forecast_result.cv_metrics])
            path = self.results_dir / 'forecast_cv_metrics.csv'
            cv_df.to_csv(path, index=False, float_format='%.6f')
            saved['cv_metrics'] = str(path)
        
        return saved

    # -----------------------------------------------------------------
    # Analysis results
    # -----------------------------------------------------------------

    def save_analysis_results(
        self,
        analysis_results: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Save analysis results including hierarchical sensitivity analysis.
        
        Saves comprehensive results:
        - Subcriteria sensitivity (28 subcriteria)
        - Criteria sensitivity (8 criteria)
        - Temporal stability (year-to-year correlation)
        - IFS uncertainty sensitivity
        - Forecast robustness
        - Overall robustness score
        """
        saved: Dict[str, str] = {}
        sens = analysis_results.get('sensitivity')
        
        if sens is None:
            return saved
        
        # Save subcriteria sensitivity (main sensitivity analysis)
        if hasattr(sens, 'subcriteria_sensitivity') and sens.subcriteria_sensitivity:
            df = pd.DataFrame([
                {'Subcriterion': k, 'Sensitivity': v}
                for k, v in sorted(
                    sens.subcriteria_sensitivity.items(),
                    key=lambda x: x[1], reverse=True,
                )
            ])
            path = self.results_dir / 'sensitivity_subcriteria.csv'
            df.to_csv(path, index=False, float_format='%.6f')
            saved['sensitivity_subcriteria'] = str(path)
        
        # Save criteria sensitivity
        if hasattr(sens, 'criteria_sensitivity') and sens.criteria_sensitivity:
            df = pd.DataFrame([
                {'Criterion': k, 'Sensitivity': v}
                for k, v in sorted(
                    sens.criteria_sensitivity.items(),
                    key=lambda x: x[1], reverse=True,
                )
            ])
            path = self.results_dir / 'sensitivity_criteria.csv'
            df.to_csv(path, index=False, float_format='%.6f')
            saved['sensitivity_criteria'] = str(path)
        
        # Save temporal stability
        if hasattr(sens, 'temporal_stability') and sens.temporal_stability:
            df = pd.DataFrame([
                {'YearPair': k, 'Correlation': v}
                for k, v in sens.temporal_stability.items()
            ])
            path = self.results_dir / 'temporal_stability.csv'
            df.to_csv(path, index=False, float_format='%.6f')
            saved['temporal_stability'] = str(path)
        
        # Save top-N stability
        if hasattr(sens, 'top_n_stability') and sens.top_n_stability:
            df = pd.DataFrame([
                {'TopN': k, 'Stability': v}
                for k, v in sorted(sens.top_n_stability.items())
            ])
            path = self.results_dir / 'top_n_stability.csv'
            df.to_csv(path, index=False, float_format='%.6f')
            saved['top_n_stability'] = str(path)
        
        # Save IFS sensitivity
        if hasattr(sens, 'ifs_membership_sensitivity'):
            ifs_df = pd.DataFrame([{
                'IFS_Membership_Sensitivity': sens.ifs_membership_sensitivity,
                'IFS_NonMembership_Sensitivity': getattr(sens, 'ifs_nonmembership_sensitivity', 0.0),
            }])
            path = self.results_dir / 'ifs_sensitivity.csv'
            ifs_df.to_csv(path, index=False, float_format='%.6f')
            saved['ifs_sensitivity'] = str(path)
        
        # Save overall robustness summary
        if hasattr(sens, 'overall_robustness'):
            robust_df = pd.DataFrame([{
                'Overall_Robustness': sens.overall_robustness,
                'Confidence_Level': getattr(sens, 'confidence_level', 0.95),
            }])
            path = self.results_dir / 'robustness_summary.csv'
            robust_df.to_csv(path, index=False, float_format='%.6f')
            saved['robustness'] = str(path)
        
        # Legacy fallback for old sensitivity results
        if hasattr(sens, 'weight_sensitivity') and not hasattr(sens, 'subcriteria_sensitivity'):
            df = pd.DataFrame([
                {'Criterion': k, 'Sensitivity': v}
                for k, v in sorted(
                    sens.weight_sensitivity.items(),
                    key=lambda x: x[1], reverse=True,
                )
            ])
            path = self.results_dir / 'sensitivity_analysis.csv'
            df.to_csv(path, index=False, float_format='%.6f')
            saved['sensitivity'] = str(path)
        
        return saved

    # -----------------------------------------------------------------
    # Execution summary
    # -----------------------------------------------------------------

    def save_execution_summary(
        self,
        execution_time: float,
    ) -> str:
        summary = {
            'timestamp': datetime.now().isoformat(),
            'execution_time_seconds': round(execution_time, 2),
        }
        path = self.results_dir / 'execution_summary.json'
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)
        return str(path)

    # -----------------------------------------------------------------
    # Config snapshot
    # -----------------------------------------------------------------

    def save_config_snapshot(self, config: Any) -> str:
        path = self.results_dir / 'config_snapshot.json'
        config.save(path)
        return str(path)


# =========================================================================
# Factory
# =========================================================================

def create_output_manager(output_dir: str = 'outputs') -> OutputManager:
    """Factory function to create an OutputManager."""
    return OutputManager(output_dir)
