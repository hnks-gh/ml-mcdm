# -*- coding: utf-8 -*-
"""
Panel Data Loader and Preprocessor
===================================

Handles loading, validation, and preprocessing of panel data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from .config import Config, get_config, PanelDataConfig
from .logger import get_logger


@dataclass
class PanelData:
    """Container for panel data with multiple views."""
    long: pd.DataFrame          # Long format: (n*T) × (2 + K)
    wide: pd.DataFrame          # Wide format: n × (K*T)
    cross_section: Dict[int, pd.DataFrame]  # Year → cross-section
    provinces: List[str]
    years: List[int]
    components: List[str]
    
    @property
    def entities(self) -> List[str]:
        """Alias for provinces."""
        return self.provinces
    
    @property
    def time_periods(self) -> List[int]:
        """Alias for years."""
        return self.years
    
    @property
    def n_provinces(self) -> int:
        return len(self.provinces)
    
    @property
    def n_years(self) -> int:
        return len(self.years)
    
    @property
    def n_components(self) -> int:
        return len(self.components)
    
    @property
    def n_observations(self) -> int:
        return self.n_provinces * self.n_years
    
    def get_year(self, year: int) -> pd.DataFrame:
        """Get cross-sectional data for specific year."""
        return self.cross_section[year]
    
    def get_province(self, province: str) -> pd.DataFrame:
        """Get time-series data for specific province."""
        return self.long[self.long['Province'] == province].set_index('Year')[self.components]
    
    def get_component(self, component: str) -> pd.DataFrame:
        """Get panel data for specific component."""
        return self.long.pivot(index='Province', columns='Year', values=component)
    
    def get_latest(self) -> pd.DataFrame:
        """Get latest year cross-section."""
        return self.cross_section[max(self.years)]


class PanelDataLoader:
    """Loads and validates panel data from various formats."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.logger = get_logger()
        self._panel_config = self.config.panel
    
    def load(self, filepath: Optional[Path] = None) -> PanelData:
        """Load panel data from file or generate if not exists."""
        filepath = filepath or self.config.paths.data_file
        
        # Convert to Path if string
        if isinstance(filepath, str):
            filepath = Path(filepath)
        
        if not filepath.exists():
            self.logger.warning(f"Panel data not found at {filepath}")
            self.logger.info("Generating new panel data...")
            self._generate_panel_data(filepath)
        
        self.logger.info(f"Loading panel data from {filepath}")
        df = pd.read_csv(filepath)
        
        # Validate and process
        df = self._validate_structure(df)
        df = self._preprocess(df)
        
        # Create multiple views
        panel_data = self._create_views(df)
        
        self.logger.info(f"✓ Loaded: {panel_data.n_provinces} provinces, "
                        f"{panel_data.n_years} years, {panel_data.n_components} components")
        
        return panel_data
    
    def _validate_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate panel data structure."""
        required_cols = [self._panel_config.year_col, self._panel_config.province_col]
        required_cols += self._panel_config.component_cols
        
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Check for duplicates
        dups = df.duplicated(subset=[self._panel_config.year_col, 
                                     self._panel_config.province_col])
        if dups.any():
            raise ValueError(f"Found {dups.sum()} duplicate province-year combinations")
        
        # Check balanced panel
        provinces = df[self._panel_config.province_col].unique()
        years = df[self._panel_config.year_col].unique()
        expected_rows = len(provinces) * len(years)
        
        if len(df) != expected_rows:
            self.logger.warning(f"Unbalanced panel: {len(df)} rows, expected {expected_rows}")
        
        return df
    
    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess panel data."""
        # Sort by province and year
        df = df.sort_values([self._panel_config.province_col, 
                           self._panel_config.year_col]).reset_index(drop=True)
        
        # Handle missing values
        component_cols = self._panel_config.component_cols
        missing_before = df[component_cols].isna().sum().sum()
        
        if missing_before > 0:
            self.logger.warning(f"Found {missing_before} missing values")
            # Forward fill within each province, then backward fill
            df[component_cols] = df.groupby(self._panel_config.province_col)[component_cols].transform(
                lambda x: x.ffill().bfill()
            )
            # If still missing, fill with mean
            df[component_cols] = df[component_cols].fillna(df[component_cols].mean())
        
        # Ensure numeric
        df[component_cols] = df[component_cols].apply(pd.to_numeric, errors='coerce')
        
        # Clip to [0, 1] if normalized data
        if df[component_cols].max().max() <= 1.0 and df[component_cols].min().min() >= 0.0:
            df[component_cols] = df[component_cols].clip(0, 1)
        
        return df
    
    def _create_views(self, df: pd.DataFrame) -> PanelData:
        """Create multiple data views for different analyses."""
        province_col = self._panel_config.province_col
        year_col = self._panel_config.year_col
        component_cols = self._panel_config.component_cols
        
        provinces = sorted(df[province_col].unique().tolist())
        years = sorted(df[year_col].unique().tolist())
        
        # Long format (already have it)
        long = df.copy()
        
        # Wide format
        wide_parts = []
        for year in years:
            year_df = df[df[year_col] == year].set_index(province_col)[component_cols]
            year_df.columns = [f"{c}_{year}" for c in component_cols]
            wide_parts.append(year_df)
        wide = pd.concat(wide_parts, axis=1).reset_index()
        
        # Cross-sections by year
        cross_section = {}
        for year in years:
            year_df = df[df[year_col] == year].set_index(province_col)[component_cols]
            cross_section[year] = year_df
        
        return PanelData(
            long=long,
            wide=wide,
            cross_section=cross_section,
            provinces=provinces,
            years=years,
            components=component_cols
        )
    
    def _generate_panel_data(self, filepath: Path) -> None:
        """Generate synthetic panel data."""
        np.random.seed(self.config.random.seed)
        
        n_provinces = self._panel_config.n_provinces
        n_components = self._panel_config.n_components
        years = self._panel_config.years
        
        provinces = [f'P{i+1:02d}' for i in range(n_provinces)]
        components = self._panel_config.component_cols
        
        # Generate base scores with regional heterogeneity
        base_scores = np.zeros((n_provinces, n_components))
        
        # Divide provinces into 3 regions proportionally
        n_high = max(1, n_provinces // 4)      # ~25% high performers
        n_mid = max(1, n_provinces // 2)       # ~50% medium performers
        n_low = n_provinces - n_high - n_mid   # ~25% low performers
        
        # Region 1: High performers
        base_scores[0:n_high, :] = np.random.uniform(0.6, 0.85, (n_high, n_components))
        # Region 2: Medium performers
        base_scores[n_high:n_high+n_mid, :] = np.random.uniform(0.4, 0.65, (n_mid, n_components))
        # Region 3: Low performers
        if n_low > 0:
            base_scores[n_high+n_mid:, :] = np.random.uniform(0.25, 0.50, (n_low, n_components))
        
        # Add component-specific noise (only for available components)
        n_comp = min(n_components, 20)
        base_scores[:, :min(5, n_comp)] += np.random.normal(0, 0.08, (n_provinces, min(5, n_comp)))
        if n_comp > 5:
            base_scores[:, 5:min(10, n_comp)] += np.random.normal(0.02, 0.05, (n_provinces, min(5, n_comp-5)))
        if n_comp > 10:
            base_scores[:, 10:min(15, n_comp)] += np.random.normal(0, 0.03, (n_provinces, min(5, n_comp-10)))
        if n_comp > 15:
            base_scores[:, 15:min(20, n_comp)] += np.random.normal(0.01, 0.04, (n_provinces, min(5, n_comp-15)))
        base_scores = np.clip(base_scores, 0, 1)
        
        all_data = []
        for t, year in enumerate(years):
            year_data = base_scores.copy()
            
            # Trend
            trend = 0.005 * t
            year_data += trend
            
            # COVID shock (2020-2021) - apply only to available components
            if year in [2020, 2021]:
                covid_impact = -0.03 * (2021 - year + 1)
                # Economic components (5:10) if they exist
                eco_start, eco_end = 5, min(10, n_components)
                if eco_end > eco_start:
                    eco_cols = eco_end - eco_start
                    year_data[:, eco_start:eco_end] += covid_impact * np.random.uniform(0.8, 1.2, (n_provinces, eco_cols))
                # Social components (10:15) if they exist
                soc_start, soc_end = 10, min(15, n_components)
                if soc_end > soc_start:
                    soc_cols = soc_end - soc_start
                    year_data[:, soc_start:soc_end] += covid_impact * 0.5 * np.random.uniform(0.8, 1.2, (n_provinces, soc_cols))
            
            # Recovery (2022+) - apply only to available components
            if year >= 2022:
                recovery = 0.02 * (year - 2021)
                rec_start, rec_end = 5, min(15, n_components)
                if rec_end > rec_start:
                    rec_cols = rec_end - rec_start
                    year_data[:, rec_start:rec_end] += recovery * np.random.uniform(0.9, 1.1, (n_provinces, rec_cols))
            
            # Green transition - apply only to available components
            green_trend = 0.008 * t
            green_end = min(5, n_components)
            if green_end > 0:
                year_data[:, 0:green_end] += green_trend * np.random.uniform(0.8, 1.2, (n_provinces, green_end))
            
            # Random shocks
            shocks = np.random.normal(0, 0.02, (n_provinces, n_components))
            year_data += shocks
            year_data = np.clip(year_data, 0, 1)
            
            df_year = pd.DataFrame(year_data, columns=components)
            df_year.insert(0, 'Year', year)
            df_year.insert(1, 'Province', provinces)
            all_data.append(df_year)
        
        df_panel = pd.concat(all_data, ignore_index=True)
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df_panel.to_csv(filepath, index=False)
        self.logger.info(f"✓ Generated panel data: {filepath}")
    
    def generate_synthetic(self, 
                          n_provinces: int = None, 
                          n_years: int = None,
                          n_components: int = None) -> PanelData:
        """
        Generate synthetic panel data and return as PanelData object.
        
        Parameters
        ----------
        n_provinces : int, optional
            Number of provinces
        n_years : int, optional
            Number of years  
        n_components : int, optional
            Number of components
        """
        # Update config if parameters provided
        if n_provinces is not None:
            self._panel_config.n_provinces = n_provinces
        if n_years is not None:
            self._panel_config.years = list(range(2020, 2020 + n_years))
        if n_components is not None:
            self._panel_config.n_components = n_components
        
        # Generate to temp file and load
        import tempfile
        temp_file = Path(tempfile.gettempdir()) / 'ml_topsis_panel.csv'
        self._generate_panel_data(temp_file)
        return self.load(temp_file)


class TemporalFeatureEngineer:
    """Creates temporal features for ML models."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
    
    def create_lag_features(self, data: PanelData, n_lags: int = 2) -> pd.DataFrame:
        """Create lagged features for each component."""
        df = data.long.copy()
        component_cols = data.components
        
        for lag in range(1, n_lags + 1):
            for col in component_cols:
                df[f'{col}_lag{lag}'] = df.groupby('Province')[col].shift(lag)
        
        return df.dropna()
    
    def create_rolling_features(self, data: PanelData, window: int = 2) -> pd.DataFrame:
        """Create rolling statistics features."""
        df = data.long.copy()
        component_cols = data.components
        
        for col in component_cols:
            df[f'{col}_roll_mean'] = df.groupby('Province')[col].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f'{col}_roll_std'] = df.groupby('Province')[col].transform(
                lambda x: x.rolling(window, min_periods=1).std().fillna(0)
            )
        
        return df
    
    def create_growth_features(self, data: PanelData) -> pd.DataFrame:
        """Create year-over-year growth features."""
        df = data.long.copy()
        component_cols = data.components
        
        for col in component_cols:
            df[f'{col}_growth'] = df.groupby('Province')[col].pct_change().fillna(0)
        
        return df
    
    def create_all_features(self, data: PanelData) -> pd.DataFrame:
        """Create comprehensive feature set for ML."""
        rf_config = self.config.random_forest
        
        df = data.long.copy()
        
        if rf_config.use_lags:
            lag_df = self.create_lag_features(data, rf_config.n_lags)
            lag_cols = [c for c in lag_df.columns if 'lag' in c]
            df = df.merge(lag_df[['Province', 'Year'] + lag_cols], 
                         on=['Province', 'Year'], how='left')
        
        if rf_config.use_rolling_features:
            roll_df = self.create_rolling_features(data, rf_config.rolling_window)
            roll_cols = [c for c in roll_df.columns if 'roll' in c]
            df = df.merge(roll_df[['Province', 'Year'] + roll_cols],
                         on=['Province', 'Year'], how='left')
        
        growth_df = self.create_growth_features(data)
        growth_cols = [c for c in growth_df.columns if 'growth' in c]
        df = df.merge(growth_df[['Province', 'Year'] + growth_cols],
                     on=['Province', 'Year'], how='left')
        
        return df.dropna()


def load_panel_data(filepath: Optional[Path] = None) -> PanelData:
    """Convenience function to load panel data."""
    loader = PanelDataLoader()
    return loader.load(filepath)
