# -*- coding: utf-8 -*-
"""Panel data loading with hierarchical structure and adaptive zero handling.

This module handles:
1. Loading multiple CSV files (one per year) from the data folder
2. Hierarchical structure: Subcriteria → Criteria → Final Score
3. Adaptive zero handling: Excludes provinces/subcriteria with 0 values from calculations
4. Composite calculation at each hierarchy level
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .config import Config, get_config
from .logger import get_logger


@dataclass
class HierarchyMapping:
    """Mapping between subcriteria and criteria."""
    subcriteria_to_criteria: Dict[str, str]  # SC01 -> C01
    criteria_to_subcriteria: Dict[str, List[str]]  # C01 -> [SC01, SC02, SC03, SC04]
    criteria_names: Dict[str, str]  # C01 -> "Participation"
    subcriteria_names: Dict[str, str]  # SC01 -> "Civic Knowledge"
    
    @property
    def all_subcriteria(self) -> List[str]:
        return sorted(self.subcriteria_to_criteria.keys())
    
    @property
    def all_criteria(self) -> List[str]:
        return sorted(self.criteria_to_subcriteria.keys())


@dataclass
class PanelData:
    """Container for hierarchical panel data with multiple views."""
    # Raw subcriteria data
    subcriteria_long: pd.DataFrame  # Long format: (n*T) × (2 + K_sub)
    subcriteria_cross_section: Dict[int, pd.DataFrame]  # Year → subcriteria data
    
    # Aggregated criteria data (calculated from subcriteria)
    criteria_long: pd.DataFrame  # Long format: (n*T) × (2 + K_criteria)
    criteria_cross_section: Dict[int, pd.DataFrame]  # Year → criteria data
    
    # Final scores (calculated from criteria)
    final_long: pd.DataFrame  # Long format: (n*T) × 3 (Year, Province, FinalScore)
    final_cross_section: Dict[int, pd.DataFrame]  # Year → final scores
    
    # Metadata
    provinces: List[str]
    years: List[int]
    hierarchy: HierarchyMapping
    
    # Data availability tracking (for adaptive calculations)
    availability: Dict  # Tracks which provinces/criteria have data each year
    
    @property
    def n_provinces(self) -> int:
        return len(self.provinces)
    
    @property
    def n_years(self) -> int:
        return len(self.years)
    
    @property
    def n_subcriteria(self) -> int:
        return len(self.hierarchy.all_subcriteria)
    
    @property
    def n_criteria(self) -> int:
        return len(self.hierarchy.all_criteria)
    
    def get_subcriteria_year(self, year: int) -> pd.DataFrame:
        """Get subcriteria data for specific year."""
        return self.subcriteria_cross_section[year]
    
    def get_criteria_year(self, year: int) -> pd.DataFrame:
        """Get criteria data for specific year."""
        return self.criteria_cross_section[year]
    
    def get_final_year(self, year: int) -> pd.DataFrame:
        """Get final scores for specific year."""
        return self.final_cross_section[year]
    
    def get_latest(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Get latest year data (subcriteria, criteria, final)."""
        latest_year = max(self.years)
        return (
            self.subcriteria_cross_section[latest_year],
            self.criteria_cross_section[latest_year],
            self.final_cross_section[latest_year]
        )


class DataLoader:
    """Loads panel data from year-based CSV files with hierarchical structure."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.logger = get_logger()
    
    def load(self) -> PanelData:
        """Load panel data from data folder."""
        data_dir = self.config.paths.data_dir
        
        self.logger.info(f"Loading data from {data_dir}")
        
        # Load hierarchy mapping from codebook
        hierarchy = self._load_hierarchy_mapping(data_dir)
        
        # Load yearly data files
        yearly_data = self._load_yearly_files(data_dir, hierarchy)
        
        # Create hierarchical views
        panel_data = self._create_hierarchical_views(yearly_data, hierarchy)
        
        self.logger.info(f"✓ Loaded: {panel_data.n_provinces} provinces, "
                        f"{panel_data.n_years} years, "
                        f"{panel_data.n_subcriteria} subcriteria, "
                        f"{panel_data.n_criteria} criteria")
        
        return panel_data
    
    def _load_hierarchy_mapping(self, data_dir: Path) -> HierarchyMapping:
        """Load hierarchy mapping from codebook files."""
        codebook_dir = data_dir / "codebook"
        
        # Load subcriteria codebook
        subcriteria_file = codebook_dir / "codebook_subcriteria.csv"
        if not subcriteria_file.exists():
            raise FileNotFoundError(f"Subcriteria codebook not found: {subcriteria_file}")
        
        sub_df = pd.read_csv(subcriteria_file)
        
        # Load criteria codebook
        criteria_file = codebook_dir / "codebook_criteria.csv"
        if not criteria_file.exists():
            raise FileNotFoundError(f"Criteria codebook not found: {criteria_file}")
        
        crit_df = pd.read_csv(criteria_file)
        
        # Build mappings
        subcriteria_to_criteria = dict(zip(sub_df['Variable_Code'], sub_df['Criteria_Code']))
        subcriteria_names = dict(zip(sub_df['Variable_Code'], sub_df['Variable_Name']))
        criteria_names = dict(zip(crit_df['Variable_Code'], crit_df['Variable_Name']))
        
        # Build reverse mapping
        criteria_to_subcriteria = {}
        for sc, c in subcriteria_to_criteria.items():
            if c not in criteria_to_subcriteria:
                criteria_to_subcriteria[c] = []
            criteria_to_subcriteria[c].append(sc)
        
        # Sort subcriteria within each criterion
        for c in criteria_to_subcriteria:
            criteria_to_subcriteria[c] = sorted(criteria_to_subcriteria[c])
        
        self.logger.info(f"✓ Loaded hierarchy: {len(criteria_to_subcriteria)} criteria, "
                        f"{len(subcriteria_to_criteria)} subcriteria")
        
        return HierarchyMapping(
            subcriteria_to_criteria=subcriteria_to_criteria,
            criteria_to_subcriteria=criteria_to_subcriteria,
            criteria_names=criteria_names,
            subcriteria_names=subcriteria_names
        )
    
    def _load_yearly_files(self, data_dir: Path, hierarchy: HierarchyMapping) -> Dict[int, pd.DataFrame]:
        """Load all yearly CSV files from data directory."""
        yearly_data = {}
        
        # Find all year CSV files
        year_files = sorted(data_dir.glob("[0-9][0-9][0-9][0-9].csv"))
        
        if not year_files:
            raise FileNotFoundError(f"No yearly CSV files found in {data_dir}")
        
        for year_file in year_files:
            # Extract year from filename
            year = int(year_file.stem)
            
            # Load CSV
            df = pd.read_csv(year_file)
            
            # Validate structure
            if 'Province' not in df.columns:
                raise ValueError(f"'Province' column not found in {year_file}")
            
            # Check for subcriteria columns
            expected_subcriteria = hierarchy.all_subcriteria
            missing_cols = set(expected_subcriteria) - set(df.columns)
            if missing_cols:
                self.logger.warning(f"Year {year}: Missing subcriteria columns: {missing_cols}")
            
            # Add Year column
            df.insert(0, 'Year', year)
            
            yearly_data[year] = df
            self.logger.info(f"  Loaded {year}: {len(df)} provinces, {len(df.columns)-2} subcriteria")
        
        return yearly_data
    
    def _create_hierarchical_views(
        self, 
        yearly_data: Dict[int, pd.DataFrame], 
        hierarchy: HierarchyMapping
    ) -> PanelData:
        """Create hierarchical panel data with adaptive composite calculations."""
        years = sorted(yearly_data.keys())
        
        # Get all provinces across all years
        all_provinces = set()
        for df in yearly_data.values():
            all_provinces.update(df['Province'].unique())
        provinces = sorted(all_provinces)
        
        # Initialize containers
        subcriteria_data = []
        criteria_data = []
        final_data = []
        
        subcriteria_cross = {}
        criteria_cross = {}
        final_cross = {}
        
        availability = {
            'province_by_year': {},
            'subcriteria_by_year': {},
            'criteria_by_year': {}
        }
        
        # Process each year
        for year in years:
            df_year = yearly_data[year]
            
            # Extract subcriteria data
            subcriteria_cols = [c for c in hierarchy.all_subcriteria if c in df_year.columns]
            df_sub = df_year[['Year', 'Province'] + subcriteria_cols].copy()
            
            # Track availability
            provinces_with_data = df_sub[df_sub[subcriteria_cols].sum(axis=1) > 0]['Province'].tolist()
            availability['province_by_year'][year] = provinces_with_data
            
            # Calculate criteria from subcriteria (adaptive - exclude zeros)
            criteria_values = {}
            criteria_availability = {}
            
            for criterion, subcrit_list in hierarchy.criteria_to_subcriteria.items():
                available_subcrit = [sc for sc in subcrit_list if sc in subcriteria_cols]
                
                if not available_subcrit:
                    criteria_values[criterion] = pd.Series(0.0, index=df_sub.index)
                    criteria_availability[criterion] = []
                    continue
                
                criterion_scores = []
                provinces_with_criterion = []
                
                for idx, row in df_sub.iterrows():
                    province = row['Province']
                    sub_values = [row[sc] for sc in available_subcrit if sc in row.index]
                    non_zero_values = [v for v in sub_values if v > 0]
                    
                    if non_zero_values:
                        criterion_score = np.mean(non_zero_values)
                        provinces_with_criterion.append(province)
                    else:
                        criterion_score = 0.0
                    
                    criterion_scores.append(criterion_score)
                
                criteria_values[criterion] = pd.Series(criterion_scores, index=df_sub.index)
                criteria_availability[criterion] = provinces_with_criterion
            
            # Create criteria DataFrame
            df_criteria = pd.DataFrame({
                'Year': df_sub['Year'],
                'Province': df_sub['Province'],
                **criteria_values
            })
            
            availability['criteria_by_year'][year] = criteria_availability
            
            # Calculate final score from criteria (adaptive - exclude zeros)
            final_scores = []
            
            for idx, row in df_criteria.iterrows():
                crit_values = [row[c] for c in hierarchy.all_criteria if c in df_criteria.columns]
                non_zero_criteria = [v for v in crit_values if v > 0]
                final_scores.append(np.mean(non_zero_criteria) if non_zero_criteria else 0.0)
            
            # Create final score DataFrame
            df_final = pd.DataFrame({
                'Year': df_criteria['Year'],
                'Province': df_criteria['Province'],
                'FinalScore': final_scores
            })
            
            # Append to long format
            subcriteria_data.append(df_sub)
            criteria_data.append(df_criteria)
            final_data.append(df_final)
            
            # Store cross-sections
            subcriteria_cross[year] = df_sub.set_index('Province')[subcriteria_cols]
            criteria_cross[year] = df_criteria.set_index('Province')[hierarchy.all_criteria]
            final_cross[year] = df_final.set_index('Province')[['FinalScore']]
        
        # Concatenate long format data
        subcriteria_long = pd.concat(subcriteria_data, ignore_index=True)
        criteria_long = pd.concat(criteria_data, ignore_index=True)
        final_long = pd.concat(final_data, ignore_index=True)
        
        return PanelData(
            subcriteria_long=subcriteria_long,
            subcriteria_cross_section=subcriteria_cross,
            criteria_long=criteria_long,
            criteria_cross_section=criteria_cross,
            final_long=final_long,
            final_cross_section=final_cross,
            provinces=provinces,
            years=years,
            hierarchy=hierarchy,
            availability=availability
        )


def load_data() -> PanelData:
    """Convenience function to load panel data."""
    loader = DataLoader()
    return loader.load()
