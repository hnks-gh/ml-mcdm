# -*- coding: utf-8 -*-
"""
Rough Set Theory - Attribute Reduction
=======================================

Reduces dimensionality by finding minimal attribute subsets that preserve
classification ability.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from itertools import combinations


@dataclass
class RoughSetResult:
    """Result container for Rough Set reduction."""
    core_attributes: List[str]           # Indispensable attributes
    reducts: List[List[str]]             # All minimal reducts
    best_reduct: List[str]               # Selected best reduct
    original_n_attributes: int
    reduced_n_attributes: int
    quality_original: float              # Quality of approximation (original)
    quality_reduced: float               # Quality of approximation (reduced)
    dependency_degree: float
    boundary_objects: List[str]          # Objects in boundary region
    
    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            "ROUGH SET ATTRIBUTE REDUCTION",
            f"{'='*60}",
            f"\nOriginal attributes: {self.original_n_attributes}",
            f"Reduced attributes: {self.reduced_n_attributes}",
            f"Reduction rate: {(1 - self.reduced_n_attributes/self.original_n_attributes)*100:.1f}%",
            f"\nCore attributes (indispensable): {len(self.core_attributes)}",
            f"  {self.core_attributes}",
            f"\nBest reduct: {self.best_reduct}",
            f"\nQuality measures:",
            f"  Original quality: {self.quality_original:.4f}",
            f"  Reduced quality: {self.quality_reduced:.4f}",
            f"  Dependency degree: {self.dependency_degree:.4f}",
            f"\nBoundary objects: {len(self.boundary_objects)}",
            "=" * 60
        ]
        return "\n".join(lines)


class RoughSetReducer:
    """
    Rough Set Theory based attribute reduction.
    
    Finds minimal attribute subsets that preserve the same classification
    ability as the full attribute set.
    """
    
    def __init__(self,
                 quality_threshold: float = 0.95,
                 n_bins: int = 5,
                 max_reducts: int = 5,
                 method: str = 'heuristic'):
        """
        Initialize Rough Set Reducer.
        
        Parameters
        ----------
        quality_threshold : float
            Minimum quality of approximation to accept (0-1)
        n_bins : int
            Number of bins for discretization
        max_reducts : int
            Maximum number of reducts to find
        method : str
            'heuristic' (fast), 'exhaustive' (complete), or 'genetic'
        """
        self.quality_threshold = quality_threshold
        self.n_bins = n_bins
        self.max_reducts = max_reducts
        self.method = method
    
    def reduce(self,
              data: pd.DataFrame,
              decision_col: Optional[str] = None,
              condition_cols: Optional[List[str]] = None) -> RoughSetResult:
        """
        Perform attribute reduction using Rough Set Theory.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data with condition attributes and decision attribute
        decision_col : str
            Decision attribute column (if None, creates from clustering)
        condition_cols : List[str]
            Condition attributes (if None, uses all except decision)
        """
        df = data.copy()
        
        # Setup decision attribute
        if decision_col is None:
            # Create decision from clustering/quantiles
            composite = df.mean(axis=1)
            df['_decision'] = pd.qcut(composite, q=4, labels=['Low', 'Medium-Low', 
                                                               'Medium-High', 'High'])
            decision_col = '_decision'
        
        # Setup condition attributes
        if condition_cols is None:
            condition_cols = [c for c in df.columns if c != decision_col]
        
        # Reset index to ensure integer indexing works consistently
        df = df.reset_index(drop=True)
        
        # Discretize continuous attributes
        df_discrete = self._discretize(df, condition_cols)
        
        # Calculate indiscernibility relations and approximations
        objects = df.index.tolist()  # Now always 0, 1, 2, ...
        
        # Find core (indispensable attributes)
        core = self._find_core(df_discrete, condition_cols, decision_col, objects)
        
        # Find reducts
        if self.method == 'heuristic':
            reducts = self._find_reducts_heuristic(
                df_discrete, condition_cols, decision_col, objects, core
            )
        elif self.method == 'exhaustive':
            reducts = self._find_reducts_exhaustive(
                df_discrete, condition_cols, decision_col, objects
            )
        else:
            reducts = self._find_reducts_heuristic(
                df_discrete, condition_cols, decision_col, objects, core
            )
        
        # Select best reduct (shortest that meets quality threshold)
        best_reduct = self._select_best_reduct(
            df_discrete, reducts, decision_col, objects
        )
        
        # Calculate quality measures
        quality_original = self._calculate_quality(
            df_discrete, condition_cols, decision_col, objects
        )
        quality_reduced = self._calculate_quality(
            df_discrete, best_reduct, decision_col, objects
        )
        
        # Find boundary objects
        boundary = self._find_boundary_region(
            df_discrete, best_reduct, decision_col, objects
        )
        
        # Dependency degree
        dep_degree = quality_reduced
        
        return RoughSetResult(
            core_attributes=core,
            reducts=reducts[:self.max_reducts],
            best_reduct=best_reduct,
            original_n_attributes=len(condition_cols),
            reduced_n_attributes=len(best_reduct),
            quality_original=quality_original,
            quality_reduced=quality_reduced,
            dependency_degree=dep_degree,
            boundary_objects=boundary
        )
    
    def _discretize(self, df: pd.DataFrame, 
                   columns: List[str]) -> pd.DataFrame:
        """Discretize continuous attributes into bins."""
        df_discrete = df.copy()
        
        for col in columns:
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                try:
                    df_discrete[col] = pd.cut(df[col], bins=self.n_bins, 
                                             labels=False, duplicates='drop')
                except:
                    df_discrete[col] = pd.qcut(df[col], q=self.n_bins, 
                                              labels=False, duplicates='drop')
        
        return df_discrete
    
    def _get_equivalence_classes(self, df: pd.DataFrame,
                                 attributes: List[str],
                                 objects: List) -> Dict[tuple, List]:
        """Get equivalence classes based on attributes."""
        classes = {}
        
        for obj in objects:
            # Get attribute values as tuple
            values = tuple(df.loc[obj, attributes].values)
            
            if values not in classes:
                classes[values] = []
            classes[values].append(obj)
        
        return classes
    
    def _calculate_quality(self, df: pd.DataFrame,
                          attributes: List[str],
                          decision: str,
                          objects: List) -> float:
        """Calculate quality of approximation (dependency degree)."""
        if not attributes:
            return 0.0
        
        # Get equivalence classes for condition attributes
        eq_classes = self._get_equivalence_classes(df, attributes, objects)
        
        # Get decision classes
        decision_classes = self._get_equivalence_classes(df, [decision], objects)
        
        # Calculate lower approximation
        positive_region = set()
        
        for eq_class_objects in eq_classes.values():
            # Check if all objects in equivalence class have same decision
            decisions = set(df.loc[eq_class_objects, decision].values)
            
            if len(decisions) == 1:
                positive_region.update(eq_class_objects)
        
        return len(positive_region) / len(objects)
    
    def _find_core(self, df: pd.DataFrame,
                  attributes: List[str],
                  decision: str,
                  objects: List) -> List[str]:
        """Find core attributes (indispensable for classification)."""
        core = []
        base_quality = self._calculate_quality(df, attributes, decision, objects)
        
        for attr in attributes:
            # Calculate quality without this attribute
            remaining = [a for a in attributes if a != attr]
            reduced_quality = self._calculate_quality(df, remaining, decision, objects)
            
            # If quality decreases, attribute is in core
            if reduced_quality < base_quality - 1e-10:
                core.append(attr)
        
        return core
    
    def _find_reducts_heuristic(self, df: pd.DataFrame,
                                attributes: List[str],
                                decision: str,
                                objects: List,
                                core: List[str]) -> List[List[str]]:
        """Find reducts using heuristic greedy approach."""
        reducts = []
        base_quality = self._calculate_quality(df, attributes, decision, objects)
        
        # Start with core
        current_reduct = list(core)
        remaining = [a for a in attributes if a not in core]
        
        # Greedily add attributes until quality is achieved
        current_quality = self._calculate_quality(df, current_reduct, decision, objects)
        
        while current_quality < base_quality - 1e-10 and remaining:
            best_attr = None
            best_quality = current_quality
            
            for attr in remaining:
                test_reduct = current_reduct + [attr]
                test_quality = self._calculate_quality(df, test_reduct, decision, objects)
                
                if test_quality > best_quality:
                    best_quality = test_quality
                    best_attr = attr
            
            if best_attr:
                current_reduct.append(best_attr)
                remaining.remove(best_attr)
                current_quality = best_quality
            else:
                break
        
        reducts.append(current_reduct)
        
        # Try to find alternative reducts by starting from different attributes
        for start_attr in attributes:
            if start_attr not in core:
                alt_reduct = list(core) + [start_attr]
                remaining = [a for a in attributes if a not in alt_reduct]
                
                current_quality = self._calculate_quality(df, alt_reduct, decision, objects)
                
                while current_quality < base_quality - 1e-10 and remaining:
                    best_attr = None
                    best_quality = current_quality
                    
                    for attr in remaining:
                        test_reduct = alt_reduct + [attr]
                        test_quality = self._calculate_quality(df, test_reduct, decision, objects)
                        
                        if test_quality > best_quality:
                            best_quality = test_quality
                            best_attr = attr
                    
                    if best_attr:
                        alt_reduct.append(best_attr)
                        remaining.remove(best_attr)
                        current_quality = best_quality
                    else:
                        break
                
                if alt_reduct not in reducts:
                    reducts.append(alt_reduct)
                
                if len(reducts) >= self.max_reducts:
                    break
        
        return reducts
    
    def _find_reducts_exhaustive(self, df: pd.DataFrame,
                                 attributes: List[str],
                                 decision: str,
                                 objects: List) -> List[List[str]]:
        """Find all reducts using exhaustive search (for small attribute sets)."""
        reducts = []
        base_quality = self._calculate_quality(df, attributes, decision, objects)
        n_attrs = len(attributes)
        
        # Limit exhaustive search for computational reasons
        max_search_size = min(n_attrs, 10)
        
        # Search from smallest to largest subsets
        for size in range(1, max_search_size + 1):
            for subset in combinations(attributes, size):
                subset_list = list(subset)
                quality = self._calculate_quality(df, subset_list, decision, objects)
                
                if quality >= base_quality - 1e-10:
                    # Check if it's minimal (no proper subset is also a reduct)
                    is_minimal = True
                    for existing in reducts:
                        if set(existing).issubset(set(subset_list)):
                            is_minimal = False
                            break
                    
                    if is_minimal:
                        reducts.append(subset_list)
                        
                        if len(reducts) >= self.max_reducts:
                            return reducts
        
        return reducts
    
    def _select_best_reduct(self, df: pd.DataFrame,
                           reducts: List[List[str]],
                           decision: str,
                           objects: List) -> List[str]:
        """Select the best reduct based on size and quality."""
        if not reducts:
            return []
        
        # Score reducts by size and quality
        scored_reducts = []
        for reduct in reducts:
            quality = self._calculate_quality(df, reduct, decision, objects)
            
            if quality >= self.quality_threshold:
                scored_reducts.append((reduct, len(reduct), quality))
        
        if not scored_reducts:
            # If none meet threshold, take best quality
            for reduct in reducts:
                quality = self._calculate_quality(df, reduct, decision, objects)
                scored_reducts.append((reduct, len(reduct), quality))
        
        # Sort by size (ascending), then quality (descending)
        scored_reducts.sort(key=lambda x: (x[1], -x[2]))
        
        return scored_reducts[0][0]
    
    def _find_boundary_region(self, df: pd.DataFrame,
                             attributes: List[str],
                             decision: str,
                             objects: List) -> List:
        """Find objects in boundary region (uncertain classification)."""
        if not attributes:
            return objects
        
        boundary = []
        eq_classes = self._get_equivalence_classes(df, attributes, objects)
        
        for eq_class_objects in eq_classes.values():
            decisions = set(df.loc[eq_class_objects, decision].values)
            
            if len(decisions) > 1:
                boundary.extend(eq_class_objects)
        
        return boundary


def reduce_attributes(data: pd.DataFrame,
                     n_bins: int = 5,
                     quality_threshold: float = 0.95) -> RoughSetResult:
    """Convenience function for attribute reduction."""
    reducer = RoughSetReducer(
        quality_threshold=quality_threshold,
        n_bins=n_bins
    )
    return reducer.reduce(data)
