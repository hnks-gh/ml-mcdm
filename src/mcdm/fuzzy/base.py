# -*- coding: utf-8 -*-
"""
Fuzzy Number Base Classes
=========================

This module provides foundational classes for fuzzy MCDM methods:
- TriangularFuzzyNumber (TFN): Core fuzzy number representation
- FuzzyDecisionMatrix: Container for fuzzy decision matrices
- Linguistic scales: Predefined mappings for linguistic terms

Mathematical Foundation:
    A Triangular Fuzzy Number (TFN) is denoted as Ã = (l, m, u) where:
    - l: lower bound (minimum possible value)
    - m: modal value (most likely value)  
    - u: upper bound (maximum possible value)
    
    The membership function μ_Ã(x) is:
        μ_Ã(x) = (x - l)/(m - l)  if l ≤ x ≤ m
               = (u - x)/(u - m)  if m ≤ x ≤ u
               = 0                otherwise
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class TriangularFuzzyNumber:
    """
    Triangular Fuzzy Number (TFN) representation.
    
    A TFN Ã = (l, m, u) represents uncertain values where:
    - l (lower): minimum possible value
    - m (modal): most likely/expected value
    - u (upper): maximum possible value
    
    Supports arithmetic operations and various defuzzification methods.
    
    Attributes:
        l: Lower bound (minimum)
        m: Modal value (most likely)
        u: Upper bound (maximum)
    
    Example:
        >>> tfn = TriangularFuzzyNumber(0.2, 0.5, 0.8)
        >>> crisp = tfn.defuzzify('centroid')  # Returns 0.5
        >>> scaled = tfn * 2  # Returns TFN(0.4, 1.0, 1.6)
    """
    l: float  # Lower bound
    m: float  # Modal value (most likely)
    u: float  # Upper bound
    
    def __post_init__(self):
        """Ensure l ≤ m ≤ u by sorting if necessary."""
        if not (self.l <= self.m <= self.u):
            self.l, self.m, self.u = sorted([self.l, self.m, self.u])
    
    def defuzzify(self, method: str = "centroid") -> float:
        """
        Convert fuzzy number to crisp value.
        
        Defuzzification Methods:
            - centroid: Center of gravity = (l + m + u) / 3
            - mom: Mean of Maximum = m
            - bisector: Bisector method = (l + 2m + u) / 4
            - graded_mean: Graded mean = (l + 4m + u) / 6
        
        Args:
            method: Defuzzification method name
        
        Returns:
            Crisp (scalar) value
        
        Raises:
            ValueError: If method is unknown
        """
        if method == "centroid":
            return (self.l + self.m + self.u) / 3
        elif method == "mom":
            return self.m
        elif method == "bisector":
            return (self.l + 2*self.m + self.u) / 4
        elif method == "graded_mean":
            return (self.l + 4*self.m + self.u) / 6
        else:
            raise ValueError(f"Unknown defuzzification method: {method}")
    
    def __add__(self, other: 'TriangularFuzzyNumber') -> 'TriangularFuzzyNumber':
        """Add two fuzzy numbers: Ã + B̃ = (a_l + b_l, a_m + b_m, a_u + b_u)"""
        return TriangularFuzzyNumber(
            self.l + other.l,
            self.m + other.m,
            self.u + other.u
        )
    
    def __sub__(self, other: 'TriangularFuzzyNumber') -> 'TriangularFuzzyNumber':
        """Subtract two fuzzy numbers: Ã - B̃ = (a_l - b_u, a_m - b_m, a_u - b_l)"""
        return TriangularFuzzyNumber(
            self.l - other.u,
            self.m - other.m,
            self.u - other.l
        )
    
    def __mul__(self, other: Union['TriangularFuzzyNumber', float]) -> 'TriangularFuzzyNumber':
        """
        Multiply fuzzy numbers or by scalar.
        
        For TFN × TFN: Uses approximate multiplication
        For TFN × scalar: Scales all bounds
        """
        if isinstance(other, TriangularFuzzyNumber):
            # Fuzzy multiplication (approximate via vertex method)
            products = [
                self.l * other.l, self.l * other.u,
                self.u * other.l, self.u * other.u
            ]
            return TriangularFuzzyNumber(
                min(products),
                self.m * other.m,
                max(products)
            )
        else:
            # Scalar multiplication
            scalar = float(other)
            if scalar >= 0:
                return TriangularFuzzyNumber(
                    self.l * scalar, self.m * scalar, self.u * scalar
                )
            else:
                return TriangularFuzzyNumber(
                    self.u * scalar, self.m * scalar, self.l * scalar
                )
    
    def __rmul__(self, scalar: float) -> 'TriangularFuzzyNumber':
        """Right multiplication by scalar."""
        return self.__mul__(scalar)
    
    def __truediv__(self, other: Union['TriangularFuzzyNumber', float]) -> 'TriangularFuzzyNumber':
        """
        Divide fuzzy numbers or by scalar.
        
        Handles division by zero with small epsilon.
        """
        if isinstance(other, TriangularFuzzyNumber):
            # Avoid division by zero
            l_safe = other.l if other.l != 0 else 1e-10
            m_safe = other.m if other.m != 0 else 1e-10
            u_safe = other.u if other.u != 0 else 1e-10
            
            quotients = [
                self.l / l_safe, self.l / u_safe,
                self.u / l_safe, self.u / u_safe
            ]
            return TriangularFuzzyNumber(
                min(quotients),
                self.m / m_safe,
                max(quotients)
            )
        else:
            scalar = float(other) if other != 0 else 1e-10
            if scalar >= 0:
                return TriangularFuzzyNumber(
                    self.l / scalar, self.m / scalar, self.u / scalar
                )
            else:
                return TriangularFuzzyNumber(
                    self.u / scalar, self.m / scalar, self.l / scalar
                )
    
    def distance(self, other: 'TriangularFuzzyNumber') -> float:
        """
        Calculate vertex distance between two fuzzy numbers.
        
        Formula: d(Ã, B̃) = √[((a_l - b_l)² + (a_m - b_m)² + (a_u - b_u)²) / 3]
        
        Args:
            other: Another TFN to compute distance to
        
        Returns:
            Vertex distance (float)
        """
        return np.sqrt(
            ((self.l - other.l) ** 2 + 
             (self.m - other.m) ** 2 + 
             (self.u - other.u) ** 2) / 3
        )
    
    def euclidean_distance(self, other: 'TriangularFuzzyNumber') -> float:
        """
        Calculate Euclidean distance between two fuzzy numbers.
        
        Formula: d(Ã, B̃) = √[(a_l - b_l)² + (a_m - b_m)² + (a_u - b_u)²]
        
        Args:
            other: Another TFN to compute distance to
        
        Returns:
            Euclidean distance (float)
        """
        return np.sqrt(
            (self.l - other.l) ** 2 + 
            (self.m - other.m) ** 2 + 
            (self.u - other.u) ** 2
        )
    
    @staticmethod
    def from_crisp(value: float, spread: float = 0.0) -> 'TriangularFuzzyNumber':
        """
        Create TFN from crisp value with symmetric spread.
        
        Args:
            value: Central (modal) value
            spread: Spread around modal value
        
        Returns:
            TFN with (value - spread, value, value + spread)
        """
        return TriangularFuzzyNumber(
            value - spread,
            value,
            value + spread
        )
    
    @staticmethod
    def from_interval(low: float, high: float) -> 'TriangularFuzzyNumber':
        """
        Create TFN from interval [low, high] with modal at center.
        
        Args:
            low: Lower bound of interval
            high: Upper bound of interval
        
        Returns:
            TFN with (low, (low+high)/2, high)
        """
        return TriangularFuzzyNumber(
            low,
            (low + high) / 2,
            high
        )
    
    def normalize(self, max_val: float) -> 'TriangularFuzzyNumber':
        """
        Normalize fuzzy number by dividing by max value.
        
        Args:
            max_val: Maximum value for normalization
        
        Returns:
            Normalized TFN with values in [0, 1] range
        """
        if max_val == 0:
            return TriangularFuzzyNumber(0, 0, 0)
        return TriangularFuzzyNumber(
            self.l / max_val,
            self.m / max_val,
            self.u / max_val
        )
    
    def __repr__(self) -> str:
        return f"TFN({self.l:.4f}, {self.m:.4f}, {self.u:.4f})"


class FuzzyDecisionMatrix:
    """
    Container for fuzzy decision matrix operations.
    
    Stores alternatives × criteria matrix of Triangular Fuzzy Numbers.
    Provides methods for accessing, manipulating, and converting fuzzy matrices.
    
    Attributes:
        matrix: Dictionary mapping alternative -> criterion -> TFN
        alternatives: List of alternative names
        criteria: List of criterion names
    
    Example:
        >>> matrix = FuzzyDecisionMatrix.from_crisp_with_uncertainty(data, spread_ratio=0.1)
        >>> crisp_df = matrix.to_crisp('centroid')
    """
    
    def __init__(self, 
                 matrix: Dict[str, Dict[str, TriangularFuzzyNumber]],
                 alternatives: List[str],
                 criteria: List[str]):
        self.matrix = matrix
        self.alternatives = alternatives
        self.criteria = criteria
    
    def get(self, alternative: str, criterion: str) -> TriangularFuzzyNumber:
        """
        Get fuzzy value for alternative and criterion.
        
        Args:
            alternative: Alternative name
            criterion: Criterion name
        
        Returns:
            TriangularFuzzyNumber for the cell
        """
        return self.matrix[alternative][criterion]
    
    def set(self, alternative: str, criterion: str, value: TriangularFuzzyNumber):
        """
        Set fuzzy value for alternative and criterion.
        
        Args:
            alternative: Alternative name
            criterion: Criterion name
            value: TFN value to set
        """
        if alternative not in self.matrix:
            self.matrix[alternative] = {}
        self.matrix[alternative][criterion] = value
    
    def to_crisp(self, method: str = "centroid") -> pd.DataFrame:
        """
        Convert to crisp decision matrix using defuzzification.
        
        Args:
            method: Defuzzification method (centroid, mom, bisector, graded_mean)
        
        Returns:
            pandas DataFrame with defuzzified values
        """
        data = {}
        for alt in self.alternatives:
            data[alt] = {}
            for crit in self.criteria:
                data[alt][crit] = self.matrix[alt][crit].defuzzify(method)
        return pd.DataFrame(data).T
    
    @staticmethod
    def from_crisp_with_uncertainty(data: pd.DataFrame,
                                    uncertainty: Optional[pd.DataFrame] = None,
                                    spread_ratio: float = 0.1) -> 'FuzzyDecisionMatrix':
        """
        Create fuzzy matrix from crisp data with uncertainty.
        
        The spread for each value is determined by:
        - If uncertainty matrix provided: uses those values directly
        - Otherwise: spread = value × spread_ratio
        
        Args:
            data: Crisp decision matrix (alternatives × criteria)
            uncertainty: Optional matrix of uncertainty values (std dev)
            spread_ratio: Default spread as ratio of value (used if no uncertainty)
        
        Returns:
            FuzzyDecisionMatrix with TFN values
        """
        alternatives = data.index.tolist()
        criteria = data.columns.tolist()
        matrix = {}
        
        for alt in alternatives:
            matrix[alt] = {}
            for crit in criteria:
                value = data.loc[alt, crit]
                
                if uncertainty is not None:
                    spread = uncertainty.loc[alt, crit]
                else:
                    spread = abs(value * spread_ratio)
                
                matrix[alt][crit] = TriangularFuzzyNumber(
                    value - spread,
                    value,
                    value + spread
                )
        
        return FuzzyDecisionMatrix(matrix, alternatives, criteria)
    
    @staticmethod
    def from_panel_temporal_variance(panel_data,
                                     spread_factor: float = 1.0) -> 'FuzzyDecisionMatrix':
        """
        Create fuzzy matrix from panel data using temporal variance.
        
        Uses the historical variance across time periods to determine
        the spread of fuzzy numbers, capturing temporal uncertainty.
        
        Args:
            panel_data: Panel data object with temporal data
            spread_factor: Multiplier for standard deviation spread
        
        Returns:
            FuzzyDecisionMatrix with TFN values based on temporal variance
        """
        alternatives = panel_data.entities
        criteria = panel_data.components
        matrix = {}
        
        # Get latest period data as modal values
        latest_data = panel_data.get_latest()
        
        # Calculate temporal variance from long-format data
        temporal_std = panel_data.long.groupby('Province').std()
        
        for alt in alternatives:
            matrix[alt] = {}
            for crit in criteria:
                if crit not in latest_data.columns:
                    continue
                    
                value = latest_data.loc[alt, crit] if alt in latest_data.index else 0
                
                # Get temporal std if available
                if alt in temporal_std.index and crit in temporal_std.columns:
                    std = temporal_std.loc[alt, crit] * spread_factor
                else:
                    std = abs(value * 0.1)  # Default 10% spread
                
                matrix[alt][crit] = TriangularFuzzyNumber(
                    max(0, value - std),
                    value,
                    value + std
                )
        
        return FuzzyDecisionMatrix(matrix, alternatives, criteria)


# =============================================================================
# LINGUISTIC SCALES
# =============================================================================

# 5-point linguistic scale for rating alternatives
LINGUISTIC_SCALE_5 = {
    'very_low': TriangularFuzzyNumber(0.0, 0.0, 0.25),
    'low': TriangularFuzzyNumber(0.0, 0.25, 0.50),
    'medium': TriangularFuzzyNumber(0.25, 0.50, 0.75),
    'high': TriangularFuzzyNumber(0.50, 0.75, 1.0),
    'very_high': TriangularFuzzyNumber(0.75, 1.0, 1.0),
}

# 7-point linguistic scale for more granular ratings
LINGUISTIC_SCALE_7 = {
    'very_poor': TriangularFuzzyNumber(0.0, 0.0, 0.1),
    'poor': TriangularFuzzyNumber(0.0, 0.1, 0.3),
    'medium_poor': TriangularFuzzyNumber(0.1, 0.3, 0.5),
    'fair': TriangularFuzzyNumber(0.3, 0.5, 0.7),
    'medium_good': TriangularFuzzyNumber(0.5, 0.7, 0.9),
    'good': TriangularFuzzyNumber(0.7, 0.9, 1.0),
    'very_good': TriangularFuzzyNumber(0.9, 1.0, 1.0),
}

# Importance scale for criteria weights
IMPORTANCE_SCALE = {
    'very_low': TriangularFuzzyNumber(0.0, 0.0, 0.25),
    'low': TriangularFuzzyNumber(0.0, 0.25, 0.50),
    'medium': TriangularFuzzyNumber(0.25, 0.50, 0.75),
    'high': TriangularFuzzyNumber(0.50, 0.75, 1.0),
    'very_high': TriangularFuzzyNumber(0.75, 1.0, 1.0),
}
