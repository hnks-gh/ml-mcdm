# -*- coding: utf-8 -*-
"""
Fuzzy MCDM Methods Module
=========================

This module provides fuzzy extensions of traditional MCDM methods using
Triangular Fuzzy Numbers (TFN) to handle uncertainty in decision making.

Available Methods:
    - FuzzyTOPSIS: Fuzzy Technique for Order Preference by Similarity to Ideal Solution
    - FuzzyVIKOR: Fuzzy VIseKriterijumska Optimizacija I Kompromisno Resenje
    - FuzzyPROMETHEE: Fuzzy Preference Ranking Organization Method
    - FuzzyCOPRAS: Fuzzy Complex Proportional Assessment
    - FuzzyEDAS: Fuzzy Evaluation based on Distance from Average Solution

Core Components:
    - TriangularFuzzyNumber: Fuzzy number representation (l, m, u)
    - FuzzyDecisionMatrix: Container for fuzzy decision matrices
    - Linguistic scales: Predefined scales for converting linguistic terms

Example Usage:
    >>> from src.mcdm.fuzzy import FuzzyTOPSIS, TriangularFuzzyNumber
    >>> 
    >>> # Create calculator with cost criteria specification
    >>> calculator = FuzzyTOPSIS(cost_criteria=['Cost', 'Risk'])
    >>> 
    >>> # Calculate from panel data (uses temporal variance for fuzziness)
    >>> result = calculator.calculate_from_panel(panel_data, weights=weights)
    >>> 
    >>> # Or from crisp data with uncertainty
    >>> result = calculator.calculate(crisp_data, spread_ratio=0.1)
"""

# Core fuzzy types
from .base import (
    TriangularFuzzyNumber,
    FuzzyDecisionMatrix,
    LINGUISTIC_SCALE_5,
    LINGUISTIC_SCALE_7,
    IMPORTANCE_SCALE
)

# Fuzzy MCDM methods
from .topsis import FuzzyTOPSIS, FuzzyTOPSISResult
from .vikor import FuzzyVIKOR, FuzzyVIKORResult
from .promethee import FuzzyPROMETHEE, FuzzyPROMETHEEResult
from .copras import FuzzyCOPRAS, FuzzyCOPRASResult
from .edas import FuzzyEDAS, FuzzyEDASResult

__all__ = [
    # Core types
    'TriangularFuzzyNumber',
    'FuzzyDecisionMatrix',
    'LINGUISTIC_SCALE_5',
    'LINGUISTIC_SCALE_7',
    'IMPORTANCE_SCALE',
    # Methods
    'FuzzyTOPSIS',
    'FuzzyTOPSISResult',
    'FuzzyVIKOR',
    'FuzzyVIKORResult',
    'FuzzyPROMETHEE',
    'FuzzyPROMETHEEResult',
    'FuzzyCOPRAS',
    'FuzzyCOPRASResult',
    'FuzzyEDAS',
    'FuzzyEDASResult',
]
