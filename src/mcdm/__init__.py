# -*- coding: utf-8 -*-
"""
Multi-Criteria Decision Making Module
=====================================

Comprehensive MCDM methods including traditional and fuzzy variants.

Submodules
----------
traditional
    Traditional MCDM methods: TOPSIS, VIKOR, PROMETHEE, COPRAS, EDAS
fuzzy
    Fuzzy MCDM methods using Triangular Fuzzy Numbers

Usage
-----
>>> from src.mcdm.traditional import TOPSISCalculator, VIKORCalculator
>>> from src.mcdm.fuzzy import FuzzyTOPSIS, FuzzyVIKOR
>>> from src.weighting import EntropyWeightCalculator, CRITICWeightCalculator
"""

# Import from traditional submodule
from .traditional import (
    TOPSISCalculator, TOPSISResult, DynamicTOPSIS, DynamicTOPSISResult,
    VIKORCalculator, VIKORResult, MultiPeriodVIKOR,
    PROMETHEECalculator, PROMETHEEResult,
    COPRASCalculator, COPRASResult,
    EDASCalculator, EDASResult,
)

# Import from fuzzy submodule
from .fuzzy import (
    TriangularFuzzyNumber, FuzzyDecisionMatrix,
    LINGUISTIC_SCALE_5, LINGUISTIC_SCALE_7, IMPORTANCE_SCALE,
    FuzzyTOPSIS, FuzzyTOPSISResult,
    FuzzyVIKOR, FuzzyVIKORResult,
    FuzzyPROMETHEE, FuzzyPROMETHEEResult,
    FuzzyCOPRAS, FuzzyCOPRASResult,
    FuzzyEDAS, FuzzyEDASResult,
)

# Import weighting methods from weighting module
from ..weighting import (
    EntropyWeightCalculator,
    CRITICWeightCalculator,
    EnsembleWeightCalculator,
    WeightResult
)

# Legacy alias
DynamicTOPSISResult = TOPSISResult


__all__ = [
    # Fuzzy base
    'TriangularFuzzyNumber',
    'FuzzyDecisionMatrix',
    'LINGUISTIC_SCALE_5',
    'LINGUISTIC_SCALE_7',
    'IMPORTANCE_SCALE',
    
    # Weights
    'EntropyWeightCalculator',
    'CRITICWeightCalculator',
    'EnsembleWeightCalculator',
    'WeightResult',
    
    # Traditional MCDM
    'TOPSISCalculator', 'TOPSISResult',
    'DynamicTOPSIS', 'DynamicTOPSISResult',
    'VIKORCalculator', 'VIKORResult', 'MultiPeriodVIKOR',
    'PROMETHEECalculator', 'PROMETHEEResult',
    'COPRASCalculator', 'COPRASResult',
    'EDASCalculator', 'EDASResult',
    
    # Fuzzy MCDM
    'FuzzyTOPSIS', 'FuzzyTOPSISResult',
    'FuzzyVIKOR', 'FuzzyVIKORResult',
    'FuzzyPROMETHEE', 'FuzzyPROMETHEEResult',
    'FuzzyCOPRAS', 'FuzzyCOPRASResult',
    'FuzzyEDAS', 'FuzzyEDASResult',
]


# Convenience function to get all MCDM calculators
def get_all_calculators(fuzzy: bool = False):
    """
    Get dictionary of all MCDM calculators.
    
    Parameters
    ----------
    fuzzy : bool, default=False
        If True, returns fuzzy versions; if False, returns traditional
    
    Returns
    -------
    Dict[str, class]
        Dictionary of calculator classes
    """
    if fuzzy:
        return {
            'topsis': FuzzyTOPSIS,
            'vikor': FuzzyVIKOR,
            'promethee': FuzzyPROMETHEE,
            'copras': FuzzyCOPRAS,
            'edas': FuzzyEDAS,
        }
    else:
        return {
            'topsis': TOPSISCalculator,
            'vikor': VIKORCalculator,
            'promethee': PROMETHEECalculator,
            'copras': COPRASCalculator,
            'edas': EDASCalculator,
        }
