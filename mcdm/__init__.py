# -*- coding: utf-8 -*-
from typing import Dict
"""
Multi-Criteria Decision Making Module
=====================================

Traditional crisp MCDM methods.

Submodules
----------
traditional
    Traditional MCDM methods: TOPSIS, VIKOR, PROMETHEE, COPRAS, EDAS, SAW

Usage
-----
>>> from mcdm.traditional import TOPSISCalculator, VIKORCalculator, SAWCalculator
>>> from weighting import EntropyWeightCalculator, CRITICWeightCalculator
"""

# Import from traditional submodule
from .traditional import (
    TOPSISCalculator, TOPSISResult,
    VIKORCalculator, VIKORResult, MultiPeriodVIKOR,
    PROMETHEECalculator, PROMETHEEResult,
    COPRASCalculator, COPRASResult,
    EDASCalculator, EDASResult,
    SAWCalculator, SAWResult,
)

# Import weighting methods from weighting module
from weighting import (
    EntropyWeightCalculator,
    CRITICWeightCalculator,
    MERECWeightCalculator,
    StandardDeviationWeightCalculator,
    HybridWeightingCalculator,
    WeightResult
)


__all__ = [
    # Weights
    'EntropyWeightCalculator',
    'CRITICWeightCalculator',
    'MERECWeightCalculator',
    'StandardDeviationWeightCalculator',
    'HybridWeightingCalculator',
    'WeightResult',
    
    # Traditional MCDM
    'TOPSISCalculator', 'TOPSISResult',
    'VIKORCalculator', 'VIKORResult', 'MultiPeriodVIKOR',
    'PROMETHEECalculator', 'PROMETHEEResult',
    'COPRASCalculator', 'COPRASResult',
    'EDASCalculator', 'EDASResult',
    'SAWCalculator', 'SAWResult',
    
]


def get_all_calculators() -> Dict[str, type]:
    """
    Get dictionary of all traditional MCDM calculators.

    Returns
    -------
    Dict[str, class]
        Dictionary of calculator classes keyed by method name.
    """
    return {
        'saw': SAWCalculator,
        'topsis': TOPSISCalculator,
        'vikor': VIKORCalculator,
        'promethee': PROMETHEECalculator,
        'copras': COPRASCalculator,
        'edas': EDASCalculator,
    }
