# -*- coding: utf-8 -*-
"""
Multi-Criteria Decision Making Module
=====================================

Comprehensive MCDM methods including traditional and IFS variants.

Submodules
----------
traditional
    Traditional MCDM methods: TOPSIS, VIKOR, PROMETHEE, COPRAS, EDAS, SAW
ifs
    Intuitionistic Fuzzy Set MCDM methods (Atanassov, 1986)

Usage
-----
>>> from src.mcdm.traditional import TOPSISCalculator, VIKORCalculator, SAWCalculator
>>> from src.mcdm.ifs import IFS_TOPSIS, IFS_VIKOR
>>> from src.weighting import EntropyWeightCalculator, CRITICWeightCalculator
"""

# Import from traditional submodule
from .traditional import (
    TOPSISCalculator, TOPSISResult, DynamicTOPSIS, DynamicTOPSISResult,
    VIKORCalculator, VIKORResult, MultiPeriodVIKOR,
    PROMETHEECalculator, PROMETHEEResult,
    COPRASCalculator, COPRASResult,
    EDASCalculator, EDASResult,
    SAWCalculator, SAWResult,
)

# Import from IFS submodule
from .ifs import (
    IFN, IFSDecisionMatrix,
    IFS_SAW, IFS_SAWResult,
    IFS_TOPSIS, IFS_TOPSISResult,
    IFS_VIKOR, IFS_VIKORResult,
    IFS_PROMETHEE, IFS_PROMETHEEResult,
    IFS_COPRAS, IFS_COPRASResult,
    IFS_EDAS, IFS_EDASResult,
)

# Import weighting methods from weighting module
from ..weighting import (
    EntropyWeightCalculator,
    CRITICWeightCalculator,
    MERECWeightCalculator,
    StandardDeviationWeightCalculator,
    RobustGlobalWeighting,
    WeightResult
)

# Legacy alias
DynamicTOPSISResult = TOPSISResult


__all__ = [
    # IFS base
    'IFN',
    'IFSDecisionMatrix',
    
    # Weights
    'EntropyWeightCalculator',
    'CRITICWeightCalculator',
    'MERECWeightCalculator',
    'StandardDeviationWeightCalculator',
    'RobustGlobalWeighting',
    'WeightResult',
    
    # Traditional MCDM
    'TOPSISCalculator', 'TOPSISResult',
    'DynamicTOPSIS', 'DynamicTOPSISResult',
    'VIKORCalculator', 'VIKORResult', 'MultiPeriodVIKOR',
    'PROMETHEECalculator', 'PROMETHEEResult',
    'COPRASCalculator', 'COPRASResult',
    'EDASCalculator', 'EDASResult',
    'SAWCalculator', 'SAWResult',
    
    # IFS MCDM
    'IFS_SAW', 'IFS_SAWResult',
    'IFS_TOPSIS', 'IFS_TOPSISResult',
    'IFS_VIKOR', 'IFS_VIKORResult',
    'IFS_PROMETHEE', 'IFS_PROMETHEEResult',
    'IFS_COPRAS', 'IFS_COPRASResult',
    'IFS_EDAS', 'IFS_EDASResult',
]


# Convenience function to get all MCDM calculators
def get_all_calculators(ifs: bool = False):
    """
    Get dictionary of all MCDM calculators.
    
    Parameters
    ----------
    ifs : bool, default=False
        If True, returns IFS versions; if False, returns traditional
    
    Returns
    -------
    Dict[str, class]
        Dictionary of calculator classes
    """
    if ifs:
        return {
            'saw': IFS_SAW,
            'topsis': IFS_TOPSIS,
            'vikor': IFS_VIKOR,
            'promethee': IFS_PROMETHEE,
            'copras': IFS_COPRAS,
            'edas': IFS_EDAS,
        }
    else:
        return {
            'saw': SAWCalculator,
            'topsis': TOPSISCalculator,
            'vikor': VIKORCalculator,
            'promethee': PROMETHEECalculator,
            'copras': COPRASCalculator,
            'edas': EDASCalculator,
        }
