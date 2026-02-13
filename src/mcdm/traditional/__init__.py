# -*- coding: utf-8 -*-
"""
Traditional MCDM Methods Module

Provides crisp (non-fuzzy) Multi-Criteria Decision Making methods:
- TOPSIS: Technique for Order Preference by Similarity to Ideal Solution
- VIKOR: Multi-criteria Optimization and Compromise Solution
- PROMETHEE: Preference Ranking Organization Method for Enrichment Evaluations
- COPRAS: Complex Proportional Assessment
- EDAS: Evaluation based on Distance from Average Solution
- SAW: Simple Additive Weighting
"""

from .topsis import TOPSISCalculator, TOPSISResult, DynamicTOPSIS, DynamicTOPSISResult
from .vikor import VIKORCalculator, VIKORResult, MultiPeriodVIKOR
from .promethee import PROMETHEECalculator, PROMETHEEResult, PreferenceFunction, MultiPeriodPROMETHEE
from .copras import COPRASCalculator, COPRASResult
from .edas import EDASCalculator, EDASResult
from .saw import SAWCalculator, SAWResult

__all__ = [
    # TOPSIS
    'TOPSISCalculator', 'TOPSISResult',
    'DynamicTOPSIS', 'DynamicTOPSISResult',
    
    # VIKOR
    'VIKORCalculator', 'VIKORResult', 'MultiPeriodVIKOR',
    
    # PROMETHEE
    'PROMETHEECalculator', 'PROMETHEEResult', 
    'PreferenceFunction', 'MultiPeriodPROMETHEE',
    
    # COPRAS
    'COPRASCalculator', 'COPRASResult',
    
    # EDAS
    'EDASCalculator', 'EDASResult',
    
    # SAW
    'SAWCalculator', 'SAWResult',
]
