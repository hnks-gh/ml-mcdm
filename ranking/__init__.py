# -*- coding: utf-8 -*-
"""
Ranking Package
===============

Hierarchical ranking pipeline combining 6 traditional MCDM methods:
TOPSIS, VIKOR, PROMETHEE, COPRAS, EDAS, and Simple Additive Weighting (SAW).

Modules
-------
hierarchical_pipeline
    HierarchicalRankingPipeline — the main ranking orchestrator.
topsis, vikor, promethee, copras, edas, saw
    Traditional MCDM method calculators.
"""

from .hierarchical_pipeline import HierarchicalRankingPipeline, HierarchicalRankingResult
from .topsis import TOPSISCalculator, TOPSISResult
from .vikor import VIKORCalculator, VIKORResult, MultiPeriodVIKOR
from .promethee import (
    PROMETHEECalculator, PROMETHEEResult,
    PreferenceFunction, MultiPeriodPROMETHEE,
)
from .copras import COPRASCalculator, COPRASResult
from .edas import EDASCalculator, EDASResult
from .saw import SAWCalculator, SAWResult

__all__ = [
    # Pipeline
    "HierarchicalRankingPipeline", "HierarchicalRankingResult",
    # TOPSIS
    "TOPSISCalculator", "TOPSISResult",
    # VIKOR
    "VIKORCalculator", "VIKORResult", "MultiPeriodVIKOR",
    # PROMETHEE
    "PROMETHEECalculator", "PROMETHEEResult",
    "PreferenceFunction", "MultiPeriodPROMETHEE",
    # COPRAS
    "COPRASCalculator", "COPRASResult",
    # EDAS
    "EDASCalculator", "EDASResult",
    # SAW
    "SAWCalculator", "SAWResult",
]
