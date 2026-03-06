# -*- coding: utf-8 -*-
"""
Ranking Package
===============

Two-stage hierarchical ranking pipeline combining 5 traditional MCDM methods
with Evidential Reasoning (Yang & Xu, 2002).

Modules
-------
hierarchical_pipeline
    HierarchicalRankingPipeline — the main ranking orchestrator.
topsis, vikor, promethee, copras, edas
    Traditional MCDM method calculators.
saw
    Simple Additive Weighting (used as a fast surrogate in the MC ensemble).
evidential_reasoning
    BeliefDistribution, EvidentialReasoningEngine, HierarchicalEvidentialReasoning.
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
from .evidential_reasoning import (
    BeliefDistribution, EvidentialReasoningEngine,
    HierarchicalEvidentialReasoning, HierarchicalERResult,
)

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
    # Evidential Reasoning
    "BeliefDistribution", "EvidentialReasoningEngine",
    "HierarchicalEvidentialReasoning", "HierarchicalERResult",
]
