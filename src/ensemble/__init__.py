# -*- coding: utf-8 -*-
"""
Ensemble Module
===============

Evidential Reasoning for combining MCDM rankings.

Submodules
----------
evidential_reasoning
    Evidential Reasoning (Yang & Xu, 2002) for hierarchical belief
    combination of multi-method MCDM evidence.

Usage
-----
>>> from src.ensemble.evidential_reasoning import (
...     EvidentialReasoningEngine,
...     HierarchicalEvidentialReasoning,
...     HierarchicalERResult,
... )
"""

from .evidential_reasoning import (
    BeliefDistribution,
    EvidentialReasoningEngine,
    HierarchicalEvidentialReasoning,
    HierarchicalERResult,
)


__all__ = [
    'BeliefDistribution',
    'EvidentialReasoningEngine',
    'HierarchicalEvidentialReasoning',
    'HierarchicalERResult',
]
