# -*- coding: utf-8 -*-
"""
Intuitionistic Fuzzy Set (IFS) MCDM Methods
============================================

Six IFS-extended MCDM methods that incorporate hesitancy from
temporal variance into decision analysis.

Methods
-------
- IFS_SAW       — Intuitionistic Fuzzy Simple Additive Weighting
- IFS_TOPSIS    — Intuitionistic Fuzzy TOPSIS
- IFS_VIKOR     — Intuitionistic Fuzzy VIKOR
- IFS_PROMETHEE — Intuitionistic Fuzzy PROMETHEE II
- IFS_COPRAS    — Intuitionistic Fuzzy COPRAS
- IFS_EDAS      — Intuitionistic Fuzzy EDAS
"""

from .base import IFN, IFSDecisionMatrix
from .saw import IFS_SAW, IFS_SAWResult
from .topsis import IFS_TOPSIS, IFS_TOPSISResult
from .vikor import IFS_VIKOR, IFS_VIKORResult
from .promethee import IFS_PROMETHEE, IFS_PROMETHEEResult
from .copras import IFS_COPRAS, IFS_COPRASResult
from .edas import IFS_EDAS, IFS_EDASResult

__all__ = [
    # Base
    'IFN', 'IFSDecisionMatrix',
    # Methods
    'IFS_SAW', 'IFS_SAWResult',
    'IFS_TOPSIS', 'IFS_TOPSISResult',
    'IFS_VIKOR', 'IFS_VIKORResult',
    'IFS_PROMETHEE', 'IFS_PROMETHEEResult',
    'IFS_COPRAS', 'IFS_COPRASResult',
    'IFS_EDAS', 'IFS_EDASResult',
]
