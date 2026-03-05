# -*- coding: utf-8 -*-
"""
Backward-compatibility shim.

The implementation has been centralised in ``analysis.stability``.
All public names are re-exported here so that existing code importing
from ``weighting.validation`` continues to work without modification.
"""
from analysis.stability import (  # noqa: F401
    StabilityResult,
    TemporalStabilityValidator,
    temporal_stability_verification,
)
