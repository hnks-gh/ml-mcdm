# -*- coding: utf-8 -*-
"""
Backward-compatibility shim.

The implementation has been centralised in ``analysis.bootstrap``.
All public names are re-exported here so that existing code importing
from ``weighting.bootstrap`` continues to work without modification.
"""
from analysis.bootstrap import (  # noqa: F401
    bayesian_bootstrap_weights,
    BayesianBootstrap,
)
