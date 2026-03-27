# -*- coding: utf-8 -*-
"""
DEPRECATED: Missing Data Mechanism Diagnostics
===============================================

🚫 DEPRECATED (2026-03-27): Use data.missing_data.assess_missing_mechanism() instead.

This module's diagnostics have been merged into the centralized validation suite
in data.missing_data.py (function assess_missing_mechanism). That implementation
provides comprehensive MCAR/MAR/MNAR diagnostics without redundant code.

For new code, use:
    from data.missing_data import assess_missing_mechanism
    report = assess_missing_mechanism(X_train)

Legacy Code (DO NOT USE):
    from data.imputation.mechanisms import test_missing_mechanism
    result = test_missing_mechanism(df)

This file retained for backward compatibility only.
Scheduled for removal: 2026-06-27 (after 3-month deprecation period)
"""

import warnings
from typing import Any, Dict


def test_missing_mechanism(df, alpha: float = 0.05) -> Dict[str, Any]:
    """
    DEPRECATED: Use data.missing_data.assess_missing_mechanism() instead.

    This function is retained only for backward compatibility.
    Will be removed 2026-06-27.
    """
    warnings.warn(
        "test_missing_mechanism is DEPRECATED. "
        "Use data.missing_data.assess_missing_mechanism(X) instead. "
        "Will be removed 2026-06-27.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Redirect to current implementation
    import numpy as np
    import pandas as pd
    from data.missing_data import assess_missing_mechanism

    X = df.values if isinstance(df, pd.DataFrame) else df
    report = assess_missing_mechanism(X, alpha=alpha, verbose=False)

    # Return legacy format for backward compatibility
    return {
        "mar_tests": [],
        "mcar_p_value": report.littles_test_pvalue,
        "suggested_mechanism": report.mechanism_assessment,
    }
