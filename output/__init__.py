# -*- coding: utf-8 -*-
"""
Output Package
==============

Centralises all persistence logic: CSV/JSON data, Markdown reports, and
orchestration thereof.

Quick start::

    from output import OutputOrchestrator
    orch = OutputOrchestrator('result')
    orch.save_all(panel_data, weights, ranking_result, ...)
"""

from .csv_writer import CsvWriter
from .report_writer import ReportWriter
from .orchestrator import OutputOrchestrator

__all__ = ['CsvWriter', 'ReportWriter', 'OutputOrchestrator']
