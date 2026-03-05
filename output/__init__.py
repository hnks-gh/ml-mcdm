# -*- coding: utf-8 -*-
"""
Output Package
==============

Centralises all persistence logic: CSV/JSON data, Markdown reports,
visualization figures, and orchestration thereof.

Quick start::

    from output import OutputOrchestrator
    orch = OutputOrchestrator('output/result')
    orch.save_all(panel_data, weights, ranking_result, ...)
"""

from pathlib import Path as _Path


def _sanitize_output_dir(raw: str, *, anchor: _Path | None = None) -> _Path:
    """Resolve *raw* and guard against ``..`` path-traversal attacks.

    When *raw* is a **relative** path the resolved result must stay
    under *anchor* (default: cwd).  Absolute paths are allowed — they
    are only rejected when ``..`` segments would cause them to escape a
    parent they logically started under.

    Raises ``ValueError`` if traversal is detected.
    """
    raw_path = _Path(raw)

    # Absolute paths are accepted as-is after resolving symlinks,
    # provided they don't contain literal '..' components.
    if raw_path.is_absolute():
        resolved = raw_path.resolve()
        # Check for '..' in the *original* string (before resolve)
        if '..' in raw_path.parts:
            raise ValueError(
                f"Output directory {str(raw)!r} contains '..' traversal "
                f"components.  Refusing to write."
            )
        return resolved

    # Relative path — must stay under anchor
    if anchor is None:
        anchor = _Path.cwd()
    resolved = (anchor / raw_path).resolve()
    anchor_resolved = anchor.resolve()
    try:
        resolved.relative_to(anchor_resolved)
    except ValueError:
        raise ValueError(
            f"Output directory {str(resolved)!r} escapes the project root "
            f"{str(anchor_resolved)!r}.  Refusing to write."
        )
    return resolved


from .csv_writer import CsvWriter
from .report_writer import ReportWriter
from .orchestrator import OutputOrchestrator
from .visualization import VisualizationOrchestrator, create_visualizer

__all__ = ['CsvWriter', 'ReportWriter', 'OutputOrchestrator',
           '_sanitize_output_dir']
