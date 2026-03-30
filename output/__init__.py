"""
Output and Persistence Layer.

This package centralizes all data serialization, reporting, and 
visualization logic for the ML-MCDM pipeline. It provides a unified 
interface for saving structured CSV/JSON data, generating publication-quality 
Markdown reports, and rendering diagnostic plots.

Package Structure
-----------------
- `orchestrator`: Central coordination hub for all writers.
- `csv_writer`: Structured numerical data persistence.
- `report_writer`: Human-readable analytical summaries.
- `visualization`: Diagnostic and result plotting engine.

Security
--------
Includes directory sanitization to prevent path-traversal attacks when 
resolving output locations.
"""

from pathlib import Path as _Path


def _sanitize_output_dir(raw: str, *, anchor: _Path | None = None) -> _Path:
    """
    Resolve and validate output directory paths.

    Guards against path-traversal attacks by ensuring that relative output 
    paths do not escape the project anchor (usually CWD) and that 
    absolute paths do not contain literal '..' components.

    Parameters
    ----------
    raw : str
        The raw path string provided by the user or config.
    anchor : Path, optional
        The reference directory for relative path resolution. Defaults 
        to `Path.cwd()`.

    Returns
    -------
    Path
        The resolved and validated absolute Path object.

    Raises
    ------
    ValueError
        If path traversal is detected or if the resolved path escapes 
        the anchor.
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
