# -*- coding: utf-8 -*-
"""
Unit tests for the output package — CsvWriter, ReportWriter, OutputOrchestrator.

Covers:
  - CsvWriter directory scaffolding and CSV/JSON round-trips
  - save_criterion_weights output correctness
  - save_weights output structure
  - ReportWriter initialisation and path
  - OutputOrchestrator wiring (csv + report writers created)
"""

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from output import _sanitize_output_dir
from output.csv_writer import CsvWriter
from output.report_writer import ReportWriter
from output.orchestrator import OutputOrchestrator


# ---------------------------------------------------------------------------
# Path sanitization guard (M11)
# ---------------------------------------------------------------------------

class TestSanitizeOutputDir:
    """Verify _sanitize_output_dir rejects traversal and accepts valid paths."""

    def test_relative_path_under_anchor(self, tmp_path):
        result = _sanitize_output_dir("sub/dir", anchor=tmp_path)
        assert result == (tmp_path / "sub" / "dir").resolve()

    def test_relative_traversal_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="escapes the project root"):
            _sanitize_output_dir("../../etc/passwd", anchor=tmp_path)

    def test_absolute_path_accepted(self, tmp_path):
        abs_dir = str(tmp_path / "output")
        result = _sanitize_output_dir(abs_dir)
        assert result == (tmp_path / "output").resolve()

    def test_absolute_path_with_dotdot_rejected(self, tmp_path):
        bad = str(tmp_path / "a" / ".." / ".." / "escape")
        with pytest.raises(ValueError, match="traversal"):
            _sanitize_output_dir(bad)


# ---------------------------------------------------------------------------
# CsvWriter
# ---------------------------------------------------------------------------

class TestCsvWriterInit:
    def test_phase_directories_created(self, tmp_path):
        """All six canonical phase directories should be created on init."""
        writer = CsvWriter(base_output_dir=str(tmp_path))
        for phase in CsvWriter.PHASES:
            assert (tmp_path / "csv" / phase).is_dir(), f"Missing dir: {phase}"

    def test_csv_dir_attribute(self, tmp_path):
        writer = CsvWriter(base_output_dir=str(tmp_path))
        assert writer.csv_dir == tmp_path / "csv"

    def test_phase_dir_attributes(self, tmp_path):
        writer = CsvWriter(base_output_dir=str(tmp_path))
        assert writer.weighting_dir == tmp_path / "csv" / "weighting"
        assert writer.ranking_dir == tmp_path / "csv" / "ranking"
        assert writer.forecasting_dir == tmp_path / "csv" / "forecasting"

    def test_saved_files_initially_empty(self, tmp_path):
        writer = CsvWriter(base_output_dir=str(tmp_path))
        assert writer.get_saved_files() == []


class TestCsvWriterRoundTrips:
    def test_save_csv_roundtrip(self, tmp_path):
        """DataFrame → CSV → re-read should preserve data."""
        writer = CsvWriter(base_output_dir=str(tmp_path))
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
        path_str = writer._save_csv(df, "test.csv", float_fmt="%.2f")

        reloaded = pd.read_csv(path_str, index_col=0)
        np.testing.assert_allclose(reloaded["A"].values, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(reloaded["B"].values, [4.0, 5.0, 6.0])

    def test_save_csv_records_path(self, tmp_path):
        writer = CsvWriter(base_output_dir=str(tmp_path))
        df = pd.DataFrame({"X": [1]})
        writer._save_csv(df, "tracked.csv")
        assert len(writer.get_saved_files()) == 1
        assert "tracked.csv" in writer.get_saved_files()[0]

    def test_save_json_roundtrip(self, tmp_path):
        """Dict → JSON → re-read should preserve content."""
        writer = CsvWriter(base_output_dir=str(tmp_path))
        data = {"key": "value", "number": 42, "nested": {"a": [1, 2, 3]}}
        path_str = writer._save_json(data, "test.json")

        with open(path_str, "r", encoding="utf-8") as f:
            reloaded = json.load(f)

        assert reloaded["key"] == "value"
        assert reloaded["number"] == 42
        assert reloaded["nested"]["a"] == [1, 2, 3]

    def test_save_json_records_path(self, tmp_path):
        writer = CsvWriter(base_output_dir=str(tmp_path))
        writer._save_json({"a": 1}, "tracked.json")
        assert len(writer.get_saved_files()) == 1


class TestCsvWriterCriterionWeights:
    def test_criterion_weights_saved_correctly(self, tmp_path):
        writer = CsvWriter(base_output_dir=str(tmp_path))
        crit_w = {"C01": 0.3, "C02": 0.5, "C03": 0.2}
        path_str = writer.save_criterion_weights(crit_w)

        df = pd.read_csv(path_str)
        assert set(df.columns) == {"C01", "C02", "C03"}
        assert df["C01"].iloc[0] == pytest.approx(0.3, abs=1e-6)
        assert df["C02"].iloc[0] == pytest.approx(0.5, abs=1e-6)
        assert df["C03"].iloc[0] == pytest.approx(0.2, abs=1e-6)


class TestCsvWriterWeights:
    def test_save_weights_structure(self, tmp_path):
        """save_weights should produce a CSV with expected columns."""
        writer = CsvWriter(base_output_dir=str(tmp_path))
        subcriteria = ["SC11", "SC12", "SC21"]
        weights_dict = {
            "global_sc_weights": {"SC11": 0.2, "SC12": 0.3, "SC21": 0.5},
            "criterion_weights": {"C01": 0.5, "C02": 0.5},
            "details": {
                "level1": {
                    "C01": {
                        "local_sc_weights": {"SC11": 0.4, "SC12": 0.6},
                        "mc_diagnostics": {
                            "mean_weights": {"SC11": 0.4, "SC12": 0.6},
                            "std_weights": {"SC11": 0.01, "SC12": 0.01},
                            "cv_weights": {"SC11": 0.02, "SC12": 0.02},
                            "ci_lower_2_5": {"SC11": 0.38, "SC12": 0.58},
                            "ci_upper_97_5": {"SC11": 0.42, "SC12": 0.62},
                        },
                    },
                    "C02": {
                        "local_sc_weights": {"SC21": 1.0},
                        "mc_diagnostics": {
                            "mean_weights": {"SC21": 1.0},
                            "std_weights": {"SC21": 0.0},
                            "cv_weights": {"SC21": 0.0},
                            "ci_lower_2_5": {"SC21": 1.0},
                            "ci_upper_97_5": {"SC21": 1.0},
                        },
                    },
                },
                "level2": {"mc_diagnostics": {}},
            },
        }
        path_str = writer.save_weights(weights_dict, subcriteria)

        df = pd.read_csv(path_str, index_col=0)
        # Subcriteria rows should be present
        assert "SC11" in df.index
        assert "SC12" in df.index
        assert "SC21" in df.index
        # Expected columns
        for col in ("Global_Weight", "Criterion_Weight",
                     "Local_SC_Weight", "Rank_Global"):
            assert col in df.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# ReportWriter
# ---------------------------------------------------------------------------

class TestReportWriter:
    def test_init_creates_reports_dir(self, tmp_path):
        rw = ReportWriter(base_output_dir=str(tmp_path))
        assert (tmp_path / "reports").is_dir()

    def test_path_attribute(self, tmp_path):
        rw = ReportWriter(base_output_dir=str(tmp_path))
        expected = tmp_path / "reports" / "report.md"
        assert Path(rw._path) == expected


# ---------------------------------------------------------------------------
# OutputOrchestrator
# ---------------------------------------------------------------------------

class TestOutputOrchestrator:
    def test_init_creates_both_writers(self, tmp_path):
        orch = OutputOrchestrator(base_output_dir=str(tmp_path))
        assert isinstance(orch.csv, CsvWriter)
        assert isinstance(orch.report, ReportWriter)

    def test_phase_dirs_exist(self, tmp_path):
        orch = OutputOrchestrator(base_output_dir=str(tmp_path))
        for phase in CsvWriter.PHASES:
            assert (tmp_path / "csv" / phase).is_dir()

    def test_get_saved_files_initially_empty(self, tmp_path):
        orch = OutputOrchestrator(base_output_dir=str(tmp_path))
        files = orch.get_saved_files()
        assert isinstance(files, list)
