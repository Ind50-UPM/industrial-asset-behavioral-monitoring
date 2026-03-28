"""CLI smoke tests for Model_C."""

from __future__ import annotations

from pathlib import Path

import pytest

import iabm_semantics.main as cli_main


def test_cli_generates_semantic_reports(
    active_sequences_report: Path,
    comparison_report: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The CLI should generate semantic assignment and summary reports."""
    output_dir = tmp_path / "semantic_reports"
    monkeypatch.setattr(
        "sys.argv",
        [
            "industrial-semantics",
            "--input",
            str(active_sequences_report),
            "--comparison-input",
            str(comparison_report),
            "--output-dir",
            str(output_dir),
            "--lang",
            "en",
        ],
    )

    cli_main.main()

    assert (output_dir / "semantic_assignments.xlsx").exists()
    assert (output_dir / "semantic_mode_summary.xlsx").exists()
