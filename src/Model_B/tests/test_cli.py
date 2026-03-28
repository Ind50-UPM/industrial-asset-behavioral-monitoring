"""CLI smoke tests for Model_B."""

from __future__ import annotations

from pathlib import Path

import pytest

import iabm_behavior.main as cli_main


def test_cli_generates_reports(
    synthetic_timeline: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The CLI should generate all expected behavioral reports."""
    output_dir = tmp_path / "reports"
    monkeypatch.setattr(
        "sys.argv",
        [
            "industrial-behavior",
            "--input",
            str(synthetic_timeline),
            "--output-dir",
            str(output_dir),
            "--smooth-short-runs",
            "--lang",
            "en",
        ],
    )

    cli_main.main()

    assert (output_dir / "state_runs.xlsx").exists()
    assert (output_dir / "active_sequences.xlsx").exists()
    assert (output_dir / "sequence_words.xlsx").exists()


def test_cli_generates_nominal_comparison_report(
    synthetic_timeline: Path,
    nominal_timeline: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The CLI should write a comparison report when a nominal timeline is provided."""
    output_dir = tmp_path / "reports_with_nominal"
    monkeypatch.setattr(
        "sys.argv",
        [
            "industrial-behavior",
            "--input",
            str(synthetic_timeline),
            "--nominal-input",
            str(nominal_timeline),
            "--output-dir",
            str(output_dir),
            "--anomaly-threshold",
            "0.5",
        ],
    )

    cli_main.main()

    assert (output_dir / "sequence_comparison.xlsx").exists()
