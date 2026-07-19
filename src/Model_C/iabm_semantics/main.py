"""Command-line entry point for Model_C semantic interpretation.

The CLI converts behavioral outputs from ``Model_B`` into semantic and
reliability-oriented labels that downstream incident processing can consume.
It supports optional anomaly-context enrichment through Model_B comparison
reports and keeps reporting concerns isolated from the interpreter itself.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import pandas as pd

from .semantics import SemanticModeInterpreter
from .utils import setup_i18n


def parse_arguments(translator: Callable[[str], str]) -> argparse.Namespace:
    """Build the CLI parser for semantic interpretation.

    Args:
        translator: Translation function used for localized help strings.

    Returns:
        Parsed command-line arguments for input selection, optional custom rule
        injection, and output formatting.
    """
    _ = translator
    parser = argparse.ArgumentParser(
        description=_("Semantic interpretation of industrial behavioral sequences")
    )
    parser.add_argument("--input", required=True, help=_("Path to the Model_B active-sequence report."))
    parser.add_argument("--comparison-input", help=_("Optional Model_B comparison report used to enrich semantic status."))
    parser.add_argument("--rules", help=_("Optional JSON file with semantic interpretation rules."))
    parser.add_argument("--output-dir", required=True, help=_("Directory where Model_C reports will be written."))
    parser.add_argument("--lang", default="en", choices=["es", "en"], help=_("Interface language."))
    parser.add_argument("--output-format", "--output_format", dest="output_format", choices=["xlsx", "csv"], default="xlsx", help=_("Report export format."))
    return parser.parse_args()


def main() -> None:
    """Run semantic interpretation and persist the resulting reports.

    Returns:
        ``None``. Output artefacts are written into the selected directory.

    Notes:
        The CLI emits both a row-level assignment table and multiple summary
        views because downstream layers often require different semantic slices:
        one for per-sequence incident-family alignment and another for broader
        operating-mode and life-regime analysis.
    """
    lang = _detect_language(sys.argv)
    translator = setup_i18n(lang)
    args = parse_arguments(translator)

    interpreter = SemanticModeInterpreter()
    if args.rules:
        interpreter.load_rules(args.rules)

    sequences = interpreter.load_active_sequences(args.input)
    comparison = interpreter.load_comparison_report(args.comparison_input) if args.comparison_input else None

    assignments = interpreter.interpret_sequences(sequences, comparison=comparison)
    summary = interpreter.summarize_modes(assignments)
    life_regime_summary = interpreter.summarize_life_regimes(assignments)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = args.output_format
    assignments_path = output_dir / f"semantic_assignments.{suffix}"
    summary_path = output_dir / f"semantic_mode_summary.{suffix}"
    family_path = output_dir / f"incident_family_assignments.{suffix}"
    life_regime_summary_path = output_dir / f"asset_life_regime_summary.{suffix}"
    _write_report(assignments, assignments_path)
    _write_report(summary, summary_path)
    _write_report(assignments, family_path)
    _write_report(life_regime_summary, life_regime_summary_path)

    print(translator("Semantic assignments saved to: {}").format(assignments_path))
    print(translator("Semantic mode summary saved to: {}").format(summary_path))
    print(translator("Incident family assignments saved to: {}").format(family_path))
    print(translator("Asset life regime summary saved to: {}").format(life_regime_summary_path))


def _write_report(frame: pd.DataFrame, path: Path) -> None:
    """Persist one semantic report in CSV or Excel form.

    Args:
        frame: Semantic dataframe to export.
        path: Output path whose suffix selects the persistence backend.

    Returns:
        ``None``. The dataframe is written to disk.
    """
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
        return
    frame.to_excel(path, index=False)


def _detect_language(argv: list[str]) -> str:
    """Infer the CLI language from the raw command line.

    Args:
        argv: Raw command-line argument list.

    Returns:
        Requested language code or ``"en"`` when no explicit language was
        provided.
    """
    if "--lang" in argv:
        try:
            return argv[argv.index("--lang") + 1]
        except (IndexError, ValueError):
            return "en"
    return "en"


if __name__ == "__main__":
    main()
