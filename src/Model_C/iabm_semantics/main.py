"""Command-line entry point for Model_C semantic interpretation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import pandas as pd

from .semantics import SemanticModeInterpreter
from .utils import setup_i18n


def parse_arguments(translator: Callable[[str], str]) -> argparse.Namespace:
    """Build the CLI parser with translated help messages.

    Args:
        translator: Translation function returned by :func:`setup_i18n`.

    Returns:
        Parsed command-line arguments for semantic interpretation.
    """
    _ = translator
    parser = argparse.ArgumentParser(
        description=_("Semantic interpretation of industrial behavioral sequences")
    )
    parser.add_argument(
        "--input",
        required=True,
        help=_("Path to the Model_B active-sequence report."),
    )
    parser.add_argument(
        "--comparison-input",
        help=_("Optional Model_B comparison report used to enrich semantic status."),
    )
    parser.add_argument(
        "--rules",
        help=_("Optional JSON file with semantic interpretation rules."),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help=_("Directory where Model_C reports will be written."),
    )
    parser.add_argument(
        "--lang",
        default="en",
        choices=["es", "en"],
        help=_("Interface language."),
    )
    return parser.parse_args()


def main() -> None:
    """Run the Model_C semantic interpretation workflow from the command line.

    The workflow consumes sequence-level reports from Model_B, maps them to
    semantic operating and working modes, and optionally enriches the result
    with anomaly status coming from Model_B comparison outputs.
    """
    lang = _detect_language(sys.argv)
    translator = setup_i18n(lang)
    args = parse_arguments(translator)

    interpreter = SemanticModeInterpreter()
    if args.rules:
        # Explicit rules allow project-specific semantic vocabularies to
        # override the default heuristic mapping.
        interpreter.load_rules(args.rules)

    sequences = interpreter.load_active_sequences(args.input)
    comparison = (
        interpreter.load_comparison_report(args.comparison_input)
        if args.comparison_input
        else None
    )

    assignments = interpreter.interpret_sequences(sequences, comparison=comparison)
    summary = interpreter.summarize_modes(assignments)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assignments_path = output_dir / "semantic_assignments.xlsx"
    summary_path = output_dir / "semantic_mode_summary.xlsx"
    assignments.to_excel(assignments_path, index=False)
    summary.to_excel(summary_path, index=False)

    print(translator("Semantic assignments saved to: {}").format(assignments_path))
    print(translator("Semantic mode summary saved to: {}").format(summary_path))


def _detect_language(argv: list[str]) -> str:
    """Extract the requested language before parsing the translated CLI.

    Args:
        argv: Raw command-line token list.

    Returns:
        The requested language code, or ``"en"`` when no valid token is found.
    """
    if "--lang" in argv:
        try:
            return argv[argv.index("--lang") + 1]
        except (IndexError, ValueError):
            return "en"
    return "en"


if __name__ == "__main__":
    main()
