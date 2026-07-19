"""Command-line entry point for Model_D incident episode analysis.

The CLI exposed by this module supports both the legacy semantic route and the
new indicator-driven route. It accepts either a set of independent tabular
inputs or a consolidated Excel workbook with well-known sheet names. The goal
is to keep operational batch execution simple while the underlying domain model
continues to evolve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import load_model_d_config
from .episodes import IncidentEpisodeBuilder
from .evaluation import EpisodeEvaluator
from .metrics import summarize_episode_metrics
from .occurrence import OccurrenceModeler
from .pipeline import IndicatorPipeline
from .registry import IncidentRegistry


DEFAULT_SHEETS = {
    'sequences': 'sequences',
    'assignments': 'assignments',
    'registry': 'registry',
}
REAL_WORKBOOK_SHEETS = {
    'window_indicators': 'Window_indicators',
    'registry': 'Incident_registry',
    'exposure': 'Observation_exposure',
}
REQUIRED_STANDARD_SEQUENCE_COLUMNS = {'start_time', 'end_time'}
REQUIRED_STANDARD_ASSIGNMENT_COLUMNS = {'semantic_status', 'incident_family'}
REQUIRED_WINDOW_INDICATOR_COLUMNS = {
    'asset_id',
    'window_start',
    'window_end',
    'sequence_divergence',
    'duration_drift',
    'recurrence_excess',
    'persistence_excess',
    'consumption_deviation',
    'state_error_rate',
    'mode_divergence',
    'data_coverage',
    'sequence_count',
    'active_sequence_count',
    'semantic_status',
    'incident_family',
}
REQUIRED_REGISTRY_COLUMNS = {
    'incident_id',
    'asset_id',
    'documented_start',
    'documented_end',
    'incident_family',
    'label_strength',
}
REQUIRED_EXPOSURE_COLUMNS = {
    'asset_id',
    'observation_start',
    'observation_end',
    'excluded_start',
    'excluded_end',
    'exclusion_reason',
}
FAMILY_ASSIGNMENT_COLUMNS = [
    'episode_id',
    'primary_family',
    'secondary_families',
    'family_confidence',
    'assignment_method',
    'evidence',
]
RECOVERY_COLUMNS = [
    'episode_id',
    'recovery_start',
    'recovery_end',
    'time_to_recovery_seconds',
    'recovery_status',
]


class WorkbookValidationError(ValueError):
    """Raised when workbook inputs do not satisfy the expected contract."""


def parse_arguments() -> argparse.Namespace:
    """Parse the CLI arguments for Model_D execution.

    Returns:
        Namespace with validated command-line options.
    """

    parser = argparse.ArgumentParser(
        description='Longitudinal incident episode construction for industrial asset monitoring'
    )
    parser.add_argument('--detection-mode', choices=['semantic', 'indicators'], default='semantic', help='Episode construction route. semantic keeps the legacy path; indicators builds windows and scores before segmentation.')
    parser.add_argument('--sequences', help='Path to Model_B active-sequence report.')
    parser.add_argument('--assignments', help='Path to Model_C assignment report.')
    parser.add_argument('--registry', help='Optional path to a canonical incident registry.')
    parser.add_argument('--analogue', help='Optional analog telemetry table for indicator-driven observation exposure.')
    parser.add_argument('--digital', help='Optional digital telemetry table for indicator-driven observation exposure.')
    parser.add_argument('--workbook', help='Optional single Excel workbook containing multiple Model_D input sheets.')
    parser.add_argument('--sequences-sheet', default=DEFAULT_SHEETS['sequences'], help='Sheet name for sequence inputs when --workbook is used.')
    parser.add_argument('--assignments-sheet', default=DEFAULT_SHEETS['assignments'], help='Sheet name for assignment inputs when --workbook is used.')
    parser.add_argument('--registry-sheet', default=DEFAULT_SHEETS['registry'], help='Sheet name for incident registry inputs when --workbook is used.')
    parser.add_argument('--config', help='Optional Model_D JSON configuration file.')
    parser.add_argument('--output-dir', required=True, help='Directory where Model_D reports will be written.')
    parser.add_argument('--output-format', '--output_format', dest='output_format', choices=['xlsx', 'csv'], default='xlsx', help='Report export format.')
    args = parser.parse_args()
    if not args.workbook and (not args.sequences or not args.assignments):
        parser.error('Provide either --workbook or both --sequences and --assignments.')
    return args


def main() -> None:
    """Run the Model_D pipeline from the command line.

    Notes:
        The command always writes fully materialized tabular outputs so that
        downstream analyses can be reproduced without re-executing the full
        segmentation logic.
    """

    args = parse_arguments()
    config = load_model_d_config(args.config)
    sequences, assignments, registry_frame, exposure_frame, prewindowed_inputs, analogue_frame, digital_frame = _resolve_inputs(args)

    builder = IncidentEpisodeBuilder(config)
    modeler = OccurrenceModeler(config.occurrence)
    registry = IncidentRegistry()
    evaluator = EpisodeEvaluator()
    pipeline = IndicatorPipeline(config)

    if args.detection_mode == 'indicators':
        if prewindowed_inputs is not None:
            window_scores = prewindowed_inputs
            observation_periods = exposure_frame if exposure_frame is not None else pd.DataFrame()
            windows = pd.DataFrame()
            episodes = builder.build_episodes_from_window_scores(window_scores)
        else:
            pipeline_result = pipeline.run(
                sequences=sequences,
                assignments=assignments,
                registry=registry_frame,
                analogue=analogue_frame,
                digital=digital_frame,
            )
            observation_periods = pipeline_result.observation_periods
            windows = pipeline_result.windows
            window_scores = pipeline_result.window_scores
            episodes = pipeline_result.episodes
    else:
        observation_periods = pd.DataFrame()
        windows = pd.DataFrame()
        window_scores = builder.build_window_scores(sequences, assignments)
        episodes = builder.build_episodes_from_semantic_assignments(sequences, assignments)
    metrics = summarize_episode_metrics(episodes)
    occurrence = modeler.summarize(episodes, exposure_frame)
    family_assignments = _select_columns(episodes, FAMILY_ASSIGNMENT_COLUMNS)
    recovery_assessment = _select_columns(episodes, RECOVERY_COLUMNS)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = args.output_format
    observation_periods_path = output_dir / f'observation_periods.{suffix}'
    windows_path = output_dir / f'analysis_windows.{suffix}'
    window_scores_path = output_dir / f'window_scores.{suffix}'
    episodes_path = output_dir / f'detected_episodes.{suffix}'
    family_assignments_path = output_dir / f'family_assignments.{suffix}'
    recovery_path = output_dir / f'recovery_assessment.{suffix}'
    metrics_path = output_dir / f'evaluation_summary.{suffix}'
    occurrence_path = output_dir / f'occurrence_summary.{suffix}'
    metadata_path = output_dir / 'model_d_run_metadata.json'

    if args.detection_mode == 'indicators':
        _write_report(observation_periods, observation_periods_path)
        _write_report(windows, windows_path)
    _write_report(window_scores, window_scores_path)
    _write_report(episodes, episodes_path)
    _write_report(family_assignments, family_assignments_path)
    _write_report(recovery_assessment, recovery_path)
    _write_report(metrics, metrics_path)
    _write_report(occurrence, occurrence_path)

    if registry_frame is not None:
        incidents = registry_frame if isinstance(registry_frame, pd.DataFrame) else registry.load(registry_frame)
        matches = evaluator.match(incidents, episodes)
        evaluation_summary = evaluator.summarize(incidents, episodes)
        _write_report(matches, output_dir / f'incident_registry_matches.{suffix}')
        _write_report(evaluation_summary, output_dir / f'registry_evaluation_summary.{suffix}')

    metadata_path.write_text(
        json.dumps(
            {
                'config': config.to_dict(),
                'inputs': {
                    'sequences': str(args.sequences) if args.sequences else None,
                    'assignments': str(args.assignments) if args.assignments else None,
                    'registry': str(args.registry) if args.registry else None,
                    'analogue': str(args.analogue) if args.analogue else None,
                    'digital': str(args.digital) if args.digital else None,
                    'workbook': str(args.workbook) if args.workbook else None,
                    'detection_mode': args.detection_mode,
                    'used_prewindowed_inputs': prewindowed_inputs is not None,
                    'sheets': {
                        'sequences': args.sequences_sheet,
                        'assignments': args.assignments_sheet,
                        'registry': args.registry_sheet,
                        'real_window_indicators': REAL_WORKBOOK_SHEETS['window_indicators'],
                        'real_exposure': REAL_WORKBOOK_SHEETS['exposure'],
                    },
                    'has_exposure_sheet': exposure_frame is not None,
                },
            },
            indent=2,
        ),
        encoding='utf-8',
    )

    if args.detection_mode == 'indicators':
        print(f'Observation periods saved to: {observation_periods_path}')
        print(f'Analysis windows saved to: {windows_path}')
    print(f'Window scores saved to: {window_scores_path}')
    print(f'Detected episodes saved to: {episodes_path}')
    print(f'Family assignments saved to: {family_assignments_path}')
    print(f'Recovery assessment saved to: {recovery_path}')
    print(f'Evaluation summary saved to: {metrics_path}')
    print(f'Occurrence summary saved to: {occurrence_path}')


def _resolve_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """Resolve Model_D inputs from flat files or a single workbook.

    Args:
        args: Parsed CLI arguments.

    Returns:
        A tuple containing sequences, assignments, optional registry, optional
        exposure frame, optional prewindowed indicator inputs, optional analog
        telemetry, and optional digital telemetry.
    """

    analogue_frame = _load_table(args.analogue) if args.analogue else None
    digital_frame = _load_table(args.digital) if args.digital else None

    if args.workbook:
        workbook_path = Path(args.workbook)
        if _sheet_exists(workbook_path, REAL_WORKBOOK_SHEETS['window_indicators']):
            window_indicators = _load_table(workbook_path, sheet_name=REAL_WORKBOOK_SHEETS['window_indicators'])
            _require_columns(
                window_indicators,
                REQUIRED_WINDOW_INDICATOR_COLUMNS,
                context=f"sheet '{REAL_WORKBOOK_SHEETS['window_indicators']}'",
            )
            sequences, assignments = _split_window_indicator_frame(window_indicators)

            registry_frame = None
            if _sheet_exists(workbook_path, REAL_WORKBOOK_SHEETS['registry']):
                registry_frame = _load_table(workbook_path, sheet_name=REAL_WORKBOOK_SHEETS['registry'])
                _require_columns(
                    registry_frame,
                    REQUIRED_REGISTRY_COLUMNS,
                    context=f"sheet '{REAL_WORKBOOK_SHEETS['registry']}'",
                )

            exposure_frame = None
            if _sheet_exists(workbook_path, REAL_WORKBOOK_SHEETS['exposure']):
                exposure_frame = _load_table(workbook_path, sheet_name=REAL_WORKBOOK_SHEETS['exposure'])
                exposure_frame = _normalize_exposure_frame(exposure_frame)
            return sequences, assignments, registry_frame, exposure_frame, _normalize_window_indicator_frame(window_indicators), analogue_frame, digital_frame

        _validate_standard_workbook(workbook_path, args)
        sequences = _load_table(workbook_path, sheet_name=args.sequences_sheet)
        assignments = _load_table(workbook_path, sheet_name=args.assignments_sheet)
        registry_frame = _load_table(workbook_path, sheet_name=args.registry_sheet) if _sheet_exists(workbook_path, args.registry_sheet) else None
        return sequences, assignments, registry_frame, None, None, analogue_frame, digital_frame

    sequences = _load_table(args.sequences)
    assignments = _load_table(args.assignments)
    _require_columns(sequences, REQUIRED_STANDARD_SEQUENCE_COLUMNS, context=f"file '{args.sequences}'")
    _require_columns(assignments, REQUIRED_STANDARD_ASSIGNMENT_COLUMNS, context=f"file '{args.assignments}'")
    registry_frame = None
    if args.registry:
        registry_frame = _load_table(args.registry)
        _require_columns(registry_frame, REQUIRED_REGISTRY_COLUMNS, context=f"file '{args.registry}'")
    return sequences, assignments, registry_frame, None, None, analogue_frame, digital_frame


def _validate_standard_workbook(workbook_path: Path, args: argparse.Namespace) -> None:
    """Validate the standard workbook layout before loading sheets."""

    available_sheets = set(pd.ExcelFile(workbook_path).sheet_names)
    required_sheets = {args.sequences_sheet, args.assignments_sheet}
    missing_sheets = sorted(required_sheets.difference(available_sheets))
    if missing_sheets:
        raise WorkbookValidationError(
            'Missing required workbook sheets: '
            f'{missing_sheets}. Available sheets: {sorted(available_sheets)}'
        )

    sequences = _load_table(workbook_path, sheet_name=args.sequences_sheet)
    assignments = _load_table(workbook_path, sheet_name=args.assignments_sheet)
    _require_columns(sequences, REQUIRED_STANDARD_SEQUENCE_COLUMNS, context=f"sheet '{args.sequences_sheet}'")
    _require_columns(assignments, REQUIRED_STANDARD_ASSIGNMENT_COLUMNS, context=f"sheet '{args.assignments_sheet}'")

    if _sheet_exists(workbook_path, args.registry_sheet):
        registry_frame = _load_table(workbook_path, sheet_name=args.registry_sheet)
        _require_columns(registry_frame, REQUIRED_REGISTRY_COLUMNS, context=f"sheet '{args.registry_sheet}'")


def _sheet_exists(workbook_path: Path, sheet_name: str) -> bool:
    """Return whether the given sheet is available in the workbook."""

    return sheet_name in pd.ExcelFile(workbook_path).sheet_names


def _split_window_indicator_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a unified window-indicator table into sequence and assignment views.

    Notes:
        Some historical workbooks store all window-level information in a single
        sheet. This helper reconstructs the two logical views expected by the
        internal API without losing columns required later for scoring.
    """

    normalized = _normalize_window_indicator_frame(frame)
    sequence_columns = [
        column
        for column in [
            'asset_id',
            'start_time',
            'end_time',
            'sequence_count',
            'active_sequence_count',
            'data_coverage',
        ]
        if column in normalized.columns
    ]
    assignment_columns = [column for column in normalized.columns if column not in sequence_columns]
    return normalized.loc[:, sequence_columns].copy(), normalized.loc[:, assignment_columns].copy()


def _normalize_window_indicator_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize workbook window columns to the internal window-score schema.

    Notes:
        When precomputed workbooks lack a global ``deviation_score``, the helper
        derives a simple aggregate by averaging the absolute indicator
        magnitudes. This preserves compatibility with the new segmentation
        route without requiring historical files to be regenerated.
    """

    normalized = frame.copy()
    if 'window_start' in normalized.columns and 'start_time' not in normalized.columns:
        normalized['start_time'] = normalized['window_start']
    if 'window_end' in normalized.columns and 'end_time' not in normalized.columns:
        normalized['end_time'] = normalized['window_end']
    normalized['start_time'] = pd.to_datetime(normalized['start_time'])
    normalized['end_time'] = pd.to_datetime(normalized['end_time'])
    if 'deviation_score' not in normalized.columns:
        deviation_components = [
            column
            for column in [
                'sequence_divergence',
                'duration_drift',
                'recurrence_excess',
                'persistence_excess',
                'consumption_deviation',
                'state_error_rate',
                'mode_divergence',
            ]
            if column in normalized.columns
        ]
        if deviation_components:
            normalized['deviation_score'] = (
                normalized.loc[:, deviation_components]
                .apply(pd.to_numeric, errors='coerce')
                .fillna(0.0)
                .abs()
                .mean(axis=1)
            )
        else:
            normalized['deviation_score'] = 0.0
    return normalized


def _normalize_exposure_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize workbook exposure sheets into the observation-period schema."""

    normalized = frame.copy()
    normalized = normalized.rename(
        columns={
            'observation_start': 'start_time',
            'observation_end': 'end_time',
            'exclusion_reason': 'reason',
        }
    )
    normalized['period_type'] = 'observed'
    normalized['source'] = 'workbook'
    normalized['confidence'] = 1.0
    normalized['exclude_from_baseline'] = False
    normalized['exclude_from_exposure'] = False
    if 'excluded_start' in normalized.columns and 'excluded_end' in normalized.columns:
        normalized.loc[
            normalized['excluded_start'].notna() & normalized['excluded_end'].notna(),
            'exclude_from_exposure',
        ] = True
    normalized['linked_incident_id'] = None
    return normalized


def _require_columns(frame: pd.DataFrame, required: set[str], *, context: str) -> None:
    """Ensure that the given table contains the required columns."""

    missing = sorted(required.difference(frame.columns))
    if missing:
        raise WorkbookValidationError(
            f'Missing required columns in {context}: {missing}. Present columns: {list(frame.columns)}'
        )


def _select_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return the requested columns or an empty frame with that schema."""

    if frame.empty:
        return pd.DataFrame(columns=columns)
    available = [column for column in columns if column in frame.columns]
    return frame.loc[:, available]


def _write_report(frame: pd.DataFrame, path: Path) -> None:
    """Write a report as CSV or Excel depending on the target extension."""

    if path.suffix.lower() == '.csv':
        frame.to_csv(path, index=False)
        return
    frame.to_excel(path, index=False)


def _load_table(file_path: str | Path, sheet_name: str | int | None = None) -> pd.DataFrame:
    """Load a report table from CSV, Parquet, or Excel."""

    path = Path(file_path)
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path)
    if path.suffix.lower() == '.parquet':
        return pd.read_parquet(path)
    if path.suffix.lower() in {'.xlsx', '.xls'}:
        return pd.read_excel(path, sheet_name=0 if sheet_name is None else sheet_name)
    raise ValueError(f'Unsupported file extension: {path.suffix}')
