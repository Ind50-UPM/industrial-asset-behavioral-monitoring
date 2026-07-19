"""Generate candidate incident registries month by month and consolidate the outputs.

This utility exists to make large-scale incident-registry generation robust on
historical industrial campaigns where full-table in-memory runs may be too slow
or too fragile. It processes each ``source_month`` independently, persists a
monthly artifact, records traceability in a manifest, and finally produces a
consolidated flat-file registry consumed by downstream analysis steps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Model_D.iabm_incidents.registry_builder import CandidateIncidentRegistryBuilder, RegistryGenerationConfig


DEFAULT_STATES = "data/estados_nonans.parquet"
DEFAULT_SEQUENCES = "data/secuencias.parquet"
DEFAULT_WORDS = "data/palabras.parquet"
DEFAULT_OUTPUT_DIR = "src/predictions/Model_D"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for batch candidate-registry generation.

    Returns:
        Parsed command-line arguments describing input parquet locations,
        optional month filters, and output behavior.

    Notes:
        The CLI is intentionally restart-friendly. ``--skip-existing-months``
        allows long campaigns to resume from already written monthly parquet
        artifacts without recomputing earlier months.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build candidate incident registries month by month from parquet inputs, "
            "persist partial outputs, and consolidate a final registry and summary."
        )
    )
    parser.add_argument('--states', default=DEFAULT_STATES, help=f'States parquet path. Default: {DEFAULT_STATES}')
    parser.add_argument('--sequences', default=DEFAULT_SEQUENCES, help=f'Sequences parquet path. Default: {DEFAULT_SEQUENCES}')
    parser.add_argument('--words', default=DEFAULT_WORDS, help=f'Words parquet path. Default: {DEFAULT_WORDS}')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, help=f'Output directory. Default: {DEFAULT_OUTPUT_DIR}')
    parser.add_argument('--months', nargs='+', help='Optional explicit source_month values to process.')
    parser.add_argument('--output-stem', default='candidate_incident_registry_refined', help='Base name for consolidated outputs.')
    parser.add_argument('--keep-monthly-csv', action='store_true', help='Also write one CSV per month in addition to monthly parquet files.')
    parser.add_argument('--skip-existing-months', action='store_true', help='Reuse existing monthly parquet outputs when present.')
    parser.add_argument('--fail-fast', action='store_true', help='Stop at the first month that raises an exception.')
    return parser.parse_args()


def collect_months(states_path: Path, sequences_path: Path, words_path: Path) -> list[str]:
    """Collect every ``source_month`` available across the three parquet inputs.

    Args:
        states_path: Path to the state-history parquet.
        sequences_path: Path to the sequence-history parquet.
        words_path: Path to the sequence-word parquet.

    Returns:
        Sorted month identifiers detected across the available sources.

    Notes:
        The union of all months is used so that installation stops or partially
        missing exports can still be represented in the manifest as empty months
        rather than silently disappearing from the orchestration layer.
    """
    months: set[str] = set()
    for parquet_path in [states_path, sequences_path, words_path]:
        if not parquet_path.is_file():
            continue
        frame = pd.read_parquet(parquet_path, columns=['source_month'])
        months.update(frame['source_month'].dropna().astype(str).unique().tolist())
    return sorted(months)


def load_month_frame(parquet_path: Path, month: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Load a single month slice from a parquet file.

    Args:
        parquet_path: Parquet file containing a ``source_month`` column.
        month: Month identifier to filter.
        columns: Optional column projection to reduce I/O and memory.

    Returns:
        A month-filtered dataframe, or an empty frame if the parquet is absent.
    """
    if not parquet_path.is_file():
        return pd.DataFrame()
    return pd.read_parquet(parquet_path, columns=columns, filters=[('source_month', '==', month)])


def build_month_registry(
    builder: CandidateIncidentRegistryBuilder,
    *,
    month: str,
    states_path: Path,
    sequences_path: Path,
    words_path: Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build the weakly labeled registry for a single month.

    Args:
        builder: Configured builder implementing the heuristic incident logic.
        month: Month identifier being processed.
        states_path: Monthly source of state and power telemetry.
        sequences_path: Monthly source of state-run sequence summaries.
        words_path: Monthly source of sequence-word summaries.

    Returns:
        A tuple containing the month registry and basic input row counts.

    Notes:
        The returned counts are written to the manifest so operational users can
        later distinguish empty installations from failed computation.
    """
    states = load_month_frame(
        states_path,
        month,
        columns=['source_month', 'date', 'RP1', 'RP2', 'RP3', 'RP4', 'estado', 'pred_estado'],
    )
    sequences = load_month_frame(sequences_path, month)
    words = load_month_frame(words_path, month)

    counts = {
        'states_rows': int(len(states)),
        'sequences_rows': int(len(sequences)),
        'words_rows': int(len(words)),
    }
    if states.empty and sequences.empty and words.empty:
        return builder._empty_final_registry(), counts

    registry = builder.build_from_frames(states=states, sequences=sequences, words=words)
    return registry, counts


def consolidate_registries(monthly_paths: list[Path]) -> pd.DataFrame:
    """Concatenate monthly registry parquet artifacts into one flat dataframe.

    Args:
        monthly_paths: Paths to monthly parquet files written during the batch.

    Returns:
        A consolidated registry sorted by ``event_time`` and reindexed with a
        fresh, sequential ``incident_id`` namespace.

    Notes:
        Incident identifiers are reassigned after concatenation so the final
        registry is deterministic regardless of whether some months were reused,
        skipped, or regenerated.
    """
    frames = [pd.read_parquet(path) for path in monthly_paths if path.is_file()]
    if not frames:
        return pd.DataFrame(
            columns=[
                'incident_id', 'source_window_start', 'source_window_end', 'documented_start', 'documented_end',
                'event_time', 'event_time_precision', 'incident_family', 'family', 'secondary_family',
                'recovery_time', 'recovery_status', 'label_strength', 'source_type', 'downtime_start',
                'downtime_end', 'maintenance_time', 'affected_subsystem', 'notes', 'asset_id',
            ]
        )
    full_registry = pd.concat(frames, ignore_index=True)
    if 'event_time' in full_registry.columns:
        full_registry = full_registry.sort_values('event_time').reset_index(drop=True)
    full_registry['incident_id'] = [f"AUTO-{index + 1:06d}" for index in range(len(full_registry))]
    return full_registry


def write_summary(registry: pd.DataFrame, summary_path: Path) -> None:
    """Write the final family/source summary for the consolidated registry.

    Args:
        registry: Consolidated candidate incident registry.
        summary_path: Destination CSV path.

    Returns:
        ``None``. The summary is persisted to disk.
    """
    summary = (
        registry.groupby(['incident_family', 'secondary_family', 'source_type'], dropna=False)
        .size()
        .reset_index(name='candidate_count')
        .sort_values(['candidate_count', 'incident_family', 'secondary_family'], ascending=[False, True, True])
    )
    summary.to_csv(summary_path, index=False)


def main() -> int:
    """Execute the month-wise candidate-registry orchestration.

    Returns:
        Process-style exit code where ``0`` means successful completion and
        ``1`` indicates missing inputs or at least one failed month.

    Notes:
        The function is intentionally verbose in its artifact writing:
        monthly parquet files, a manifest CSV, a metadata JSON file, the
        consolidated registry, and the consolidated summary are all written so
        long historical campaigns remain auditable and restartable.
    """
    args = parse_args()
    states_path = Path(args.states).resolve()
    sequences_path = Path(args.sequences).resolve()
    words_path = Path(args.words).resolve()
    output_dir = Path(args.output_dir).resolve()
    monthly_dir = output_dir / 'monthly_candidate_registry'
    output_dir.mkdir(parents=True, exist_ok=True)
    monthly_dir.mkdir(parents=True, exist_ok=True)

    months = args.months or collect_months(states_path, sequences_path, words_path)
    if not months:
        print('No source_month values found in the provided parquet files.', file=sys.stderr)
        return 1

    builder = CandidateIncidentRegistryBuilder(RegistryGenerationConfig())
    monthly_paths: list[Path] = []
    manifest_rows: list[dict[str, object]] = []
    failures = 0

    for month in months:
        monthly_parquet = monthly_dir / f'{month}.parquet'
        monthly_csv = monthly_dir / f'{month}.csv'
        if args.skip_existing_months and monthly_parquet.is_file():
            existing = pd.read_parquet(monthly_parquet)
            monthly_paths.append(monthly_parquet)
            manifest_rows.append({
                'source_month': month,
                'status': 'reused',
                'registry_rows': int(len(existing)),
                'states_rows': None,
                'sequences_rows': None,
                'words_rows': None,
                'secondary_family_non_null': int(existing['secondary_family'].notna().sum()) if 'secondary_family' in existing.columns else 0,
                'error': None,
            })
            print(f'[{month}] reused {len(existing)} rows from {monthly_parquet}')
            continue

        try:
            registry, counts = build_month_registry(
                builder,
                month=month,
                states_path=states_path,
                sequences_path=sequences_path,
                words_path=words_path,
            )
            if registry.empty:
                manifest_rows.append({
                    'source_month': month,
                    'status': 'empty',
                    'registry_rows': 0,
                    'states_rows': counts['states_rows'],
                    'sequences_rows': counts['sequences_rows'],
                    'words_rows': counts['words_rows'],
                    'secondary_family_non_null': 0,
                    'error': None,
                })
                print(f'[{month}] empty month or installation stop; skipped final monthly artifact')
                continue

            registry.to_parquet(monthly_parquet, index=False)
            if args.keep_monthly_csv:
                registry.to_csv(monthly_csv, index=False)
            monthly_paths.append(monthly_parquet)
            manifest_rows.append({
                'source_month': month,
                'status': 'written',
                'registry_rows': int(len(registry)),
                'states_rows': counts['states_rows'],
                'sequences_rows': counts['sequences_rows'],
                'words_rows': counts['words_rows'],
                'secondary_family_non_null': int(registry['secondary_family'].notna().sum()) if 'secondary_family' in registry.columns else 0,
                'error': None,
            })
            print(
                f'[{month}] wrote {len(registry)} rows '
                f'(secondary_family non-null: {int(registry["secondary_family"].notna().sum()) if "secondary_family" in registry.columns else 0})'
            )
        except Exception as exc:  # pragma: no cover - operational CLI safeguard
            failures += 1
            manifest_rows.append({
                'source_month': month,
                'status': 'failed',
                'registry_rows': None,
                'states_rows': None,
                'sequences_rows': None,
                'words_rows': None,
                'secondary_family_non_null': None,
                'error': f'{type(exc).__name__}: {exc}',
            })
            print(f'[{month}] failed: {type(exc).__name__}: {exc}', file=sys.stderr)
            if args.fail_fast:
                break

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / f'{args.output_stem}_manifest.csv'
    manifest.to_csv(manifest_path, index=False)

    successful_paths = [path for path in monthly_paths if path.is_file()]
    full_registry = consolidate_registries(successful_paths)
    registry_path = output_dir / f'{args.output_stem}.csv'
    summary_path = output_dir / f'{args.output_stem}_summary.csv'
    full_registry.to_csv(registry_path, index=False)
    write_summary(full_registry, summary_path)

    metadata = {
        'months_requested': months,
        'months_written': manifest.loc[manifest['status'].isin(['written', 'reused']), 'source_month'].astype(str).tolist(),
        'empty_months': manifest.loc[manifest['status'] == 'empty', 'source_month'].astype(str).tolist(),
        'failed_months': manifest.loc[manifest['status'] == 'failed', 'source_month'].astype(str).tolist(),
        'failure_count': failures,
        'final_registry_rows': int(len(full_registry)),
        'outputs': {
            'registry_csv': str(registry_path),
            'summary_csv': str(summary_path),
            'manifest_csv': str(manifest_path),
            'monthly_dir': str(monthly_dir),
        },
    }
    metadata_path = output_dir / f'{args.output_stem}_metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    print(f'Manifest saved to: {manifest_path}')
    print(f'Consolidated registry saved to: {registry_path}')
    print(f'Summary saved to: {summary_path}')
    print(f'Metadata saved to: {metadata_path}')
    if failures:
        print(f'Completed with {failures} failed months.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
