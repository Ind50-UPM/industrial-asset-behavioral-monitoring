"""Concatenate monthly CSV datasets and export one Parquet file per CSV type."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


DEFAULT_FILENAMES = (
    "palabras.csv",
    "secuencias.csv",
    "estados_nonans.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan monthly subdirectories, concatenate matching CSV files, "
            "and write one Parquet file per CSV name."
        )
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Base directory containing monthly subdirectories. Default: data",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where Parquet files will be written. Default: same as --data-dir",
    )
    parser.add_argument(
        "--filenames",
        nargs="+",
        default=list(DEFAULT_FILENAMES),
        help=(
            "CSV filenames to aggregate across monthly folders. "
            f"Default: {' '.join(DEFAULT_FILENAMES)}"
        ),
    )
    parser.add_argument(
        "--include-source-month",
        action="store_true",
        help="Add a source_month column derived from the subdirectory name.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the files that would be processed without writing output.",
    )
    return parser.parse_args()


def discover_month_dirs(data_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in data_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    )


def collect_existing_files(month_dirs: list[Path], filename: str) -> list[Path]:
    return [month_dir / filename for month_dir in month_dirs if (month_dir / filename).is_file()]


def load_and_concat(csv_paths: list[Path], include_source_month: bool) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        if include_source_month:
            frame.insert(0, "source_month", csv_path.parent.name)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def parquet_name_for(filename: str) -> str:
    return f"{Path(filename).stem}.parquet"


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else data_dir

    if not data_dir.is_dir():
        print(f"Data directory does not exist: {data_dir}", file=sys.stderr)
        return 1

    month_dirs = discover_month_dirs(data_dir)
    if not month_dirs:
        print(f"No monthly subdirectories found in: {data_dir}", file=sys.stderr)
        return 1

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    exit_code = 0
    for filename in args.filenames:
        csv_paths = collect_existing_files(month_dirs, filename)
        if not csv_paths:
            print(f"Skipped {filename}: no matching files found.")
            exit_code = 1
            continue

        print(f"{filename}: found {len(csv_paths)} files")
        for csv_path in csv_paths:
            print(f"  - {csv_path}")

        if args.dry_run:
            continue

        combined = load_and_concat(csv_paths, args.include_source_month)
        parquet_path = output_dir / parquet_name_for(filename)
        combined.to_parquet(parquet_path, index=False)
        print(f"Wrote {len(combined)} rows to {parquet_path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
