from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from Model_D.iabm_incidents import AnticipationConfig, AnticipatoryMaintenanceForecaster

DATASET_ROOTS = {
    "probe": Path("src/predictions/episode_eval_probe"),
    "full": Path("src/predictions/episode_eval_full"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate anticipatory maintenance report from retained Model_D outputs.")
    parser.add_argument("--dataset", choices=sorted(DATASET_ROOTS), default="probe")
    parser.add_argument("--horizons", nargs="*", type=int, default=[24, 72, 168])
    parser.add_argument("--risk-threshold", type=float, default=0.55)
    parser.add_argument("--warning-threshold", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = DATASET_ROOTS[args.dataset]
    window_scores = pd.read_csv(root / "Model_D_episodes_localized_fallback" / "window_scores.csv", parse_dates=["start_time", "end_time"])
    episodes = pd.read_csv(root / "Model_D_episodes_localized_fallback" / "detected_episodes.csv", parse_dates=["event_start", "event_end"])
    forecaster = AnticipatoryMaintenanceForecaster(
        AnticipationConfig(
            horizons_hours=tuple(args.horizons),
            risk_threshold=args.risk_threshold,
            warning_threshold=args.warning_threshold,
        )
    )
    result = forecaster.run(window_scores=window_scores, episodes=episodes)
    output_dir = root / "anticipation_report"
    forecaster.export_report(result, output_dir)
    print(f"Anticipation report written to: {output_dir}")


if __name__ == "__main__":
    main()
