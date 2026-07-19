"""Episode-level summary metrics."""

from __future__ import annotations

import pandas as pd


def summarize_episode_metrics(episodes: pd.DataFrame) -> pd.DataFrame:
    """Summarize episode metrics grouped by primary family."""

    if episodes.empty:
        return pd.DataFrame(
            columns=[
                "primary_family",
                "episode_count",
                "mean_duration_seconds",
                "median_duration_seconds",
                "mean_time_to_recovery_seconds",
                "median_time_to_recovery_seconds",
                "recovery_rate",
                "mean_peak_score",
            ]
        )

    frame = episodes.copy()
    frame["duration_seconds"] = pd.to_numeric(frame.get("duration_seconds"), errors="coerce")
    frame["time_to_recovery_seconds"] = pd.to_numeric(frame.get("time_to_recovery_seconds"), errors="coerce")
    if "peak_score" not in frame.columns:
        fallback = frame.get("mean_score", pd.Series(0.0, index=frame.index))
        frame["peak_score"] = fallback
    frame["peak_score"] = pd.to_numeric(frame["peak_score"], errors="coerce")
    if "recovery_status" not in frame.columns:
        frame["recovery_status"] = "unknown"

    rows: list[dict[str, object]] = []
    for family, family_frame in frame.groupby("primary_family", dropna=False):
        recovery_series = family_frame["time_to_recovery_seconds"].dropna()
        rows.append(
            {
                "primary_family": family,
                "episode_count": int(family_frame["episode_id"].size),
                "mean_duration_seconds": float(family_frame["duration_seconds"].mean()),
                "median_duration_seconds": float(family_frame["duration_seconds"].median()),
                "mean_time_to_recovery_seconds": float(recovery_series.mean()) if not recovery_series.empty else None,
                "median_time_to_recovery_seconds": float(recovery_series.median()) if not recovery_series.empty else None,
                "recovery_rate": float(family_frame["recovery_status"].eq("stable").mean()),
                "mean_peak_score": float(family_frame["peak_score"].mean()),
            }
        )

    return pd.DataFrame(rows).sort_values("episode_count", ascending=False)
