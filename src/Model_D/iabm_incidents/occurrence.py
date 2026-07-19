"""Family-specific occurrence modelling scaffolding."""

from __future__ import annotations

import pandas as pd

from .config import OccurrenceConfig


class OccurrenceModeler:
    """Derive simple family-specific occurrence indicators."""

    def __init__(self, config: OccurrenceConfig | None = None) -> None:
        self._config = config or OccurrenceConfig()

    def summarize(
        self,
        episodes: pd.DataFrame,
        exposure: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Summarize recurrence intervals and exposure-aware event rates by family."""

        if episodes.empty:
            return pd.DataFrame(
                columns=[
                    "primary_family",
                    "occurrence_count",
                    "mean_occurrence_interval_seconds",
                    "median_occurrence_interval_seconds",
                    "minimum_occurrence_interval_seconds",
                    "maximum_occurrence_interval_seconds",
                    "exposure_hours",
                    "excluded_hours",
                    "event_rate_per_hour",
                    "fit_status",
                ]
            )

        exposure_summary = _summarize_exposure(exposure)
        rows = []
        for family, family_frame in episodes.sort_values("event_start").groupby("primary_family"):
            starts = pd.to_datetime(family_frame["event_start"]).tolist()
            intervals = [
                (starts[index] - starts[index - 1]).total_seconds()
                for index in range(1, len(starts))
            ]
            asset_ids = family_frame["asset_id"].dropna().unique().tolist() if "asset_id" in family_frame.columns else []
            exposure_hours, excluded_hours = _exposure_for_assets(exposure_summary, asset_ids)
            rows.append(
                {
                    "primary_family": family,
                    "occurrence_count": len(family_frame),
                    "mean_occurrence_interval_seconds": sum(intervals) / len(intervals) if intervals else None,
                    "median_occurrence_interval_seconds": pd.Series(intervals).median() if intervals else None,
                    "minimum_occurrence_interval_seconds": min(intervals) if intervals else None,
                    "maximum_occurrence_interval_seconds": max(intervals) if intervals else None,
                    "exposure_hours": exposure_hours,
                    "excluded_hours": excluded_hours,
                    "event_rate_per_hour": len(family_frame) / exposure_hours if exposure_hours and exposure_hours > 0 else None,
                    "fit_status": "insufficient_events"
                    if len(family_frame) < self._config.minimum_events_for_fit
                    else "ready_for_distribution_fit",
                }
            )
        return pd.DataFrame(rows).sort_values("occurrence_count", ascending=False)


def _summarize_exposure(exposure: pd.DataFrame | None) -> pd.DataFrame:
    """Convert raw exposure intervals into effective observation summaries."""

    if exposure is None or exposure.empty:
        return pd.DataFrame(columns=["asset_id", "exposure_hours", "excluded_hours"])

    frame = exposure.copy()
    frame["observation_start"] = pd.to_datetime(frame["observation_start"])
    frame["observation_end"] = pd.to_datetime(frame["observation_end"])
    if "excluded_start" in frame.columns:
        frame["excluded_start"] = pd.to_datetime(frame["excluded_start"])
    if "excluded_end" in frame.columns:
        frame["excluded_end"] = pd.to_datetime(frame["excluded_end"])

    rows: list[dict[str, object]] = []
    for asset_id, asset_frame in frame.groupby("asset_id", dropna=False):
        observation_hours = (
            (asset_frame["observation_end"] - asset_frame["observation_start"]).dt.total_seconds().sum() / 3600.0
        )
        excluded_hours = 0.0
        if {"excluded_start", "excluded_end"}.issubset(asset_frame.columns):
            valid_exclusions = asset_frame[asset_frame["excluded_start"].notna() & asset_frame["excluded_end"].notna()]
            excluded_hours = (
                (valid_exclusions["excluded_end"] - valid_exclusions["excluded_start"]).dt.total_seconds().sum() / 3600.0
            )
        rows.append(
            {
                "asset_id": asset_id,
                "exposure_hours": max(observation_hours - excluded_hours, 0.0),
                "excluded_hours": excluded_hours,
            }
        )
    return pd.DataFrame(rows)


def _exposure_for_assets(summary: pd.DataFrame, asset_ids: list[str]) -> tuple[float | None, float | None]:
    """Resolve effective exposure for the assets participating in a family."""

    if summary.empty:
        return None, None
    if not asset_ids:
        return float(summary["exposure_hours"].sum()), float(summary["excluded_hours"].sum())
    filtered = summary[summary["asset_id"].isin(asset_ids)]
    if filtered.empty:
        return None, None
    return float(filtered["exposure_hours"].sum()), float(filtered["excluded_hours"].sum())
