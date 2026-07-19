"""Deviation scoring utilities for Model_D."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DeviationWeights, WindowConfig


INDICATOR_COLUMNS = {
    "sequence_divergence": "sequence",
    "duration_drift": "duration",
    "recurrence_excess": "recurrence",
    "persistence_excess": "persistence",
    "consumption_deviation": "consumption",
    "state_error_rate": "state_error",
    "mode_divergence": "mode",
}
PREWINDOWED_COLUMNS = {"window_start", "window_end", "data_coverage", "sequence_count", "active_sequence_count"}
TIME_COLUMN_ALIASES = {
    "window_start": "start_time",
    "window_end": "end_time",
}
COUNT_COLUMN_ALIASES = {
    "active_sequence_count": "run_count",
}


@dataclass(frozen=True)
class WindowIndicators:
    """Container describing a single window-level anomaly profile."""

    start_time: pd.Timestamp
    end_time: pd.Timestamp
    sequence_divergence: float
    duration_drift: float
    recurrence_excess: float
    persistence_excess: float
    consumption_deviation: float
    state_error_rate: float
    mode_divergence: float


def compute_deviation_score(frame: pd.DataFrame, weights: DeviationWeights) -> pd.Series:
    """Compute a composite deviation score from available indicators."""

    if any(column in frame.columns for column in INDICATOR_COLUMNS):
        score = pd.Series(0.0, index=frame.index, dtype=float)
        for column, weight_name in INDICATOR_COLUMNS.items():
            values = pd.to_numeric(
                frame[column] if column in frame.columns else pd.Series(0.0, index=frame.index),
                errors="coerce",
            ).fillna(0.0)
            score = score + getattr(weights, weight_name) * values
        return score

    if "semantic_status" in frame.columns:
        return frame["semantic_status"].eq("ANOMALOUS").astype(float)

    raise ValueError(
        "Unable to compute deviation score: expected semantic_status or at least one deviation indicator column."
    )


def build_window_scores(
    sequences: pd.DataFrame,
    assignments: pd.DataFrame,
    weights: DeviationWeights,
    window_config: WindowConfig,
) -> pd.DataFrame:
    """Build scored windows from raw timed rows or accept pre-windowed inputs."""

    sequence_frame = _normalize_window_input(sequences).reset_index(drop=True).copy()
    assignment_frame = _normalize_window_input(assignments).reset_index(drop=True).copy()

    if _is_prewindowed(sequence_frame):
        merged = sequence_frame.copy()
        for column in assignment_frame.columns:
            if column not in merged.columns:
                merged[column] = assignment_frame[column]
        return _finalize_window_frame(merged, weights)

    overlapping = [column for column in assignment_frame.columns if column in sequence_frame.columns]
    protected = {'semantic_status', 'incident_family', 'anomaly_score'}
    drop_from_assignment = [column for column in overlapping if column not in protected]
    if drop_from_assignment:
        assignment_frame = assignment_frame.drop(columns=drop_from_assignment)

    merged = pd.concat([sequence_frame, assignment_frame], axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()].copy()
    return _build_rolling_windows(merged, weights, window_config)


def _build_rolling_windows(
    merged: pd.DataFrame,
    weights: DeviationWeights,
    window_config: WindowConfig,
) -> pd.DataFrame:
    """Construct explicit rolling windows from timed sequence rows."""

    frame = merged.loc[:, ~merged.columns.duplicated()].copy()
    frame["start_time"] = pd.to_datetime(frame["start_time"], errors="coerce", utc=True).dt.tz_convert(None)
    frame["end_time"] = pd.to_datetime(frame["end_time"], errors="coerce", utc=True).dt.tz_convert(None)
    frame = frame.dropna(subset=["start_time", "end_time"]).reset_index(drop=True)
    if frame.empty:
        return _finalize_window_frame(frame, weights)

    if not any(column in frame.columns for column in INDICATOR_COLUMNS):
        return _finalize_window_frame(_build_semantic_windows_fast(frame, window_config), weights)

    min_start = frame["start_time"].min().floor(window_config.step)
    max_end = frame["end_time"].max()
    rows: list[dict[str, object]] = []
    current_start = min_start
    while current_start <= max_end:
        current_end = current_start + window_config.length
        window_rows = frame[(frame["start_time"] < current_end) & (frame["end_time"] > current_start)]
        window_payload = _summarize_window(window_rows, current_start, current_end, window_config)
        if window_payload is not None:
            rows.append(window_payload)
        current_start = current_start + window_config.step

    return _finalize_window_frame(pd.DataFrame(rows), weights)


def _build_semantic_windows_fast(
    frame: pd.DataFrame,
    window_config: WindowConfig,
) -> pd.DataFrame:
    """Build semantic-only windows using array operations instead of DataFrame slicing."""

    if frame.empty:
        return pd.DataFrame()

    starts = frame["start_time"].to_numpy(dtype="datetime64[ns]")
    ends = frame["end_time"].to_numpy(dtype="datetime64[ns]")
    semantic_values = (
        frame["semantic_status"].fillna("NORMAL").astype(str).to_numpy()
        if "semantic_status" in frame.columns
        else np.full(len(frame), "NORMAL", dtype=object)
    )
    family_values = (
        frame["incident_family"].fillna("unclassified_incident").astype(str).to_numpy()
        if "incident_family" in frame.columns
        else np.full(len(frame), "unclassified_incident", dtype=object)
    )
    asset_values = (
        frame["asset_id"].fillna("asset-unknown").astype(str).to_numpy()
        if "asset_id" in frame.columns
        else np.full(len(frame), "asset-unknown", dtype=object)
    )

    starts_ns = starts.astype("int64")
    ends_ns = ends.astype("int64")
    length_seconds = window_config.length.total_seconds()
    step = window_config.step
    min_start = frame["start_time"].min().floor(step)
    max_end = frame["end_time"].max()

    rows: list[dict[str, object]] = []
    current_start = min_start
    while current_start <= max_end:
        current_end = current_start + window_config.length
        current_start_ns = current_start.to_datetime64().astype("datetime64[ns]").astype("int64")
        current_end_ns = current_end.to_datetime64().astype("datetime64[ns]").astype("int64")
        mask = (starts_ns < current_end_ns) & (ends_ns > current_start_ns)
        if not bool(mask.any()):
            current_start = current_start + step
            continue

        overlap_start = np.maximum(starts_ns[mask], current_start_ns)
        overlap_end = np.minimum(ends_ns[mask], current_end_ns)
        overlap_seconds = np.maximum((overlap_end - overlap_start) / 1_000_000_000.0, 0.0)
        active_mask = overlap_seconds > 0.0
        if not bool(active_mask.any()):
            current_start = current_start + step
            continue

        overlap_seconds = overlap_seconds[active_mask]
        coverage = min(float(overlap_seconds.sum()) / length_seconds, 1.0)
        if coverage < window_config.min_coverage:
            current_start = current_start + step
            continue
        active_count = int(active_mask.sum())
        if active_count < window_config.min_active_sequences:
            current_start = current_start + step
            continue

        active_semantic = semantic_values[mask][active_mask]
        active_family = family_values[mask][active_mask]
        anomalous_mask = active_semantic == "ANOMALOUS"
        family_source = active_family[anomalous_mask] if anomalous_mask.any() else active_family
        if family_source.size:
            family_mode = pd.Series(family_source).mode(dropna=True)
            dominant_family = str(family_mode.iloc[0]) if not family_mode.empty else "unclassified_incident"
        else:
            dominant_family = "unclassified_incident"

        rows.append(
            {
                "asset_id": str(asset_values[mask][active_mask][0]) if active_count else None,
                "start_time": current_start,
                "end_time": current_end,
                "data_coverage": coverage,
                "sequence_count": active_count,
                "active_sequence_count": active_count,
                "semantic_status": "ANOMALOUS" if anomalous_mask.any() else "NORMAL",
                "incident_family": dominant_family,
            }
        )
        current_start = current_start + step

    return pd.DataFrame(rows)


def _summarize_window(
    window_rows: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    window_config: WindowConfig,
) -> dict[str, object] | None:
    """Summarize a single rolling window."""

    if window_rows.empty:
        return None

    start_clip = window_rows["start_time"].where(window_rows["start_time"] > window_start, window_start)
    end_clip = window_rows["end_time"].where(window_rows["end_time"] < window_end, window_end)
    overlap_seconds = (end_clip - start_clip).dt.total_seconds().clip(lower=0.0)
    active_mask = overlap_seconds.gt(0)
    if not bool(active_mask.any()):
        return None

    active_rows = window_rows.loc[active_mask]
    overlap_seconds = overlap_seconds.loc[active_mask]
    coverage = min(float(overlap_seconds.sum()) / window_config.length.total_seconds(), 1.0)
    if coverage < window_config.min_coverage:
        return None
    if len(active_rows) < window_config.min_active_sequences:
        return None

    weighted = {
        column: _weighted_mean(active_rows[column], overlap_seconds)
        for column in INDICATOR_COLUMNS
        if column in active_rows.columns
    }

    if "semantic_status" in active_rows.columns:
        semantic_series = active_rows["semantic_status"].fillna("NORMAL")
    else:
        semantic_series = pd.Series("NORMAL", index=active_rows.index)
    is_anomalous = semantic_series.eq("ANOMALOUS")

    dominant_family = None
    if "incident_family" in active_rows.columns:
        family_source = active_rows.loc[is_anomalous, "incident_family"] if bool(is_anomalous.any()) else active_rows["incident_family"]
        family_mode = family_source.dropna().mode()
        dominant_family = family_mode.iloc[0] if not family_mode.empty else None

    return {
        "asset_id": active_rows["asset_id"].iloc[0] if "asset_id" in active_rows.columns else None,
        "start_time": window_start,
        "end_time": window_end,
        "data_coverage": coverage,
        "sequence_count": int(len(active_rows)),
        "active_sequence_count": int(len(active_rows)),
        "semantic_status": "ANOMALOUS" if bool(is_anomalous.any()) else "NORMAL",
        "incident_family": dominant_family or "unclassified_incident",
        **weighted,
    }


def _finalize_window_frame(frame: pd.DataFrame, weights: DeviationWeights) -> pd.DataFrame:
    """Normalize and score a window frame regardless of its source."""

    if frame.empty:
        return pd.DataFrame(
            columns=[
                "asset_id",
                "start_time",
                "end_time",
                "data_coverage",
                "sequence_count",
                "active_sequence_count",
                "semantic_status",
                "incident_family",
                *INDICATOR_COLUMNS.keys(),
                "deviation_score",
            ]
        )

    window_frame = frame.copy()
    window_frame["start_time"] = pd.to_datetime(window_frame["start_time"])
    window_frame["end_time"] = pd.to_datetime(window_frame["end_time"])
    data_coverage = window_frame["data_coverage"] if "data_coverage" in window_frame.columns else pd.Series(1.0, index=window_frame.index)
    active_sequence_count = window_frame["run_count"] if "run_count" in window_frame.columns else pd.Series(1, index=window_frame.index)
    sequence_count = window_frame["sequence_count"] if "sequence_count" in window_frame.columns else pd.Series(1, index=window_frame.index)
    window_frame["data_coverage"] = pd.to_numeric(data_coverage, errors="coerce").fillna(1.0)
    window_frame["sequence_count"] = pd.to_numeric(sequence_count, errors="coerce").fillna(1).astype(int)
    window_frame["active_sequence_count"] = pd.to_numeric(active_sequence_count, errors="coerce").fillna(1).astype(int)
    window_frame["deviation_score"] = compute_deviation_score(window_frame, weights)
    return window_frame.sort_values("start_time").reset_index(drop=True)


def _normalize_window_input(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize supported window input column conventions."""

    normalized = frame.copy()
    for source, target in TIME_COLUMN_ALIASES.items():
        if source in normalized.columns and target not in normalized.columns:
            normalized[target] = normalized[source]
    for source, target in COUNT_COLUMN_ALIASES.items():
        if source in normalized.columns and target not in normalized.columns:
            normalized[target] = normalized[source]
    return normalized


def _is_prewindowed(frame: pd.DataFrame) -> bool:
    """Return whether the input already represents analysis windows."""

    return bool(PREWINDOWED_COLUMNS.intersection(frame.columns))


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Compute a weighted mean with safe fallbacks for empty or invalid data."""

    numeric_values = pd.to_numeric(values, errors="coerce")
    numeric_weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    mask = numeric_values.notna() & numeric_weights.gt(0)
    if not bool(mask.any()):
        return 0.0
    return float((numeric_values[mask] * numeric_weights[mask]).sum() / numeric_weights[mask].sum())


def _overlap_seconds(
    left_start: pd.Timestamp,
    left_end: pd.Timestamp,
    right_start: pd.Timestamp,
    right_end: pd.Timestamp,
) -> float:
    """Compute the temporal overlap in seconds between two intervals."""

    overlap_start = max(left_start, pd.to_datetime(right_start))
    overlap_end = min(left_end, pd.to_datetime(right_end))
    return max((overlap_end - overlap_start).total_seconds(), 0.0)
