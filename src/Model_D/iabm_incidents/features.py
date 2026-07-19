"""Episode-level feature extraction for Model_D classification and reporting."""

from __future__ import annotations

import pandas as pd


FEATURE_COLUMNS = [
    "episode_id",
    "asset_id",
    "duration_hours",
    "recovery_hours",
    "onset_slope",
    "peak_score",
    "mean_score",
    "max_sequence_divergence",
    "median_sequence_divergence",
    "max_duration_drift",
    "median_duration_drift",
    "max_recurrence_excess",
    "median_recurrence_excess",
    "max_persistence_excess",
    "median_persistence_excess",
    "signed_consumption_deviation",
    "min_consumption_deviation",
    "max_state_error_rate",
    "mean_state_error_rate",
    "max_mode_divergence",
    "mean_mode_divergence",
    "dominant_family",
    "mixed_family_evidence",
]


def build_episode_features(episodes: pd.DataFrame, window_scores: pd.DataFrame) -> pd.DataFrame:
    """Convert scored windows plus episode spans into one feature row per episode."""

    if episodes.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    rows: list[dict[str, object]] = []
    for _, episode in episodes.iterrows():
        window_ids = episode.get("source_window_ids", tuple())
        if isinstance(window_ids, str):
            window_ids = (window_ids,)
        episode_windows = _slice_episode_windows(window_scores, window_ids, episode)
        if episode_windows.empty:
            episode_windows = window_scores.iloc[0:0].copy()
        row = {
            "episode_id": episode["episode_id"],
            "asset_id": episode.get("asset_id"),
            "duration_hours": float(pd.to_numeric(pd.Series([episode.get("duration_seconds")]), errors="coerce").fillna(0.0).iloc[0] / 3600.0),
            "recovery_hours": _seconds_to_hours(episode.get("time_to_recovery_seconds")),
            "onset_slope": _onset_slope(episode_windows),
            "peak_score": _safe_float(episode.get("peak_score")),
            "mean_score": _safe_float(episode.get("mean_score")),
            "max_sequence_divergence": _agg(episode_windows, "sequence_divergence", "max"),
            "median_sequence_divergence": _agg(episode_windows, "sequence_divergence", "median"),
            "max_duration_drift": _agg(episode_windows, "duration_drift", "max"),
            "median_duration_drift": _agg(episode_windows, "duration_drift", "median"),
            "max_recurrence_excess": _agg(episode_windows, "recurrence_excess", "max"),
            "median_recurrence_excess": _agg(episode_windows, "recurrence_excess", "median"),
            "max_persistence_excess": _agg(episode_windows, "persistence_excess", "max"),
            "median_persistence_excess": _agg(episode_windows, "persistence_excess", "median"),
            "signed_consumption_deviation": _agg(episode_windows, "consumption_deviation", "mean"),
            "min_consumption_deviation": _agg(episode_windows, "consumption_deviation", "min"),
            "max_state_error_rate": _agg(episode_windows, "state_error_rate", "max"),
            "mean_state_error_rate": _agg(episode_windows, "state_error_rate", "mean"),
            "max_mode_divergence": _agg(episode_windows, "mode_divergence", "max"),
            "mean_mode_divergence": _agg(episode_windows, "mode_divergence", "mean"),
            "dominant_family": _dominant_family(episode_windows),
            "mixed_family_evidence": _mixed_family_evidence(episode_windows),
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def _slice_episode_windows(window_scores: pd.DataFrame, window_ids: tuple[str, ...], episode: pd.Series) -> pd.DataFrame:
    if window_scores.empty:
        return window_scores.copy()
    if window_ids:
        index_labels = [int(item) for item in window_ids if str(item).isdigit()]
        if index_labels:
            existing = [index for index in index_labels if index in window_scores.index]
            if existing:
                return window_scores.loc[existing].copy()
    start = pd.to_datetime(episode.get("event_start"))
    end = pd.to_datetime(episode.get("event_end"))
    frame = window_scores.copy()
    frame["start_time"] = pd.to_datetime(frame["start_time"])
    frame["end_time"] = pd.to_datetime(frame["end_time"])
    return frame[(frame["start_time"] <= end) & (frame["end_time"] >= start)].copy()


def _agg(frame: pd.DataFrame, column: str, op: str) -> float:
    if column not in frame.columns or frame.empty:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return 0.0
    if op == "max":
        return float(values.max())
    if op == "min":
        return float(values.min())
    if op == "median":
        return float(values.median())
    return float(values.mean())


def _dominant_family(frame: pd.DataFrame) -> str:
    if "incident_family" not in frame.columns or frame.empty:
        return "unclassified_incident"
    mode = frame["incident_family"].mode(dropna=True)
    return str(mode.iloc[0]) if not mode.empty else "unclassified_incident"


def _mixed_family_evidence(frame: pd.DataFrame) -> float:
    if "incident_family" not in frame.columns or frame.empty:
        return 0.0
    return 1.0 if frame["incident_family"].nunique(dropna=True) >= 2 else 0.0


def _onset_slope(frame: pd.DataFrame) -> float:
    if frame.empty or "deviation_score" not in frame.columns or len(frame) < 2:
        return 0.0
    scores = pd.to_numeric(frame["deviation_score"], errors="coerce").fillna(0.0).reset_index(drop=True)
    return float(scores.iloc[1] - scores.iloc[0])


def _seconds_to_hours(value: object) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric) / 3600.0


def _safe_float(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0.0).iloc[0]
    return float(numeric)
