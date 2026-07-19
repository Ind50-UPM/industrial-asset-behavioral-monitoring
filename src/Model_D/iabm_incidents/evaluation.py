"""Temporal evaluation against a reference incident registry.

The utilities in this module compare detected episode windows against a
reference registry of incidents documented by operators or downstream business
systems. The main goal is not only to compute overlap, but also to expose
practical operational quantities such as onset error, recovery error, and early
warning lead time.

The matching logic intentionally remains deterministic and transparent. For a
given reference incident it selects the candidate episode with the highest
temporal IoU among the episodes belonging to the same family. This keeps the
evaluation easy to audit while still producing stable metrics for early
iteration on the segmentation and scoring stack.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class EpisodeMatch:
    """Alignment details between a detected episode and a reference incident.

    Attributes:
        detected_episode_id: Identifier of the detected episode.
        reference_incident_id: Identifier of the registry event.
        temporal_iou: Intersection-over-union computed over the time spans.
        temporal_overlap_seconds: Absolute duration of the intersection.
        onset_error_seconds: Signed error between detected and documented onset.
        end_error_seconds: Signed error between detected and documented end.
        recovery_error_seconds: Signed error on recovery end, when both sources
            expose recovery timestamps.
        lead_time_seconds: Positive anticipation time when detection starts
            before the documented onset.
        family_match: Whether the detected family equals the reference family.
        reference_family: Family stored in the incident registry.
        detected_family: Family assigned to the detected episode.
    """

    detected_episode_id: str
    reference_incident_id: str
    temporal_iou: float
    temporal_overlap_seconds: float
    onset_error_seconds: float
    end_error_seconds: float | None
    recovery_error_seconds: float | None
    lead_time_seconds: float | None
    family_match: bool
    reference_family: str
    detected_family: str


class EpisodeEvaluator:
    """Evaluate detected episodes against a canonical incident registry."""

    def match(self, incidents: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
        """Match incidents to episodes using family and temporal overlap.

        Args:
            incidents: Canonical registry dataframe.
            episodes: Detected episode dataframe.

        Returns:
            A dataframe of per-incident best matches.

        Notes:
            The method performs a family-constrained search first and only then
            ranks episode candidates by temporal IoU. This means the resulting
            table is most suitable for measuring quality of already classified
            episodes rather than open-world retrieval performance.
        """

        if incidents.empty or episodes.empty:
            return pd.DataFrame(
                columns=[
                    'detected_episode_id',
                    'reference_incident_id',
                    'temporal_iou',
                    'temporal_overlap_seconds',
                    'onset_error_seconds',
                    'end_error_seconds',
                    'recovery_error_seconds',
                    'lead_time_seconds',
                    'family_match',
                    'reference_family',
                    'detected_family',
                ]
            )

        normalized_incidents = incidents.copy()
        normalized_incidents['event_time'] = pd.to_datetime(normalized_incidents['event_time'])
        if 'documented_start' in normalized_incidents.columns:
            normalized_incidents['documented_start'] = pd.to_datetime(normalized_incidents['documented_start'])
        if 'documented_end' in normalized_incidents.columns:
            normalized_incidents['documented_end'] = pd.to_datetime(normalized_incidents['documented_end'])
        if 'recovery_time' in normalized_incidents.columns:
            normalized_incidents['recovery_time'] = pd.to_datetime(normalized_incidents['recovery_time'])

        rows: list[dict[str, object]] = []
        for _, incident in normalized_incidents.iterrows():
            family_episodes = episodes[episodes['primary_family'] == incident['incident_family']]
            best_match = self._best_match(incident, family_episodes)
            if best_match is not None:
                rows.append(best_match.__dict__)
        return pd.DataFrame(rows)

    def summarize(self, incidents: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
        """Produce a compact evaluation summary.

        Args:
            incidents: Canonical registry dataframe.
            episodes: Detected episode dataframe.

        Returns:
            A one-row dataframe with precision, recall, timing, and family
            agreement metrics.
        """

        matches = self.match(incidents, episodes)
        matched_incidents = matches['reference_incident_id'].nunique() if not matches.empty else 0
        false_alarms = max(len(episodes) - matched_incidents, 0)
        precision = matched_incidents / len(episodes) if len(episodes) else 0.0
        recall = matched_incidents / len(incidents) if len(incidents) else 0.0
        family_precision = float(matches['family_match'].mean()) if not matches.empty else 0.0
        return pd.DataFrame(
            [
                {
                    'episode_precision': precision,
                    'episode_recall': recall,
                    'false_alarm_count': false_alarms,
                    'mean_temporal_iou': _safe_mean(matches, 'temporal_iou', default=0.0),
                    'median_temporal_iou': _safe_median(matches, 'temporal_iou', default=0.0),
                    'mean_onset_error_seconds': _safe_mean(matches, 'onset_error_seconds'),
                    'mean_absolute_onset_error_seconds': _safe_abs_mean(matches, 'onset_error_seconds'),
                    'mean_end_error_seconds': _safe_mean(matches, 'end_error_seconds'),
                    'mean_recovery_error_seconds': _safe_mean(matches, 'recovery_error_seconds'),
                    'mean_lead_time_seconds': _safe_mean(matches, 'lead_time_seconds'),
                    'positive_lead_matches': int(matches['lead_time_seconds'].gt(0).sum()) if not matches.empty else 0,
                    'family_precision': family_precision,
                    'family_recall': matched_incidents / len(incidents) if len(incidents) else 0.0,
                }
            ]
        )

    def _best_match(self, incident: pd.Series, episodes: pd.DataFrame) -> EpisodeMatch | None:
        """Return the highest-IoU episode match for a single incident."""
        if episodes.empty:
            return None

        best_match: EpisodeMatch | None = None
        best_iou = -1.0
        reference_start = self._reference_start(incident)
        reference_end = self._reference_end(incident)
        reference_recovery = pd.to_datetime(incident['recovery_time']) if 'recovery_time' in incident.index and pd.notna(incident['recovery_time']) else None
        for _, episode in episodes.iterrows():
            overlap_seconds = _temporal_overlap_seconds(reference_start, reference_end, episode['event_start'], episode['event_end'])
            iou = _temporal_iou(reference_start, reference_end, episode['event_start'], episode['event_end'])
            if iou > best_iou:
                best_iou = iou
                detected_start = pd.to_datetime(episode['event_start'])
                detected_end = pd.to_datetime(episode['event_end'])
                detected_recovery = pd.to_datetime(episode['recovery_end']) if pd.notna(episode.get('recovery_end')) else None
                onset_error = (detected_start - reference_start).total_seconds()
                lead_time = -onset_error if onset_error < 0 else 0.0
                recovery_error = None
                if reference_recovery is not None and detected_recovery is not None:
                    recovery_error = (detected_recovery - reference_recovery).total_seconds()
                best_match = EpisodeMatch(
                    detected_episode_id=episode['episode_id'],
                    reference_incident_id=str(incident.get('incident_id', incident.name)),
                    temporal_iou=iou,
                    temporal_overlap_seconds=overlap_seconds,
                    onset_error_seconds=onset_error,
                    end_error_seconds=(detected_end - reference_end).total_seconds(),
                    recovery_error_seconds=recovery_error,
                    lead_time_seconds=lead_time,
                    family_match=episode['primary_family'] == incident['incident_family'],
                    reference_family=str(incident['incident_family']),
                    detected_family=str(episode['primary_family']),
                )
        return best_match

    @staticmethod
    def _reference_start(incident: pd.Series) -> pd.Timestamp:
        """Resolve the best available reference start timestamp."""
        if 'documented_start' in incident.index and pd.notna(incident['documented_start']):
            return pd.to_datetime(incident['documented_start'])
        return pd.to_datetime(incident['event_time'])

    @staticmethod
    def _reference_end(incident: pd.Series) -> pd.Timestamp:
        """Resolve the best available reference end timestamp."""
        if 'documented_end' in incident.index and pd.notna(incident['documented_end']):
            return pd.to_datetime(incident['documented_end'])
        return pd.to_datetime(incident['event_time'])


def _temporal_overlap_seconds(
    left_start: pd.Timestamp,
    left_end: pd.Timestamp,
    right_start: pd.Timestamp,
    right_end: pd.Timestamp,
) -> float:
    """Return temporal overlap in seconds between two intervals."""
    intersection_start = max(left_start, pd.to_datetime(right_start))
    intersection_end = min(left_end, pd.to_datetime(right_end))
    return max((intersection_end - intersection_start).total_seconds(), 0.0)


def _temporal_iou(
    left_start: pd.Timestamp,
    left_end: pd.Timestamp,
    right_start: pd.Timestamp,
    right_end: pd.Timestamp,
) -> float:
    """Return the temporal intersection-over-union for two intervals."""
    intersection = _temporal_overlap_seconds(left_start, left_end, right_start, right_end)
    union_start = min(left_start, pd.to_datetime(right_start))
    union_end = max(left_end, pd.to_datetime(right_end))
    union = max((union_end - union_start).total_seconds(), 0.0)
    if union == 0.0:
        return 1.0 if intersection == 0.0 else 0.0
    return intersection / union


def _safe_mean(frame: pd.DataFrame, column: str, default: float | None = None) -> float | None:
    """Return the mean of a column without emitting warnings for all-missing data."""

    if frame.empty or column not in frame.columns:
        return default
    series = pd.to_numeric(frame[column], errors='coerce').dropna()
    if series.empty:
        return default
    return float(series.mean())


def _safe_abs_mean(frame: pd.DataFrame, column: str) -> float | None:
    """Return the absolute mean of a column without warnings for all-missing data."""

    if frame.empty or column not in frame.columns:
        return None
    series = pd.to_numeric(frame[column], errors='coerce').dropna()
    if series.empty:
        return None
    return float(series.abs().mean())


def _safe_median(frame: pd.DataFrame, column: str, default: float | None = None) -> float | None:
    """Return the median of a column without warnings for all-missing data."""

    if frame.empty or column not in frame.columns:
        return default
    series = pd.to_numeric(frame[column], errors='coerce').dropna()
    if series.empty:
        return default
    return float(series.median())
