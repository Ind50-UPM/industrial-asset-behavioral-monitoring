"""Incident registry normalization and matching helpers.

The registry abstraction used by ``Model_D`` must reconcile several flavours of
incident annotation. Some sources provide precise event timestamps, others
document longer operational windows, and exploratory candidate registries may
only carry weak labels derived from state words or duration patterns.

This module offers a small compatibility layer over those heterogeneous inputs.
Its responsibilities are deliberately focused on I/O, timestamp normalization,
and simple temporal alignment so the rest of the incident pipeline can consume a
stable schema regardless of the upstream source.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass
class IncidentRecord:
    """Normalized incident record supporting both source windows and real events.

    Attributes:
        incident_id: Stable identifier of the registry entry.
        source_window_start: Start of the source analytical window when the
            incident comes from a generated candidate registry.
        source_window_end: End of the source analytical window.
        documented_start: Human-validated start timestamp, if known.
        documented_end: Human-validated end timestamp, if known.
        event_time: Representative timestamp for point-like events.
        event_time_precision: Optional description of timestamp precision.
        family: Primary incident family.
        secondary_family: Optional secondary family.
        recovery_time: Recovery timestamp when available.
        recovery_status: Outcome or completeness of the recovery information.
        label_strength: Confidence or governance label of the annotation.
        source_type: Provenance of the registry row.
        downtime_start: Downtime start timestamp when distinct from incident
            onset.
        downtime_end: Downtime end timestamp.
        maintenance_time: Maintenance intervention timestamp.
        affected_subsystem: Optional subsystem scope.
        notes: Free-text operator notes.
        asset_id: Asset or node identifier.
    """

    incident_id: str
    source_window_start: pd.Timestamp | None = None
    source_window_end: pd.Timestamp | None = None
    documented_start: pd.Timestamp | None = None
    documented_end: pd.Timestamp | None = None
    event_time: pd.Timestamp | None = None
    event_time_precision: str | None = None
    family: str | None = None
    secondary_family: str | None = None
    recovery_time: pd.Timestamp | None = None
    recovery_status: str | None = None
    label_strength: str | None = None
    source_type: str | None = None
    downtime_start: pd.Timestamp | None = None
    downtime_end: pd.Timestamp | None = None
    maintenance_time: pd.Timestamp | None = None
    affected_subsystem: str | None = None
    notes: str | None = None
    asset_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize the dataclass as a plain dictionary."""
        return asdict(self)


class IncidentRegistry:
    """Load, normalize, and align incident tables with longitudinal episodes.

    Notes:
        The methods in this class return pandas dataframes instead of dedicated
        domain objects because they are intended to integrate directly with the
        tabular outputs produced by the rest of the repository.
    """

    def load(self, file_path: str | Path) -> pd.DataFrame:
        """Load and normalize a registry table from a supported file format."""
        path = Path(file_path)
        if path.suffix.lower() == '.parquet':
            frame = pd.read_parquet(path)
        elif path.suffix.lower() in {'.xlsx', '.xls'}:
            frame = pd.read_excel(path)
        elif path.suffix.lower() == '.csv':
            frame = pd.read_csv(path)
        else:
            raise ValueError(f'Unsupported file extension: {path.suffix}')

        return self.normalize(frame)

    def normalize(self, incidents: pd.DataFrame) -> pd.DataFrame:
        """Normalize timestamps and fill compatible optional fields."""

        if incidents.empty:
            return incidents.copy()

        frame = incidents.copy()
        timestamp_columns = [
            'source_window_start',
            'source_window_end',
            'documented_start',
            'documented_end',
            'event_time',
            'recovery_time',
            'downtime_start',
            'downtime_end',
            'maintenance_time',
        ]
        for column in timestamp_columns:
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column])

        if 'incident_family' not in frame.columns and 'family' in frame.columns:
            frame['incident_family'] = frame['family']
        if 'family' not in frame.columns and 'incident_family' in frame.columns:
            frame['family'] = frame['incident_family']

        if 'source_window_start' not in frame.columns:
            frame['source_window_start'] = frame.get('documented_start')
        else:
            frame['source_window_start'] = frame['source_window_start'].fillna(frame.get('documented_start'))
        if 'source_window_end' not in frame.columns:
            frame['source_window_end'] = frame.get('documented_end')
        else:
            frame['source_window_end'] = frame['source_window_end'].fillna(frame.get('documented_end'))

        for column in [
            'secondary_family',
            'event_time_precision',
            'recovery_status',
            'label_strength',
            'source_type',
            'affected_subsystem',
            'notes',
        ]:
            if column not in frame.columns:
                frame[column] = None

        if 'incident_id' not in frame.columns:
            frame['incident_id'] = [f'incident_{index}' for index in range(len(frame))]

        return frame

    def confirmed(self, incidents: pd.DataFrame) -> pd.DataFrame:
        """Return only registry entries explicitly marked as confirmed."""
        frame = self.normalize(incidents)
        return frame[frame['label_strength'].eq('confirmed')].copy()

    def weakly_labelled(self, incidents: pd.DataFrame) -> pd.DataFrame:
        """Return weakly or strongly labelled entries usable for exploration."""
        frame = self.normalize(incidents)
        return frame[frame['label_strength'].isin(['weak', 'strong'])].copy()

    def unresolved(self, incidents: pd.DataFrame) -> pd.DataFrame:
        """Return rows that still lack family or label-strength attribution."""
        frame = self.normalize(incidents)
        return frame[frame['family'].isna() | frame['label_strength'].isna()].copy()

    def overlapping(self, incidents: pd.DataFrame, start: object, end: object) -> pd.DataFrame:
        """Return incidents whose documented span overlaps the target interval."""
        frame = self.normalize(incidents)
        start_time = pd.to_datetime(start)
        end_time = pd.to_datetime(end)
        reference_start = frame['documented_start'].fillna(frame['event_time'])
        reference_end = frame['documented_end'].fillna(frame['event_time'])
        mask = reference_start.le(end_time) & reference_end.ge(start_time)
        return frame[mask].copy()

    def match_candidates(
        self,
        incidents: pd.DataFrame,
        candidates: pd.DataFrame,
        tolerance: pd.Timedelta = pd.Timedelta(hours=1),
    ) -> pd.DataFrame:
        """Match candidate windows or episodes to registry events with tolerance."""

        frame = self.normalize(incidents)
        if frame.empty or candidates.empty:
            return pd.DataFrame(columns=['incident_id', 'candidate_id', 'matched', 'time_delta_seconds'])

        rows: list[dict[str, object]] = []
        for _, candidate in candidates.iterrows():
            candidate_start = pd.to_datetime(
                candidate.get('event_start', candidate.get('source_window_start', candidate.get('start_time')))
            )
            candidate_id = candidate.get('episode_id', candidate.get('candidate_id', candidate.name))
            for _, incident in frame.iterrows():
                incident_time = incident.get('event_time')
                if pd.isna(incident_time) and pd.notna(incident.get('documented_start')):
                    incident_time = incident.get('documented_start')
                if pd.isna(incident_time) or pd.isna(candidate_start):
                    continue
                delta = abs((pd.to_datetime(candidate_start) - pd.to_datetime(incident_time)).total_seconds())
                rows.append(
                    {
                        'incident_id': incident['incident_id'],
                        'candidate_id': candidate_id,
                        'matched': delta <= tolerance.total_seconds(),
                        'time_delta_seconds': delta,
                    }
                )
        return pd.DataFrame(rows)

    def align_with_episodes(
        self,
        incidents: pd.DataFrame,
        episodes: pd.DataFrame,
    ) -> pd.DataFrame:
        """Check whether incident events fall inside derived episode windows.

        Args:
            incidents: Registry dataframe.
            episodes: Episode dataframe generated by ``Model_D``.

        Returns:
            A dataframe indicating whether each incident time falls within a
            compatible detected episode.
        """
        if incidents.empty or episodes.empty:
            return pd.DataFrame(
                columns=[
                    'incident_id',
                    'event_time',
                    'incident_family',
                    'matched_episode_id',
                    'window_contains_event',
                ]
            )

        normalized = self.normalize(incidents)
        episode_frame = episodes.copy()
        episode_frame['pre_event_start'] = pd.to_datetime(episode_frame['pre_event_start'])
        episode_frame['recovery_end'] = pd.to_datetime(episode_frame['recovery_end'])
        episode_frame['event_end'] = pd.to_datetime(episode_frame['event_end'])
        family_column = 'primary_family' if 'primary_family' in episode_frame.columns else 'incident_family'

        rows: list[dict[str, object]] = []
        for index, incident in normalized.iterrows():
            family_matches = episode_frame[episode_frame[family_column] == incident['incident_family']]
            match_id = None
            contains_event = False
            for _, episode in family_matches.iterrows():
                window_end = episode['recovery_end']
                if pd.isna(window_end):
                    window_end = episode['event_end']
                if episode['pre_event_start'] <= incident['event_time'] <= window_end:
                    match_id = episode['episode_id']
                    contains_event = True
                    break

            rows.append(
                {
                    'incident_id': incident.get('incident_id', index),
                    'event_time': incident['event_time'],
                    'incident_family': incident['incident_family'],
                    'matched_episode_id': match_id,
                    'window_contains_event': contains_event,
                }
            )

        return pd.DataFrame(rows)
