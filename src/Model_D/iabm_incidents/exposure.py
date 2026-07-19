"""Observation-exposure derivation for indicator-driven Model_D workflows."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ExposureConfig:
    """Controls how observation periods are derived from source tables."""

    gap_tolerance: pd.Timedelta = pd.Timedelta(hours=2)
    minimum_period: pd.Timedelta = pd.Timedelta(minutes=5)
    exclude_documented_downtime: bool = True
    exclude_documented_maintenance: bool = True


def derive_observation_periods(
    analogue: pd.DataFrame,
    digital: pd.DataFrame,
    states: pd.DataFrame | None = None,
    registry: pd.DataFrame | None = None,
    config: ExposureConfig | None = None,
) -> pd.DataFrame:
    """Infer observation and exclusion periods from telemetry and registry data."""

    effective_config = config or ExposureConfig()
    telemetry_ranges = _collect_time_ranges([analogue, digital, states])
    if telemetry_ranges.empty:
        return pd.DataFrame(
            columns=[
                'period_id', 'asset_id', 'start_time', 'end_time', 'period_type', 'reason',
                'source', 'confidence', 'exclude_from_baseline', 'exclude_from_exposure', 'linked_incident_id',
            ]
        )

    observation = _merge_contiguous_ranges(
        telemetry_ranges,
        gap_tolerance=effective_config.gap_tolerance,
        minimum_period=effective_config.minimum_period,
    )
    observation['period_type'] = 'observed'
    observation['reason'] = 'telemetry_available'
    observation['source'] = 'telemetry'
    observation['confidence'] = 1.0
    observation['exclude_from_baseline'] = False
    observation['exclude_from_exposure'] = False
    observation['linked_incident_id'] = None

    exclusions = _registry_exclusions(registry, effective_config)
    combined = pd.concat([observation, exclusions], ignore_index=True, sort=False)
    combined['start_time'] = _normalize_timestamp_series(combined['start_time'])
    combined['end_time'] = _normalize_timestamp_series(combined['end_time'])
    combined['asset_id'] = combined['asset_id'].fillna('asset-unknown').astype(str)
    combined['period_type'] = combined['period_type'].fillna('observed').astype(str)
    combined = combined.dropna(subset=['start_time', 'end_time']).reset_index(drop=True)
    combined['period_id'] = [f'period_{index:05d}' for index in range(len(combined))]
    return combined[
        [
            'period_id', 'asset_id', 'start_time', 'end_time', 'period_type', 'reason', 'source',
            'confidence', 'exclude_from_baseline', 'exclude_from_exposure', 'linked_incident_id',
        ]
    ].sort_values(['asset_id', 'start_time', 'period_type']).reset_index(drop=True)


def _collect_time_ranges(frames: list[pd.DataFrame | None]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for frame in frames:
        if frame is None or frame.empty:
            continue
        normalized = frame.copy()
        start_column = _resolve_time_column(normalized, ['start_time', 'window_start', 'timestamp', 'time', 'date'])
        end_column = _resolve_time_column(normalized, ['end_time', 'window_end', 'timestamp', 'time', 'date'])
        if start_column is None:
            continue
        normalized['_start_time'] = _normalize_timestamp_series(normalized[start_column])
        normalized['_end_time'] = _normalize_timestamp_series(normalized[end_column]) if end_column is not None else normalized['_start_time']
        normalized = normalized.dropna(subset=['_start_time', '_end_time'])
        asset_series = normalized['asset_id'] if 'asset_id' in normalized.columns else pd.Series('asset-unknown', index=normalized.index)
        for asset_id, asset_frame in normalized.groupby(asset_series, dropna=False):
            rows.append({'asset_id': asset_id, 'start_time': asset_frame['_start_time'].min(), 'end_time': asset_frame['_end_time'].max()})
    return pd.DataFrame(rows)


def _merge_contiguous_ranges(ranges: pd.DataFrame, *, gap_tolerance: pd.Timedelta, minimum_period: pd.Timedelta) -> pd.DataFrame:
    if ranges.empty:
        return pd.DataFrame(columns=['asset_id', 'start_time', 'end_time'])

    rows: list[dict[str, object]] = []
    for asset_id, asset_ranges in ranges.sort_values('start_time').groupby('asset_id', dropna=False):
        current_start = None
        current_end = None
        for _, row in asset_ranges.iterrows():
            if current_start is None:
                current_start = row['start_time']
                current_end = row['end_time']
                continue
            if row['start_time'] - current_end <= gap_tolerance:
                current_end = max(current_end, row['end_time'])
                continue
            if current_end - current_start >= minimum_period:
                rows.append({'asset_id': asset_id, 'start_time': current_start, 'end_time': current_end})
            current_start = row['start_time']
            current_end = row['end_time']
        if current_start is not None and current_end - current_start >= minimum_period:
            rows.append({'asset_id': asset_id, 'start_time': current_start, 'end_time': current_end})
    return pd.DataFrame(rows)


def _registry_exclusions(registry: pd.DataFrame | None, config: ExposureConfig) -> pd.DataFrame:
    if registry is None or registry.empty:
        return pd.DataFrame(columns=['asset_id', 'start_time', 'end_time', 'period_type', 'reason', 'source', 'confidence', 'exclude_from_baseline', 'exclude_from_exposure', 'linked_incident_id'])

    frame = registry.copy()
    for column in ['downtime_start', 'downtime_end', 'maintenance_time']:
        if column in frame.columns:
            frame[column] = _normalize_timestamp_series(frame[column])

    rows: list[dict[str, object]] = []
    if config.exclude_documented_downtime and {'downtime_start', 'downtime_end'}.issubset(frame.columns):
        valid = frame[frame['downtime_start'].notna() & frame['downtime_end'].notna()]
        for _, row in valid.iterrows():
            rows.append({'asset_id': row.get('asset_id'), 'start_time': row['downtime_start'], 'end_time': row['downtime_end'], 'period_type': 'excluded', 'reason': 'documented_downtime', 'source': 'registry', 'confidence': 1.0, 'exclude_from_baseline': True, 'exclude_from_exposure': True, 'linked_incident_id': row.get('incident_id')})
    if config.exclude_documented_maintenance and 'maintenance_time' in frame.columns:
        valid = frame[frame['maintenance_time'].notna()]
        for _, row in valid.iterrows():
            maintenance_time = row['maintenance_time']
            rows.append({'asset_id': row.get('asset_id'), 'start_time': maintenance_time, 'end_time': maintenance_time, 'period_type': 'excluded', 'reason': 'documented_maintenance', 'source': 'registry', 'confidence': 0.8, 'exclude_from_baseline': True, 'exclude_from_exposure': False, 'linked_incident_id': row.get('incident_id')})
    return pd.DataFrame(rows)


def _normalize_timestamp_series(values: pd.Series) -> pd.Series:
    converted = pd.to_datetime(values, errors='coerce', utc=True)
    if getattr(converted.dtype, 'tz', None) is not None:
        converted = converted.dt.tz_convert(None)
    return converted


def _resolve_time_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None
