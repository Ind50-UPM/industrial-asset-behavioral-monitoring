"""Rolling-window generation for indicator-driven Model_D workflows."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WindowBuildConfig:
    """Configuration for rolling observation windows."""

    length: pd.Timedelta = pd.Timedelta(hours=24)
    step: pd.Timedelta = pd.Timedelta(hours=1)
    min_coverage: float = 0.90


def build_rolling_windows(
    observation_periods: pd.DataFrame,
    config: WindowBuildConfig | None = None,
) -> pd.DataFrame:
    """Create rolling windows over observation periods, excluding invalid intervals."""

    effective_config = config or WindowBuildConfig()
    if observation_periods.empty:
        return pd.DataFrame(columns=['window_id', 'asset_id', 'start_time', 'end_time', 'data_coverage', 'is_valid'])

    frame = observation_periods.copy()
    frame = frame.loc[:, ~frame.columns.duplicated()].copy()
    frame['start_time'] = pd.to_datetime(frame['start_time'], errors='coerce', utc=True).dt.tz_convert(None)
    frame['end_time'] = pd.to_datetime(frame['end_time'], errors='coerce', utc=True).dt.tz_convert(None)
    frame = frame.dropna(subset=['start_time', 'end_time']).reset_index(drop=True)
    frame['asset_id'] = frame.get('asset_id', pd.Series('asset-unknown', index=frame.index)).fillna('asset-unknown').astype(str)

    rows: list[dict[str, object]] = []
    for asset_id, asset_frame in frame.groupby('asset_id', dropna=False):
        observed = asset_frame[asset_frame['period_type'].eq('observed')].sort_values('start_time').reset_index(drop=True)
        excluded = asset_frame[asset_frame['exclude_from_exposure'].fillna(False)].sort_values('start_time').reset_index(drop=True)
        if observed.empty:
            continue
        for period in observed.itertuples(index=False):
            current_start = period.start_time
            while current_start + effective_config.length <= period.end_time:
                current_end = current_start + effective_config.length
                coverage = _window_coverage(current_start, current_end, observed, excluded)
                rows.append(
                    {
                        'window_id': f'{asset_id}_{current_start.isoformat()}',
                        'asset_id': asset_id,
                        'start_time': current_start,
                        'end_time': current_end,
                        'data_coverage': coverage,
                        'is_valid': coverage >= effective_config.min_coverage,
                    }
                )
                current_start = current_start + effective_config.step
    return pd.DataFrame(rows)


def _window_coverage(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    observed: pd.DataFrame,
    excluded: pd.DataFrame,
) -> float:
    total_seconds = max((end_time - start_time).total_seconds(), 1.0)

    observed_slice = observed[(observed['start_time'] < end_time) & (observed['end_time'] > start_time)]
    excluded_slice = excluded[(excluded['start_time'] < end_time) & (excluded['end_time'] > start_time)] if not excluded.empty else excluded

    observed_seconds = 0.0
    for row in observed_slice.itertuples(index=False):
        observed_seconds += _overlap_seconds(start_time, end_time, row.start_time, row.end_time)

    excluded_seconds = 0.0
    for row in excluded_slice.itertuples(index=False):
        excluded_seconds += _overlap_seconds(start_time, end_time, row.start_time, row.end_time)

    effective = max(observed_seconds - excluded_seconds, 0.0)
    return min(effective / total_seconds, 1.0)


def _overlap_seconds(left_start: pd.Timestamp, left_end: pd.Timestamp, right_start: pd.Timestamp, right_end: pd.Timestamp) -> float:
    overlap_start = max(left_start, pd.to_datetime(right_start))
    overlap_end = min(left_end, pd.to_datetime(right_end))
    return max((overlap_end - overlap_start).total_seconds(), 0.0)
