"""Recovery assessment primitives for Model_D episodes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class RecoveryStatus(str, Enum):
    """Recovery outcome categories for a detected episode."""

    NOT_RECOVERED = "not_recovered"
    PARTIAL = "partial"
    STABLE = "stable"
    RECONFIGURED = "reconfigured"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RecoveryAssessment:
    """Outcome of recovery confirmation for one episode."""

    recovery_start: pd.Timestamp | None
    recovery_end: pd.Timestamp | None
    recovery_status: RecoveryStatus
    time_to_recovery_seconds: float | None


class RecoveryAssessor:
    """Confirm recovery using threshold hysteresis and consecutive windows."""

    def assess(
        self,
        window_frame: pd.DataFrame,
        *,
        last_event_index: int,
        recovery_threshold: float,
        recovery_windows: int,
    ) -> RecoveryAssessment:
        """Assess whether a stable recovery is observed after the event."""

        following = window_frame.iloc[last_event_index + 1 :].copy()
        if following.empty:
            return RecoveryAssessment(None, None, RecoveryStatus.NOT_RECOVERED, None)

        below = following["deviation_score"].lt(recovery_threshold)
        sustained = (
            below.rolling(recovery_windows, min_periods=recovery_windows).sum().eq(recovery_windows)
            if len(following) >= recovery_windows
            else pd.Series(False, index=following.index)
        )
        if bool(sustained.any()):
            stable_index = sustained[sustained].index[0]
            stable_position = following.index.get_loc(stable_index)
            start_index = following.index[stable_position - recovery_windows + 1]
            stable_start = window_frame.loc[start_index]
            stable_end = window_frame.loc[stable_index]
            return RecoveryAssessment(
                recovery_start=stable_start["start_time"],
                recovery_end=stable_end["end_time"],
                recovery_status=RecoveryStatus.STABLE,
                time_to_recovery_seconds=(
                    stable_end["end_time"] - window_frame.loc[last_event_index, "end_time"]
                ).total_seconds(),
            )

        if bool(below.any()):
            first_index = below[below].index[0]
            first_row = window_frame.loc[first_index]
            return RecoveryAssessment(
                recovery_start=first_row["start_time"],
                recovery_end=first_row["end_time"],
                recovery_status=RecoveryStatus.PARTIAL,
                time_to_recovery_seconds=(
                    first_row["end_time"] - window_frame.loc[last_event_index, "end_time"]
                ).total_seconds(),
            )

        return RecoveryAssessment(None, None, RecoveryStatus.NOT_RECOVERED, None)
