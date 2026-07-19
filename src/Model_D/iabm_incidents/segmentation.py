"""Temporal episode segmentation for Model_D."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import EpisodeDetectionConfig


@dataclass(frozen=True)
class EpisodeSegment:
    """Rich temporal segment describing one detected episode in the window frame."""

    start_index: int
    peak_index: int
    end_index: int
    recovery_start_index: int | None = None
    recovery_end_index: int | None = None


class EpisodeSegmenter:
    """Create temporally robust segments from scored windows."""

    def segment(
        self,
        window_frame: pd.DataFrame,
        detection_config: EpisodeDetectionConfig,
    ) -> list[EpisodeSegment]:
        if window_frame.empty:
            return []

        frame = window_frame.copy()
        frame["start_time"] = pd.to_datetime(frame["start_time"])
        frame["end_time"] = pd.to_datetime(frame["end_time"])
        scores = pd.to_numeric(frame["deviation_score"], errors="coerce").fillna(0.0)
        candidate_mask = scores.ge(detection_config.onset_threshold)

        if "semantic_status" in frame.columns:
            return self._segment_semantic(frame, candidate_mask, detection_config)
        return self._segment_indicator(frame, candidate_mask, detection_config)

    def _segment_semantic(
        self,
        frame: pd.DataFrame,
        candidate_mask: pd.Series,
        detection_config: EpisodeDetectionConfig,
    ) -> list[EpisodeSegment]:
        segments: list[EpisodeSegment] = []
        current_indices: list[int] = []
        previous_end_time: pd.Timestamp | None = None

        index_values = frame.index.to_list()
        start_times = frame['start_time'].to_list()
        end_times = frame['end_time'].to_list()
        candidate_values = candidate_mask.reindex(frame.index, fill_value=False).to_numpy(dtype=bool)

        for position, index in enumerate(index_values):
            if not candidate_values[position]:
                if current_indices:
                    self._append_if_valid(frame, current_indices, segments, detection_config)
                    current_indices = []
                    previous_end_time = None
                continue

            if current_indices and previous_end_time is not None:
                gap = start_times[position] - previous_end_time
                if gap > detection_config.maximum_gap:
                    self._append_if_valid(frame, current_indices, segments, detection_config)
                    current_indices = []

            current_indices.append(index)
            previous_end_time = end_times[position]

        if current_indices:
            self._append_if_valid(frame, current_indices, segments, detection_config)
        return segments

    def _segment_indicator(
        self,
        frame: pd.DataFrame,
        candidate_mask: pd.Series,
        detection_config: EpisodeDetectionConfig,
    ) -> list[EpisodeSegment]:
        segments: list[EpisodeSegment] = []
        current_candidate_indices: list[int] = []
        pending_gap_indices: list[int] = []
        previous_end_time: pd.Timestamp | None = None

        for index, row in frame.iterrows():
            if current_candidate_indices and previous_end_time is not None:
                gap_duration = row["start_time"] - previous_end_time
                if gap_duration > detection_config.maximum_gap:
                    self._append_if_valid(frame, current_candidate_indices, segments, detection_config)
                    current_candidate_indices = []
                    pending_gap_indices = []

            if bool(candidate_mask.loc[index]):
                current_candidate_indices.append(index)
                pending_gap_indices = []
                previous_end_time = row["end_time"]
                continue

            if current_candidate_indices:
                pending_gap_indices.append(index)
                previous_end_time = row["end_time"]
                if len(pending_gap_indices) >= detection_config.recovery_windows:
                    self._append_if_valid(frame, current_candidate_indices, segments, detection_config)
                    current_candidate_indices = []
                    pending_gap_indices = []
                    previous_end_time = None

        if current_candidate_indices:
            self._append_if_valid(frame, current_candidate_indices, segments, detection_config)
        return segments

    @staticmethod
    def _append_if_valid(
        window_frame: pd.DataFrame,
        indices: list[int],
        segments: list[EpisodeSegment],
        detection_config: EpisodeDetectionConfig,
    ) -> None:
        if len(indices) < detection_config.onset_windows:
            return
        start_row = window_frame.loc[indices[0]]
        end_row = window_frame.loc[indices[-1]]
        if end_row["end_time"] - start_row["start_time"] < detection_config.minimum_duration:
            return

        scores = pd.to_numeric(window_frame.loc[indices, "deviation_score"], errors="coerce").fillna(0.0)
        peak_index = int(scores.idxmax())
        recovery_start_index, recovery_end_index = EpisodeSegmenter._find_recovery_indices(
            window_frame,
            end_index=indices[-1],
            recovery_threshold=detection_config.recovery_threshold,
            recovery_windows=detection_config.recovery_windows,
        )
        segments.append(
            EpisodeSegment(
                start_index=indices[0],
                peak_index=peak_index,
                end_index=indices[-1],
                recovery_start_index=recovery_start_index,
                recovery_end_index=recovery_end_index,
            )
        )

    @staticmethod
    def _find_recovery_indices(
        window_frame: pd.DataFrame,
        *,
        end_index: int,
        recovery_threshold: float,
        recovery_windows: int,
    ) -> tuple[int | None, int | None]:
        following = window_frame.iloc[end_index + 1 :].copy()
        if following.empty:
            return None, None

        below = pd.to_numeric(following["deviation_score"], errors="coerce").fillna(0.0).lt(recovery_threshold)
        if len(following) >= recovery_windows:
            sustained = below.rolling(recovery_windows, min_periods=recovery_windows).sum().eq(recovery_windows)
            if bool(sustained.any()):
                recovery_end_index = int(sustained[sustained].index[0])
                recovery_position = following.index.get_loc(recovery_end_index)
                recovery_start_index = int(following.index[recovery_position - recovery_windows + 1])
                return recovery_start_index, recovery_end_index
        if bool(below.any()):
            recovery_index = int(below[below].index[0])
            return recovery_index, recovery_index
        return None, None
