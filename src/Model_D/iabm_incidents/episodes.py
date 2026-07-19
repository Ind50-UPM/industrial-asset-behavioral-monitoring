"""Longitudinal episode construction logic.

This module turns window-level deviation evidence into incident episodes. It is
responsible for chaining segmentation, recovery assessment, and family
assignment into a stable tabular contract that downstream evaluation and
occurrence modules can consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from .classification import RuleBasedFamilyClassifier
from .config import ModelDConfig
from .detection import build_window_scores
from .recovery import RecoveryAssessor
from .segmentation import EpisodeSegmenter


@dataclass(frozen=True)
class CandidateSegment:
    """Serializable segment contract for the indicator-driven route.

    Attributes:
        candidate_id: Stable identifier for the candidate segment.
        start_index: Inclusive start index in the source score table.
        end_index: Inclusive end index in the source score table.
        event_start: Timestamp of the first anomalous window.
        peak_time: Timestamp of the segment peak.
        event_end: Timestamp of the last anomalous window.
        recovery_start: Optional timestamp when recovery begins.
        recovery_end: Optional timestamp when recovery ends.
        source_window_ids: Window identifiers participating in the segment.
    """

    candidate_id: str
    start_index: int
    end_index: int
    event_start: pd.Timestamp
    peak_time: pd.Timestamp
    event_end: pd.Timestamp
    recovery_start: pd.Timestamp | None = None
    recovery_end: pd.Timestamp | None = None
    source_window_ids: tuple[str, ...] = tuple()


@dataclass(frozen=True)
class IncidentEpisode:
    """Typed representation of one detected incident episode."""

    episode_id: str
    event_start: pd.Timestamp
    event_end: pd.Timestamp
    pre_event_start: pd.Timestamp
    peak_time: pd.Timestamp
    recovery_start: pd.Timestamp | None
    recovery_end: pd.Timestamp | None
    primary_family: str | None
    secondary_families: tuple[str, ...]
    family_confidence: float | None
    assignment_method: str | None
    evidence: tuple[str, ...]
    onset_score: float
    peak_score: float
    mean_score: float
    duration_seconds: float
    time_to_recovery_seconds: float | None
    recovery_status: str
    source_sequence_indices: tuple[int, ...]
    source_window_ids: tuple[str, ...]
    asset_id: str | None
    registry_incident_id: str | None = None
    label_strength: str | None = None


class IncidentEpisodeBuilder:
    """Build incident episodes from sequence-level and semantic outputs.

    Args:
        config: Optional typed configuration governing window and detection
            behavior.

    Notes:
        The builder supports two routes: a backward-compatible semantic route
        where sequence tables and assignment tables are merged into window
        scores, and an indicator-driven route where window scores are already
        available from upstream processing.
    """

    def __init__(self, config: ModelDConfig | None = None) -> None:
        self._config = config or ModelDConfig()
        self._segmenter = EpisodeSegmenter()
        self._recovery = RecoveryAssessor()
        self._classifier = RuleBasedFamilyClassifier()

    def build(
        self,
        sequences: pd.DataFrame,
        assignments: pd.DataFrame,
        *,
        pre_event_window: pd.Timedelta | None = None,
    ) -> pd.DataFrame:
        """Compatibility wrapper for the semantic assignment route."""
        return self.build_episodes_from_semantic_assignments(
            sequences,
            assignments,
            pre_event_window=pre_event_window,
        )

    def build_episodes_from_semantic_assignments(
        self,
        sequences: pd.DataFrame,
        assignments: pd.DataFrame,
        *,
        pre_event_window: pd.Timedelta | None = None,
        window_scores: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Construct episodes from sequences plus semantic assignments.

        Args:
            sequences: Model_B sequence table.
            assignments: Model_C assignment table.
            pre_event_window: Optional look-back horizon attached to each
                episode for contextual analysis.
            window_scores: Optional precomputed semantic window-score table.
                When provided, the method reuses it instead of rebuilding the
                same windows internally.

        Returns:
            Episode dataframe derived from merged semantic window scores.
        """
        if sequences.empty or assignments.empty:
            return self._empty_episode_frame()

        merged = window_scores.copy() if window_scores is not None else self.build_window_scores(sequences, assignments)
        effective_pre_event_window = pre_event_window or self._config.window.step
        segments = self._segmenter.segment(merged, self._config.detection)
        candidate_segments = self._candidate_segments_from_frame(segments, merged)
        return self.build_episodes_from_segments(
            candidate_segments,
            merged,
            pre_event_window=effective_pre_event_window,
        )

    def build_episodes_from_window_scores(
        self,
        window_scores: pd.DataFrame,
        *,
        pre_event_window: pd.Timedelta | None = None,
    ) -> pd.DataFrame:
        """Construct episodes directly from a precomputed window-score table."""
        if window_scores.empty:
            return self._empty_episode_frame()

        effective_pre_event_window = pre_event_window or self._config.window.step
        segments = self._segmenter.segment(window_scores, self._config.detection)
        candidate_segments = self._candidate_segments_from_frame(segments, window_scores)
        return self.build_episodes_from_segments(
            candidate_segments,
            window_scores,
            pre_event_window=effective_pre_event_window,
        )

    def build_episodes_from_segments(
        self,
        segments: Sequence[CandidateSegment],
        indicators: pd.DataFrame,
        *,
        pre_event_window: pd.Timedelta | None = None,
    ) -> pd.DataFrame:
        """Finalize typed episodes from candidate segments and indicator rows.

        Args:
            segments: Candidate segments emitted by the segmenter.
            indicators: Window-score or indicator dataframe used as source.
            pre_event_window: Optional context window attached before each onset.

        Returns:
            Final episode dataframe ready for evaluation and occurrence analysis.

        Notes:
            This method is the convergence point for both semantic and
            indicator-driven processing routes.
        """
        if not segments or indicators.empty:
            return self._empty_episode_frame()

        effective_pre_event_window = pre_event_window or self._config.window.step
        episodes: list[dict[str, object]] = []
        for segment in segments:
            episode = self._finalize_episode(
                merged=indicators,
                start_index=segment.start_index,
                end_index=segment.end_index,
                pre_event_window=effective_pre_event_window,
                source_window_ids=segment.source_window_ids,
                candidate_segment=segment,
            )
            episodes.append(episode.__dict__)
        return pd.DataFrame(episodes)

    def build_window_scores(
        self,
        sequences: pd.DataFrame,
        assignments: pd.DataFrame,
    ) -> pd.DataFrame:
        """Derive window scores from sequence and assignment tables."""
        return build_window_scores(sequences, assignments, self._config.weights, self._config.window)

    def _finalize_episode(
        self,
        merged: pd.DataFrame,
        start_index: int,
        end_index: int,
        pre_event_window: pd.Timedelta,
        source_window_ids: tuple[str, ...] = tuple(),
        candidate_segment: CandidateSegment | None = None,
    ) -> IncidentEpisode:
        """Finalize one incident episode from a segmented interval.

        Notes:
            Recovery assessment and family assignment are intentionally computed
            after segmentation so they can observe the whole interval rather than
            only the onset window.
        """
        indices = list(range(start_index, end_index + 1))
        episode_frame = merged.loc[indices]
        start_row = merged.loc[start_index]
        end_row = merged.loc[end_index]
        recovery = self._recovery.assess(
            merged,
            last_event_index=end_index,
            recovery_threshold=self._config.detection.recovery_threshold,
            recovery_windows=self._config.detection.recovery_windows,
        )
        if candidate_segment is not None and candidate_segment.recovery_start is not None:
            recovery_end = candidate_segment.recovery_end
            recovery = type(recovery)(
                recovery_start=candidate_segment.recovery_start,
                recovery_end=recovery_end,
                recovery_status=recovery.recovery_status,
                time_to_recovery_seconds=(recovery_end - end_row["end_time"]).total_seconds() if recovery_end is not None else recovery.time_to_recovery_seconds,
            )
        family_assignment = self._classifier.assign(episode_frame, self._config.family_assignment)
        peak_index = candidate_segment.start_index if candidate_segment is not None else start_index
        if candidate_segment is not None:
            peak_index = candidate_segment.start_index + (0 if candidate_segment.peak_time is None else 0)
            peak_matches = merged.index[merged["start_time"].eq(candidate_segment.peak_time)]
            if len(peak_matches) > 0:
                peak_index = int(peak_matches[0])
        else:
            peak_index = int(pd.to_numeric(episode_frame["deviation_score"], errors="coerce").fillna(0.0).idxmax())

        return IncidentEpisode(
            episode_id=f"episode_{start_index}_{end_index}",
            event_start=start_row["start_time"],
            event_end=end_row["end_time"],
            pre_event_start=start_row["start_time"] - pre_event_window,
            peak_time=merged.loc[peak_index, "start_time"],
            recovery_start=recovery.recovery_start,
            recovery_end=recovery.recovery_end,
            primary_family=family_assignment.primary_family,
            secondary_families=family_assignment.secondary_families,
            family_confidence=family_assignment.family_confidence,
            assignment_method=family_assignment.assignment_method,
            evidence=family_assignment.evidence,
            onset_score=float(start_row["deviation_score"]),
            peak_score=float(episode_frame["deviation_score"].max()),
            mean_score=float(episode_frame["deviation_score"].mean()),
            duration_seconds=(end_row["end_time"] - start_row["start_time"]).total_seconds(),
            time_to_recovery_seconds=recovery.time_to_recovery_seconds,
            recovery_status=recovery.recovery_status.value,
            source_sequence_indices=tuple(indices),
            source_window_ids=source_window_ids or tuple(str(index) for index in indices),
            asset_id=start_row.get("asset_id") if "asset_id" in merged.columns else None,
        )

    @staticmethod
    def _candidate_segments_from_frame(segments, frame: pd.DataFrame) -> list[CandidateSegment]:
        """Convert segmenter output into a stable candidate-segment contract."""
        return [
            CandidateSegment(
                candidate_id=f"segment_{segment.start_index}_{segment.end_index}",
                start_index=segment.start_index,
                end_index=segment.end_index,
                event_start=pd.to_datetime(frame.loc[segment.start_index, "start_time"]),
                peak_time=pd.to_datetime(frame.loc[segment.peak_index, "start_time"]),
                event_end=pd.to_datetime(frame.loc[segment.end_index, "end_time"]),
                recovery_start=pd.to_datetime(frame.loc[segment.recovery_start_index, "start_time"]) if segment.recovery_start_index is not None else None,
                recovery_end=pd.to_datetime(frame.loc[segment.recovery_end_index, "end_time"]) if segment.recovery_end_index is not None else None,
                source_window_ids=tuple(str(index) for index in range(segment.start_index, segment.end_index + 1)),
            )
            for segment in segments
        ]

    @staticmethod
    def _empty_episode_frame() -> pd.DataFrame:
        """Return an empty episode dataframe with the canonical output schema."""
        return pd.DataFrame(
            columns=[
                "episode_id",
                "event_start",
                "event_end",
                "pre_event_start",
                "peak_time",
                "recovery_start",
                "recovery_end",
                "primary_family",
                "secondary_families",
                "family_confidence",
                "assignment_method",
                "evidence",
                "onset_score",
                "peak_score",
                "mean_score",
                "duration_seconds",
                "time_to_recovery_seconds",
                "recovery_status",
                "source_sequence_indices",
                "source_window_ids",
                "asset_id",
                "registry_incident_id",
                "label_strength",
            ]
        )
