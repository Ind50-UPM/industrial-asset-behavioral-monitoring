"""Sequence-analysis utilities for Model_B behavioral modeling.

This module contains the typed data structures and core logic that translate a
state timeline into progressively richer behavioral abstractions: contiguous
runs, active sequences, nominal references, anomaly comparisons, longitudinal
metrics, recovery metrics, and family-signature summaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class StateRun:
    """Contiguous temporal segment where the decoded state remains constant.

    Attributes:
        state: Integer state identifier observed throughout the run.
        start_time: Timestamp of the first sample in the run.
        end_time: Timestamp of the last sample in the run.
        sample_count: Number of samples belonging to the run.
        duration_seconds: Duration covered by the run.
    """

    state: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    sample_count: int
    duration_seconds: float


@dataclass(frozen=True)
class ActiveSequence:
    """Higher-level sequence formed by consecutive non-zero runs.

    Attributes:
        start_time: Timestamp of the first run in the sequence.
        end_time: Timestamp of the last run in the sequence.
        states: Ordered tuple of non-zero states composing the sequence.
        total_duration_seconds: Total duration accumulated by the sequence.
        run_count: Number of runs participating in the sequence.
    """

    start_time: pd.Timestamp
    end_time: pd.Timestamp
    states: tuple[int, ...]
    total_duration_seconds: float
    run_count: int


@dataclass(frozen=True)
class NominalSequenceReference:
    """Condensed representation of a recurrent nominal sequence word.

    Attributes:
        states: Canonical state word used as nominal reference.
        count: Number of observations supporting the reference.
        avg_duration_seconds: Average observed duration for the word.
    """

    states: tuple[int, ...]
    count: int
    avg_duration_seconds: float


@dataclass(frozen=True)
class SequenceComparison:
    """Anomaly-oriented comparison between one observed and one nominal word.

    Attributes:
        observed_states: Observed state word.
        nominal_states: Best-matching nominal word.
        exact_match: Whether both words are identical.
        state_distance: Edit-distance-like difference between words.
        dtw_distance: Dynamic-time-warping-inspired mismatch score.
        duration_ratio_delta: Relative duration deviation from nominal.
        anomaly_score: Combined score used for thresholding.
        is_anomalous: Flag indicating whether the score crosses the threshold.
    """

    observed_states: tuple[int, ...]
    nominal_states: tuple[int, ...]
    exact_match: bool
    state_distance: int
    dtw_distance: float
    duration_ratio_delta: float
    anomaly_score: float
    is_anomalous: bool


@dataclass(frozen=True)
class LongitudinalSequenceMetrics:
    """Rolling metrics describing the temporal behavior of one sequence event."""

    sequence_index: int
    states: tuple[int, ...]
    recurrence_interval_seconds: float | None
    sequence_persistence_ratio: float
    over_activation_rate: float
    transition_instability: float
    duration_drift: float
    rolling_divergence: float
    nominal_match: tuple[int, ...]
    anomaly_flag: bool


@dataclass(frozen=True)
class RecoveryMetrics:
    """Recovery-oriented summary anchored on one anomalous sequence."""

    anomaly_sequence_index: int
    recovered_sequence_index: int | None
    time_to_recovery_seconds: float | None
    partial_recovery_score: float
    post_intervention_regime_shift: bool


class BehavioralSequenceAnalyzer:
    """Load state timelines and derive run-, sequence-, and longitudinal-level features.

    Args:
        state_column: Name of the state column used when reading input tables.

    Notes:
        The analyzer is intentionally deterministic and side-effect free so it
        can be used both by small CLIs and by the larger month-wise batch
        utilities introduced for historical campaign processing.
    """

    def __init__(self, state_column: str = "Predicted_State") -> None:
        self.state_column = state_column

    def load_state_timeline(self, file_path: str | Path) -> pd.DataFrame:
        """Load a state timeline from Parquet, Excel, or CSV.

        Args:
            file_path: Path to the timeline source.

        Returns:
            Time-indexed dataframe sorted chronologically.

        Raises:
            ValueError: If the file format is unsupported or no usable temporal
                axis and state column can be resolved.

        Notes:
            The helper supports the repository's real datasets where the state
            column may be named either ``Predicted_State`` or ``estado``.
        """
        path = Path(file_path)
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        elif path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file extension: {path.suffix}")

        if self.state_column not in df.columns and "estado" in df.columns:
            df = df.rename(columns={"estado": self.state_column})

        if self.state_column not in df.columns:
            raise ValueError(f"State column '{self.state_column}' not found in {path}")

        if "Time" in df.columns:
            df["Time"] = pd.to_datetime(df["Time"])
            df = df.set_index("Time")
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("A datetime index or a 'Time' column is required.")

        return df.sort_index()

    def smooth_short_runs(
        self,
        timeline: pd.DataFrame,
        *,
        min_duration_seconds: float = 1.0,
        min_samples: int = 1,
    ) -> pd.DataFrame:
        """Merge short transient runs into the following active state.

        Args:
            timeline: Time-indexed state timeline.
            min_duration_seconds: Maximum duration treated as transient.
            min_samples: Maximum sample count treated as transient.

        Returns:
            Smoothed copy of the input timeline.

        Notes:
            The heuristic is intentionally asymmetric: transient active runs are
            merged into the following active run, not into the previous one, so
            later sequence extraction remains compatible with the operational
            progression of the signal.
        """
        smoothed = timeline.copy()
        runs = self.extract_runs(smoothed)
        if len(runs) < 2:
            return smoothed

        values = smoothed[self.state_column].to_numpy(copy=True)
        run_boundaries = self._compute_run_boundaries(values)

        for index, run in enumerate(runs[:-1]):
            if (
                run.state > 0
                and runs[index + 1].state > 0
                and run.state != runs[index + 1].state
                and run.duration_seconds < min_duration_seconds
                and run.sample_count <= min_samples
            ):
                start_idx, stop_idx = run_boundaries[index]
                values[start_idx:stop_idx] = runs[index + 1].state

        smoothed[self.state_column] = values
        return smoothed

    def extract_runs(self, timeline: pd.DataFrame) -> list[StateRun]:
        """Extract contiguous runs of the same state from a timeline."""
        states = timeline[self.state_column].astype(int).to_numpy()
        boundaries = self._compute_run_boundaries(states)
        runs: list[StateRun] = []
        for start_idx, stop_idx in boundaries:
            start_time = timeline.index[start_idx]
            end_time = timeline.index[stop_idx - 1]
            duration_seconds = max((end_time - start_time).total_seconds(), 0.0)
            runs.append(
                StateRun(
                    state=int(states[start_idx]),
                    start_time=start_time,
                    end_time=end_time,
                    sample_count=stop_idx - start_idx,
                    duration_seconds=duration_seconds,
                )
            )
        return runs

    def extract_active_sequences(self, timeline: pd.DataFrame) -> list[ActiveSequence]:
        """Group consecutive non-zero runs into higher-level active sequences."""
        runs = self.extract_runs(timeline)
        sequences: list[ActiveSequence] = []
        current_runs: list[StateRun] = []

        for run in runs:
            if run.state > 0:
                current_runs.append(run)
                continue
            if current_runs:
                sequences.append(self._build_active_sequence(current_runs))
                current_runs = []

        if current_runs:
            sequences.append(self._build_active_sequence(current_runs))

        return sequences

    def summarize_sequence_words(self, sequences: Iterable[ActiveSequence]) -> pd.DataFrame:
        """Aggregate repeated sequence words and their average duration."""
        rows = [
            {
                "word": sequence.states,
                "count": 1,
                "duration_seconds": sequence.total_duration_seconds,
            }
            for sequence in sequences
        ]
        if not rows:
            return pd.DataFrame(columns=["word", "count", "avg_duration_seconds"])

        frame = pd.DataFrame(rows)
        return (
            frame.groupby("word", dropna=False)
            .agg(count=("count", "sum"), avg_duration_seconds=("duration_seconds", "mean"))
            .reset_index()
            .sort_values(["count", "avg_duration_seconds"], ascending=[False, False])
        )

    def build_nominal_reference(self, sequences: Iterable[ActiveSequence]) -> list[NominalSequenceReference]:
        """Convert observed sequences into a condensed nominal reference list."""
        summary = self.summarize_sequence_words(sequences)
        return [
            NominalSequenceReference(
                states=tuple(word),
                count=int(count),
                avg_duration_seconds=float(avg_duration_seconds),
            )
            for word, count, avg_duration_seconds in summary.itertuples(index=False)
        ]

    def compare_to_nominal(
        self,
        observed_sequences: Iterable[ActiveSequence],
        nominal_references: Iterable[NominalSequenceReference],
        *,
        anomaly_threshold: float = 1.0,
    ) -> pd.DataFrame:
        """Compare observed sequence words against a nominal reference set."""
        references = list(nominal_references)
        return pd.DataFrame(
            [
                self._compare_single_sequence(
                    sequence,
                    references,
                    anomaly_threshold=anomaly_threshold,
                ).__dict__
                for sequence in observed_sequences
            ]
        )

    def compute_longitudinal_metrics(
        self,
        observed_sequences: Iterable[ActiveSequence],
        nominal_references: Iterable[NominalSequenceReference],
        *,
        anomaly_threshold: float = 1.0,
        rolling_window: int = 3,
    ) -> pd.DataFrame:
        """Compute temporal recurrence and divergence metrics for each sequence."""
        sequences = list(observed_sequences)
        references = list(nominal_references)
        rows: list[dict[str, object]] = []

        for index, sequence in enumerate(sequences):
            comparison = self._compare_single_sequence(
                sequence,
                references,
                anomaly_threshold=anomaly_threshold,
            )
            previous_same = self._find_previous_same_sequence(sequences, index)
            previous_sequence = sequences[index - 1] if index > 0 else None
            rows.append(
                LongitudinalSequenceMetrics(
                    sequence_index=index,
                    states=sequence.states,
                    recurrence_interval_seconds=(
                        None
                        if previous_same is None
                        else (sequence.start_time - previous_same.start_time).total_seconds()
                    ),
                    sequence_persistence_ratio=self._sequence_persistence_ratio(sequence),
                    over_activation_rate=self._over_activation_rate(sequence),
                    transition_instability=self._transition_instability(sequence, previous_sequence),
                    duration_drift=comparison.duration_ratio_delta,
                    rolling_divergence=self._rolling_divergence(
                        sequences,
                        references,
                        index=index,
                        anomaly_threshold=anomaly_threshold,
                        window=rolling_window,
                    ),
                    nominal_match=comparison.nominal_states,
                    anomaly_flag=comparison.is_anomalous,
                ).__dict__
            )
        return pd.DataFrame(rows)

    def compute_recovery_metrics(
        self,
        observed_sequences: Iterable[ActiveSequence],
        nominal_references: Iterable[NominalSequenceReference],
        *,
        anomaly_threshold: float = 1.0,
    ) -> pd.DataFrame:
        """Estimate recovery behavior after anomalous sequences."""
        sequences = list(observed_sequences)
        references = list(nominal_references)
        comparisons = [
            self._compare_single_sequence(sequence, references, anomaly_threshold=anomaly_threshold)
            for sequence in sequences
        ]
        rows: list[dict[str, object]] = []

        for index, comparison in enumerate(comparisons):
            if not comparison.is_anomalous:
                continue

            recovered_index = None
            recovery_time = None
            partial_recovery_score = 0.0
            regime_shift = False

            for future_index in range(index + 1, len(comparisons)):
                future = comparisons[future_index]
                if future.nominal_states != comparison.nominal_states:
                    continue
                partial_recovery_score = max(
                    0.0,
                    1.0 - min(future.anomaly_score, comparison.anomaly_score) / max(comparison.anomaly_score, 1e-9),
                )
                if not future.is_anomalous:
                    recovered_index = future_index
                    recovery_time = (
                        sequences[future_index].end_time - sequences[index].end_time
                    ).total_seconds()
                    regime_shift = future.observed_states != comparison.observed_states
                    break

            rows.append(
                RecoveryMetrics(
                    anomaly_sequence_index=index,
                    recovered_sequence_index=recovered_index,
                    time_to_recovery_seconds=recovery_time,
                    partial_recovery_score=partial_recovery_score,
                    post_intervention_regime_shift=regime_shift,
                ).__dict__
            )
        return pd.DataFrame(rows)

    def summarize_family_signatures(self, longitudinal_metrics: pd.DataFrame) -> pd.DataFrame:
        """Summarize sequence signatures into coarse incident-family candidates."""
        if longitudinal_metrics.empty:
            return pd.DataFrame(
                columns=[
                    "incident_family_candidate",
                    "signature_word",
                    "count",
                    "mean_transition_instability",
                    "mean_duration_drift",
                ]
            )

        metrics = longitudinal_metrics.copy()
        metrics["incident_family_candidate"] = metrics.apply(
            lambda row: self._infer_family_candidate(
                row["transition_instability"],
                row["duration_drift"],
                bool(row["anomaly_flag"]),
            ),
            axis=1,
        )
        return (
            metrics.groupby(["incident_family_candidate", "states"], dropna=False)
            .agg(
                count=("states", "size"),
                mean_transition_instability=("transition_instability", "mean"),
                mean_duration_drift=("duration_drift", "mean"),
            )
            .reset_index()
            .rename(columns={"states": "signature_word"})
            .sort_values(["count", "mean_transition_instability"], ascending=[False, False])
        )

    @staticmethod
    def _compute_run_boundaries(states: pd.Series | list[int] | pd.Index | pd.array) -> list[tuple[int, int]]:
        """Compute contiguous-state boundaries from a one-dimensional state series."""
        values = list(states)
        if not values:
            return []

        boundaries: list[tuple[int, int]] = []
        start = 0
        for index in range(1, len(values)):
            if values[index] != values[index - 1]:
                boundaries.append((start, index))
                start = index
        boundaries.append((start, len(values)))
        return boundaries

    @staticmethod
    def _build_active_sequence(runs: list[StateRun]) -> ActiveSequence:
        """Assemble one active sequence from a list of non-zero runs."""
        return ActiveSequence(
            start_time=runs[0].start_time,
            end_time=runs[-1].end_time,
            states=tuple(run.state for run in runs),
            total_duration_seconds=sum(run.duration_seconds for run in runs),
            run_count=len(runs),
        )

    def _compare_single_sequence(
        self,
        sequence: ActiveSequence,
        references: list[NominalSequenceReference],
        *,
        anomaly_threshold: float,
    ) -> SequenceComparison:
        """Compare one observed sequence against the best nominal reference."""
        if not references:
            anomaly_score = float(len(sequence.states) + 1.0)
            return SequenceComparison(
                observed_states=sequence.states,
                nominal_states=(),
                exact_match=False,
                state_distance=len(sequence.states),
                dtw_distance=float(len(sequence.states)),
                duration_ratio_delta=1.0,
                anomaly_score=anomaly_score,
                is_anomalous=anomaly_score >= anomaly_threshold,
            )

        best_reference = min(
            references,
            key=lambda reference: (
                self._sequence_edit_distance(sequence.states, reference.states),
                abs(self._duration_ratio_delta(sequence.total_duration_seconds, reference.avg_duration_seconds)),
                -reference.count,
            ),
        )
        state_distance = self._sequence_edit_distance(sequence.states, best_reference.states)
        dtw_distance = self._sequence_dtw_distance(sequence.states, best_reference.states)
        duration_ratio_delta = self._duration_ratio_delta(
            sequence.total_duration_seconds,
            best_reference.avg_duration_seconds,
        )
        anomaly_score = float(dtw_distance + abs(duration_ratio_delta))

        return SequenceComparison(
            observed_states=sequence.states,
            nominal_states=best_reference.states,
            exact_match=sequence.states == best_reference.states,
            state_distance=state_distance,
            dtw_distance=dtw_distance,
            duration_ratio_delta=duration_ratio_delta,
            anomaly_score=anomaly_score,
            is_anomalous=anomaly_score >= anomaly_threshold,
        )

    @staticmethod
    def _sequence_edit_distance(observed: tuple[int, ...], nominal: tuple[int, ...]) -> int:
        """Compute a Levenshtein-like edit distance between two state words."""
        rows = len(observed) + 1
        cols = len(nominal) + 1
        distance = [[0] * cols for _ in range(rows)]

        for row in range(rows):
            distance[row][0] = row
        for col in range(cols):
            distance[0][col] = col

        for row in range(1, rows):
            for col in range(1, cols):
                substitution_cost = 0 if observed[row - 1] == nominal[col - 1] else 1
                distance[row][col] = min(
                    distance[row - 1][col] + 1,
                    distance[row][col - 1] + 1,
                    distance[row - 1][col - 1] + substitution_cost,
                )
        return distance[-1][-1]

    @staticmethod
    def _duration_ratio_delta(observed_seconds: float, nominal_seconds: float) -> float:
        """Measure relative duration drift between observed and nominal sequence durations."""
        if nominal_seconds <= 0:
            return 0.0 if observed_seconds <= 0 else 1.0
        return float((observed_seconds - nominal_seconds) / nominal_seconds)

    @staticmethod
    def _sequence_dtw_distance(observed: tuple[int, ...], nominal: tuple[int, ...]) -> float:
        """Compute a simple DTW-inspired mismatch score between state words."""
        if not observed and not nominal:
            return 0.0
        if not observed or not nominal:
            return float(max(len(observed), len(nominal)))

        rows = len(observed)
        cols = len(nominal)
        dtw = [[float("inf")] * (cols + 1) for _ in range(rows + 1)]
        dtw[0][0] = 0.0

        for row in range(1, rows + 1):
            for col in range(1, cols + 1):
                cost = 0.0 if observed[row - 1] == nominal[col - 1] else 1.0
                dtw[row][col] = cost + min(
                    dtw[row - 1][col],
                    dtw[row][col - 1],
                    dtw[row - 1][col - 1],
                )
        return float(dtw[rows][cols])

    @staticmethod
    def _sequence_persistence_ratio(sequence: ActiveSequence) -> float:
        """Estimate persistence as duration per contributing run."""
        if sequence.run_count <= 0:
            return 0.0
        return float(sequence.total_duration_seconds / max(sequence.run_count, 1))

    @staticmethod
    def _over_activation_rate(sequence: ActiveSequence) -> float:
        """Estimate how densely a sequence reuses active states."""
        if not sequence.states:
            return 0.0
        active_runs = sum(1 for state in sequence.states if state > 0)
        unique_states = len(set(sequence.states))
        return float(active_runs / max(unique_states, 1))

    def _transition_instability(
        self,
        sequence: ActiveSequence,
        previous_sequence: ActiveSequence | None,
    ) -> float:
        """Quantify state-word change against the immediately previous sequence."""
        if previous_sequence is None:
            return 0.0
        distance = self._sequence_edit_distance(sequence.states, previous_sequence.states)
        scale = max(len(sequence.states), len(previous_sequence.states), 1)
        return float(distance / scale)

    def _rolling_divergence(
        self,
        sequences: list[ActiveSequence],
        references: list[NominalSequenceReference],
        *,
        index: int,
        anomaly_threshold: float,
        window: int,
    ) -> float:
        """Average recent anomaly scores over a rolling sequence window."""
        start = max(0, index - window + 1)
        scores = [
            self._compare_single_sequence(sequence, references, anomaly_threshold=anomaly_threshold).anomaly_score
            for sequence in sequences[start : index + 1]
        ]
        return float(sum(scores) / len(scores)) if scores else 0.0

    @staticmethod
    def _find_previous_same_sequence(sequences: list[ActiveSequence], index: int) -> ActiveSequence | None:
        """Return the most recent prior sequence with the same state word."""
        current = sequences[index]
        for previous_index in range(index - 1, -1, -1):
            if sequences[previous_index].states == current.states:
                return sequences[previous_index]
        return None

    @staticmethod
    def _infer_family_candidate(
        transition_instability: float,
        duration_drift: float,
        anomaly_flag: bool,
    ) -> str:
        """Map sequence-level metrics into a coarse family candidate."""
        if not anomaly_flag:
            return "nominal_cycle"
        if transition_instability >= 0.7:
            return "recurrent_spurious_cycle"
        if duration_drift >= 0.5:
            return "process_saturation"
        return "pump_abrupt_failure"
