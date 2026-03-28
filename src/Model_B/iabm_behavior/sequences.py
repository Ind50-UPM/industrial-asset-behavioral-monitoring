"""Sequence-analysis utilities for Model_B behavioral modeling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class StateRun:
    """Represent a contiguous run of the same predicted or measured state.

    Attributes:
        state: State identifier associated with the run.
        start_time: Timestamp of the first sample in the run.
        end_time: Timestamp of the last sample in the run.
        sample_count: Number of rows belonging to the run.
        duration_seconds: Elapsed time covered by the run.
    """

    state: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    sample_count: int
    duration_seconds: float


@dataclass(frozen=True)
class ActiveSequence:
    """Represent a higher-level active behavioral sequence.

    Attributes:
        start_time: Timestamp of the first active run.
        end_time: Timestamp of the last active run.
        states: Ordered tuple of state identifiers composing the sequence.
        total_duration_seconds: Total duration of the sequence in seconds.
        run_count: Number of state runs contained in the sequence.
    """

    start_time: pd.Timestamp
    end_time: pd.Timestamp
    states: tuple[int, ...]
    total_duration_seconds: float
    run_count: int


@dataclass(frozen=True)
class NominalSequenceReference:
    """Represent a nominal behavioral word learned from historical sequences.

    Attributes:
        states: Ordered state tuple that defines the nominal word.
        count: Number of times the word appears in the reference dataset.
        avg_duration_seconds: Mean duration of the word across occurrences.
    """

    states: tuple[int, ...]
    count: int
    avg_duration_seconds: float


@dataclass(frozen=True)
class SequenceComparison:
    """Store the comparison of an observed sequence against a nominal reference.

    Attributes:
        observed_states: Observed sequence word.
        nominal_states: Closest nominal word found in the reference set.
        exact_match: Whether both words are identical.
        state_distance: Discrete distance between words using edit distance.
        dtw_distance: Alignment distance between the observed and nominal words.
        duration_ratio_delta: Relative duration deviation against the nominal word.
        anomaly_score: Aggregate score combining state mismatch and duration drift.
        is_anomalous: Whether the aggregate score exceeds the configured threshold.
    """

    observed_states: tuple[int, ...]
    nominal_states: tuple[int, ...]
    exact_match: bool
    state_distance: int
    dtw_distance: float
    duration_ratio_delta: float
    anomaly_score: float
    is_anomalous: bool


class BehavioralSequenceAnalyzer:
    """Load state timelines and derive run- and sequence-level behavior features.

    The analyzer transforms Model_A predictions or digital state traces into
    contiguous runs and active state sequences. It also includes a lightweight
    smoothing step inspired by the original legacy scripts, where very short
    transient runs can be merged into the following state to reduce noise.
    """

    def __init__(self, state_column: str = "Predicted_State") -> None:
        """Initialize the analyzer with the configured state column name.

        Args:
            state_column: Column holding the discrete state identifiers in
                input timelines and exported reports.
        """
        self.state_column = state_column

    def load_state_timeline(self, file_path: str | Path) -> pd.DataFrame:
        """Load a time-indexed state timeline from CSV, Excel, or Parquet.

        Args:
            file_path: Path to the state timeline file.

        Returns:
            A DataFrame indexed by timestamps and containing the configured
            state column.
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
        """Merge short transient runs into the following run when possible.

        Args:
            timeline: Input state timeline.
            min_duration_seconds: Maximum duration treated as transient noise.
            min_samples: Maximum sample count treated as transient noise.

        Returns:
            A copy of the timeline with eligible short runs reassigned.
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
                and
                run.duration_seconds < min_duration_seconds
                and run.sample_count <= min_samples
            ):
                # Very short active transients are treated as sequence noise and
                # reassigned to the following active run, mirroring the legacy
                # behavioral-cleaning logic.
                start_idx, stop_idx = run_boundaries[index]
                next_state = runs[index + 1].state
                values[start_idx:stop_idx] = next_state

        smoothed[self.state_column] = values
        return smoothed

    def extract_runs(self, timeline: pd.DataFrame) -> list[StateRun]:
        """Convert a state timeline into contiguous state runs.

        Args:
            timeline: Time-indexed DataFrame containing the configured state column.

        Returns:
            A list of :class:`StateRun` objects ordered by time.
        """
        states = timeline[self.state_column].astype(int).to_numpy()
        boundaries = self._compute_run_boundaries(states)
        runs: list[StateRun] = []
        for start_idx, stop_idx in boundaries:
            start_time = timeline.index[start_idx]
            end_time = timeline.index[stop_idx - 1]
            duration_seconds = max(
                (end_time - start_time).total_seconds(),
                0.0,
            )
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
        """Group consecutive non-zero runs into active behavioral sequences.

        Args:
            timeline: Time-indexed DataFrame containing state values.

        Returns:
            A list of active sequences. Each sequence stores the ordered state
            pattern and its total duration.
        """
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
        """Count repeated behavioral words extracted from active sequences.

        Args:
            sequences: Iterable of active sequences.

        Returns:
            A DataFrame with the sequence word, occurrence count, and average
            duration.
        """
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
        summary = (
            frame.groupby("word", dropna=False)
            .agg(count=("count", "sum"), avg_duration_seconds=("duration_seconds", "mean"))
            .reset_index()
            .sort_values(["count", "avg_duration_seconds"], ascending=[False, False])
        )
        return summary

    def build_nominal_reference(
        self,
        sequences: Iterable[ActiveSequence],
    ) -> list[NominalSequenceReference]:
        """Build nominal references from repeated behavioral words.

        Args:
            sequences: Historical active sequences representing nominal behavior.

        Returns:
            A list of nominal references sorted by occurrence count and duration.
        """
        summary = self.summarize_sequence_words(sequences)
        references = [
            NominalSequenceReference(
                states=tuple(word),
                count=int(count),
                avg_duration_seconds=float(avg_duration_seconds),
            )
            for word, count, avg_duration_seconds in summary.itertuples(index=False)
        ]
        return references

    def compare_to_nominal(
        self,
        observed_sequences: Iterable[ActiveSequence],
        nominal_references: Iterable[NominalSequenceReference],
        *,
        anomaly_threshold: float = 1.0,
    ) -> pd.DataFrame:
        """Compare observed sequences against the closest nominal references.

        Args:
            observed_sequences: Sequences extracted from the timeline under study.
            nominal_references: Reference words representing nominal behavior.
            anomaly_threshold: Score threshold used to flag anomalous sequences.

        Returns:
            A DataFrame where each row quantifies the difference between an
            observed sequence and its closest nominal counterpart.
        """
        references = list(nominal_references)
        rows = [
            self._compare_single_sequence(
                sequence,
                references,
                anomaly_threshold=anomaly_threshold,
            ).__dict__
            for sequence in observed_sequences
        ]
        return pd.DataFrame(rows)

    @staticmethod
    def _compute_run_boundaries(states: pd.Series | list[int] | pd.Index | pd.array) -> list[tuple[int, int]]:
        """Compute half-open boundaries for contiguous equal-value runs.

        Args:
            states: Ordered state identifiers from a timeline.

        Returns:
            A list of ``(start, stop)`` pairs using half-open interval
            notation, suitable for slicing NumPy arrays and DataFrames.
        """
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
        """Create an active sequence object from a contiguous list of runs.

        Args:
            runs: Consecutive non-zero runs belonging to the same active window.

        Returns:
            One aggregated active-sequence object ready for downstream
            summarization or nominal comparison.
        """
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
        """Compare a single observed sequence against the closest nominal word.

        Args:
            sequence: Observed sequence under evaluation.
            references: Nominal reference words available for matching.
            anomaly_threshold: Threshold used to flag the sequence as anomalous.

        Returns:
            A structured comparison result for one observed sequence.
        """
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
                # Structural similarity is prioritized over duration; nominal
                # frequency acts as a final tie breaker across equivalent words.
                self._sequence_edit_distance(sequence.states, reference.states),
                abs(
                    self._duration_ratio_delta(
                        sequence.total_duration_seconds,
                        reference.avg_duration_seconds,
                    )
                ),
                -reference.count,
            ),
        )
        state_distance = self._sequence_edit_distance(
            sequence.states,
            best_reference.states,
        )
        dtw_distance = self._sequence_dtw_distance(
            sequence.states,
            best_reference.states,
        )
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
    def _sequence_edit_distance(
        observed: tuple[int, ...],
        nominal: tuple[int, ...],
    ) -> int:
        """Compute a discrete edit distance between two state words.

        Args:
            observed: Observed sequence word.
            nominal: Nominal reference word.

        Returns:
            Levenshtein-style edit distance between both words.
        """
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
        """Return relative duration deviation against the nominal duration.

        Args:
            observed_seconds: Duration of the observed sequence.
            nominal_seconds: Average duration of the nominal reference word.

        Returns:
            Relative deviation expressed as ``(observed - nominal) / nominal``.
        """
        if nominal_seconds <= 0:
            return 0.0 if observed_seconds <= 0 else 1.0
        return float((observed_seconds - nominal_seconds) / nominal_seconds)

    @staticmethod
    def _sequence_dtw_distance(
        observed: tuple[int, ...],
        nominal: tuple[int, ...],
    ) -> float:
        """Compute a simple DTW alignment distance between two state words.

        Args:
            observed: Observed sequence word.
            nominal: Nominal reference word.

        Returns:
            Dynamic Time Warping distance using a binary match or mismatch cost.
        """
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
                # DTW allows local stretching so similar words with shifted
                # transitions can still align with a limited penalty.
                dtw[row][col] = cost + min(
                    dtw[row - 1][col],
                    dtw[row][col - 1],
                    dtw[row - 1][col - 1],
                )
        return float(dtw[rows][cols])
