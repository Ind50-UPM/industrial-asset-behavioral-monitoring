"""Unit tests for Model_B sequence analysis."""

from __future__ import annotations

import pandas as pd

from iabm_behavior import ActiveSequence, BehavioralSequenceAnalyzer


def test_extract_runs_and_sequences(synthetic_timeline) -> None:
    """The analyzer should extract runs and active sequences from a timeline."""
    analyzer = BehavioralSequenceAnalyzer()
    timeline = analyzer.load_state_timeline(synthetic_timeline)

    runs = analyzer.extract_runs(timeline)
    sequences = analyzer.extract_active_sequences(timeline)
    summary = analyzer.summarize_sequence_words(sequences)

    assert len(runs) == 6
    assert runs[1].state == 1
    assert runs[2].state == 2
    assert len(sequences) == 2
    assert sequences[0].states == (1, 2)
    assert sequences[1].states == (4,)
    assert list(summary["count"]) == [1, 1]


def test_smooth_short_runs_merges_transients() -> None:
    """Short transient runs should be merged into the following run."""
    analyzer = BehavioralSequenceAnalyzer()
    timeline = pd.DataFrame(
        {"Predicted_State": [0, 1, 2, 2, 0]},
        index=pd.date_range("2022-01-01", periods=5, freq="500ms"),
    )

    smoothed = analyzer.smooth_short_runs(
        timeline,
        min_duration_seconds=1.0,
        min_samples=1,
    )

    assert smoothed["Predicted_State"].tolist() == [0, 2, 2, 2, 0]


def test_compare_to_nominal_quantifies_sequence_differences() -> None:
    """Observed sequences should be scored against the closest nominal word."""
    analyzer = BehavioralSequenceAnalyzer()
    nominal_sequences = [
        ActiveSequence(
            start_time=pd.Timestamp("2022-01-01 00:00:00"),
            end_time=pd.Timestamp("2022-01-01 00:00:03"),
            states=(1, 2),
            total_duration_seconds=3.0,
            run_count=2,
        )
    ]
    observed_sequences = [
        ActiveSequence(
            start_time=pd.Timestamp("2022-01-02 00:00:00"),
            end_time=pd.Timestamp("2022-01-02 00:00:03"),
            states=(1, 4),
            total_duration_seconds=3.0,
            run_count=2,
        )
    ]

    nominal_reference = analyzer.build_nominal_reference(nominal_sequences)
    comparison = analyzer.compare_to_nominal(observed_sequences, nominal_reference)

    assert len(comparison) == 1
    assert comparison.loc[0, "observed_states"] == (1, 4)
    assert comparison.loc[0, "nominal_states"] == (1, 2)
    assert comparison.loc[0, "state_distance"] >= 1
    assert comparison.loc[0, "dtw_distance"] >= comparison.loc[0, "state_distance"]
    assert comparison.loc[0, "anomaly_score"] >= comparison.loc[0, "dtw_distance"]
    assert bool(comparison.loc[0, "is_anomalous"]) is True


def test_compare_to_nominal_accepts_exact_nominal_sequence() -> None:
    """Exact nominal words with matching durations should not be flagged."""
    analyzer = BehavioralSequenceAnalyzer()
    nominal_sequences = [
        ActiveSequence(
            start_time=pd.Timestamp("2022-01-01 00:00:00"),
            end_time=pd.Timestamp("2022-01-01 00:00:03"),
            states=(1, 2),
            total_duration_seconds=3.0,
            run_count=2,
        )
    ]
    observed_sequences = [
        ActiveSequence(
            start_time=pd.Timestamp("2022-01-02 00:00:00"),
            end_time=pd.Timestamp("2022-01-02 00:00:03"),
            states=(1, 2),
            total_duration_seconds=3.0,
            run_count=2,
        )
    ]

    nominal_reference = analyzer.build_nominal_reference(nominal_sequences)
    comparison = analyzer.compare_to_nominal(
        observed_sequences,
        nominal_reference,
        anomaly_threshold=0.5,
    )

    assert bool(comparison.loc[0, "exact_match"]) is True
    assert comparison.loc[0, "dtw_distance"] == 0.0
    assert comparison.loc[0, "duration_ratio_delta"] == 0.0
    assert bool(comparison.loc[0, "is_anomalous"]) is False
