"""Tests for heuristic candidate incident registry generation."""

from __future__ import annotations

import pandas as pd

from iabm_incidents.registry_builder import CandidateIncidentRegistryBuilder, RegistryGenerationConfig


def test_candidate_registry_builder_generates_multiple_candidate_types() -> None:
    states = pd.DataFrame(
        {
            "source_month": ["202201"] * 5,
            "date": [
                "2022-01-01 00:00:00+01:00",
                "2022-01-01 00:00:01+01:00",
                "2022-01-01 00:20:00+01:00",
                "2022-01-01 00:20:01+01:00",
                "2022-01-01 00:20:02+01:00",
            ],
            "RP1": [100.0, 90.0, 10.0, 9.0, 9.0],
            "RP2": [100.0, 90.0, 10.0, 9.0, 9.0],
            "RP3": [100.0, 90.0, 10.0, 9.0, 9.0],
            "RP4": [0.0, 0.0, 0.0, 0.0, 0.0],
            "pred_estado": [1, 1, 2, 2, 2],
        }
    )
    sequences = pd.DataFrame(
        {
            "source_month": ["202201", "202201", "202201"],
            "date": [
                "2022-01-01 01:00:00+01:00",
                "2022-01-01 02:00:00+01:00",
                "2022-01-01 03:00:00+01:00",
            ],
            "Runs": [1, 1, 1],
            "Values": [19, 19, 19],
            "span": [10.0, 15.0, 500.0],
        }
    )
    words = pd.DataFrame(
        {
            "source_month": ["202201"] * 4,
            "date": [
                "2022-01-01 02:55:00+01:00",
                "2022-01-01 03:05:00+01:00",
                "2022-01-01 03:15:00+01:00",
                "2022-01-01 06:00:00+01:00",
            ],
            "word": ["[19, 31, 12]", "[99, 99]", "[99, 99]", "[99, 99]"],
            "span": [10.0, 5.0, 5.0, 5.0],
        }
    )
    builder = CandidateIncidentRegistryBuilder(
        RegistryGenerationConfig(
            downtime_gap_seconds=60.0,
            low_power_quantile=0.5,
            abrupt_drop_fraction=0.2,
            saturation_span_quantile=0.8,
            rare_word_frequency_quantile=0.5,
            recurrence_window_hours=3.0,
            minimum_recurrence_count=2,
            supporting_word_window_hours=6.0,
            abrupt_recovery_window_seconds=5.0,
            abrupt_recovery_fraction=0.2,
        )
    )

    registry = builder.build_from_frames(states=states, sequences=sequences, words=words)

    assert not registry.empty
    assert set(registry["incident_family"]).issuperset(
        {"external_ambiguous_disturbance", "process_saturation", "float_recurrent_disturbance"}
    )
    assert registry["incident_id"].str.startswith("AUTO-").all()
    assert (registry["source_type"] == "derived_state_power_supported").sum() <= 1


def test_candidate_registry_builder_merges_close_candidates() -> None:
    builder = CandidateIncidentRegistryBuilder(RegistryGenerationConfig(candidate_merge_gap_seconds=3600.0))
    states = pd.DataFrame(columns=["source_month", "date", "RP1", "RP2", "RP3", "RP4"])
    sequences = pd.DataFrame(
        {
            "source_month": ["202201", "202201"],
            "date": ["2022-01-01 01:00:00+01:00", "2022-01-01 01:10:00+01:00"],
            "Runs": [1, 1],
            "Values": [19, 19],
            "span": [500.0, 600.0],
        }
    )
    words = pd.DataFrame(columns=["source_month", "date", "word", "span"])

    registry = builder.build_from_frames(states=states, sequences=sequences, words=words)

    assert len(registry) == 1
