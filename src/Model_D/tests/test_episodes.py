"""Unit tests for Model_D episode construction."""

from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from iabm_incidents import EpisodeEvaluator, IncidentEpisodeBuilder, IncidentRegistry, OccurrenceModeler
from iabm_incidents.classification import RuleBasedFamilyClassifier
from iabm_incidents.config import FamilyAssignmentConfig, ModelDConfig, WindowConfig
from iabm_incidents.episodes import CandidateSegment
from iabm_incidents.metrics import summarize_episode_metrics
from iabm_incidents.taxonomy import FAMILY_SIGNATURES, is_known_family


def _prewindowed_frame(assignments_frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asset_id": ["WW-01"] * 6,
            "window_start": [
                "2022-01-01 00:00:00",
                "2022-01-01 01:00:00",
                "2022-01-01 02:00:00",
                "2022-01-01 03:00:00",
                "2022-01-01 04:00:00",
                "2022-01-01 05:00:00",
            ],
            "window_end": [
                "2022-01-01 00:59:59",
                "2022-01-01 01:59:59",
                "2022-01-01 02:59:59",
                "2022-01-01 03:59:59",
                "2022-01-01 04:59:59",
                "2022-01-01 05:59:59",
            ],
            "data_coverage": [0.99] * 6,
            "sequence_count": [2] * 6,
            "active_sequence_count": [2] * 6,
            **assignments_frame.to_dict(orient="list"),
        }
    )


def test_build_synthetic_incident_episodes_from_prewindowed_input(assignments_frame) -> None:
    builder = IncidentEpisodeBuilder()
    window_indicators = _prewindowed_frame(assignments_frame)
    episodes = builder.build(window_indicators, window_indicators)

    assert len(episodes) == 2
    assert episodes.loc[0, "primary_family"] == "process_saturation"
    assert episodes.loc[1, "primary_family"] == "float_recurrent_disturbance"
    assert episodes.loc[0, "time_to_recovery_seconds"] == 3600.0
    assert episodes.loc[1, "time_to_recovery_seconds"] == 3600.0
    assert "peak_time" in episodes.columns
    assert "source_window_ids" in episodes.columns


def test_build_window_scores_constructs_real_rolling_windows_from_timed_rows(
    sequences_frame,
    assignments_frame,
) -> None:
    config = ModelDConfig(
        window=WindowConfig(
            length=pd.Timedelta(minutes=20),
            step=pd.Timedelta(minutes=10),
            min_coverage=0.0,
            min_active_sequences=1,
        )
    )
    builder = IncidentEpisodeBuilder(config)

    window_scores = builder.build_window_scores(sequences_frame, assignments_frame)

    assert len(window_scores) >= 3
    assert window_scores.loc[0, "start_time"] == pd.Timestamp("2022-01-01 00:00:00")
    assert window_scores.loc[0, "end_time"] == pd.Timestamp("2022-01-01 00:20:00")
    assert window_scores.loc[0, "sequence_count"] >= 2
    assert window_scores["deviation_score"].max() > 0.0


def test_build_episodes_from_segments(assignments_frame) -> None:
    builder = IncidentEpisodeBuilder()
    window_scores = builder.build_window_scores(_prewindowed_frame(assignments_frame), _prewindowed_frame(assignments_frame))
    segment = CandidateSegment(
        candidate_id="candidate-1",
        start_index=0,
        end_index=1,
        event_start=pd.Timestamp("2022-01-01 00:00:00"),
        peak_time=pd.Timestamp("2022-01-01 01:00:00"),
        event_end=pd.Timestamp("2022-01-01 01:59:59"),
        source_window_ids=("0", "1"),
    )

    episodes = builder.build_episodes_from_segments([segment], window_scores)

    assert len(episodes) == 1
    assert episodes.loc[0, "episode_id"] == "episode_0_1"
    assert episodes.loc[0, "source_window_ids"] == ("0", "1")


def test_rule_based_classifier_detects_pump_failure_signature(
    pump_failure_assignments_frame,
) -> None:
    classifier = RuleBasedFamilyClassifier()
    window_frame = pd.DataFrame(
        {
            "asset_id": ["WW-01", "WW-01"],
            "window_start": ["2022-01-01 00:00:00", "2022-01-01 01:00:00"],
            "window_end": ["2022-01-01 00:59:59", "2022-01-01 01:59:59"],
            "data_coverage": [0.99, 0.99],
            "sequence_count": [2, 2],
            "active_sequence_count": [2, 2],
            **pump_failure_assignments_frame.to_dict(orient="list"),
        }
    )
    builder = IncidentEpisodeBuilder()
    scored = builder.build_window_scores(window_frame, window_frame)

    assignment = classifier.assign(scored, FamilyAssignmentConfig(minimum_confidence=0.6))

    assert assignment.primary_family == "pump_abrupt_failure"
    assert assignment.family_confidence >= 0.8
    assert "large consumption drop" in assignment.evidence
    assert "state prediction degradation" in assignment.evidence


def test_episode_windows_align_with_known_incidents(assignments_frame, known_incidents_frame) -> None:
    builder = IncidentEpisodeBuilder()
    registry = IncidentRegistry()
    window_indicators = _prewindowed_frame(assignments_frame)
    episodes = builder.build(window_indicators, window_indicators)
    alignment = registry.align_with_episodes(known_incidents_frame, episodes)

    assert alignment["window_contains_event"].tolist() == [True, True]
    assert alignment["matched_episode_id"].notna().all()


def test_occurrence_summary_and_reproducible_episode_metrics(assignments_frame) -> None:
    builder = IncidentEpisodeBuilder()
    modeler = OccurrenceModeler()
    window_indicators = _prewindowed_frame(assignments_frame)
    episodes = builder.build(window_indicators, window_indicators)
    summary = summarize_episode_metrics(episodes).reset_index(drop=True)
    occurrence = modeler.summarize(episodes).reset_index(drop=True)

    expected_summary = pd.DataFrame(
        {
            "primary_family": ["float_recurrent_disturbance", "process_saturation"],
            "episode_count": [1, 1],
            "mean_duration_seconds": [7199.0, 7199.0],
            "median_duration_seconds": [7199.0, 7199.0],
            "mean_time_to_recovery_seconds": [3600.0, 3600.0],
            "median_time_to_recovery_seconds": [3600.0, 3600.0],
            "recovery_rate": [0.0, 0.0],
            "mean_peak_score": [1.95, 1.94],
        }
    )
    expected_occurrence = pd.DataFrame(
        {
            "primary_family": ["float_recurrent_disturbance", "process_saturation"],
            "occurrence_count": [1, 1],
            "mean_occurrence_interval_seconds": [None, None],
            "median_occurrence_interval_seconds": [None, None],
            "minimum_occurrence_interval_seconds": [None, None],
            "maximum_occurrence_interval_seconds": [None, None],
            "exposure_hours": [None, None],
            "excluded_hours": [None, None],
            "event_rate_per_hour": [None, None],
            "fit_status": ["insufficient_events", "insufficient_events"],
        }
    )

    assert_frame_equal(summary, expected_summary, check_dtype=False)
    assert_frame_equal(occurrence, expected_occurrence, check_dtype=False)


def test_occurrence_summary_uses_observation_exposure() -> None:
    modeler = OccurrenceModeler()
    episodes = pd.DataFrame(
        {
            "episode_id": ["ep-1", "ep-2"],
            "primary_family": ["process_saturation", "process_saturation"],
            "event_start": ["2022-01-01 00:00:00", "2022-01-03 00:00:00"],
            "asset_id": ["WW-01", "WW-01"],
        }
    )
    exposure = pd.DataFrame(
        {
            "asset_id": ["WW-01", "WW-01"],
            "observation_start": ["2022-01-01 00:00:00", "2022-01-01 00:00:00"],
            "observation_end": ["2022-01-06 00:00:00", "2022-01-06 00:00:00"],
            "excluded_start": ["2022-01-02 00:00:00", None],
            "excluded_end": ["2022-01-02 12:00:00", None],
            "exclusion_reason": ["maintenance", None],
        }
    )

    summary = modeler.summarize(episodes, exposure).reset_index(drop=True)

    assert summary.loc[0, "exposure_hours"] == 228.0
    assert summary.loc[0, "excluded_hours"] == 12.0
    assert round(summary.loc[0, "event_rate_per_hour"], 6) == round(2.0 / 228.0, 6)


def test_evaluation_summary_reports_temporal_metrics(assignments_frame, known_incidents_frame) -> None:
    builder = IncidentEpisodeBuilder()
    evaluator = EpisodeEvaluator()
    window_indicators = _prewindowed_frame(assignments_frame)
    episodes = builder.build(window_indicators, window_indicators)

    matches = evaluator.match(known_incidents_frame, episodes)
    summary = evaluator.summarize(known_incidents_frame, episodes)

    assert matches.loc[0, "temporal_overlap_seconds"] > 0.0
    assert summary.loc[0, "episode_precision"] == 1.0
    assert summary.loc[0, "episode_recall"] == 1.0




def test_segmenter_tolerates_brief_gap_before_recovery() -> None:
    from iabm_incidents.config import EpisodeDetectionConfig
    from iabm_incidents.segmentation import EpisodeSegmenter

    frame = pd.DataFrame(
        {
            "start_time": pd.to_datetime([
                "2022-01-01 00:00:00",
                "2022-01-01 01:00:00",
                "2022-01-01 02:00:00",
                "2022-01-01 03:00:00",
                "2022-01-01 04:00:00",
            ]),
            "end_time": pd.to_datetime([
                "2022-01-01 00:59:59",
                "2022-01-01 01:59:59",
                "2022-01-01 02:59:59",
                "2022-01-01 03:59:59",
                "2022-01-01 04:59:59",
            ]),
            "deviation_score": [1.4, 1.3, 0.9, 0.3, 0.2],
        }
    )
    segmenter = EpisodeSegmenter()
    config = EpisodeDetectionConfig(
        onset_threshold=1.0,
        recovery_threshold=0.5,
        onset_windows=2,
        recovery_windows=2,
        minimum_duration=pd.Timedelta(minutes=10),
        maximum_gap=pd.Timedelta(hours=2),
    )

    segments = segmenter.segment(frame, config)

    assert len(segments) == 1
    assert segments[0].start_index == 0
    assert segments[0].peak_index == 0
    assert segments[0].end_index == 1
    assert segments[0].recovery_start_index == 3
    assert segments[0].recovery_end_index == 4

def test_build_episode_features_for_classification(assignments_frame) -> None:
    from iabm_incidents.features import build_episode_features

    builder = IncidentEpisodeBuilder()
    window_indicators = _prewindowed_frame(assignments_frame)
    window_scores = builder.build_window_scores(window_indicators, window_indicators)
    episodes = builder.build(window_indicators, window_indicators)
    features = build_episode_features(episodes, window_scores)

    assert len(features) == len(episodes)
    assert "max_sequence_divergence" in features.columns
    assert features["dominant_family"].notna().all()


def test_nominal_baseline_is_computed_from_nominal_rows(assignments_frame) -> None:
    from iabm_incidents.baseline import fit_nominal_baseline

    baseline = fit_nominal_baseline(assignments_frame)

    assert baseline.medians()["sequence_divergence"] >= 0.0

def test_registry_normalization_and_matching_helpers(known_incidents_frame) -> None:
    registry = IncidentRegistry()
    normalized = registry.normalize(known_incidents_frame)
    matches = registry.match_candidates(
        known_incidents_frame,
        pd.DataFrame({"candidate_id": ["cand-1"], "event_start": ["2022-01-01 00:12:30"]}),
    )

    assert "source_window_start" in normalized.columns
    assert len(registry.confirmed(known_incidents_frame)) == 1
    assert len(registry.weakly_labelled(known_incidents_frame)) == 1
    assert matches["matched"].any()



def test_signature_layer_contributes_evidence_for_known_family() -> None:
    classifier = RuleBasedFamilyClassifier()
    features = pd.Series(
        {
            "onset_slope": 0.3,
            "min_consumption_deviation": -0.7,
            "max_state_error_rate": 0.25,
            "max_sequence_divergence": 0.8,
            "peak_score": 1.8,
        }
    )
    assignment = classifier.assign(pd.DataFrame(), FamilyAssignmentConfig(minimum_confidence=0.3), episode_features=features)

    assert assignment.primary_family == "pump_abrupt_failure"
    assert any("required feature:" in item or "preferred feature:" in item for item in assignment.evidence)

def test_taxonomy_exposes_signatures() -> None:
    assert is_known_family("pump_abrupt_failure")
    assert FAMILY_SIGNATURES["pump_abrupt_failure"].minimum_assignment_score == 0.70
