"""Heuristic generation of candidate incident registries from operational history."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

import pandas as pd


@dataclass(frozen=True)
class RegistryGenerationConfig:
    """Thresholds controlling heuristic incident candidate generation."""

    downtime_gap_seconds: float = 600.0
    low_power_quantile: float = 0.01
    abrupt_drop_fraction: float = 0.50
    saturation_span_quantile: float = 0.995
    rare_word_frequency_quantile: float = 0.10
    recurrence_window_hours: float = 24.0
    minimum_recurrence_count: int = 3
    candidate_merge_gap_seconds: float = 1800.0
    supporting_word_window_hours: float = 6.0
    downtime_operational_stop_seconds: float = 3600.0
    abrupt_cluster_gap_seconds: float = 1800.0
    abrupt_recovery_window_seconds: float = 1800.0
    abrupt_recovery_fraction: float = 0.75
    abrupt_local_dominant_word_fraction: float = 0.60
    abrupt_local_transition_rate: float = 0.40
    abrupt_nominal_match_floor: float = 0.50
    abrupt_persistent_nominal_match_floor: float = 0.75
    abrupt_degraded_nominal_match_ceiling: float = 0.35
    abrupt_recurrence_count_floor: int = 3
    abrupt_specific_drop_fraction: float = 0.85
    abrupt_specific_rebound_fraction: float = 0.85
    abrupt_specific_transition_rate: float = 0.55
    abrupt_specific_diversity_floor: int = 3
    abrupt_specific_evidence_min_score: int = 2
    abrupt_persistence_penalty_max_score: int = 0
    competing_family_penalty_threshold: int = 2


class CandidateIncidentRegistryBuilder:
    """Build a weakly labelled incident registry from states, sequences, and words."""

    def __init__(self, config: RegistryGenerationConfig | None = None) -> None:
        self._config = config or RegistryGenerationConfig()

    def build_from_files(
        self,
        *,
        states_path: str | Path,
        sequences_path: str | Path,
        words_path: str | Path,
    ) -> pd.DataFrame:
        states = pd.read_parquet(states_path, columns=["source_month", "date", "RP1", "RP2", "RP3", "RP4", "estado", "pred_estado"])
        sequences = pd.read_parquet(sequences_path)
        words = pd.read_parquet(words_path)
        return self.build_from_frames(states=states, sequences=sequences, words=words)

    def build_from_frames(
        self,
        *,
        states: pd.DataFrame,
        sequences: pd.DataFrame,
        words: pd.DataFrame,
    ) -> pd.DataFrame:
        normalized_states = self._normalize_states(states)
        normalized_sequences = self._normalize_sequences(sequences)
        normalized_words = self._normalize_words(words)

        downtime = self._build_downtime_candidates(normalized_states, normalized_sequences)
        saturation = self._build_process_saturation_candidates(normalized_sequences)
        recurrence = self._build_recurrent_disturbance_candidates(normalized_words)
        abrupt = self._build_abrupt_failure_candidates(normalized_states, normalized_sequences, normalized_words)
        saturation = self._filter_saturation_overlaps_with_redirects(saturation, abrupt)

        candidate_parts = [frame for frame in [downtime, saturation, recurrence, abrupt] if not frame.empty]
        candidates = pd.concat(candidate_parts, ignore_index=True, sort=False) if candidate_parts else self._empty_candidate_frame()
        if candidates.empty:
            return self._empty_final_registry()

        candidates = self._enrich_with_word_evidence(candidates, normalized_words)
        candidates = self._merge_nearby_candidates(candidates)
        candidates = candidates.sort_values("event_time").reset_index(drop=True)
        candidates["incident_id"] = [f"AUTO-{index + 1:06d}" for index in range(len(candidates))]
        candidates["event_time_precision"] = candidates.get("event_time_precision", "estimated")
        candidates["label_strength"] = candidates.get("label_strength", "weak")
        candidates["source_type"] = candidates.get("source_type", "derived_candidate")
        candidates["secondary_family"] = candidates.get("secondary_family", None)
        candidates["recovery_status"] = candidates.get("recovery_status", None)
        candidates["affected_subsystem"] = candidates.get("affected_subsystem", "industrial_asset")
        candidates["asset_id"] = candidates.get("asset_id", "asset-unknown")
        candidates["family"] = candidates.get("incident_family")
        for column in ["recovery_time", "maintenance_time", "downtime_start", "downtime_end"]:
            if column not in candidates.columns:
                candidates[column] = None
        return candidates[
            [
                "incident_id",
                "source_window_start",
                "source_window_end",
                "documented_start",
                "documented_end",
                "event_time",
                "event_time_precision",
                "incident_family",
                "family",
                "secondary_family",
                "recovery_time",
                "recovery_status",
                "label_strength",
                "source_type",
                "downtime_start",
                "downtime_end",
                "maintenance_time",
                "affected_subsystem",
                "notes",
                "asset_id",
            ]
        ]

    def _build_downtime_candidates(self, states: pd.DataFrame, sequences: pd.DataFrame) -> pd.DataFrame:
        if states.empty:
            return self._empty_candidate_frame()
        frame = states.sort_values("event_time").copy()
        frame["gap_seconds"] = frame["event_time"].diff().dt.total_seconds()
        gaps = frame[frame["gap_seconds"].gt(self._config.downtime_gap_seconds)].copy()
        if gaps.empty:
            return self._empty_candidate_frame()
        previous_times = frame["event_time"].shift(1)
        gaps["downtime_start"] = previous_times.loc[gaps.index].values
        gaps["downtime_end"] = gaps["event_time"]
        gaps["event_time"] = gaps["downtime_start"]
        gaps["source_window_start"] = gaps["downtime_start"]
        gaps["source_window_end"] = gaps["downtime_end"]
        gaps["documented_start"] = gaps["downtime_start"]
        gaps["documented_end"] = gaps["downtime_end"]
        gaps["incident_family"] = "external_ambiguous_disturbance"
        gaps["source_type"] = "derived_gap"
        gaps["label_strength"] = "weak"
        gap_classes: list[str] = []
        gap_notes: list[str] = []
        for _, row in gaps.iterrows():
            gap_class = self._classify_gap(
                pd.to_datetime(row["downtime_start"]),
                pd.to_datetime(row["downtime_end"]),
                float(row["gap_seconds"]),
                sequences,
            )
            gap_classes.append(gap_class)
            gap_notes.append(f"Detected {gap_class} gap of {row['gap_seconds']:.1f} seconds")
        gaps["secondary_family"] = gap_classes
        gaps["notes"] = gap_notes
        return self._with_candidate_schema(gaps)

    def _build_abrupt_failure_candidates(
        self,
        states: pd.DataFrame,
        sequences: pd.DataFrame,
        words: pd.DataFrame,
    ) -> pd.DataFrame:
        if states.empty or "total_real_power" not in states.columns:
            return self._empty_candidate_frame()
        frame = states.sort_values("event_time").copy()
        frame["relative_power_drop"] = frame["total_real_power"].pct_change().fillna(0.0)
        frame["state_changed"] = (
            frame.groupby("source_month")["pred_estado"].diff().fillna(0).ne(0)
            if "pred_estado" in frame.columns
            else False
        )
        monthly_threshold = frame.groupby("source_month")["total_real_power"].transform(
            lambda series: series.quantile(self._config.low_power_quantile)
        )
        raw = frame[
            frame["relative_power_drop"].le(-self._config.abrupt_drop_fraction)
            & frame["total_real_power"].le(monthly_threshold)
            & frame["state_changed"]
        ].copy()
        if raw.empty:
            return self._empty_candidate_frame()

        support_parts = []
        saturation_support = self._build_process_saturation_candidates(sequences)
        if not saturation_support.empty:
            support_parts.append(saturation_support[["source_window_start", "source_window_end", "source_month"]])
        recurrence_support = self._build_recurrent_disturbance_candidates(words)
        if not recurrence_support.empty:
            support_parts.append(recurrence_support[["source_window_start", "source_window_end", "source_month"]])
        support_windows = pd.concat(support_parts, ignore_index=True) if support_parts else pd.DataFrame(columns=["source_window_start", "source_window_end", "source_month"])
        if support_windows.empty:
            return self._empty_candidate_frame()

        support_delta = pd.Timedelta(hours=self._config.supporting_word_window_hours)
        monthly_sequence_duration_threshold = sequences.groupby("source_month")["duration_seconds"].quantile(
            self._config.saturation_span_quantile
        ).to_dict() if not sequences.empty else {}
        monthly_word_frequency = words.groupby(["source_month", "word"]).size().rename("word_count").reset_index() if not words.empty else pd.DataFrame(columns=["source_month", "word", "word_count"])
        words_with_frequency = words.merge(monthly_word_frequency, on=["source_month", "word"], how="left") if not words.empty else words.copy()
        words_by_month = {
            month: month_frame.sort_values("start_time").reset_index(drop=True)
            for month, month_frame in words_with_frequency.groupby("source_month", sort=False)
        }
        sequences_by_month = {
            month: month_frame.sort_values("start_time").reset_index(drop=True)
            for month, month_frame in sequences.groupby("source_month", sort=False)
        }
        supported_parts: list[pd.DataFrame] = []
        redirected_parts: list[pd.DataFrame] = []
        grouped_support = {month: month_frame.sort_values("source_window_start") for month, month_frame in support_windows.groupby("source_month", sort=False)}
        for month, month_raw in raw.groupby("source_month", sort=False):
            month_support = grouped_support.get(month)
            if month_support is None or month_support.empty:
                continue
            support_start = month_support["source_window_start"].to_numpy(dtype="datetime64[ns]").astype("int64")
            support_end = month_support["source_window_end"].to_numpy(dtype="datetime64[ns]").astype("int64")
            event_ns = month_raw["event_time"].to_numpy(dtype="datetime64[ns]").astype("int64")
            lower = event_ns - int(support_delta.value)
            upper = event_ns + int(support_delta.value)
            keep_mask = np.zeros(len(month_raw), dtype=bool)
            for idx, (lo, hi) in enumerate(zip(lower, upper)):
                left = np.searchsorted(support_start, hi, side="right")
                if left == 0:
                    continue
                overlap_idx = np.flatnonzero(support_end[:left] >= lo)
                if overlap_idx.size:
                    keep_mask[idx] = True
            if keep_mask.any():
                supported = month_raw.loc[keep_mask].copy()
                kept, redirected = self._separate_abrupt_candidates_by_context(
                    supported,
                    sequences_by_month.get(month),
                    words_by_month.get(month),
                    monthly_sequence_duration_threshold.get(month),
                    support_delta,
                )
                if not kept.empty:
                    supported_parts.append(kept)
                if not redirected.empty:
                    redirected_parts.append(redirected)
        redirected_candidates = pd.concat(redirected_parts, ignore_index=True) if redirected_parts else self._empty_candidate_frame()
        if not supported_parts:
            return redirected_candidates
        candidates = pd.concat(supported_parts, ignore_index=True)
        candidates = self._filter_abrupt_candidates_with_recovery(candidates, frame)
        if candidates.empty:
            return self._empty_candidate_frame()
        candidates = self._filter_abrupt_candidates_by_specificity(
            candidates,
            sequences_by_month,
            words_by_month,
            support_delta,
        )
        if candidates.empty:
            return redirected_candidates
        candidates = self._cluster_abrupt_candidates(candidates)
        if candidates.empty:
            return redirected_candidates
        candidates["source_window_start"] = candidates["event_time"]
        candidates["source_window_end"] = candidates["event_time"]
        candidates["documented_start"] = candidates["event_time"]
        candidates["documented_end"] = candidates["event_time"]
        candidates["incident_family"] = "pump_abrupt_failure"
        candidates["notes"] = candidates.apply(
            lambda row: (
                f"Abrupt power drop {row['relative_power_drop']:.2%} with specific abrupt evidence "
                f"(score={int(row.get('abrupt_specificity_score', 0))}, penalty={int(row.get('abrupt_persistence_penalty', 0))}), "
                f"low power {row['total_real_power']:.2f} and rebound {row['recovery_power']:.2f}"
            ),
            axis=1,
        )
        candidates["source_type"] = "derived_state_power_supported"
        candidates["label_strength"] = "weak"
        abrupt_candidates = self._with_candidate_schema(candidates)
        if redirected_candidates.empty:
            return abrupt_candidates
        return pd.concat([abrupt_candidates, redirected_candidates], ignore_index=True, sort=False)

    def _separate_abrupt_candidates_by_context(
        self,
        candidates: pd.DataFrame,
        month_sequences: pd.DataFrame | None,
        month_words: pd.DataFrame | None,
        sequence_duration_threshold: float | None,
        support_delta: pd.Timedelta,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if candidates.empty:
            return candidates, self._empty_candidate_frame()
        keep_rows: list[pd.Series] = []
        redirected_rows: list[dict[str, object]] = []
        for _, row in candidates.iterrows():
            event_time = pd.to_datetime(row["event_time"])
            sequence_window = self._nearby_sequence_window(month_sequences, event_time, support_delta)
            word_window = self._nearby_word_window(month_words, event_time, support_delta)
            penalty = self._persistent_regime_penalty(sequence_window, word_window, sequence_duration_threshold)
            preferred_family = self._preferred_competing_family(sequence_window, word_window, sequence_duration_threshold, penalty)
            if penalty > self._config.abrupt_persistence_penalty_max_score and preferred_family is not None:
                redirected_rows.append(
                    self._build_competing_candidate(row, sequence_window, preferred_family, penalty)
                )
            else:
                keep_rows.append(row)
        kept = pd.DataFrame(keep_rows) if keep_rows else candidates.iloc[0:0].copy()
        redirected = pd.DataFrame(redirected_rows) if redirected_rows else self._empty_candidate_frame()
        return kept, redirected

    def _filter_abrupt_candidates_by_specificity(
        self,
        candidates: pd.DataFrame,
        sequences_by_month: dict[object, pd.DataFrame],
        words_by_month: dict[object, pd.DataFrame],
        support_delta: pd.Timedelta,
    ) -> pd.DataFrame:
        if candidates.empty:
            return candidates
        keep_mask = np.zeros(len(candidates), dtype=bool)
        evidence_scores = np.zeros(len(candidates), dtype=int)
        penalty_scores = np.zeros(len(candidates), dtype=int)
        for position, (_, row) in enumerate(candidates.iterrows()):
            event_time = pd.to_datetime(row["event_time"])
            month = row.get("source_month")
            sequence_window = self._nearby_sequence_window(sequences_by_month.get(month), event_time, support_delta)
            word_window = self._nearby_word_window(words_by_month.get(month), event_time, support_delta)
            evidence_score = self._abrupt_specificity_score(row, sequence_window, word_window)
            penalty_score = self._persistent_regime_penalty(sequence_window, word_window, None)
            evidence_scores[position] = evidence_score
            penalty_scores[position] = penalty_score
            keep_mask[position] = evidence_score >= self._config.abrupt_specific_evidence_min_score and evidence_score > penalty_score
        kept = candidates.loc[keep_mask].copy()
        if kept.empty:
            return kept
        kept["abrupt_specificity_score"] = evidence_scores[keep_mask]
        kept["abrupt_persistence_penalty"] = penalty_scores[keep_mask]
        return kept

    def _abrupt_specificity_score(
        self,
        candidate_row: pd.Series,
        nearby_sequences: pd.DataFrame,
        nearby_words: pd.DataFrame,
    ) -> int:
        score = 0
        relative_drop = abs(float(candidate_row.get("relative_power_drop", 0.0)))
        if relative_drop >= self._config.abrupt_specific_drop_fraction:
            score += 1

        current_power = float(candidate_row.get("total_real_power", 0.0))
        previous_power = current_power / max(1.0 + float(candidate_row.get("relative_power_drop", 0.0)), 1e-6)
        rebound_power = float(candidate_row.get("recovery_power", 0.0)) if pd.notna(candidate_row.get("recovery_power")) else 0.0
        if previous_power > 0.0 and rebound_power / previous_power >= self._config.abrupt_specific_rebound_fraction:
            score += 1

        word_metrics = self._word_regime_metrics(nearby_words)
        if word_metrics["transition_rate"] >= self._config.abrupt_specific_transition_rate:
            score += 1
        if word_metrics["diversity"] >= self._config.abrupt_specific_diversity_floor:
            score += 1

        if not nearby_sequences.empty and "state_value" in nearby_sequences.columns:
            state_values = pd.to_numeric(nearby_sequences["state_value"], errors="coerce").dropna()
            if state_values.nunique() >= 2:
                score += 1
        return score

    def _preferred_competing_family(
        self,
        nearby_sequences: pd.DataFrame,
        nearby_words: pd.DataFrame,
        sequence_duration_threshold: float | None,
        penalty_score: int,
    ) -> str | None:
        if penalty_score < self._config.competing_family_penalty_threshold:
            return None
        if self._has_long_sequence(nearby_sequences, sequence_duration_threshold) or self._has_state17(nearby_sequences):
            return "process_saturation"
        nominal_match_mean = self._mean_nominal_match(nearby_sequences)
        word_metrics = self._word_regime_metrics(nearby_words)
        if nominal_match_mean <= self._config.abrupt_degraded_nominal_match_ceiling and word_metrics["dominant_count"] >= self._config.abrupt_recurrence_count_floor:
            return "process_saturation"
        if nominal_match_mean >= self._config.abrupt_persistent_nominal_match_floor and word_metrics["dominant_fraction"] >= self._config.abrupt_local_dominant_word_fraction:
            return "process_saturation"
        return None

    def _build_competing_candidate(
        self,
        candidate_row: pd.Series,
        nearby_sequences: pd.DataFrame,
        preferred_family: str,
        penalty_score: int,
    ) -> dict[str, object]:
        if not nearby_sequences.empty:
            source_window_start = nearby_sequences["start_time"].min()
            source_window_end = nearby_sequences["end_time"].max()
            duration_seconds = float(
                pd.to_numeric(nearby_sequences.get("duration_seconds"), errors="coerce").fillna(0.0).max()
            )
            state_value = pd.to_numeric(nearby_sequences.get("state_value"), errors="coerce").dropna()
            dominant_state = int(state_value.mode().iloc[0]) if not state_value.empty else None
        else:
            source_window_start = pd.to_datetime(candidate_row["event_time"])
            source_window_end = pd.to_datetime(candidate_row["event_time"])
            duration_seconds = 0.0
            dominant_state = None
        return {
            "source_window_start": source_window_start,
            "source_window_end": source_window_end,
            "documented_start": source_window_start,
            "documented_end": source_window_end,
            "event_time": pd.to_datetime(candidate_row["event_time"]),
            "incident_family": preferred_family,
            "source_month": candidate_row.get("source_month"),
            "secondary_family": "abrupt_context_redirect",
            "notes": f"Context redirected from abrupt candidate to {preferred_family} (penalty={penalty_score}, state={dominant_state}, span={duration_seconds:.1f}s)",
            "source_type": "derived_abrupt_context_redirect",
            "label_strength": "weak",
        }

    def _persistent_regime_penalty(
        self,
        nearby_sequences: pd.DataFrame,
        nearby_words: pd.DataFrame,
        sequence_duration_threshold: float | None,
    ) -> int:
        if nearby_sequences.empty and nearby_words.empty:
            return 0

        penalty = 0
        word_metrics = self._word_regime_metrics(nearby_words)
        stable_word_regime = (
            word_metrics["dominant_fraction"] >= self._config.abrupt_local_dominant_word_fraction
            and word_metrics["transition_rate"] <= self._config.abrupt_local_transition_rate
        )
        recurrent_word_regime = word_metrics["dominant_count"] >= self._config.abrupt_recurrence_count_floor
        nominal_match_mean = self._mean_nominal_match(nearby_sequences)
        long_sequence = self._has_long_sequence(nearby_sequences, sequence_duration_threshold)
        state17_present = self._has_state17(nearby_sequences)

        if stable_word_regime and recurrent_word_regime:
            penalty += 1
        if stable_word_regime and nominal_match_mean >= self._config.abrupt_persistent_nominal_match_floor:
            penalty += 1
        if stable_word_regime and long_sequence:
            penalty += 1
        if state17_present and stable_word_regime:
            penalty += 1
        if recurrent_word_regime and nominal_match_mean <= self._config.abrupt_degraded_nominal_match_ceiling:
            penalty += 1
        return penalty

    @staticmethod
    def _has_long_sequence(nearby_sequences: pd.DataFrame, sequence_duration_threshold: float | None) -> bool:
        if nearby_sequences.empty or sequence_duration_threshold is None or pd.isna(sequence_duration_threshold):
            return False
        return bool(
            pd.to_numeric(nearby_sequences.get("duration_seconds"), errors="coerce")
            .fillna(0.0)
            .ge(float(sequence_duration_threshold))
            .any()
        )

    @staticmethod
    def _has_state17(nearby_sequences: pd.DataFrame) -> bool:
        if nearby_sequences.empty or "state_value" not in nearby_sequences.columns:
            return False
        state_values = pd.to_numeric(nearby_sequences["state_value"], errors="coerce")
        return bool(state_values.eq(17).any())

    @staticmethod
    def _mean_nominal_match(nearby_sequences: pd.DataFrame) -> float:
        if nearby_sequences.empty or "nominal_match" not in nearby_sequences.columns:
            return 0.5
        nominal_match = pd.to_numeric(nearby_sequences["nominal_match"], errors="coerce").dropna()
        if nominal_match.empty:
            return 0.5
        return float(nominal_match.mean())

    @staticmethod
    def _word_regime_metrics(nearby_words: pd.DataFrame) -> dict[str, float]:
        if nearby_words.empty or "word" not in nearby_words.columns:
            return {
                "dominant_fraction": 0.0,
                "transition_rate": 1.0,
                "diversity": 0.0,
                "dominant_count": 0.0,
            }
        ordered_words = nearby_words.sort_values("start_time")["word"].dropna().astype(str)
        if ordered_words.empty:
            return {
                "dominant_fraction": 0.0,
                "transition_rate": 1.0,
                "diversity": 0.0,
                "dominant_count": 0.0,
            }
        counts = ordered_words.value_counts()
        transitions = int((ordered_words != ordered_words.shift(1)).sum() - 1) if len(ordered_words) > 1 else 0
        return {
            "dominant_fraction": float(counts.iloc[0] / max(int(counts.sum()), 1)),
            "transition_rate": float(transitions / max(len(ordered_words) - 1, 1)),
            "diversity": float(ordered_words.nunique()),
            "dominant_count": float(counts.iloc[0]),
        }

    @staticmethod
    def _nearby_sequence_window(
        month_sequences: pd.DataFrame | None,
        event_time: pd.Timestamp,
        support_delta: pd.Timedelta,
    ) -> pd.DataFrame:
        if month_sequences is None or month_sequences.empty:
            return pd.DataFrame()
        mask = (
            month_sequences["start_time"].le(event_time + support_delta)
            & month_sequences["end_time"].ge(event_time - support_delta)
        )
        return month_sequences.loc[mask].copy()

    @staticmethod
    def _nearby_word_window(
        month_words: pd.DataFrame | None,
        event_time: pd.Timestamp,
        support_delta: pd.Timedelta,
    ) -> pd.DataFrame:
        if month_words is None or month_words.empty:
            return pd.DataFrame()
        mask = (
            month_words["start_time"].le(event_time + support_delta)
            & month_words["end_time"].ge(event_time - support_delta)
        )
        return month_words.loc[mask].copy()

    def _build_process_saturation_candidates(self, sequences: pd.DataFrame) -> pd.DataFrame:
        if sequences.empty:
            return self._empty_candidate_frame()
        frame = sequences.copy()
        monthly_threshold = frame.groupby("source_month")["duration_seconds"].transform(
            lambda series: series.quantile(self._config.saturation_span_quantile)
        )
        candidates = frame[frame["duration_seconds"].ge(monthly_threshold)].copy()
        if candidates.empty:
            return self._empty_candidate_frame()
        candidates["event_time"] = candidates["start_time"]
        candidates["source_window_start"] = candidates["start_time"]
        candidates["source_window_end"] = candidates["end_time"]
        candidates["documented_start"] = candidates["start_time"]
        candidates["documented_end"] = candidates["end_time"]
        candidates["incident_family"] = "process_saturation"
        candidates["source_month"] = candidates.get("source_month")
        candidates["notes"] = candidates.apply(
            lambda row: f"Long sequence span {row['duration_seconds']:.1f}s for value {row.get('state_value')}",
            axis=1,
        )
        candidates["source_type"] = "derived_sequence_duration"
        candidates["label_strength"] = "weak"
        return self._with_candidate_schema(candidates)

    def _filter_saturation_overlaps_with_redirects(
        self,
        saturation: pd.DataFrame,
        abrupt: pd.DataFrame,
    ) -> pd.DataFrame:
        if saturation.empty or abrupt.empty or "secondary_family" not in abrupt.columns:
            return saturation
        redirects = abrupt[abrupt["secondary_family"] == "abrupt_context_redirect"].copy()
        if redirects.empty:
            return saturation
        keep_mask = np.ones(len(saturation), dtype=bool)
        for position, (_, sat_row) in enumerate(saturation.iterrows()):
            sat_month = sat_row.get("source_month")
            sat_start = pd.to_datetime(sat_row.get("source_window_start"))
            sat_end = pd.to_datetime(sat_row.get("source_window_end"))
            month_redirects = redirects[redirects["source_month"] == sat_month]
            if month_redirects.empty:
                continue
            overlap = month_redirects[
                (pd.to_datetime(month_redirects["source_window_start"]) <= sat_end)
                & (pd.to_datetime(month_redirects["source_window_end"]) >= sat_start)
            ]
            if not overlap.empty:
                keep_mask[position] = False
        return saturation.loc[keep_mask].copy()

    def _build_recurrent_disturbance_candidates(self, words: pd.DataFrame) -> pd.DataFrame:
        if words.empty:
            return self._empty_candidate_frame()
        frame = words.copy()
        frequency = frame.groupby(["source_month", "word"]).size().rename("word_count").reset_index()
        frame = frame.merge(frequency, on=["source_month", "word"], how="left")
        monthly_cutoff = frame.groupby("source_month")["word_count"].transform(
            lambda series: series.quantile(self._config.rare_word_frequency_quantile)
        )
        rare = frame[frame["word_count"].le(monthly_cutoff)].copy()
        if rare.empty:
            return self._empty_candidate_frame()
        rare = rare.sort_values(["word", "start_time"])
        window = pd.Timedelta(hours=self._config.recurrence_window_hours)
        window_ns = int(window.value)
        rows: list[dict[str, object]] = []
        for word, word_frame in rare.groupby("word", sort=False):
            starts = word_frame["start_time"].to_numpy(dtype="datetime64[ns]").astype("int64")
            ends = word_frame["end_time"].to_numpy(dtype="datetime64[ns]").astype("int64")
            if len(starts) < self._config.minimum_recurrence_count:
                continue
            right_idx = np.searchsorted(starts, starts + window_ns, side="right")
            counts = right_idx - np.arange(len(starts))
            valid = np.flatnonzero(counts >= self._config.minimum_recurrence_count)
            if valid.size == 0:
                continue
            idx = int(valid[0])
            end_idx = int(right_idx[idx] - 1)
            rows.append(
                {
                    "source_window_start": word_frame["start_time"].iloc[idx],
                    "source_window_end": word_frame["end_time"].iloc[end_idx],
                    "documented_start": word_frame["start_time"].iloc[idx],
                    "documented_end": word_frame["end_time"].iloc[end_idx],
                    "event_time": word_frame["start_time"].iloc[idx],
                    "incident_family": "float_recurrent_disturbance",
                    "source_month": word_frame["source_month"].iloc[idx],
                    "notes": f"Rare word burst '{word}' repeated {int(counts[idx])} times in {self._config.recurrence_window_hours:.0f}h",
                    "source_type": "derived_word_recurrence",
                    "label_strength": "weak",
                }
            )
        return pd.DataFrame(rows) if rows else self._empty_candidate_frame()

    def _enrich_with_word_evidence(self, candidates: pd.DataFrame, words: pd.DataFrame) -> pd.DataFrame:
        if candidates.empty or words.empty:
            return candidates
        frame = candidates.copy()
        word_support_window = pd.Timedelta(hours=self._config.supporting_word_window_hours)
        word_frequency = words.groupby(["source_month", "word"]).size().rename("word_count").reset_index()
        words_with_frequency = words.merge(word_frequency, on=["source_month", "word"], how="left")
        words_by_month = {month: month_frame.sort_values(["start_time", "word_count", "duration_seconds"]) for month, month_frame in words_with_frequency.groupby("source_month")}
        enriched_notes = []
        for _, row in frame.iterrows():
            start = pd.to_datetime(row["source_window_start"])
            end = pd.to_datetime(row["source_window_end"])
            month_words = words_by_month.get(row.get("source_month"))
            if month_words is None or month_words.empty:
                enriched_notes.append(row.get("notes"))
                continue
            mask = (
                month_words["start_time"].le(end + word_support_window)
                & month_words["end_time"].ge(start - word_support_window)
            )
            nearby_words = month_words.loc[mask, ["word", "word_count", "duration_seconds"]]
            if nearby_words.empty:
                enriched_notes.append(row.get("notes"))
                continue
            top_words = nearby_words.nlargest(3, ["word_count", "duration_seconds"])
            word_note = "; supporting words: " + ", ".join(
                f"{word} (count={count})" for word, count in zip(top_words["word"], top_words["word_count"])
            )
            enriched_notes.append(f"{row.get('notes', '')}{word_note}")
        frame["notes"] = enriched_notes
        return frame

    def _merge_nearby_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        if candidates.empty:
            return candidates
        frame = candidates.sort_values(["incident_family", "event_time"]).reset_index(drop=True)
        merge_gap = pd.Timedelta(seconds=self._config.candidate_merge_gap_seconds)
        rows: list[dict[str, object]] = []
        current: dict[str, object] | None = None
        for _, row in frame.iterrows():
            row_dict = row.to_dict()
            row_start = pd.to_datetime(row_dict["source_window_start"])
            row_end = pd.to_datetime(row_dict["source_window_end"])
            if current is None:
                current = row_dict
                continue
            current_end = pd.to_datetime(current["source_window_end"])
            same_family = current["incident_family"] == row_dict["incident_family"]
            same_secondary_family = current.get("secondary_family") == row_dict.get("secondary_family")
            close_in_time = row_start - current_end <= merge_gap
            if same_family and same_secondary_family and close_in_time:
                current["source_window_end"] = max(current_end, row_end)
                current["documented_end"] = max(pd.to_datetime(current["documented_end"]), pd.to_datetime(row_dict["documented_end"]))
                current["merge_count"] = int(current.get("merge_count", 1)) + int(row_dict.get("merge_count", 1))
                current["notes"] = self._merge_notes(current.get("notes"), row_dict.get("notes"), current["merge_count"])
                continue
            rows.append(current)
            current = row_dict
        if current is not None:
            rows.append(current)
        return pd.DataFrame(rows)


    @staticmethod
    def _merge_notes(current_note: object, new_note: object, merge_count: int) -> str:
        current_text = str(current_note or "").strip()
        new_text = str(new_note or "").strip()
        if merge_count <= 2:
            merged = " | merged: ".join(part for part in [current_text, new_text] if part)
            return merged[:1200]
        if not current_text:
            return new_text[:1200]
        if " | merged summary:" in current_text:
            summary_base = current_text.split(" | merged summary:", 1)[0]
        else:
            summary_base = current_text
        if new_text and new_text != summary_base:
            return f"{summary_base} | merged summary: {merge_count} nearby observations"[:1200]
        return f"{summary_base} | merged summary: {merge_count} nearby observations"[:1200]

    @staticmethod
    def _normalize_states(states: pd.DataFrame) -> pd.DataFrame:
        frame = states.copy()
        frame["event_time"] = _normalize_timestamp_series(frame["date"])
        power_columns = [column for column in ["RP1", "RP2", "RP3", "RP4"] if column in frame.columns]
        if power_columns:
            frame["total_real_power"] = frame[power_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
        for column in ["estado", "pred_estado"]:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame

    def _classify_gap(
        self,
        gap_start: pd.Timestamp,
        gap_end: pd.Timestamp,
        gap_seconds: float,
        sequences: pd.DataFrame,
    ) -> str:
        if gap_seconds >= self._config.downtime_operational_stop_seconds:
            return "operational_stop_candidate"
        if sequences.empty or "start_time" not in sequences.columns:
            return "telemetry"
        nearby = sequences[
            (sequences["start_time"] <= gap_end + pd.Timedelta(minutes=5))
            & (sequences["end_time"] >= gap_start - pd.Timedelta(minutes=5))
        ]
        if nearby.empty:
            return "telemetry"
        state_values = pd.to_numeric(nearby.get("state_value"), errors="coerce")
        if state_values.eq(0).any():
            return "operational_stop_candidate"
        return "telemetry"

    def _filter_abrupt_candidates_with_recovery(self, candidates: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
        if candidates.empty:
            return candidates
        rebound_fraction = self._config.abrupt_recovery_fraction
        recovery_window_ns = int(pd.Timedelta(seconds=self._config.abrupt_recovery_window_seconds).value)
        rows: list[pd.DataFrame] = []
        grouped_states = {month: month_frame.sort_values("event_time") for month, month_frame in states.groupby("source_month", sort=False)}
        for month, month_candidates in candidates.groupby("source_month", sort=False):
            month_states = grouped_states.get(month)
            if month_states is None or month_states.empty:
                continue
            state_times = month_states["event_time"].to_numpy(dtype="datetime64[ns]").astype("int64")
            state_power = month_states["total_real_power"].to_numpy(dtype=float)
            event_ns = month_candidates["event_time"].to_numpy(dtype="datetime64[ns]").astype("int64")
            start_idx = np.searchsorted(state_times, event_ns, side="right")
            end_idx = np.searchsorted(state_times, event_ns + recovery_window_ns, side="right")
            recovery_power = np.full(len(month_candidates), np.nan, dtype=float)
            keep_mask = np.zeros(len(month_candidates), dtype=bool)
            previous_power = month_candidates["total_real_power"].to_numpy(dtype=float) / np.maximum(
                1.0 + month_candidates["relative_power_drop"].to_numpy(dtype=float),
                1e-6,
            )
            for idx, (left, right) in enumerate(zip(start_idx, end_idx)):
                if right <= left:
                    continue
                max_power = float(state_power[left:right].max())
                recovery_power[idx] = max_power
                if max_power >= previous_power[idx] * rebound_fraction:
                    keep_mask[idx] = True
            if keep_mask.any():
                kept = month_candidates.loc[keep_mask].copy()
                kept["recovery_power"] = recovery_power[keep_mask]
                rows.append(kept)
        return pd.concat(rows, ignore_index=True) if rows else candidates.iloc[0:0].copy()

    def _cluster_abrupt_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        if candidates.empty:
            return candidates
        frame = candidates.sort_values(["source_month", "event_time"]).reset_index(drop=True)
        cluster_gap = pd.Timedelta(seconds=self._config.abrupt_cluster_gap_seconds)
        rows: list[dict[str, object]] = []
        current: dict[str, object] | None = None
        for _, row in frame.iterrows():
            row_dict = row.to_dict()
            row_time = pd.to_datetime(row_dict["event_time"])
            if current is None:
                current = row_dict
                continue
            current_time = pd.to_datetime(current["event_time"])
            same_month = current["source_month"] == row_dict["source_month"]
            if same_month and row_time - current_time <= cluster_gap:
                if row_dict["relative_power_drop"] < current["relative_power_drop"]:
                    current = row_dict
                continue
            rows.append(current)
            current = row_dict
        if current is not None:
            rows.append(current)
        return pd.DataFrame(rows)

    @staticmethod
    def _normalize_sequences(sequences: pd.DataFrame) -> pd.DataFrame:
        frame = sequences.copy()
        frame["start_time"] = _normalize_timestamp_series(frame["date"])
        frame["duration_seconds"] = pd.to_numeric(frame.get("span"), errors="coerce").fillna(0.0)
        frame["end_time"] = frame["start_time"] + pd.to_timedelta(frame["duration_seconds"], unit="s")
        if "Values" in frame.columns:
            frame = frame.rename(columns={"Values": "state_value", "Runs": "run_count"})
        return frame

    @staticmethod
    def _normalize_words(words: pd.DataFrame) -> pd.DataFrame:
        frame = words.copy()
        frame["start_time"] = _normalize_timestamp_series(frame["date"])
        frame["duration_seconds"] = pd.to_numeric(frame.get("span"), errors="coerce").fillna(0.0)
        frame["end_time"] = frame["start_time"] + pd.to_timedelta(frame["duration_seconds"], unit="s")
        return frame

    @classmethod
    def _with_candidate_schema(cls, frame: pd.DataFrame) -> pd.DataFrame:
        normalized = frame.copy()
        for column in cls._candidate_columns():
            if column not in normalized.columns:
                normalized[column] = None
        return normalized.loc[:, cls._candidate_columns()]

    @staticmethod
    def _candidate_columns() -> list[str]:
        return [
            "source_window_start", "source_window_end", "documented_start", "documented_end", "event_time",
            "incident_family", "source_month", "downtime_start", "downtime_end", "secondary_family", "notes", "source_type", "label_strength",
        ]

    @classmethod
    def _empty_candidate_frame(cls) -> pd.DataFrame:
        return pd.DataFrame(columns=cls._candidate_columns())

    @staticmethod
    def _empty_final_registry() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "incident_id", "source_window_start", "source_window_end", "documented_start", "documented_end",
                "event_time", "event_time_precision", "incident_family", "family", "secondary_family",
                "recovery_time", "recovery_status", "label_strength", "source_type", "downtime_start",
                "downtime_end", "maintenance_time", "affected_subsystem", "notes", "asset_id",
            ]
        )


def _normalize_timestamp_series(series: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(series, utc=True, errors="coerce")
    return timestamps.dt.tz_convert(None)
