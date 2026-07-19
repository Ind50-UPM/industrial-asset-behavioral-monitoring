"""Heuristic generation of candidate incident registries from operational history."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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

        candidates = pd.concat([downtime, saturation, recurrence, abrupt], ignore_index=True, sort=False)
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

        support_windows = pd.concat(
            [
                self._build_process_saturation_candidates(sequences)[["source_window_start", "source_window_end", "source_month"]],
                self._build_recurrent_disturbance_candidates(words)[["source_window_start", "source_window_end", "source_month"]],
            ],
            ignore_index=True,
        )
        if support_windows.empty:
            return self._empty_candidate_frame()

        supported_rows = []
        support_delta = pd.Timedelta(hours=self._config.supporting_word_window_hours)
        support_windows = support_windows.sort_values("source_window_start")
        for _, row in raw.iterrows():
            event_time = row["event_time"]
            month_support = support_windows[support_windows["source_month"] == row.get("source_month")] if "source_month" in support_windows.columns else support_windows
            overlaps = month_support[
                (month_support["source_window_start"] <= event_time + support_delta)
                & (month_support["source_window_end"] >= event_time - support_delta)
            ]
            if overlaps.empty:
                continue
            supported_rows.append(row)
        if not supported_rows:
            return self._empty_candidate_frame()
        candidates = pd.DataFrame(supported_rows)
        candidates = self._filter_abrupt_candidates_with_recovery(candidates, frame)
        if candidates.empty:
            return self._empty_candidate_frame()
        candidates = self._cluster_abrupt_candidates(candidates)
        if candidates.empty:
            return self._empty_candidate_frame()
        candidates["source_window_start"] = candidates["event_time"]
        candidates["source_window_end"] = candidates["event_time"]
        candidates["documented_start"] = candidates["event_time"]
        candidates["documented_end"] = candidates["event_time"]
        candidates["incident_family"] = "pump_abrupt_failure"
        candidates["notes"] = candidates.apply(
            lambda row: (
                f"Abrupt power drop {row['relative_power_drop']:.2%} with contextual anomaly support, "
                f"low power {row['total_real_power']:.2f} and rebound {row['recovery_power']:.2f}"
            ),
            axis=1,
        )
        candidates["source_type"] = "derived_state_power_supported"
        candidates["label_strength"] = "weak"
        return self._with_candidate_schema(candidates)

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
        rare = rare.sort_values("start_time")
        window = pd.Timedelta(hours=self._config.recurrence_window_hours)
        rows: list[dict[str, object]] = []
        for word, word_frame in rare.groupby("word"):
            for idx in range(len(word_frame)):
                current_start = word_frame["start_time"].iloc[idx]
                current_end = current_start + window
                burst = word_frame[(word_frame["start_time"] >= current_start) & (word_frame["start_time"] <= current_end)]
                if len(burst) < self._config.minimum_recurrence_count:
                    continue
                rows.append(
                    {
                        "source_window_start": burst["start_time"].iloc[0],
                        "source_window_end": burst["end_time"].iloc[-1],
                        "documented_start": burst["start_time"].iloc[0],
                        "documented_end": burst["end_time"].iloc[-1],
                        "event_time": burst["start_time"].iloc[0],
                        "incident_family": "float_recurrent_disturbance",
                        "source_month": burst["source_month"].iloc[0],
                        "notes": f"Rare word burst '{word}' repeated {len(burst)} times in {self._config.recurrence_window_hours:.0f}h",
                        "source_type": "derived_word_recurrence",
                        "label_strength": "weak",
                    }
                )
                break
        return pd.DataFrame(rows) if rows else self._empty_candidate_frame()

    def _enrich_with_word_evidence(self, candidates: pd.DataFrame, words: pd.DataFrame) -> pd.DataFrame:
        if candidates.empty or words.empty:
            return candidates
        frame = candidates.copy()
        word_support_window = pd.Timedelta(hours=self._config.supporting_word_window_hours)
        word_frequency = words.groupby(["source_month", "word"]).size().rename("word_count").reset_index()
        words_with_frequency = words.merge(word_frequency, on=["source_month", "word"], how="left")
        words_by_month = {month: month_frame.sort_values("start_time") for month, month_frame in words_with_frequency.groupby("source_month")}
        enriched_notes = []
        for _, row in frame.iterrows():
            start = pd.to_datetime(row["source_window_start"])
            end = pd.to_datetime(row["source_window_end"])
            month_words = words_by_month.get(row.get("source_month"))
            if month_words is None:
                enriched_notes.append(row.get("notes"))
                continue
            nearby_words = month_words[
                (month_words["start_time"] <= end + word_support_window)
                & (month_words["end_time"] >= start - word_support_window)
            ]
            if nearby_words.empty:
                enriched_notes.append(row.get("notes"))
                continue
            top_words = nearby_words.sort_values(["word_count", "duration_seconds"]).head(3)
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
        recovery_window = pd.Timedelta(seconds=self._config.abrupt_recovery_window_seconds)
        rows: list[dict[str, object]] = []
        for _, row in candidates.iterrows():
            event_time = pd.to_datetime(row["event_time"])
            month_states = states[states["source_month"] == row["source_month"]]
            recovery_slice = month_states[
                (month_states["event_time"] > event_time)
                & (month_states["event_time"] <= event_time + recovery_window)
            ]
            if recovery_slice.empty:
                continue
            recovery_power = float(recovery_slice["total_real_power"].max())
            previous_power = float(row["total_real_power"] / max(1.0 + row["relative_power_drop"], 1e-6))
            if recovery_power < previous_power * rebound_fraction:
                continue
            row_dict = row.to_dict()
            row_dict["recovery_power"] = recovery_power
            rows.append(row_dict)
        return pd.DataFrame(rows)

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
