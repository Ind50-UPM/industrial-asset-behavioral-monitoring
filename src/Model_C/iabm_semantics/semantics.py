"""Semantic interpretation utilities for Model_C.

The module bridges behavioral outputs from ``Model_B`` and the longitudinal
incident-processing logic in ``Model_D``. It enriches raw state words with
component semantics, operating modes, incident-family proxies, reliability
interpretations, and life/recovery regimes.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_COMPONENT_MAP = {
    1: "DIVING_PUMP_1",
    2: "DIVING_PUMP_2",
    4: "FEEDBACK_PUMP_1",
    8: "FEEDBACK_PUMP_2",
    16: "FLOCCULANT_PUMP",
    32: "BASIN_PUMP",
}


@dataclass(frozen=True)
class SemanticRule:
    """Declarative mapping from component requirements to semantic modes."""

    required_components: tuple[str, ...]
    operating_mode: str
    working_mode: str


@dataclass(frozen=True)
class SemanticAssignment:
    """Row-level semantic interpretation emitted for one behavioral sequence."""

    sequence_states: tuple[int, ...]
    components: tuple[str, ...]
    operating_mode: str
    working_mode: str
    semantic_status: str
    anomaly_score: float | None
    incident_family: str
    reliability_interpretation: str
    life_regime: str
    recovery_regime: str


class SemanticModeInterpreter:
    """Interpret Model_B sequences as semantic and reliability-oriented labels.

    Args:
        component_map: Optional override for the bitmask-to-component mapping.
        rules: Optional declarative semantic rules evaluated before heuristics.

    Notes:
        The interpreter first applies explicit rules, then falls back to compact
        heuristic logic. This makes it easy to start from transparent defaults
        while still supporting future domain-specific rule injections.
    """

    def __init__(self, component_map: dict[int, str] | None = None, rules: Iterable[SemanticRule] | None = None) -> None:
        self.component_map = dict(component_map or DEFAULT_COMPONENT_MAP)
        self.rules = list(rules or [])

    def load_active_sequences(self, file_path: str | Path) -> pd.DataFrame:
        """Load a Model_B sequence report from parquet, Excel, or CSV."""
        path = Path(file_path)
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        raise ValueError(f"Unsupported file extension: {path.suffix}")

    def load_comparison_report(self, file_path: str | Path) -> pd.DataFrame:
        """Load a comparison report using the same tabular backends as sequences."""
        return self.load_active_sequences(file_path)

    def load_rules(self, file_path: str | Path) -> list[SemanticRule]:
        """Load declarative semantic rules from JSON."""
        payload = json.loads(Path(file_path).read_text(encoding="utf-8"))
        self.rules = [
            SemanticRule(
                required_components=tuple(item["required_components"]),
                operating_mode=item["operating_mode"],
                working_mode=item["working_mode"],
            )
            for item in payload
        ]
        return self.rules

    def interpret_sequences(self, sequences: pd.DataFrame, *, comparison: pd.DataFrame | None = None) -> pd.DataFrame:
        """Interpret every behavioral sequence row into semantic and family labels.

        Args:
            sequences: Model_B active-sequence dataframe.
            comparison: Optional anomaly comparison dataframe aligned row-wise.

        Returns:
            One row per input sequence with semantic, reliability, and family
            fields required by downstream incident processing.

        Notes:
            The current implementation aligns comparison rows by positional
            order. That contract is acceptable because the repository pipeline
            generates both tables from the same ordered sequence export.
        """
        comparison = comparison.reset_index(drop=True) if comparison is not None else None
        assignments = []
        for index, row in sequences.reset_index(drop=True).iterrows():
            states = self._parse_states(row["states"])
            components = self._decode_sequence_components(states)
            operating_mode, working_mode = self._assign_modes(components)

            anomaly_score = None
            semantic_status = "NORMAL"
            if comparison is not None and index < len(comparison):
                anomaly_score = float(comparison.loc[index, "anomaly_score"])
                semantic_status = "ANOMALOUS" if bool(comparison.loc[index, "is_anomalous"]) else "NORMAL"

            incident_family = self._assign_incident_family(
                components,
                anomaly_score=anomaly_score,
                semantic_status=semantic_status,
            )
            assignments.append(
                SemanticAssignment(
                    sequence_states=states,
                    components=components,
                    operating_mode=operating_mode,
                    working_mode=working_mode,
                    semantic_status=semantic_status,
                    anomaly_score=anomaly_score,
                    incident_family=incident_family,
                    reliability_interpretation=self._assign_reliability_interpretation(incident_family, semantic_status=semantic_status),
                    life_regime=self._assign_life_regime(incident_family, components, semantic_status=semantic_status),
                    recovery_regime=self._assign_recovery_regime(incident_family, semantic_status=semantic_status),
                ).__dict__
            )

        return pd.DataFrame(assignments)

    def summarize_modes(self, assignments: pd.DataFrame) -> pd.DataFrame:
        """Aggregate semantic assignments by mode, status, family, and life regime."""
        if assignments.empty:
            return pd.DataFrame(
                columns=["operating_mode", "working_mode", "semantic_status", "incident_family", "life_regime", "count"]
            )
        return (
            assignments.groupby(
                ["operating_mode", "working_mode", "semantic_status", "incident_family", "life_regime"],
                dropna=False,
            )
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )

    def summarize_life_regimes(self, assignments: pd.DataFrame) -> pd.DataFrame:
        """Aggregate assignments by family, life regime, and recovery regime."""
        if assignments.empty:
            return pd.DataFrame(
                columns=["incident_family", "life_regime", "recovery_regime", "reliability_interpretation", "count"]
            )
        return (
            assignments.groupby(
                ["incident_family", "life_regime", "recovery_regime", "reliability_interpretation"],
                dropna=False,
            )
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )

    def _decode_sequence_components(self, states: tuple[int, ...]) -> tuple[str, ...]:
        """Decode a state word into the set of participating industrial components."""
        components: set[str] = set()
        for state in states:
            for bitmask, component in self.component_map.items():
                if state & bitmask:
                    components.add(component)
        return tuple(sorted(components))

    def _assign_modes(self, components: tuple[str, ...]) -> tuple[str, str]:
        """Assign operating and working modes from the decoded component set."""
        component_set = set(components)
        for rule in self.rules:
            if set(rule.required_components).issubset(component_set):
                return rule.operating_mode, rule.working_mode

        if not components:
            return "IDLE", "NO_ACTIVE_COMPONENTS"
        if "BASIN_PUMP" in component_set:
            return "TRANSFER_MODE", "BASIN_TRANSFER"
        if "FLOCCULANT_PUMP" in component_set and component_set & {"DIVING_PUMP_1", "DIVING_PUMP_2"}:
            return "TREATMENT_MODE", "FLOCCULANT_ASSISTED_CYCLE"
        if component_set & {"FEEDBACK_PUMP_1", "FEEDBACK_PUMP_2"} and component_set & {"DIVING_PUMP_1", "DIVING_PUMP_2"}:
            return "RECIRCULATION_MODE", "DIVING_FEEDBACK_CYCLE"
        if component_set <= {"DIVING_PUMP_1", "DIVING_PUMP_2"}:
            return "PUMPING_MODE", "DIVING_ONLY"
        if component_set <= {"FEEDBACK_PUMP_1", "FEEDBACK_PUMP_2"}:
            return "RECIRCULATION_MODE", "FEEDBACK_ONLY"
        return "COMPOSITE_MODE", "+".join(components)

    @staticmethod
    def _assign_incident_family(
        components: tuple[str, ...],
        *,
        anomaly_score: float | None,
        semantic_status: str,
    ) -> str:
        """Map semantic status and component mix into a coarse incident-family proxy."""
        component_set = set(components)
        score = anomaly_score or 0.0
        if semantic_status == "NORMAL":
            return "post_intervention_recovery"
        if "FLOCCULANT_PUMP" in component_set and score >= 1.0:
            return "process_saturation"
        if component_set & {"FEEDBACK_PUMP_1", "FEEDBACK_PUMP_2"} and score >= 1.0:
            return "float_recurrent_disturbance"
        if component_set & {"DIVING_PUMP_1", "DIVING_PUMP_2"} and score >= 1.5:
            return "pump_abrupt_failure"
        return "external_ambiguous_disturbance"

    @staticmethod
    def _assign_reliability_interpretation(incident_family: str, *, semantic_status: str) -> str:
        """Translate family-level semantics into a reliability interpretation label."""
        if semantic_status == "NORMAL":
            return "nominal_or_stable_regime"
        mapping = {
            "pump_abrupt_failure": "high_severity_failure_proxy",
            "float_recurrent_disturbance": "recurrent_control_disturbance_proxy",
            "process_saturation": "load_or_capacity_stress_proxy",
            "post_intervention_recovery": "recovery_stabilization_proxy",
            "external_ambiguous_disturbance": "ambiguous_external_disturbance_proxy",
        }
        return mapping.get(incident_family, "undetermined_reliability_proxy")

    @staticmethod
    def _assign_life_regime(incident_family: str, components: tuple[str, ...], *, semantic_status: str) -> str:
        """Assign a high-level life-regime interpretation for one semantic row."""
        if semantic_status == "NORMAL":
            return "nominal_operation"
        if incident_family == "pump_abrupt_failure":
            return "degradation_acceleration"
        if incident_family == "process_saturation":
            return "stress_accumulation"
        if incident_family == "float_recurrent_disturbance":
            return "recurrent_instability"
        if not components:
            return "idle_or_missing_signal"
        return "externally_perturbed_operation"

    @staticmethod
    def _assign_recovery_regime(incident_family: str, *, semantic_status: str) -> str:
        """Assign a coarse recovery regime label for downstream longitudinal use."""
        if semantic_status == "NORMAL":
            return "stable_recovered"
        if incident_family == "post_intervention_recovery":
            return "recovering"
        if incident_family == "pump_abrupt_failure":
            return "needs_intervention"
        return "under_observation"

    @staticmethod
    def _parse_states(value: str | tuple[int, ...] | list[int]) -> tuple[int, ...]:
        """Normalize serialized or in-memory state words into an integer tuple."""
        if isinstance(value, tuple):
            return tuple(int(item) for item in value)
        if isinstance(value, list):
            return tuple(int(item) for item in value)
        parsed = ast.literal_eval(value)
        return tuple(int(item) for item in parsed)
