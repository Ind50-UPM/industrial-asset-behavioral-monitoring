"""Incident-family taxonomy primitives for Model_D."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FamilySignature:
    """Formal feature-level expectations used by family assigners."""

    family: str
    required_features: tuple[str, ...] = tuple()
    preferred_features: tuple[str, ...] = tuple()
    incompatible_features: tuple[str, ...] = tuple()
    minimum_assignment_score: float = 0.70


@dataclass(frozen=True)
class IncidentFamilyDefinition:
    """Semantic and operational description of one incident family."""

    family: str
    severity: str
    life_regime: str
    recovery_regime: str
    signature: FamilySignature

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["signature"] = asdict(self.signature)
        return payload


DEFAULT_INCIDENT_TAXONOMY = {
    "pump_abrupt_failure": IncidentFamilyDefinition(
        family="pump_abrupt_failure",
        severity="high",
        life_regime="degradation_acceleration",
        recovery_regime="needs_intervention",
        signature=FamilySignature(
            family="pump_abrupt_failure",
            required_features=("abrupt_onset", "negative_consumption_deviation"),
            preferred_features=(
                "digital_electrical_mismatch",
                "high_sequence_divergence",
                "high_state_disagreement",
            ),
            minimum_assignment_score=0.70,
        ),
    ),
    "float_recurrent_disturbance": IncidentFamilyDefinition(
        family="float_recurrent_disturbance",
        severity="medium",
        life_regime="recurrent_instability",
        recovery_regime="under_observation",
        signature=FamilySignature(
            family="float_recurrent_disturbance",
            required_features=("recurrence_excess",),
            preferred_features=("persistence_excess", "sequence_divergence", "state_stability"),
            minimum_assignment_score=0.60,
        ),
    ),
    "process_saturation": IncidentFamilyDefinition(
        family="process_saturation",
        severity="medium",
        life_regime="stress_accumulation",
        recovery_regime="under_observation",
        signature=FamilySignature(
            family="process_saturation",
            required_features=("duration_drift", "persistence_excess"),
            preferred_features=("mode_divergence", "non_negative_consumption_deviation"),
            minimum_assignment_score=0.60,
        ),
    ),
    "post_intervention_recovery": IncidentFamilyDefinition(
        family="post_intervention_recovery",
        severity="low",
        life_regime="nominal_operation",
        recovery_regime="stable_recovered",
        signature=FamilySignature(
            family="post_intervention_recovery",
            preferred_features=("moderate_residual_deviation", "restored_state_agreement"),
            incompatible_features=("abrupt_onset",),
            minimum_assignment_score=0.55,
        ),
    ),
    "external_ambiguous_disturbance": IncidentFamilyDefinition(
        family="external_ambiguous_disturbance",
        severity="unknown",
        life_regime="externally_perturbed_operation",
        recovery_regime="under_observation",
        signature=FamilySignature(
            family="external_ambiguous_disturbance",
            preferred_features=("mixed_family_evidence", "cross_domain_deviation"),
            minimum_assignment_score=0.50,
        ),
    ),
}


FAMILY_SIGNATURES = {
    family: definition.signature for family, definition in DEFAULT_INCIDENT_TAXONOMY.items()
}


def is_known_family(family: str) -> bool:
    """Return whether a family identifier is registered in the taxonomy."""

    return family in DEFAULT_INCIDENT_TAXONOMY
