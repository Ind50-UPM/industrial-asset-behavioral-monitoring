"""Public package surface for Model_B behavioral sequence analysis.

The package intentionally exposes a compact API centered on sequence-oriented
building blocks so downstream layers can depend on explicit typed dataclasses
instead of loosely structured intermediate dictionaries.
"""

from .sequences import (
    ActiveSequence,
    BehavioralSequenceAnalyzer,
    LongitudinalSequenceMetrics,
    NominalSequenceReference,
    RecoveryMetrics,
    SequenceComparison,
    StateRun,
)

__all__ = [
    "ActiveSequence",
    "BehavioralSequenceAnalyzer",
    "LongitudinalSequenceMetrics",
    "NominalSequenceReference",
    "RecoveryMetrics",
    "SequenceComparison",
    "StateRun",
]
