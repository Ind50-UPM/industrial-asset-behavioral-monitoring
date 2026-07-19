"""Public package surface for Model_D incident and episode analysis.

The package exports the typed configuration objects, builders, evaluators,
pipeline orchestrators, and registry helpers required to assemble longitudinal
incident-processing workflows from either semantic assignments or richer
indicator-driven inputs.
"""

from .config import ModelDConfig
from .episodes import CandidateSegment, IncidentEpisodeBuilder
from .experimental import ExperimentResult, ModelDExperimentRunner
from .evaluation import EpisodeEvaluator
from .exposure import ExposureConfig, derive_observation_periods
from .indicators import IndicatorBaseline, compute_window_indicators, fit_indicator_baseline
from .occurrence import OccurrenceModeler
from .pipeline import IndicatorPipeline, IndicatorPipelineResult
from .registry import IncidentRecord, IncidentRegistry
from .registry_builder import CandidateIncidentRegistryBuilder, RegistryGenerationConfig
from .scoring import DeviationScorer, WeightedDeviationScorer
from .taxonomy import DEFAULT_INCIDENT_TAXONOMY, FAMILY_SIGNATURES, FamilySignature
from .windows import WindowBuildConfig, build_rolling_windows

__all__ = [
    "CandidateSegment",
    "compute_window_indicators",
    "DEFAULT_INCIDENT_TAXONOMY",
    "derive_observation_periods",
    "ExperimentResult",
    "DeviationScorer",
    "EpisodeEvaluator",
    "ExposureConfig",
    "FAMILY_SIGNATURES",
    "FamilySignature",
    "fit_indicator_baseline",
    "IndicatorBaseline",
    "ModelDExperimentRunner",
    "IndicatorPipeline",
    "IndicatorPipelineResult",
    "CandidateIncidentRegistryBuilder",
    "IncidentEpisodeBuilder",
    "IncidentRecord",
    "RegistryGenerationConfig",
    "IncidentRegistry",
    "ModelDConfig",
    "OccurrenceModeler",
    "build_rolling_windows",
    "WeightedDeviationScorer",
    "WindowBuildConfig",
]
