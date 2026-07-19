"""Indicator-driven orchestration for Model_D.

This module contains the first fully materialized implementation of the
indicator-based route proposed for ``Model_D``. The pipeline starts from
state sequences and semantic assignments, derives observation exposure,
constructs rolling windows, computes deviation indicators, scores each window,
segments anomalous periods into candidate episodes, and enriches those episodes
with aggregate descriptive features.

The implementation intentionally remains lightweight and dataframe-oriented so
that it can be executed in batch mode over historical campaigns without
introducing external workflow dependencies. Every stage returns explicit
tables, which makes the module convenient for exploratory notebooks, CLI batch
processing, and Sphinx-generated API documentation.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import ModelDConfig
from .episodes import IncidentEpisodeBuilder
from .exposure import ExposureConfig, derive_observation_periods
from .baseline import fit_nominal_baseline
from .features import build_episode_features
from .indicators import compute_window_indicators
from .scoring import WeightedDeviationScorer
from .windows import WindowBuildConfig, build_rolling_windows


@dataclass(frozen=True)
class IndicatorPipelineResult:
    """Materialized outputs of the indicator-driven pipeline.

    Attributes:
        observation_periods: Exposure-aware availability table describing the
            intervals where the asset is considered observed and therefore
            eligible for baseline estimation and episode detection.
        windows: Rolling analysis windows built from the observation periods.
        window_scores: Window-level indicator table with raw features and the
            final aggregate deviation score.
        episodes: Incident episodes segmented from scored windows and enriched
            with aggregate episode-level descriptors when available.
    """

    observation_periods: pd.DataFrame
    windows: pd.DataFrame
    window_scores: pd.DataFrame
    episodes: pd.DataFrame


class IndicatorPipeline:
    """Build indicator-driven episode outputs from source tables.

    Args:
        config: Full ``Model_D`` configuration. When omitted, the default
            in-package configuration is used.

    Notes:
        The class is intentionally state-light. It only keeps the resolved
        configuration plus the reusable scorer and episode builder so that
        callers can instantiate it once and execute it for many monthly or
        asset-specific batches.
    """

    def __init__(self, config: ModelDConfig | None = None) -> None:
        self._config = config or ModelDConfig()
        self._builder = IncidentEpisodeBuilder(self._config)
        self._scorer = WeightedDeviationScorer(self._config.weights)

    def run(
        self,
        *,
        sequences: pd.DataFrame,
        assignments: pd.DataFrame,
        registry: pd.DataFrame | None = None,
        analogue: pd.DataFrame | None = None,
        digital: pd.DataFrame | None = None,
    ) -> IndicatorPipelineResult:
        """Execute the end-to-end indicator pipeline.

        Args:
            sequences: Sequence-level table, typically produced by ``Model_B``,
                containing temporal state information.
            assignments: Semantic assignment table, typically produced by
                ``Model_C``, aligned with the sequence rows or the derived
                analysis windows.
            registry: Optional canonical incident registry used to mark exposure
                periods that should be excluded from nominal baseline fitting.
            analogue: Optional analog telemetry table. When omitted, the method
                falls back to the ``sequences`` dataframe as the observable time
                base.
            digital: Optional digital telemetry table. When omitted, the method
                falls back to the ``sequences`` dataframe as the observable time
                base.

        Returns:
            A fully materialized :class:`IndicatorPipelineResult`.
        """

        analogue_frame = analogue if analogue is not None else sequences
        digital_frame = digital if digital is not None else sequences
        observation_periods = derive_observation_periods(
            analogue_frame,
            digital_frame,
            states=sequences,
            registry=registry,
            config=ExposureConfig(gap_tolerance=self._config.detection.maximum_gap),
        )
        windows = build_rolling_windows(
            observation_periods,
            config=WindowBuildConfig(
                length=self._config.window.length,
                step=self._config.window.step,
                min_coverage=self._config.window.min_coverage,
            ),
        )

        source_rows = self._merge_sources(sequences, assignments)
        baseline = fit_nominal_baseline(source_rows)
        window_scores = compute_window_indicators(
            source_rows,
            windows,
            None if baseline is None else type('_Baseline', (), {'medians': baseline.medians()})(),
        )
        if window_scores.empty:
            return IndicatorPipelineResult(observation_periods, windows, window_scores, pd.DataFrame())
        normal_mask = window_scores.get('semantic_status', pd.Series(dtype=str)).ne('ANOMALOUS')
        self._scorer.fit(window_scores[normal_mask])
        window_scores = window_scores.copy()
        window_scores['deviation_score'] = self._scorer.score(window_scores)
        episodes = self._builder.build_episodes_from_window_scores(window_scores)
        episode_features = build_episode_features(episodes, window_scores)
        if not episode_features.empty and not episodes.empty:
            episodes = episodes.merge(episode_features, on=['episode_id', 'asset_id'], how='left')
        return IndicatorPipelineResult(observation_periods, windows, window_scores, episodes)

    @staticmethod
    def _merge_sources(sequences: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
        """Create a unified row-wise source table from sequences and assignments."""

        sequence_frame = sequences.copy().reset_index(drop=True)
        assignment_frame = assignments.copy().reset_index(drop=True)

        # Backfill compatibility aliases on the assignment side before merging.
        if 'window_start' in assignment_frame.columns and 'start_time' not in assignment_frame.columns:
            assignment_frame['start_time'] = assignment_frame['window_start']
        if 'window_end' in assignment_frame.columns and 'end_time' not in assignment_frame.columns:
            assignment_frame['end_time'] = assignment_frame['window_end']
        if 'start_time' not in assignment_frame.columns and 'start_time' in sequence_frame.columns:
            assignment_frame['start_time'] = sequence_frame['start_time']
        if 'end_time' not in assignment_frame.columns and 'end_time' in sequence_frame.columns:
            assignment_frame['end_time'] = sequence_frame['end_time']
        if 'asset_id' not in assignment_frame.columns and 'asset_id' in sequence_frame.columns:
            assignment_frame['asset_id'] = sequence_frame['asset_id']

        # Avoid duplicate column labels, which later break pandas datetime assembly.
        overlapping = [column for column in assignment_frame.columns if column in sequence_frame.columns]
        protected = {'semantic_status', 'incident_family', 'anomaly_score'}
        drop_from_assignment = [column for column in overlapping if column not in protected]
        if drop_from_assignment:
            assignment_frame = assignment_frame.drop(columns=drop_from_assignment)

        return pd.concat([sequence_frame, assignment_frame], axis=1)
