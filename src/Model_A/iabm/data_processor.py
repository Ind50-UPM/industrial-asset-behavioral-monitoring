"""Data preparation utilities for Model_A industrial-state classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrainingDataset:
    """Bundle supervised features and labels for classifier training.

    The dataclass keeps the public API explicit and avoids passing loosely
    coupled tuples around the codebase when training workflows evolve.
    """

    features: pd.DataFrame
    labels: pd.Series


@dataclass(frozen=True)
class InferenceDataset:
    """Bundle inference-ready features together with activity bookkeeping.

    The source frame and activity mask allow the CLI to reconstruct outputs
    aligned with the original timestamps, including optional inactive rows.
    """

    features: pd.DataFrame
    active_mask: pd.Series
    source_frame: pd.DataFrame


@dataclass(frozen=True)
class EvaluationDataset:
    """Bundle features, labels, and alignment data for quality assessment."""

    features: pd.DataFrame
    labels: pd.Series | None
    active_mask: pd.Series
    source_frame: pd.DataFrame


class IndustrialDataProcessor:
    """Prepare industrial analog and digital signals for Model_A workflows.

    The processor encapsulates the study-specific preprocessing rules so the
    rest of the package can work with clean training and inference datasets
    through a stable, object-oriented interface.
    """

    DEFAULT_FEATURE_COLUMNS = [
        "Vrms1",
        "Vrms2",
        "Vrms3",
        "Irms1",
        "Irms2",
        "Irms3",
        "PF1",
        "PF2",
        "PF3",
    ]
    POWER_COLUMNS = ["RP1", "RP2", "RP3"]
    THREE_PHASE_BLOCKS = [
        ["Vrms1", "Vrms2", "Vrms3"],
        ["RP1", "RP2", "RP3"],
        ["Irms1", "Irms2", "Irms3"],
        ["PF1", "PF2", "PF3"],
    ]
    SINGLE_PHASE_BLOCKS = [["Vrms4"], ["RP4"], ["Irms4"], ["PF4"]]

    def __init__(
        self,
        analog_path: str,
        digital_path: Optional[str] = None,
        *,
        threshold: float = 50.0,
        feature_columns: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize the processor and eagerly load the configured datasets.

        Args:
            analog_path: Path to the analog Parquet dataset.
            digital_path: Optional path to the digital Parquet dataset used for labels.
            threshold: Minimum active power threshold used to discard inactive rows.
            feature_columns: Feature columns to expose to the classifier.
        """
        self.analog_df = pd.read_parquet(analog_path).sort_index()
        self.digital_df = (
            pd.read_parquet(digital_path).sort_index() if digital_path else None
        )
        self.threshold = threshold
        self.feature_columns = list(feature_columns or self.DEFAULT_FEATURE_COLUMNS)

    def prepare_training_data(self, start: str, end: str) -> TrainingDataset:
        """Return supervised features and labels for the requested time range.

        Args:
            start: Inclusive lower timestamp bound.
            end: Inclusive upper timestamp bound.

        Returns:
            A :class:`TrainingDataset` containing active rows only, with labels
            synchronized from the digital signal stream.
        """
        analog_window = self._get_analog_window(start, end)
        labeled_window = self._attach_labels(analog_window)
        active_window = labeled_window[labeled_window["estado"] != 0].copy()

        return TrainingDataset(
            features=active_window[self.feature_columns],
            labels=active_window["estado"].astype(np.int32),
        )

    def prepare_inference_data(
        self,
        start: str,
        end: str,
        *,
        drop_inactive: bool = True,
    ) -> InferenceDataset:
        """Return inference-ready analog features without requiring digital labels.

        Args:
            start: Inclusive lower timestamp bound.
            end: Inclusive upper timestamp bound.
            drop_inactive: Whether to keep only rows above the activity threshold.

        Returns:
            An :class:`InferenceDataset` with the feature matrix, a boolean mask
            identifying active rows, and the imputed source analog window.
        """
        analog_window = self._get_analog_window(start, end)
        active_mask = self._build_activity_mask(analog_window)
        features = analog_window.loc[active_mask, self.feature_columns].copy()

        if not drop_inactive:
            features = analog_window[self.feature_columns].copy()

        return InferenceDataset(
            features=features,
            active_mask=active_mask,
            source_frame=analog_window,
        )

    def prepare_evaluation_data(self, start: str, end: str) -> EvaluationDataset:
        """Return aligned features and optional labels for model evaluation.

        Args:
            start: Inclusive lower timestamp bound.
            end: Inclusive upper timestamp bound.

        Returns:
            An :class:`EvaluationDataset` containing the active feature matrix,
            optional real labels aligned to the full analog window, the activity
            mask, and the imputed source analog frame.
        """
        inference_dataset = self.prepare_inference_data(start, end)
        labels: pd.Series | None = None
        if self.digital_df is not None:
            labeled_frame = self._attach_labels(inference_dataset.source_frame)
            labels = labeled_frame["estado"].astype(np.int32)

        return EvaluationDataset(
            features=inference_dataset.features,
            labels=labels,
            active_mask=inference_dataset.active_mask,
            source_frame=inference_dataset.source_frame,
        )

    def _get_analog_window(self, start: str, end: str) -> pd.DataFrame:
        """Filter analog signals by period and apply the configured imputations.

        Args:
            start: Inclusive lower timestamp bound.
            end: Inclusive upper timestamp bound.

        Returns:
            An imputed analog window ready for activity filtering or labeling.
        """
        analog_window = self._slice_by_period(self.analog_df, start, end)
        return self._impute_nans(analog_window)

    @staticmethod
    def _slice_by_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        """Return a copy of the rows included in the requested time interval.

        Args:
            df: Time-indexed DataFrame to slice.
            start: Inclusive lower timestamp bound.
            end: Inclusive upper timestamp bound.

        Returns:
            A copy of the requested time window.
        """
        mask = (df.index >= start) & (df.index <= end)
        return df.loc[mask].copy()

    def _build_activity_mask(self, df: pd.DataFrame) -> pd.Series:
        """Mark rows whose maximum active power exceeds the configured threshold.

        Args:
            df: Analog signal window containing the configured power columns.

        Returns:
            A boolean Series where ``True`` identifies active rows.
        """
        return df[self.POWER_COLUMNS].max(axis=1) >= self.threshold

    def _attach_labels(self, analog_df: pd.DataFrame) -> pd.DataFrame:
        """Synchronize analog rows with the nearest previous active digital label.

        The implementation mirrors the original Random Forest labeling logic by
        using the nearest earlier non-zero digital state for active analog rows.
        """
        if self.digital_df is None:
            raise ValueError("Digital data is required to prepare supervised labels.")

        labeled_df = analog_df.copy()
        labeled_df["estado"] = 0
        active_mask = self._build_activity_mask(labeled_df)
        if not active_mask.any():
            return labeled_df

        relevant_digitals = self.digital_df[self.digital_df["estado"] != 0]
        if relevant_digitals.empty:
            raise ValueError("Digital dataset does not contain active state labels.")

        # Label transfer uses the nearest previous non-zero digital event so the
        # analog timeline inherits the operational state known at that instant.
        indexer = relevant_digitals.index.get_indexer(
            labeled_df.index[active_mask], method="pad"
        )
        valid_positions = indexer >= 0
        if valid_positions.any():
            active_index = labeled_df.index[active_mask]
            labeled_df.loc[active_index[valid_positions], "estado"] = (
                relevant_digitals["estado"].iloc[indexer[valid_positions]].to_numpy()
            )

        labeled_df["estado"] = labeled_df["estado"].astype(np.int32)
        return labeled_df

    def _impute_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing analog values with the legacy study-compatible strategy.

        Three-phase variables follow the original heuristic of using the minimum
        available phase value and then propagating neighbouring observations. The
        optional fourth-channel variables are interpolated when present.
        """
        imputed_df = df.copy()
        self._impute_three_phase_blocks(imputed_df)
        self._impute_single_phase_blocks(imputed_df)
        return imputed_df.dropna(subset=self.feature_columns + self.POWER_COLUMNS)

    def _impute_three_phase_blocks(self, df: pd.DataFrame) -> None:
        """Impute three-phase channels via phase minimum, next, and previous values.

        Args:
            df: DataFrame updated in place.
        """
        for block in self.THREE_PHASE_BLOCKS:
            available_block = [column for column in block if column in df.columns]
            if not available_block:
                continue

            block_frame = df[available_block]
            for column in available_block:
                nan_mask = df[column].isna()
                if nan_mask.any():
                    # The legacy heuristic first borrows the minimum valid value
                    # from the remaining phases before applying temporal fills.
                    df.loc[nan_mask, column] = block_frame.loc[nan_mask].min(axis=1)
                df[column] = df[column].bfill().ffill()

    def _impute_single_phase_blocks(self, df: pd.DataFrame) -> None:
        """Impute single-channel variables with linear interpolation when present.

        Args:
            df: DataFrame updated in place.
        """
        for block in self.SINGLE_PHASE_BLOCKS:
            column = block[0]
            if column not in df.columns:
                continue

            # Single auxiliary channels do not have multi-phase redundancy, so
            # interpolation is the least invasive default imputation rule here.
            df[column] = df[column].interpolate(
                method="linear", limit_direction="both"
            )
