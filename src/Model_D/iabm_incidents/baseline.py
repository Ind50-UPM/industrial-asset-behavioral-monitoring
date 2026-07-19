"""Nominal baseline helpers for Model_D indicator workflows."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class NominalBaseline:
    """Compact nominal reference statistics for indicator centering."""

    sequence_divergence: pd.Series
    duration_drift: pd.Series
    recurrence_excess: pd.Series
    persistence_excess: pd.Series
    consumption_deviation: pd.Series
    state_error_rate: pd.Series
    mode_divergence: pd.Series

    def medians(self) -> dict[str, float]:
        return {
            "sequence_divergence": float(self.sequence_divergence.get("median", 0.0)),
            "duration_drift": float(self.duration_drift.get("median", 0.0)),
            "recurrence_excess": float(self.recurrence_excess.get("median", 0.0)),
            "persistence_excess": float(self.persistence_excess.get("median", 0.0)),
            "consumption_deviation": float(self.consumption_deviation.get("median", 0.0)),
            "state_error_rate": float(self.state_error_rate.get("median", 0.0)),
            "mode_divergence": float(self.mode_divergence.get("median", 0.0)),
        }


def fit_nominal_baseline(frame: pd.DataFrame) -> NominalBaseline:
    """Fit a simple robust baseline from nominal or lowest-deviation rows."""

    reference = frame.copy()
    if "semantic_status" in reference.columns:
        nominal = reference[reference["semantic_status"].ne("ANOMALOUS")]
        if not nominal.empty:
            reference = nominal
    elif "deviation_score" in reference.columns and not reference.empty:
        threshold = pd.to_numeric(reference["deviation_score"], errors="coerce").quantile(0.25)
        subset = reference[pd.to_numeric(reference["deviation_score"], errors="coerce") <= threshold]
        if not subset.empty:
            reference = subset

    return NominalBaseline(
        sequence_divergence=_summarize(reference, "sequence_divergence"),
        duration_drift=_summarize(reference, "duration_drift"),
        recurrence_excess=_summarize(reference, "recurrence_excess"),
        persistence_excess=_summarize(reference, "persistence_excess"),
        consumption_deviation=_summarize(reference, "consumption_deviation"),
        state_error_rate=_summarize(reference, "state_error_rate"),
        mode_divergence=_summarize(reference, "mode_divergence"),
    )


def _summarize(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series({"median": 0.0, "mad": 0.0})
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return pd.Series({"median": 0.0, "mad": 0.0})
    median = float(values.median())
    mad = float((values - median).abs().median())
    return pd.Series({"median": median, "mad": mad})
