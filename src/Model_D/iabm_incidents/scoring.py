"""Scoring helpers for the indicator-driven Model_D route."""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from .config import DeviationWeights
from .detection import compute_deviation_score


class DeviationScorer(Protocol):
    """Protocol for reproducible window-level scoring methods."""

    def fit(self, nominal_indicators: pd.DataFrame) -> None: ...
    def score(self, indicators: pd.DataFrame) -> pd.Series: ...


class WeightedDeviationScorer:
    """Simple deterministic scorer backed by the existing weighted score logic."""

    def __init__(self, weights: DeviationWeights) -> None:
        self._weights = weights

    def fit(self, nominal_indicators: pd.DataFrame) -> None:
        return None

    def score(self, indicators: pd.DataFrame) -> pd.Series:
        return compute_deviation_score(indicators, self._weights)
