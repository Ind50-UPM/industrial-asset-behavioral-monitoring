"""Model abstractions for Model_A industrial-state classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier


@dataclass(frozen=True)
class CrossValidationResult:
    """Summarize fold-wise validation scores produced by the classifier API.

    Exposing the result as a dataclass makes downstream reporting clearer and
    keeps aggregate statistics close to the original fold-level scores.
    """

    scores: np.ndarray

    @property
    def mean(self) -> float:
        """Return the average score across all folds."""
        return float(self.scores.mean())

    @property
    def std(self) -> float:
        """Return the standard deviation across all folds."""
        return float(self.scores.std())


class FoldLabelEncoderClassifier(BaseEstimator, ClassifierMixin):
    """Wrap an estimator so each fit uses fold-local contiguous class labels.

    XGBoost expects class labels presented during ``fit`` to be contiguous
    integers starting at zero. During cross-validation, some training folds may
    not contain every class present in the global dataset, which makes a
    globally encoded target vector invalid for that fold. This wrapper applies a
    fresh label encoding on every fit and maps predictions back to the original
    labels expected by scikit-learn scorers.
    """

    def __init__(self, estimator: BaseEstimator) -> None:
        """Store the wrapped estimator used for fold-local training.

        Args:
            estimator: Base estimator cloned inside each cross-validation fit.
        """
        self.estimator = estimator

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "FoldLabelEncoderClassifier":
        """Fit the wrapped estimator with a fold-local label encoding.

        Args:
            X: Fold-local feature matrix.
            y: Fold-local label vector.

        Returns:
            The fitted wrapper instance.
        """
        self.label_encoder_ = LabelEncoder()
        encoded_y = self.label_encoder_.fit_transform(np.asarray(y))
        self.estimator_ = clone(self.estimator)
        # XGBoost requires contiguous zero-based labels inside each fold, which
        # is not guaranteed when rare classes disappear from a split.
        self.estimator_ = _configure_xgb_estimator(self.estimator_, encoded_y)
        self.estimator_.fit(X, encoded_y)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict labels and map them back to the original fold label space.

        Args:
            X: Fold-local feature matrix.

        Returns:
            Predictions expressed in the original label space expected by the
            scoring function.
        """
        encoded_predictions = np.asarray(self.estimator_.predict(X))
        if encoded_predictions.ndim > 1:
            encoded_predictions = encoded_predictions.argmax(axis=1)
        return self.label_encoder_.inverse_transform(np.asarray(encoded_predictions))

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Expose wrapped-estimator parameters for scikit-learn compatibility.

        Args:
            deep: Whether to include nested estimator parameters.

        Returns:
            A parameter dictionary compatible with scikit-learn cloning.
        """
        params = {"estimator": self.estimator}
        if deep and hasattr(self.estimator, "get_params"):
            for key, value in self.estimator.get_params(deep=True).items():
                params[f"estimator__{key}"] = value
        return params

    def set_params(self, **params: Any) -> "FoldLabelEncoderClassifier":
        """Propagate parameter updates to the wrapped estimator when requested.

        Args:
            **params: Wrapper or nested estimator parameters.

        Returns:
            The updated wrapper instance.
        """
        estimator = params.pop("estimator", None)
        if estimator is not None:
            self.estimator = estimator
        nested_params = {
            key.split("__", 1)[1]: value
            for key, value in params.items()
            if key.startswith("estimator__")
        }
        if nested_params and hasattr(self.estimator, "set_params"):
            self.estimator.set_params(**nested_params)
        return self


def _configure_xgb_estimator(
    estimator: BaseEstimator,
    labels: np.ndarray,
) -> BaseEstimator:
    """Inject class-count settings required by XGBoost when fitting labels.

    Args:
        estimator: Estimator about to be fitted.
        labels: Encoded label vector used for fitting.

    Returns:
        The original estimator for non-XGBoost models, or a configured clone
        with fold-local class-count metadata for XGBoost.
    """
    if not isinstance(estimator, XGBClassifier):
        return estimator

    configured = clone(estimator)
    unique_classes = np.unique(labels)
    configured.set_params(
        num_class=max(len(unique_classes), 1),
        objective="multi:softprob",
    )
    return configured


class StateClassifier:
    """High-level wrapper around the estimator lifecycle used by Model_A.

    The class keeps scaling, label encoding, validation, persistence, and
    inference in one cohesive object so command-line orchestration stays thin
    and future model variants can share the same interface.
    """

    def __init__(
        self,
        model_type: str = "rf",
        params: Optional[Dict[str, Any]] = None,
        translator: Optional[Callable[[str], str]] = None,
    ) -> None:
        """Initialize the classifier wrapper for the selected algorithm.

        Args:
            model_type: Short identifier of the underlying estimator family.
            params: Optional hyperparameter overrides.
            translator: Optional translation function for user-facing errors.
        """
        self._ = translator or (lambda s: s)
        self.model_type = model_type
        self.params = params or self._get_default_params(model_type)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.estimator = self._initialize_estimator()
        self.feature_columns: list[str] = []

    @staticmethod
    def _get_default_params(model_type: str) -> Dict[str, Any]:
        """Return sensible default hyperparameters for the requested model.

        Args:
            model_type: Short identifier of the underlying estimator family.

        Returns:
            A dictionary of baseline hyperparameters.
        """
        if model_type == "rf":
            return {
                "n_estimators": 100,
                "criterion": "entropy",
                "random_state": 42,
                "n_jobs": 8,
            }
        if model_type == "xgb":
            return {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "random_state": 42,
                "n_jobs": 8,
                "eval_metric": "mlogloss",
            }
        raise ValueError(f"Unsupported model type: {model_type}")

    def _initialize_estimator(self) -> RandomForestClassifier | XGBClassifier:
        """Build the underlying scikit-learn compatible estimator.

        Returns:
            A scikit-learn compatible classifier instance matching
            ``self.model_type``.
        """
        if self.model_type == "rf":
            return RandomForestClassifier(**self.params)
        if self.model_type == "xgb":
            return XGBClassifier(**self.params, objective="multi:softprob")
        raise ValueError(
            self._("Unsupported algorithm: {}.").format(self.model_type)
        )

    def _build_cv_estimator(self) -> BaseEstimator:
        """Return the estimator instance that should be used inside CV folds.

        Returns:
            A cloned estimator, optionally wrapped to handle fold-local label
            encoding requirements.
        """
        if self.model_type == "xgb":
            return FoldLabelEncoderClassifier(self.estimator)
        return clone(self.estimator)

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> float:
        """Fit the scaler and estimator and return the training accuracy.

        Args:
            X: Training feature matrix.
            y: Original training labels.

        Returns:
            In-sample accuracy measured on the fitted training data.
        """
        self.feature_columns = list(X.columns)
        encoded_labels = self.label_encoder.fit_transform(np.asarray(y))
        scaled_features = self.scaler.fit_transform(X)
        # The training artifact stores globally fitted preprocessing plus the
        # final estimator configured for the full label set.
        self.estimator = _configure_xgb_estimator(self.estimator, encoded_labels)
        self.estimator.fit(scaled_features, encoded_labels)
        predictions = np.asarray(self.estimator.predict(scaled_features))
        if predictions.ndim > 1:
            predictions = predictions.argmax(axis=1)
        return float(accuracy_score(encoded_labels, predictions))

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        *,
        splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
    ) -> CrossValidationResult:
        """Evaluate the configured estimator with stratified cross-validation.

        Args:
            X: Feature matrix.
            y: Original state labels before encoding.
            splits: Number of folds in the validation scheme.
            shuffle: Whether to shuffle the folds before splitting.
            random_state: Seed used when shuffling folds.

        Returns:
            A :class:`CrossValidationResult` with per-fold scores and summary
            statistics.
        """
        y_array = np.asarray(y)
        encoded_labels = self.label_encoder.fit_transform(y_array)
        min_class_count = int(pd.Series(encoded_labels).value_counts().min())
        effective_splits = min(splits, len(encoded_labels))
        if effective_splits < 2:
            raise ValueError(
                self._("Cross-validation requires at least two total samples.")
            )

        if min_class_count >= 2:
            # Preserve class proportions whenever the smallest class supports
            # stratification; otherwise fall back to plain K-Fold.
            effective_splits = min(effective_splits, min_class_count)
            splitter = StratifiedKFold(
                n_splits=effective_splits,
                shuffle=shuffle,
                random_state=random_state,
            )
        else:
            splitter = KFold(
                n_splits=effective_splits,
                shuffle=shuffle,
                random_state=random_state,
            )
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", self._build_cv_estimator()),
            ]
        )
        scores = cross_val_score(pipeline, X, encoded_labels, cv=splitter)
        return CrossValidationResult(scores=np.asarray(scores, dtype=float))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict original-state labels for new analog observations.

        Args:
            X: Inference feature matrix.

        Returns:
            Predicted labels mapped back to the original state identifiers.
        """
        scaled_features = self.scaler.transform(X)
        encoded_predictions = np.asarray(self.estimator.predict(scaled_features))
        if encoded_predictions.ndim > 1:
            encoded_predictions = encoded_predictions.argmax(axis=1)
        return self.label_encoder.inverse_transform(encoded_predictions)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities aligned with the original label space.

        Args:
            X: Inference feature matrix.

        Returns:
            A two-dimensional array whose columns follow
            ``self.label_encoder.classes_``.
        """
        scaled_features = self.scaler.transform(X)
        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError(
                self._("The selected model does not expose probability estimates.")
            )
        return np.asarray(self.estimator.predict_proba(scaled_features), dtype=float)

    def save(self, file_path: str) -> None:
        """Persist the full inference artifact required for later reuse.

        The saved payload contains every object needed to run predictions on
        unseen data without retraining: estimator, scaler, label encoder, and
        feature ordering metadata.
        """
        payload = {
            "model": self.estimator,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "model_type": self.model_type,
            "params": self.params,
            "feature_columns": self.feature_columns,
        }
        joblib.dump(payload, file_path)

    @classmethod
    def load(
        cls,
        file_path: str,
        translator: Optional[Callable[[str], str]] = None,
    ) -> "StateClassifier":
        """Restore a persisted classifier artifact from disk.

        Args:
            file_path: Serialized artifact path created with :meth:`save`.
            translator: Optional translation function for user-facing errors.

        Returns:
            A ready-to-use :class:`StateClassifier` instance.
        """
        payload = joblib.load(file_path)
        classifier = cls(
            model_type=payload["model_type"],
            params=payload.get("params"),
            translator=translator,
        )
        classifier.estimator = payload["model"]
        classifier.scaler = payload["scaler"]
        classifier.label_encoder = payload["label_encoder"]
        classifier.feature_columns = payload.get("feature_columns", [])
        return classifier
