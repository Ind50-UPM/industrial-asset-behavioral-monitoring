"""Command-line entry point for training and using Model_A classifiers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .data_processor import IndustrialDataProcessor
from .models import StateClassifier
from .utils import setup_i18n


def parse_arguments(translator: Callable[[str], str]) -> argparse.Namespace:
    """Build the CLI parser with translated help messages.

    Args:
        translator: Translation function returned by :func:`setup_i18n`.

    Returns:
        Parsed command-line arguments ready to drive the main workflow.
    """
    _ = translator
    parser = argparse.ArgumentParser(
        description=_("Industrial state identification framework for IIoT assets")
    )
    parser.add_argument(
        "--mode",
        choices=["train", "predict"],
        required=True,
        help=_("Execution mode: train a new model or generate predictions."),
    )
    parser.add_argument(
        "--algo",
        choices=["rf", "xgb"],
        default="rf",
        help=_("Algorithm to use for Model_A training or inference."),
    )
    parser.add_argument(
        "--lang",
        default="en",
        choices=["es", "en"],
        help=_("Interface language."),
    )
    parser.add_argument(
        "--data-ana",
        "--data_ana",
        dest="data_ana",
        type=str,
        help=_("Path to the analog Parquet dataset."),
    )
    parser.add_argument(
        "--data-dig",
        "--data_dig",
        dest="data_dig",
        type=str,
        help=_("Path to the digital Parquet dataset used for training labels."),
    )
    parser.add_argument(
        "--model-out",
        "--model_out",
        dest="model_out",
        type=str,
        help=_("Output path for the trained model artifact."),
    )
    parser.add_argument(
        "--pred-out",
        "--pred_out",
        dest="pred_out",
        type=str,
        help=_("Output directory for metrics or predictions."),
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help=_("Start date (YYYY-MM-DD HH:MM:SS)."),
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help=_("End date (YYYY-MM-DD HH:MM:SS)."),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=50.0,
        help=_("Power threshold used to mark inactive rows."),
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help=_("Number of cross-validation folds."),
    )
    parser.add_argument(
        "--include-inactive",
        action="store_true",
        help=_("Include inactive rows with state 0 in the prediction output."),
    )
    return parser.parse_args()


def main() -> None:
    """Run the end-to-end Model_A workflow for training or prediction.

    The entry point keeps orchestration concerns in one place while delegating
    data preparation and model lifecycle logic to their respective classes.

    Training mode prepares labeled features, runs cross-validation, fits the
    final classifier, and persists both the model artifact and fold metrics.
    Prediction mode loads a previously trained artifact and applies it to a new
    analog time window without requiring digital labels at inference time.
    """
    lang = _detect_language(sys.argv)
    translator = setup_i18n(lang)
    args = parse_arguments(translator)

    analog_path, digital_path, model_output, prediction_output = _resolve_paths(args)

    if args.mode == "train":
        processor = IndustrialDataProcessor(
            analog_path=str(analog_path),
            digital_path=str(digital_path),
            threshold=args.threshold,
        )
        training_data = processor.prepare_training_data(args.start, args.end)
        classifier = StateClassifier(model_type=args.algo, translator=translator)

        # Validation is reported before the final fit so experiment quality and
        # deployable artifact generation remain clearly separated.
        print(translator("Starting cross-validation from {} to {}").format(args.start, args.end))
        cv_result = classifier.cross_validate(
            training_data.features,
            training_data.labels,
            splits=args.cv_splits,
        )
        for fold_index, score in enumerate(cv_result.scores, start=1):
            print(translator("Fold {}: accuracy = {:.4f}").format(fold_index, score))
        print(
            translator("Average accuracy: {:.4f} (+/- {:.4f})").format(
                cv_result.mean,
                cv_result.std,
            )
        )

        training_accuracy = classifier.fit(training_data.features, training_data.labels)
        print(translator("Training accuracy: {:.4f}").format(training_accuracy))

        _save_metrics_report(
            prediction_output,
            analog_path,
            args.algo,
            args.start,
            cv_result.scores,
            translator,
        )
        model_output.parent.mkdir(parents=True, exist_ok=True)
        classifier.save(str(model_output))
        print(translator("Model saved to: {}").format(model_output))
        return

    if not args.model_out:
        sys.exit(translator("Model path is required for prediction mode (--model-out)."))
    if not args.pred_out:
        sys.exit(translator("Prediction output directory is required (--pred-out)."))

    processor = IndustrialDataProcessor(
        analog_path=str(analog_path),
        threshold=args.threshold,
    )
    classifier = StateClassifier.load(str(model_output), translator=translator)
    print(translator("Loading model from: {}").format(model_output))
    print(
        translator("Loading data for prediction from {} to {}").format(
            args.start,
            args.end,
        )
    )

    inference_data = processor.prepare_inference_data(args.start, args.end)
    prediction_features = inference_data.features
    if classifier.feature_columns:
        # Reindex to the training-time feature order stored with the artifact.
        prediction_features = prediction_features.reindex(
            columns=classifier.feature_columns
        )
    prediction_frame = _build_prediction_frame(
        source_frame=inference_data.source_frame,
        active_mask=inference_data.active_mask,
        predictions=_predict_or_empty(classifier, prediction_features),
        include_inactive=args.include_inactive,
    )

    prediction_output.mkdir(parents=True, exist_ok=True)
    prediction_file = prediction_output / (
        f"predictions_{analog_path.name}_{classifier.model_type.upper()}.xlsx"
    )
    _prepare_excel_frame(prediction_frame).to_excel(prediction_file, index=True)
    print(translator("Predictions saved to: {}").format(prediction_file))


def _detect_language(argv: list[str]) -> str:
    """Extract the requested language before parsing the translated CLI.

    Args:
        argv: Raw command-line token list.

    Returns:
        The requested language code, or ``"en"`` when no valid language was
        provided on the command line.
    """
    for option in ("--lang",):
        if option in argv:
            try:
                return argv[argv.index(option) + 1]
            except (IndexError, ValueError):
                return "en"
    return "en"


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    """Resolve input and output paths relative to the repository layout.

    Args:
        args: Parsed command-line arguments.

    Returns:
        A tuple with analog input path, digital input path, model artifact path,
        and report/prediction output directory.
    """
    repo_root = Path(__file__).resolve().parents[3]
    analog_path = Path(args.data_ana) if args.data_ana else repo_root / "data" / "analogicas_nonans.parquet"
    digital_path = Path(args.data_dig) if args.data_dig else repo_root / "data" / "digitales.parquet"
    default_model = repo_root / "src" / "models" / "Modela_A" / f"{analog_path.name}_{args.algo.upper()}.joblib"
    default_reports = repo_root / "src" / "predictions" / "Modela_A"
    model_output = Path(args.model_out) if args.model_out else default_model
    prediction_output = Path(args.pred_out) if args.pred_out else default_reports
    return analog_path, digital_path, model_output, prediction_output


def _save_metrics_report(
    output_dir: Path,
    analog_path: Path,
    algorithm: str,
    start: str,
    scores: np.ndarray,
    translator: Callable[[str], str],
) -> None:
    """Persist fold-wise cross-validation metrics to an Excel report.

    Args:
        output_dir: Directory where the report should be written.
        analog_path: Input analog dataset path used to derive the report name.
        algorithm: Short identifier of the trained algorithm.
        start: Training-period start timestamp used for report naming.
        scores: Fold-wise cross-validation scores.
        translator: Translation function for user-facing messages.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(
        {
            "Fold": np.arange(1, len(scores) + 1),
            "Accuracy": scores,
        }
    )
    report_path = output_dir / f"{analog_path.name}_{algorithm.upper()}_metrics_{start[:10]}.xlsx"
    report.to_excel(report_path, index=False)
    print(translator("Metrics saved to: {}").format(report_path))


def _build_prediction_frame(
    *,
    source_frame: pd.DataFrame,
    active_mask: pd.Series,
    predictions: np.ndarray,
    include_inactive: bool,
) -> pd.DataFrame:
    """Create a prediction table aligned with the original analog timestamps.

    Args:
        source_frame: Full analog window used for inference.
        active_mask: Boolean mask selecting active rows in the source frame.
        predictions: Predicted labels for active rows.
        include_inactive: Whether to emit inactive rows with a default zero state.

    Returns:
        A DataFrame ready to be exported or further post-processed.
    """
    active_index = source_frame.index[active_mask]
    active_predictions = pd.DataFrame(
        {"Predicted_State": predictions},
        index=active_index,
    )
    if not include_inactive:
        active_predictions.index.name = "Time"
        return active_predictions

    full_predictions = pd.DataFrame(
        {"Predicted_State": np.zeros(len(source_frame), dtype=np.int32)},
        index=source_frame.index,
    )
    full_predictions.loc[active_index, "Predicted_State"] = predictions
    full_predictions.index.name = "Time"
    return full_predictions


def _predict_or_empty(
    classifier: StateClassifier,
    features: pd.DataFrame,
) -> np.ndarray:
    """Predict labels or return an empty array when no active rows are available.

    Args:
        classifier: Fitted classifier artifact.
        features: Inference feature matrix.

    Returns:
        A NumPy array of predicted labels. The array is empty when the requested
        time window contains no active rows after preprocessing.
    """
    if features.empty:
        return np.array([], dtype=np.int32)
    return classifier.predict(features)


def _prepare_excel_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert timezone-aware indices into Excel-safe string representations.

    Args:
        frame: DataFrame about to be exported to Excel.

    Returns:
        A copy of the DataFrame whose index can be serialized by Excel writers.
    """
    excel_frame = frame.copy()
    if isinstance(excel_frame.index, pd.DatetimeIndex) and excel_frame.index.tz is not None:
        excel_frame.index = excel_frame.index.strftime("%Y-%m-%d %H:%M:%S%z")
    return excel_frame


if __name__ == "__main__":
    main()
