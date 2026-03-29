"""Command-line entry point for training and using Model_A classifiers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data_processor import EvaluationDataset, IndustrialDataProcessor
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
        choices=["train", "predict", "evaluate"],
        required=True,
        help=_("Execution mode: train a new model, generate predictions, or evaluate saved models."),
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
        "--model-paths",
        "--model_paths",
        dest="model_paths",
        nargs="+",
        help=_("One or more saved model artifacts to compare during evaluation."),
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

    if args.mode == "evaluate":
        _run_evaluation(
            args=args,
            analog_path=analog_path,
            digital_path=digital_path,
            prediction_output=prediction_output,
            translator=translator,
        )
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


def _run_evaluation(
    *,
    args: argparse.Namespace,
    analog_path: Path,
    digital_path: Path,
    prediction_output: Path,
    translator: Callable[[str], str],
) -> None:
    """Compare one or more saved models against the requested time window."""
    model_paths = _resolve_evaluation_model_paths(args, analog_path)
    if not model_paths:
        sys.exit(
            translator(
                "At least one model artifact is required for evaluation (--model-paths or --model-out)."
            )
        )

    processor = IndustrialDataProcessor(
        analog_path=str(analog_path),
        digital_path=str(digital_path) if digital_path.exists() else None,
        threshold=args.threshold,
    )
    evaluation_data = processor.prepare_evaluation_data(args.start, args.end)
    evaluation_frame, summary_frames = _build_evaluation_outputs(
        evaluation_data=evaluation_data,
        model_paths=model_paths,
        translator=translator,
    )

    prediction_output.mkdir(parents=True, exist_ok=True)
    report_stem = (
        f"evaluation_{analog_path.stem}_{_sanitize_period_component(args.start)}"
        f"_{_sanitize_period_component(args.end)}"
    )
    report_path = prediction_output / f"{report_stem}.xlsx"

    with pd.ExcelWriter(report_path) as writer:
        _prepare_excel_frame(evaluation_frame).to_excel(
            writer,
            sheet_name="predictions",
            index=True,
        )
        if summary_frames:
            summary_frames["summary"].to_excel(writer, sheet_name="summary", index=False)
            for sheet_name, frame in summary_frames.items():
                if sheet_name == "summary":
                    continue
                frame.to_excel(writer, sheet_name=sheet_name, index=True)

    print(translator("Evaluation report saved to: {}").format(report_path))

    if evaluation_data.labels is None:
        print(
            translator(
                "Real labels were not available, so confusion matrices and ROC curves were skipped."
            )
        )
        return

    for model_name, roc_plot in _save_roc_plots(
        evaluation_frame=evaluation_frame,
        evaluation_data=evaluation_data,
        model_paths=model_paths,
        output_dir=prediction_output,
        translator=translator,
    ):
        print(translator("ROC curve saved to: {}").format(roc_plot))


def _resolve_evaluation_model_paths(
    args: argparse.Namespace,
    analog_path: Path,
) -> list[Path]:
    """Return the list of model artifacts to compare in evaluation mode."""
    if args.model_paths:
        return [Path(path) for path in args.model_paths]
    if args.model_out:
        return [Path(args.model_out)]

    repo_root = Path(__file__).resolve().parents[3]
    return [
        repo_root / "src" / "models" / "Model_A" / f"{analog_path.name}_RF.joblib",
        repo_root / "src" / "models" / "Model_A" / f"{analog_path.name}_XGB.joblib",
    ]


def _build_evaluation_outputs(
    *,
    evaluation_data: EvaluationDataset,
    model_paths: list[Path],
    translator: Callable[[str], str],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Assemble timestamp-level predictions and tabular quality reports."""
    evaluation_frame = pd.DataFrame(index=evaluation_data.source_frame.index)
    evaluation_frame.index.name = "Time"
    if evaluation_data.labels is not None:
        evaluation_frame["Real_Class"] = evaluation_data.labels.astype("Int64")
    else:
        evaluation_frame["Real_Class"] = pd.Series(pd.NA, index=evaluation_frame.index, dtype="Int64")

    summaries: list[dict[str, object]] = []
    sheet_frames: dict[str, pd.DataFrame] = {}

    for model_path in model_paths:
        if not model_path.exists():
            raise FileNotFoundError(
                translator("Model artifact not found: {}").format(model_path)
            )
        classifier = StateClassifier.load(str(model_path), translator=translator)
        model_name = _build_model_label(classifier, model_path, evaluation_frame.columns)
        feature_frame = evaluation_data.features
        if classifier.feature_columns:
            feature_frame = feature_frame.reindex(columns=classifier.feature_columns)

        full_prediction_frame = _build_prediction_frame(
            source_frame=evaluation_data.source_frame,
            active_mask=evaluation_data.active_mask,
            predictions=_predict_or_empty(classifier, feature_frame),
            include_inactive=True,
        ).rename(columns={"Predicted_State": model_name})
        evaluation_frame[model_name] = full_prediction_frame[model_name].astype(np.int32)

        if evaluation_data.labels is None:
            continue

        labels = evaluation_data.labels.astype(np.int32)
        predictions = evaluation_frame[model_name].astype(np.int32)
        class_labels = np.unique(np.concatenate([labels.to_numpy(), predictions.to_numpy()]))
        matrix = confusion_matrix(labels, predictions, labels=class_labels)
        confusion_frame = pd.DataFrame(
            matrix,
            index=[f"Real_{label}" for label in class_labels],
            columns=[f"Predicted_{label}" for label in class_labels],
        )
        sheet_frames[_sheet_name(f"cm_{model_name}")] = confusion_frame

        summary_row: dict[str, object] = {
            "Model": model_name,
            "Accuracy": float(accuracy_score(labels, predictions)),
        }
        probabilities = _predict_probabilities_or_empty(classifier, feature_frame)
        roc_summary = _build_roc_summary(
            labels=labels,
            active_mask=evaluation_data.active_mask,
            classifier=classifier,
            probabilities=probabilities,
        )
        for class_name, class_auc in roc_summary.items():
            summary_row[f"ROC_AUC_Class_{class_name}"] = class_auc
        summaries.append(summary_row)

    if summaries:
        sheet_frames["summary"] = pd.DataFrame(summaries)
    return evaluation_frame, sheet_frames


def _build_model_label(
    classifier: StateClassifier,
    model_path: Path,
    existing_columns: pd.Index,
) -> str:
    """Build a readable, unique prediction-column name for a model artifact."""
    base_name = f"{classifier.model_type.upper()}_Prediction"
    if base_name not in existing_columns:
        return base_name
    return f"{base_name}_{model_path.stem}"


def _predict_probabilities_or_empty(
    classifier: StateClassifier,
    features: pd.DataFrame,
) -> np.ndarray:
    """Predict class probabilities or return an empty 2D array."""
    if features.empty:
        class_count = len(getattr(classifier.label_encoder, "classes_", []))
        return np.empty((0, class_count), dtype=float)
    return classifier.predict_proba(features)


def _build_roc_summary(
    *,
    labels: pd.Series,
    active_mask: pd.Series,
    classifier: StateClassifier,
    probabilities: np.ndarray,
) -> dict[int, float]:
    """Compute one-vs-rest ROC AUC values for each evaluable active class."""
    active_labels = labels.loc[active_mask].astype(np.int32)
    nonzero_mask = active_labels != 0
    active_labels = active_labels.loc[nonzero_mask]
    if active_labels.empty or probabilities.size == 0:
        return {}

    active_probabilities = probabilities[nonzero_mask.to_numpy()]
    classes = np.asarray(classifier.label_encoder.classes_)
    if len(classes) < 2 or len(active_labels.unique()) < 2:
        return {}

    binarized = label_binarize(active_labels, classes=classes)
    if binarized.ndim == 1:
        binarized = np.column_stack([1 - binarized, binarized])

    roc_summary: dict[int, float] = {}
    for column_index, class_label in enumerate(classes):
        y_true = binarized[:, column_index]
        if np.unique(y_true).size < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, active_probabilities[:, column_index])
        roc_summary[int(class_label)] = float(auc(fpr, tpr))
    return roc_summary


def _save_roc_plots(
    *,
    evaluation_frame: pd.DataFrame,
    evaluation_data: EvaluationDataset,
    model_paths: list[Path],
    output_dir: Path,
    translator: Callable[[str], str],
) -> list[tuple[str, Path]]:
    """Render one ROC plot per model when real active labels are available."""
    saved_plots: list[tuple[str, Path]] = []
    labels = evaluation_data.labels
    if labels is None:
        return saved_plots

    for model_path in model_paths:
        classifier = StateClassifier.load(str(model_path), translator=translator)
        model_name = _build_model_label(classifier, model_path, pd.Index([]))
        feature_frame = evaluation_data.features
        if classifier.feature_columns:
            feature_frame = feature_frame.reindex(columns=classifier.feature_columns)
        probabilities = _predict_probabilities_or_empty(classifier, feature_frame)

        active_labels = labels.loc[evaluation_data.active_mask].astype(np.int32)
        nonzero_mask = active_labels != 0
        active_labels = active_labels.loc[nonzero_mask]
        if active_labels.empty or probabilities.size == 0:
            continue

        classes = np.asarray(classifier.label_encoder.classes_)
        if len(classes) < 2 or len(active_labels.unique()) < 2:
            continue

        active_probabilities = probabilities[nonzero_mask.to_numpy()]
        binarized = label_binarize(active_labels, classes=classes)
        if binarized.ndim == 1:
            binarized = np.column_stack([1 - binarized, binarized])

        figure, axis = plt.subplots(figsize=(7, 5))
        plotted_any = False
        for column_index, class_label in enumerate(classes):
            y_true = binarized[:, column_index]
            if np.unique(y_true).size < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true, active_probabilities[:, column_index])
            axis.plot(
                fpr,
                tpr,
                label=f"Class {class_label} (AUC={auc(fpr, tpr):.3f})",
            )
            plotted_any = True

        if not plotted_any:
            plt.close(figure)
            continue

        axis.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
        axis.set_title(f"ROC - {model_name}")
        axis.set_xlabel("False Positive Rate")
        axis.set_ylabel("True Positive Rate")
        axis.legend(loc="lower right")
        axis.grid(alpha=0.2)
        plot_path = output_dir / f"roc_{_sheet_name(model_name)}.png"
        figure.tight_layout()
        figure.savefig(plot_path, dpi=150)
        plt.close(figure)
        saved_plots.append((model_name, plot_path))

    return saved_plots


def _sanitize_period_component(value: str) -> str:
    """Convert a timestamp string into a filename-safe suffix."""
    return value.replace(":", "-").replace(" ", "_")


def _sheet_name(value: str) -> str:
    """Return an Excel-safe sheet/file stem fragment."""
    sanitized = "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in value)
    return sanitized[:31]


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
