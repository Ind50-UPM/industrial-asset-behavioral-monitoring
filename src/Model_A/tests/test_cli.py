"""CLI-level smoke tests for the Model_A package."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import iabm.main as cli_main


@pytest.mark.parametrize("algorithm", ["rf", "xgb"])
def test_cli_train_and_predict_workflow(
    algorithm: str,
    synthetic_datasets: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The CLI should train, save, load, and predict for both supported models."""
    model_path = tmp_path / f"{algorithm}_model.joblib"
    report_dir = tmp_path / "reports"
    prediction_dir = tmp_path / "predictions"

    monkeypatch.setattr(
        "sys.argv",
        [
            "industrial-id",
            "--mode",
            "train",
            "--algo",
            algorithm,
            "--data-ana",
            str(synthetic_datasets["analog_path"]),
            "--data-dig",
            str(synthetic_datasets["digital_path"]),
            "--model-out",
            str(model_path),
            "--pred-out",
            str(report_dir),
            "--start",
            synthetic_datasets["start"],
            "--end",
            synthetic_datasets["train_end"],
            "--cv-splits",
            "3",
        ],
    )
    cli_main.main()

    assert model_path.exists()
    report_files = list(report_dir.glob("*.xlsx"))
    assert len(report_files) == 1

    monkeypatch.setattr(
        "sys.argv",
        [
            "industrial-id",
            "--mode",
            "predict",
            "--algo",
            algorithm,
            "--data-ana",
            str(synthetic_datasets["analog_path"]),
            "--model-out",
            str(model_path),
            "--pred-out",
            str(prediction_dir),
            "--start",
            synthetic_datasets["predict_start"],
            "--end",
            synthetic_datasets["predict_end"],
            "--include-inactive",
        ],
    )
    cli_main.main()

    prediction_files = list(prediction_dir.glob("*.xlsx"))
    assert len(prediction_files) == 1

    prediction_frame = pd.read_excel(prediction_files[0])
    assert "Predicted_State" in prediction_frame.columns
    assert not prediction_frame.empty
