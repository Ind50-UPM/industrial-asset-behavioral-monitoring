# Model_A: Industrial State Identification

`Model_A` is the first reusable package in the Industrial Asset Behavioral Monitoring framework. It focuses on supervised operating-state identification from analog electrical variables and digital state labels, with an installable CLI and a Python API built around the `iabm` package.

## Scope

This package currently supports:

- analog and digital Parquet input datasets
- legacy-compatible NaN imputation for industrial electrical signals
- supervised labeling of active analog rows from digital states
- Random Forest and XGBoost classifiers
- stratified cross-validation
- model persistence including scaler, label encoder, and feature ordering
- prediction on new analog data without requiring digital labels at inference time
- bilingual CLI messaging through English and Spanish catalogs

## Package Layout

The installable package is named `iabm`, short for `Industrial Asset Behavioral Monitoring`.

```text
src/Model_A/
├── iabm/
│   ├── __init__.py
│   ├── data_processor.py
│   ├── main.py
│   ├── models.py
│   └── utils.py
├── locales/
├── pyproject.toml
└── README.md
```

## Installation

```bash
git clone https://github.com/Ind50-UPM/industrial-asset-behavioral-monitoring.git
cd industrial-asset-behavioral-monitoring/src/Model_A
poetry install
```

After installation, the package can be used in either of these ways:

```bash
poetry run industrial-id --help
python -m iabm.main --help
```

## Command-Line Usage

Train a classifier and save both the model artifact and the cross-validation report:

```bash
poetry run industrial-id \
  --mode train \
  --algo rf \
  --start "2022-01-18 00:00:00" \
  --end "2022-02-18 00:00:00"
```

Generate predictions from a saved artifact using analog data only:

```bash
poetry run industrial-id \
  --mode predict \
  --algo rf \
  --start "2022-02-21 00:00:00" \
  --end "2022-02-22 00:00:00" \
  --model-out /path/to/analogicas_nonans.parquet_RF.joblib \
  --pred-out /path/to/predictions
```

Switch the interface language to Spanish:

```bash
poetry run industrial-id --lang es --help
```

## Default Inputs and Outputs

Unless overridden by CLI arguments, `Model_A` resolves paths relative to the repository root:

- analog data: `data/analogicas_nonans.parquet`
- digital data: `data/digitales.parquet`
- trained artifacts: `src/models/Model_A/`
- validation reports and predictions: `src/predictions/Model_A/`

## Python API

```python
from iabm import IndustrialDataProcessor, StateClassifier

processor = IndustrialDataProcessor(
    analog_path="data/analogicas_nonans.parquet",
    digital_path="data/digitales.parquet",
)
training_data = processor.prepare_training_data(
    start="2022-01-18 00:00:00",
    end="2022-02-18 00:00:00",
)

classifier = StateClassifier(model_type="rf")
cv_result = classifier.cross_validate(training_data.features, training_data.labels)
classifier.fit(training_data.features, training_data.labels)
classifier.save("rf_model.joblib")
```

## Notes for Reproducibility

- The saved artifact includes the estimator, fitted scaler, fitted label encoder, model type, hyperparameters, and feature order.
- Cross-validation is stratified and uses a fixed random seed by default.
- Excel exports use timezone-safe timestamp formatting for prediction outputs.
- Documentation docstrings in the `iabm` package are written in English, while user-facing CLI messages remain translatable.

## Current Status

- The package name is `iabm`.
- The CLI works through both `poetry run industrial-id` and `python -m iabm.main`.
- Random Forest and XGBoost share the same object-oriented training, validation, persistence, and inference interface.
- Training and prediction have been smoke-tested end to end with the datasets currently stored in the repository.

## Recommended Next Steps

- Add automated tests for artifact loading, feature-order validation, and low-support class scenarios.
- Decide whether cross-validation fold selection should adapt automatically to the minimum class frequency.
- Introduce a formal benchmark configuration layer if `Model_A` will compare additional ML or DL estimators.
- Keep the package API stable so `Model_B` can consume state predictions without depending on CLI internals.
