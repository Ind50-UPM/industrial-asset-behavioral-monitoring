# Industrial Asset Behavioral Monitoring

[![CI](https://github.com/Ind50-UPM/industrial-asset-behavioral-monitoring/actions/workflows/ci.yml/badge.svg)](https://github.com/Ind50-UPM/industrial-asset-behavioral-monitoring/actions/workflows/ci.yml)

Implementation, datasets, and experimental workflows supporting a data-driven framework for industrial asset monitoring based on operational-state identification and hierarchical behavioral modeling with Industrial Internet of Things (IIoT) data.

This repository accompanies the research work:

> **"A Data-Driven Behavioral Monitoring Framework for Industrial Assets"**  
> submitted to *Computers in Industry*.

## Overview

The repository is organized around the progressive modeling layers described in the paper:

- `Model_A`: elementary operating-state identification from analog and digital signals
- `Model_B`: temporal and behavioral modeling on top of state sequences
- `Model_C`: higher-level semantic, generalization, or deployment-oriented extensions

At the moment, `Model_A` is the most complete reusable package in the repository.

## Repository Structure

```text
industrial-asset-behavioral-monitoring/
├── data/
│   ├── analogicas_nonans.parquet
│   └── digitales.parquet
├── src/
│   ├── Model_A/
│   │   ├── iabm/
│   │   ├── locales/
│   │   ├── pyproject.toml
│   │   └── README.md
│   ├── models/
│   ├── predictions/
│   └── README.md
└── README.md
```

## Model_A

`Model_A` is packaged as `iabm`, short for `Industrial Asset Behavioral Monitoring`.

It currently provides:

- supervised state identification with Random Forest and XGBoost
- study-compatible preprocessing and NaN imputation
- stratified cross-validation
- model persistence with scaler, label encoder, and feature ordering
- prediction on new analog data without digital labels at inference time
- bilingual CLI support in English and Spanish

Package-specific documentation is available in [`src/Model_A/README.md`](src/Model_A/README.md).

## Installation

To work with the reusable `Model_A` package:

```bash
git clone https://github.com/Ind50-UPM/industrial-asset-behavioral-monitoring.git
cd industrial-asset-behavioral-monitoring/src/Model_A
poetry install
```

You can then use either the installed CLI or the package module directly:

```bash
poetry run industrial-id --help
python -m iabm.main --help
```

## Data

The datasets included in `data/` represent industrial monitoring signals from a real deployment. Typical variables include:

- electrical measurements such as voltage, current, power, and power factor
- digital control or status signals
- complementary process-related variables

These signals support both elementary state identification and later behavioral aggregation.

## Reproducibility Notes

- Default `Model_A` inputs are resolved from `data/analogicas_nonans.parquet` and `data/digitales.parquet`.
- Default trained artifacts are written under `src/models/Modela_A/`.
- Default validation reports and prediction exports are written under `src/predictions/Modela_A/`.
- The `iabm` package uses English docstrings and translatable CLI messages.

## Current Status

- `Model_A` is installable and runnable as the `iabm` package.
- The training workflow includes stratified cross-validation, artifact persistence, and report export.
- The inference workflow reuses saved artifacts and does not require digital labels at prediction time.
- Repository-level documentation has been aligned with the current package structure.

## Next Steps

- Extend the same package-oriented structure to `Model_B` and `Model_C`.
- Add explicit benchmark comparisons for additional ML and DL approaches where required by the paper.
- Expand automated tests around training windows, class imbalance, and artifact compatibility.
- Refine the higher-level narrative so code, experiments, and manuscript sections evolve together.
