# Smart System ID: Industrial Asset Operating Mode Identification

## Overview

This framework provides a data-driven approach for identifying the operational states of industrial assets using Electrical Signatures (Voltage, Current, Power, Phase Angle) and complementary IoT sensor data. 
It is designed for legacy or weakly instrumented assets where internal process visibility is restricted.

The system leverages machine learning (Random Forest and XGBoost) to classify elementary states and aggregate them into interpretable behavioral sequences for predictive maintenance and anomaly detection.

## Key Features


* Non-Intrusive Monitoring: Based on electrical signatures and IIoT architecture.
* Hierarchical Modeling: Aggregation of states into high-level operating and working modes.
* Robust Imputation: Advanced handling of missing data (NaNs) in multi-phase electrical signals.
* Industrial Deployment: Validated on a real-world wheel-washing installation at the Port of Gijón.

## Technical Architecture

The framework is organized into four layers:
1. Data Acquisition: Capturing V, I, $\phi$, and digital signals via MQTT.
2. State Identification: Supervised learning using RF and XGBoost algorithms.
3. Behavioral Modeling: Temporal analysis of state sequences.
4. Anomaly Detection: Identification of malfunctions through sequence duration deviations.

## Installation

This project uses Poetry for robust dependency management.Bash# Clone the repository
```bash
git clone https://github.com/Ind50-UPM/industrial-asset-behavioral-monitoring.git
cd src/Model_A/

# Install dependencies
poetry install

```

## Usage

1. Data PreparationConvert your raw data to Apache Parquet format for efficient processing (recommended for large IIoT datasets).
2. Training the ModelTrain a classifier using a specific time window and algorithm:

```bash
poetry run industrial-id --mode train --algo rf --start "2022-01-18 00:00:00" --end "2022-02-18 00:00:00"

```
Supported algorithms: rf (Random Forest), xgb (XGBoost).
3. Inference & PredictionIdentify states for a new monitoring period:
```bash
poetry run industrial-id --mode predict --algo rf --start "2022-02-21 00:00:00" --end "2022-02-22 00:00:00"
```

## Dataset

The system expects two main data sources in the data/ directory:

* analogicas.parquet: Time-series of electrical variables (RP#, Vrms#, Irms#, PF#).
* digitales.parquet: Control signals used for state labeling during training.


