# Industrial Asset Behavioral Monitoring

Implementation, datasets, and experimental workflows supporting a **data-driven framework for industrial asset monitoring** based on **operational-state identification and hierarchical behavioral modeling** using Industrial Internet of Things (IIoT) data.

This repository accompanies the research work:

> **"A Data-Driven Behavioral Monitoring Framework for Industrial Assets"**  
> being submitted to *Computers in Industry*.

The proposed approach enables the identification of operational states from sensor measurements (e.g., electrical signatures and process variables) and aggregates them into interpretable behavioral patterns that support **industrial monitoring, anomaly detection, and maintenance-oriented decision-making**.

---

# Overview

Industrial assets operating in remote or weakly instrumented environments often lack sufficient monitoring capabilities to fully characterize their operational behavior.

This repository provides the **datasets, algorithms, and analysis workflows** used to:

- infer **operational states** from sensor measurements
- aggregate elementary states into **behavioral sequences**
- characterize **operational modes**
- detect **abnormal operational behavior**
- support **maintenance-oriented interpretation**

The methodology integrates:

- Industrial IoT data acquisition
- Machine learning–based state identification
- Hierarchical behavioral modeling
- Temporal sequence analysis
- Anomaly detection

The approach has been validated on a **real industrial installation**, demonstrating its ability to reconstruct asset behavior and detect abnormal conditions.

---

# Repository Structure

Pendiente el tree para cuando esté montada la estructura


---

# Dataset Description

The dataset contains measurements obtained from a **real industrial asset monitoring system** deployed in an operational environment.

Typical recorded variables include:

- electrical measurements (voltage, current, phase angle)
- energy consumption indicators
- digital control signals
- complementary process variables

These signals enable the reconstruction of **operational states and behavioral sequences** describing the functioning of the asset.

Due to potential industrial sensitivity, the repository may contain:

- a **sample dataset** for reproducibility
- preprocessed datasets used for experiments

---

# Reproducing the Experiments

## 1. Clone the repository

```bash
git clone https://github.com/Ind50-UPM/industrial-asset-behavioral-monitoring.git
cd industrial-asset-behavioral-monitoring
```

## 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r environment/requirements.txt
```



