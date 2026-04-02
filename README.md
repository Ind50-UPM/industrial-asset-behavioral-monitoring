# Industrial Asset Behavioral Monitoring

Implementation, datasets, semantic artifacts, and documentation for a data-driven framework for industrial asset monitoring based on operational-state inference, hierarchical behavioral modeling, semantic contextualization, and abnormality detection using Industrial Internet of Things (IIoT) data.

This repository accompanies a research workflow on **data-driven identification of industrial asset operating modes for predictive maintenance in IoT-enabled industrial environments**. It is intended both as a **reproducibility resource** and as a **reusable software scaffold** for behavioral monitoring of weakly instrumented industrial assets.

---

## Overview

The repository is organized around three progressive modeling layers that mirror the analytical workflow described in the paper:

- **Model_A** (`iabm`)  
  Supervised identification of elementary operational states from synchronized analog and digital industrial signals.

- **Model_B** (`iabm_behavior`)  
  Extraction of contiguous state runs, behavioral sequences, nominal sequence references, and anomaly-oriented comparison metrics.

- **Model_C** (`iabm_semantics`)  
  Ontology-aligned semantic contextualization of behavioral sequences into operating modes, working modes, and higher-level contextualized summaries.

Together, these three packages implement a coherent layered workflow:

1. **Model_A** transforms synchronized industrial observations into operational-state predictions.
2. **Model_B** organizes those predicted states into runs, behavioral sequences, and nominal-comparison reports.
3. **Model_C** interprets behavioral sequences through a lightweight contextualization layer that links structural anchors, functional roles, and monitored evidence.

---

## Relation to the paper

The repository mirrors the main methodological stages of the paper:

- **Operational-state inference** -> `Model_A`
- **Behavioral sequence construction and anomaly-oriented comparison** -> `Model_B`
- **Semantic contextualization and operating/working mode interpretation** -> `Model_C`
- **Ontology-aligned contextualization schema** -> `ontology/`
- **User and developer documentation** -> `docs/`

In this sense, the repository is not only a software release, but also a reproducibility resource for the empirical workflow described in the manuscript.

---

## Repository structure

```text
industrial-asset-behavioral-monitoring/
├── CHANGELOG.md
├── LICENSE
├── README.md
├── data/
│   ├── README.md
│   ├── analogicas_nonans.parquet
│   └── digitales.parquet
├── docs/
│   ├── README.md
│   ├── conf.py
│   ├── index.rst
│   ├── overview.rst
│   ├── reproducibility.rst
│   ├── semantic_schema.rst
│   ├── model_a.rst
│   ├── model_b.rst
│   ├── model_c.rst
│   ├── requirements.txt
│   └── _static/
├── ontology/
│   ├── README.md
│   ├── iabm.ttl
│   ├── examples/
│   │   └── wheel_washer.ttl
│   └── queries/
│       └── example_queries.rq
└── src/
    ├── Model_A/
    │   ├── README.md
    │   ├── iabm/
    │   ├── tests/
    │   └── pyproject.toml
    ├── Model_B/
    │   ├── README.md
    │   ├── iabm_behavior/
    │   ├── tests/
    │   └── pyproject.toml
    └── Model_C/
        ├── README.md
        ├── iabm_semantics/
        ├── tests/
        └── pyproject.toml
```
Data

The repository includes the industrial datasets used to support the behavioral monitoring workflow:

data/analogicas_nonans.parquet
Preprocessed analog industrial monitoring signals.
data/digitales.parquet
Synchronized digital/control-layer signals.

These datasets provide the basis for state inference, sequence extraction, and contextualized operational interpretation.

Semantic schema

The repository includes a lightweight ontology-aligned contextualization schema under ontology/.

Contents
ontology/iabm.ttl
Core schema defining the main classes and relations used to contextualize observations, operational states, behavioral sequences, operating modes, working modes, and abnormality evidence.
ontology/examples/wheel_washer.ttl
Minimal instance-level example derived from the wheel-washing industrial use case.
ontology/queries/example_queries.rq
Illustrative SPARQL queries showing how semantic entities can be explored and retrieved.
Scope

The semantic layer should be understood as a lightweight contextualization scheme, not as a complete ontology-driven reasoning system. Its role is to make the semantic structure explicit, inspectable, and reusable while remaining consistent with the data-driven analytical workflow implemented in Model_A, Model_B, and Model_C.

Installation

Each modeling layer is maintained as its own Poetry package. Install the package that matches the layer you want to run.

Clone the repository:

git clone https://github.com/Ind50-UPM/industrial-asset-behavioral-monitoring.git
cd industrial-asset-behavioral-monitoring
Model_A
cd src/Model_A
poetry install
poetry run industrial-id --help
python -m iabm.main --help
Model_B
cd src/Model_B
poetry install
poetry run iabm-behavior --help
python -m iabm_behavior.main --help
Model_C
cd src/Model_C
poetry install
poetry run iabm-semantics --help
python -m iabm_semantics.main --help
Minimal workflow

A minimal end-to-end workflow is:

Run Model_A to train or load the state-identification model and generate state predictions.
Run Model_B on the resulting state timeline to obtain contiguous runs, active sequences, and optional anomaly-comparison reports.
Run Model_C on the sequence outputs to generate contextualized operating-mode and working-mode summaries.

This layered structure allows the repository to be used incrementally, depending on whether the user is interested in state inference only, behavioral monitoring, or contextualized semantic interpretation.

Model summaries
Model_A

Model_A is packaged as iabm and focuses on:

loading synchronized analog and digital industrial signals,
preprocessing and preparation of state-identification datasets,
training and evaluating supervised models for operational-state inference,
generating state predictions and validation outputs,
supporting practical deployment under partial instrumentation.
Model_B

Model_B is packaged as iabm_behavior and focuses on:

loading state timelines from Excel, CSV, or Parquet,
smoothing short transient runs before behavioral aggregation,
extracting contiguous state runs,
identifying active multi-state behavioral sequences,
generating sequence summaries and nominal sequence references,
comparing observed sequences against nominal patterns with anomaly-oriented scoring.

### Model_C

Model_C is packaged as iabm_semantics and focuses on:

* loading active-sequence reports generated by Model_B,
* optionally incorporating anomaly-comparison outputs,
* decoding behavioral sequences into structural and functional interpretations,
* assigning contextualized operating modes and working modes,
* producing aggregate semantic summaries consistent with the ontology-aligned contextualization schema.

## Documentation

Project documentation is available in the docs/ folder and can also be published as a Sphinx site.

The documentation currently includes:

* repository overview,
* reproducibility guidance,
* semantic schema description,
* package-oriented pages for Model_A, Model_B, and Model_C.

If Sphinx is installed, documentation can be built locally from docs/ with:

````bash
make html
```

If your GitHub Pages deployment is active, you may also include the public documentation URL here.

## Reproducibility

The repository is intended to support reproducibility of the main analytical workflow reported in the paper.

In particular, it provides:

* the original industrial datasets,
* the refactored software layers used in the study,
* semantic artifacts for contextual interpretation,
* and technical documentation describing the relation between code, data, and analytical stages.

## Citation

Citation details should be added once a stable public reference is available.

For the time being, the repository should be cited as the software and data companion to the corresponding research work on industrial asset behavioral monitoring and operating-mode identification.

## License

This repository is distributed under the terms of the AGPL-3.0 license. See LICENSE for details.

## Contact

For scientific or technical questions related to the repository, please use the GitHub issue tracker or contact the corresponding author through the institutional details provided in the associated manuscript.
