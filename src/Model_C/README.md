# Model_C: Semantic Mode Interpretation

`Model_C` is the third modeling layer in the Industrial Asset Behavioral Monitoring framework. It interprets behavioral sequences from `Model_B` as semantic operating modes and working modes, with optional enrichment from anomaly-comparison outputs.

## Scope

This package currently provides:

- loading of `Model_B` active-sequence reports
- optional loading of `Model_B` anomaly-comparison reports
- decoding of state words into industrial components
- heuristic or rule-based assignment of operating modes and working modes
- semantic status enrichment with anomaly context
- summary reports for interpreted semantic modes

## Package Layout

The installable package is named `iabm_semantics`.

```text
src/Model_C/
├── iabm_semantics/
│   ├── __init__.py
│   ├── main.py
│   ├── semantics.py
│   └── utils.py
├── locales/
├── pyproject.toml
└── README.md
```

## Installation

```bash
cd industrial-asset-behavioral-monitoring/src/Model_C
poetry install
```

## Command-Line Usage

Interpret a `Model_B` active-sequence report:

```bash
poetry run industrial-semantics \
  --input ../Model_B/reports/active_sequences.xlsx \
  --output-dir ./reports
```

Interpret the same report while enriching semantic status with anomaly information:

```bash
poetry run industrial-semantics \
  --input ../Model_B/reports/active_sequences.xlsx \
  --comparison-input ../Model_B/reports/sequence_comparison.xlsx \
  --output-dir ./reports
```

## Outputs

- `semantic_assignments.xlsx`: per-sequence semantic interpretation
- `semantic_mode_summary.xlsx`: aggregated counts by operating mode, working mode, and semantic status

## Python API

```python
from iabm_semantics import SemanticModeInterpreter

interpreter = SemanticModeInterpreter()
sequences = interpreter.load_active_sequences("active_sequences.xlsx")
assignments = interpreter.interpret_sequences(sequences)
summary = interpreter.summarize_modes(assignments)
```
