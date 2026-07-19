# Predictions Directory Guide

This folder contains generated artifacts produced by the layered monitoring
workflow. After cleanup, the remaining files are intended to reflect the current
canonical outputs rather than legacy exploratory exports.

## Structure

- `Model_A/`
  Reserved for state-identification prediction artifacts. Legacy presentation
  exports were removed during repository cleanup because the current workflow is
  centered on downstream CSV, parquet, and consolidated reporting artifacts.

- `Model_B/`
  Contains behavior-layer outputs.
  - `active_sequences.csv`: consolidated sequence-level export.
  - `monthly/`: per-month CSV artifacts such as `state_runs`, `active_sequences`,
    `sequence_words`, and, where available, `sequence_comparison`.

- `Model_C/`
  Contains semantic-layer outputs.
  - `semantic_assignments.csv`: consolidated semantic assignment export.
  - `semantic_mode_summary.csv`: consolidated semantic summary.
  - `monthly/`: per-month semantic assignments, incident-family assignments,
    and semantic-mode summaries.

- `Model_D/`
  Contains incident-layer outputs.
  - `candidate_incident_registry_refined.csv`: canonical consolidated refined
    weak-label registry.
  - `candidate_incident_registry_refined_manifest.csv`: month-level registry
    generation traceability.
  - `candidate_incident_registry_refined_metadata.json`: registry generation
    metadata.
  - `candidate_incident_registry_refined_summary.csv`: family/source summary for
    the refined registry.
  - `monthly_candidate_registry/`: per-month parquet candidate registries.
  - `final_reassessment/`: final full-horizon Model_D evaluation artifacts,
    including executive report, monthly and mode summaries, metadata, caveats,
    and charts.

- `model_bc_batch_manifest.csv`
  Operational manifest for the consolidated Model_B and Model_C batch process.

- `model_bc_batch_metadata.json`
  Metadata for the consolidated Model_B and Model_C batch process.

## Canonical outputs

For most current analyses, the recommended reference artifacts are:

- `Model_B/monthly/*.csv` for month-level behavior analysis
- `Model_C/monthly/*.csv` for month-level semantic interpretation
- `Model_D/candidate_incident_registry_refined.csv` for the consolidated weakly
  labeled candidate incident registry
- `Model_D/final_reassessment/model_d_final_report.html` for the executive
  full-horizon reassessment report
- `Model_D/final_reassessment/model_d_final_month_summary.csv` for month-level
  Model_D outcomes across the requested horizon
- `Model_D/final_reassessment/model_d_reassessment_metadata.json` and
  `model_d_reassessment_salvedades.json` for horizon coverage and caveat tracking

## Notes

- Missing months in the requested 2022-01 to 2026-06 horizon are tracked in the
  final reassessment metadata rather than represented as synthetic empty files.
- Low-data months are explicitly flagged in the final reassessment caveat file.
- Root-level Excel, PDF, and PNG exports were intentionally removed because they
  were either duplicated, exploratory, or superseded by the current structured
  outputs.
