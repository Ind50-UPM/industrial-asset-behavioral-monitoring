# Changelog

## e6692fc - Refactor modeling stack into reusable packages

This update restructures the repository into a reusable three-layer modeling stack for industrial asset behavioral monitoring.

`Model_A` has been refactored into the installable `iabm` package, with a cleaner object-oriented design for preprocessing, training, persistence, and inference. Random Forest and XGBoost now share a unified API, saved artifacts include all required preprocessing metadata, and the end-to-end workflow has been validated for both algorithms. Documentation, i18n, and package metadata were also aligned, and automated tests were added for persistence, CLI workflows, and cross-validation edge cases.

`Model_B` is introduced as the `iabm_behavior` package for behavioral sequence analysis. It supports run extraction, active sequence construction, sequence-word summaries, nominal reference building, anomaly-oriented comparison, DTW-style alignment metrics, and configurable anomaly thresholds. A CLI, tests, and bilingual user-facing messaging are included.

`Model_C` is introduced as the `iabm_semantics` package for semantic interpretation of behavioral sequences. It maps sequence-level patterns into components, operating modes, and working modes, supports optional anomaly enrichment from `Model_B`, and allows explicit semantic rules through JSON configuration. It also includes a CLI, tests, and localization support.

Documentation has been substantially strengthened across the three packages, with English docstrings, improved helper documentation, and targeted inline comments for complex logic. A minimal Sphinx documentation site has been added under `docs/`, and a GitHub Actions workflow now runs package tests and builds the documentation automatically.

Generated XGBoost artifacts and metrics were intentionally left out of this commit so that code, tests, documentation, and CI remain separate from experiment outputs.
