Reproducibility
===============

This repository implements a layered workflow for industrial asset behavioral
monitoring based on operational-state inference, behavioral sequence analysis,
semantic contextualization, and longitudinal incident processing. It accompanies
research work on industrial asset monitoring in IIoT-enabled environments where
direct reliability labels may be sparse or incomplete.

Repository Architecture
-----------------------

The implementation is organized into four package-oriented modeling layers.

``Model_A``
   Package: ``iabm``

   Purpose: supervised identification of elementary operating states from
   industrial analog and digital signals.

``Model_B``
   Package: ``iabm_behavior``

   Purpose: extraction of contiguous runs, active sequences, nominal behavioral
   references, and longitudinal comparison metrics.

``Model_C``
   Package: ``iabm_semantics``

   Purpose: semantic and reliability-informed interpretation of behavioral
   sequences into operating modes, working modes, and incident-family-aware
   assignments.

``Model_D``
   Package: ``iabm_incidents``

   Purpose: explicit scoring, segmentation, family assignment, recovery
   assessment, registry alignment, and occurrence summaries for longitudinal
   episodes.

Layered Workflow
----------------

The repository supports a progressive analysis workflow:

1. ``Model_A`` transforms synchronized industrial observations into discrete operating-state predictions.
2. ``Model_B`` organizes those predicted states into runs and behavioral sequences and estimates divergence-oriented longitudinal indicators.
3. ``Model_C`` interprets the extracted behavioral sequences through higher-level semantic categories such as operating modes, working modes, and incident families.
4. ``Model_D`` converts those sequence and semantic outputs into scored windows, detected episodes, recovery assessments, and registry-aware evaluation tables.

Reproducible Model_D assets
---------------------------

The current repository now includes reproducibility-oriented assets for the
longitudinal layer:

- a typed configuration system in ``iabm_incidents.config``;
- a versionable example configuration file at ``src/Model_D/configs/model_d.example.json``;
- metadata export through ``model_d_run_metadata.json``;
- synthetic tests covering configuration loading, CLI execution, episode construction, occurrence summaries, and evaluation baselines.

Recommended execution pattern for Model_D
-----------------------------------------

For reproducible longitudinal runs, the recommended invocation pattern is:

.. code-block:: bash

   poetry run industrial-incidents \
     --config configs/model_d.example.json \
     --sequences path/to/active_sequences.csv \
     --assignments path/to/window_indicators.csv \
     --registry path/to/incident_registry.csv \
     --output-dir outputs/model_d \
     --output-format csv

This keeps threshold choices, weighting assumptions, and output artifacts fully
traceable.

Relation to the Paper
---------------------

The repository mirrors the main empirical workflow reported in the manuscript.

- Data preprocessing and state-identification tasks are associated with ``Model_A``.
- Behavioral sequence extraction and longitudinal comparison are associated with ``Model_B``.
- Semantic contextualization and family-oriented interpretation are associated with ``Model_C``.
- Episode-oriented validation and recurrence analysis are associated with ``Model_D``.

The public codebase is therefore not only a software artifact, but also a
reproducibility resource for the experimental pipeline discussed in the paper.

Where to Start
--------------

For most users, the recommended reading order is:

- :doc:`overview`
- :doc:`model_a`
- :doc:`model_b`
- :doc:`model_c`
- :doc:`model_d`

Documentation Strategy
----------------------

This Sphinx configuration relies on ``autodoc`` and ``napoleon`` so that
Google-style docstrings embedded in the codebase become the primary API
reference. The intent is to keep implementation and formal documentation
synchronized while providing an accessible entry point to the structure and
purpose of the repository.
