Overview
========

This repository implements a layered workflow for industrial asset behavioral
monitoring oriented toward **longitudinal trajectories**, **incident families**,
and **reliability-informed interpretation**. It accompanies research work on how
state, sequence, mode, and episode proxies can support reasoning about asset-life
evolution in IIoT-enabled industrial environments.

The repository is organized to support both methodological understanding and
practical reproducibility. Its structure mirrors the updated progression described
in the paper: from telemetry, to operational states, to behavioral sequences, to
semantic interpretation, and finally to longitudinal incident processing.

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
   references, longitudinal divergence metrics, and recovery-oriented indicators.

``Model_C``
   Package: ``iabm_semantics``

   Purpose: semantic and reliability-informed interpretation of behavioral
   sequences into operating modes, working modes, incident families, life regimes,
   and recovery regimes.

``Model_D``
   Package: ``iabm_incidents``

   Purpose: explicit scoring, segmentation, family assignment, recovery
   assessment, registry alignment, and occurrence summaries for longitudinal
   incidents.

Updated pipeline
----------------

The repository supports a progressive hierarchical workflow:

.. code-block:: text

   telemetry + contextual signals
             |
             v
   Model_A: state inference
             |
             v
   Model_B: runs -> active sequences -> nominal comparison
             |                      |
             |                      +--> recurrence / persistence /
             |                           duration drift / recovery
             v
   Model_C: semantic and reliability-informed interpretation
             |
             +--> operating modes
             +--> incident families
             +--> life regimes
             +--> recovery regimes
             v
   Model_D: longitudinal incident processing
             |
             +--> window scores
             +--> detected episodes
             +--> recovery assessment
             +--> registry-based evaluation
             +--> occurrence summaries

Relation to the Paper
---------------------

The repository mirrors the main empirical workflow reported in the updated
manuscript.

- ``Model_A`` supports operational-state inference.
- ``Model_B`` supports behavioral sequence extraction and longitudinal metrics.
- ``Model_C`` supports reliability-informed semantic interpretation.
- ``Model_D`` supports episode detection, evaluation, and occurrence-oriented summaries.

The public codebase should therefore be understood not only as a software
artifact, but as a reproducibility resource for a longitudinal and
family-oriented monitoring pipeline.

Where to Start
--------------

For most users, the recommended reading order is:

- :doc:`overview`
- :doc:`incident_taxonomy`
- :doc:`longitudinal_validation`
- :doc:`reliability_interpretation`
- :doc:`reproducibility`
- :doc:`model_a`
- :doc:`model_b`
- :doc:`model_c`
- :doc:`model_d`

Documentation Strategy
----------------------

This Sphinx configuration relies on ``autodoc`` and ``napoleon`` so that
Google-style docstrings embedded in the codebase become the primary API
reference. The intent is to keep implementation and formal documentation
synchronized while also exposing the conceptual transition from states and modes
toward incidents, episodes, and reliability-informed interpretation.


Mermaid Workflow View
---------------------

.. mermaid::

   flowchart LR
       T[Telemetry] --> MA[Model_A]
       MA --> MB[Model_B]
       MB --> MC[Model_C]
       MB --> RG[Registry Builder]
       MC --> MD[Model_D]
       RG --> MD
       MD --> R[Research and operational reports]
