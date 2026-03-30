Reproducibility
===============

This repository implements a layered workflow for industrial asset behavioral monitoring based on operational-state inference, behavioral sequence analysis, and semantic contextualization. It accompanies the research work on data-driven identification of industrial asset operating modes for predictive maintenance in IIoT-enabled industrial environments.

The repository is organized to support both methodological understanding and practical reproducibility. Its structure mirrors the progression described in the paper: from synchronized observations, to operational states, to behavioral sequences, and finally to semantically contextualized operational modes.

Repository Architecture
-----------------------

The implementation is organized into three package-oriented modeling layers.

``Model_A``
   Package: ``iabm``

   Purpose: supervised identification of elementary operating states from industrial analog and digital signals.

``Model_B``
   Package: ``iabm_behavior``

   Purpose: extraction of contiguous runs, active sequences, nominal behavioral references, and anomaly-oriented comparison metrics.

``Model_C``
   Package: ``iabm_semantics``

   Purpose: semantic interpretation of behavioral sequences into operating modes, working modes, and anomaly-aware semantic assignments.

Layered Workflow
----------------

The repository supports a progressive analysis workflow:

1. ``Model_A`` transforms synchronized industrial observations into discrete operating-state predictions.
2. ``Model_B`` organizes those predicted states into runs and behavioral sequences, and supports comparison against nominal references.
3. ``Model_C`` interprets the extracted behavioral sequences through higher-level semantic categories such as operating modes and working modes.

This organization is intended to preserve the logic of the monitoring pipeline while keeping each layer independently usable.

Relation to the Paper
---------------------

The repository mirrors the main empirical workflow reported in the manuscript.

- Data preprocessing and state-identification tasks are associated with ``Model_A``.
- Behavioral sequence extraction and anomaly-oriented comparison are associated with ``Model_B``.
- Semantic contextualization of behaviors into higher-level operational categories is associated with ``Model_C``.

The public codebase is therefore not only a software artifact, but also a reproducibility resource for the experimental pipeline discussed in the paper.

Where to Start
--------------

For most users, the recommended reading order is:

- :doc:`model_a`
- :doc:`model_b`
- :doc:`model_c`

If the goal is to understand the repository at a high level before going into the API details, this page should be read first, followed by the model-specific pages.

Documentation Strategy
----------------------

This Sphinx configuration relies on ``autodoc`` and ``napoleon`` so that Google-style docstrings embedded in the codebase become the primary API reference. The intent is to keep implementation and formal documentation synchronized while providing an accessible entry point to the structure and purpose of the repository.

