Overview
========

Repository Architecture
-----------------------

The implementation is organized into three package-oriented modeling layers.

``Model_A``
   Package: ``iabm``

   Purpose: supervised identification of elementary operating states from
   industrial analog and digital signals.

``Model_B``
   Package: ``iabm_behavior``

   Purpose: extraction of contiguous runs, active sequences, nominal behavioral
   references, and anomaly-oriented comparison metrics.

``Model_C``
   Package: ``iabm_semantics``

   Purpose: semantic interpretation of behavioral sequences into operating
   modes, working modes, and anomaly-aware semantic assignments.

Documentation Strategy
----------------------

This Sphinx configuration relies on ``autodoc`` and ``napoleon`` so the Google
style docstrings embedded in the codebase become the primary API reference.
The intent is to keep implementation and formal documentation synchronized.
