"""Public package surface for Model_C semantic interpretation.

The package exports only the typed semantic interpreter and its rule and output
structures so downstream incident-processing layers can rely on a stable
semantic contract.
"""

from .semantics import (
    SemanticAssignment,
    SemanticModeInterpreter,
    SemanticRule,
)

__all__ = [
    "SemanticAssignment",
    "SemanticModeInterpreter",
    "SemanticRule",
]
