"""Sphinx configuration for the Industrial Asset Behavioral Monitoring docs."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PATHS = [
    REPO_ROOT / "src" / "Model_A",
    REPO_ROOT / "src" / "Model_B",
    REPO_ROOT / "src" / "Model_C",
]

for package_path in PACKAGE_PATHS:
    sys.path.insert(0, str(package_path))


project = "Industrial Asset Behavioral Monitoring"
author = "Antonio Bello-García, Javier Villalba-Díez, Ana González-Marcos, Joaquín Ordieres-Meré"
copyright = "2026, Industrial Asset Behavioral Monitoring contributors"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_static_path = ["_static"]
