"""Sphinx configuration for the Industrial Asset Behavioral Monitoring docs.

This configuration exposes the four modelling layers, their batch utilities,
and the cross-layer architectural narratives used during longitudinal runs.
The project intentionally keeps API reference generation close to the source
code so that typed Google-style docstrings become the canonical technical
specification for developers and researchers.
"""

from __future__ import annotations

import sys
from importlib.util import find_spec
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PATHS = [
    REPO_ROOT / "src",
    REPO_ROOT / "src" / "Model_A",
    REPO_ROOT / "src" / "Model_B",
    REPO_ROOT / "src" / "Model_C",
    REPO_ROOT / "src" / "Model_D",
]

for package_path in PACKAGE_PATHS:
    sys.path.insert(0, str(package_path))


class MermaidDirective(Directive):
    """Fallback Mermaid directive used when the optional extension is absent.

    Notes:
        The implementation emits a raw HTML ``<pre class="mermaid">`` block so
        Mermaid diagrams remain visible in generated HTML even when
        ``sphinxcontrib-mermaid`` is not installed. When the real extension is
        available, this fallback is not used.
    """

    has_content = True

    def run(self):
        code = "\n".join(self.content)
        html = f'<pre class="mermaid">{code}</pre>'
        return [nodes.raw('', html, format='html')]


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
try:
    _mermaid_spec = find_spec("sphinxcontrib.mermaid")
except ModuleNotFoundError:
    _mermaid_spec = None

if _mermaid_spec is not None:
    extensions.append("sphinxcontrib.mermaid")

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_inherit_docstrings = False
autodoc_preserve_defaults = True
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False

mermaid_version = "11.4.1"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
suppress_warnings = ["ref.ref"]

html_theme = "furo" if find_spec("furo") is not None else "alabaster"
html_static_path = ["_static"]


def setup(app):
    """Register additional static assets used across the documentation site.

    Args:
        app: Active Sphinx application instance.

    Returns:
        ``None``. The function mutates the Sphinx application in place.

    Notes:
        The custom stylesheet currently centralizes small presentational tweaks
        while Mermaid support is handled through the optional extension listed
        above when the dependency is installed in the docs environment.
    """
    app.add_css_file("custom.css")
    if _mermaid_spec is None:
        app.add_js_file("https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js")
        app.add_js_file(None, body="document.addEventListener(\'DOMContentLoaded\', function () { if (window.mermaid) { mermaid.initialize({ startOnLoad: true }); } });")
        app.add_directive("mermaid", MermaidDirective)
