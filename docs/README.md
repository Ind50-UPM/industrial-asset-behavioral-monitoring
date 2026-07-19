# Sphinx Documentation

This directory contains the Sphinx configuration, narrative pages, and the
GitHub Pages-compatible publication flow for the Industrial Asset Behavioral
Monitoring repository.

## Scope

The site documents:

- the four modelling layers (`Model_A` to `Model_D`),
- the month-wise batch utilities used in large historical runs,
- the longitudinal reliability interpretation proposed by the project,
- the architectural relationships between classes, pipelines, and artefacts.

## Documentation flow

```mermaid
flowchart LR
    A[Source code with typed Google docstrings] --> B[Sphinx autodoc and narrative rst pages]
    B --> C[docs/_build/html]
    C --> D[docs/ root via make github-pages]
    D --> E[GitHub Pages served from /docs]
```

## Build locally

Install the documentation requirements and build the HTML site:

```bash
cd docs
pip install -r requirements.txt
make clean
make html
```

The intermediate HTML build is written to:

```text
docs/_build/html/
```

## Publish for GitHub Pages

To leave the generated HTML directly inside `docs/` so the repository is
compatible with GitHub Pages configured to serve from `/docs`, run:

```bash
cd docs
make github-pages
```

This target:

- builds the Sphinx site,
- copies the generated HTML artefacts into `docs/`,
- creates `.nojekyll` so GitHub Pages serves `_static/` and related folders.

## Mermaid support

The documentation supports Mermaid diagrams when `sphinxcontrib-mermaid` is
installed. The repository already uses Mermaid for:

- architecture views,
- sequence diagrams,
- class collaboration snapshots,
- publication workflow descriptions.
