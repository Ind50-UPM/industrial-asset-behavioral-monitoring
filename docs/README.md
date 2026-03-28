# Sphinx Documentation

This directory contains a minimal Sphinx configuration for the three package
layers currently implemented in the repository:

- `iabm`
- `iabm_behavior`
- `iabm_semantics`

## Build

If Sphinx is available in the active environment, build the HTML site with:

```bash
cd docs
make html
```

The generated site will be written to:

```text
docs/_build/html/
```

## Continuous Integration

The repository also includes a GitHub Actions workflow that:

- installs the three package environments with Poetry
- runs the `Model_A`, `Model_B`, and `Model_C` test suites
- builds the Sphinx site from this directory
