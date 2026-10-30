# Documentation

This directory contains the sources for the VMEC++ documentation. The HTML version is built with Sphinx.

To build the documentation locally run:

```shell
pip install -e .[docs]
sphinx-apidoc --module-first --no-toc --force --separate -o docs/api src/vmecpp/
sphinx-build docs html_docs
```

The GitHub Actions workflow `docs.yaml` builds and deploys the documentation when changes are pushed to the `main` branch.
