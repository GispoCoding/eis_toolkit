# Development with Poetry

If you do not have docker, you can setup your local development environment as a python virtual environment using Poetry.

## Prerequisites

0. Make sure that GDAL's dependencies

-   libgdal (3.5.1 or greater)
-   header files (gdal-devel)
-   See `.github/workflows/tests.yml` to see how these dependencies are
    installed in CI

are satisfied. If not, install them.

1. Install [poetry](https://python-poetry.org/>) according to your platform's
[instructions](https://python-poetry.org/docs/#installation>).

## Set up a local environment

*Run all commands in the repository root unless instructed otherwise*

1. Install dependencies and create a virtual environment

```bash
poetry install
```

2. To use the virtual environment you can either enter it with

```bash
poetry shell
```

or prefix your normal shell commands with

```bash
poetry run
```

If you want to use jupyterlab see the [instructions](./using_jupyterlab.md)

