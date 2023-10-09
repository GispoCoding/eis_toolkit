# Development with Poetryr
If you do not have docker, you can setup your local development environment as a python virtual environment using Poetry.

## Prerequisites

0. Make sure that GDAL's dependencies

- libgdal (3.5.1 or greater)
- header files (gdal-devel)

are satisfied. If not, install them.

1. Install [poetry](https://python-poetry.org/>) according to your platform's
[instructions](https://python-poetry.org/docs/#installation>).

## Set up a local environment

*Run all commands in the repository root unless instructed otherwise*

1. Install dependencies and create a virtual environment

```shell
poetry install
```

2. To use the virtual environment you can either enter it with

```shell
poetry shell
```

or prefix your normal shell commands with

```shell
poetry run
```

If you want to use jpyterlab see the [instructions](./using_jupyterlab.md)

