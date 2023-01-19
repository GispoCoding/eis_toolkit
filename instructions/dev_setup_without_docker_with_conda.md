# Development without docker with `conda`

If you do not have docker, you can setup your local development
environment with `conda`.

## Prerequisites

A recent version of `conda` must be installed. See:

-   <https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html>

## Set up a local `conda` environment

*Run all commands in the repository root unless instructed otherwise*

1.  Install dependencies and create a new `conda` environment using the
    provided `environment.yml` file. The environment name is defined in
    `environment.yml` (eis_toolkit).

``` shell
conda env create -f environment.yml
```

2.  Activate the environment.

``` shell
conda activate eis_toolkit
```

3.  With the environment active, the package and all its dependencies
    should be available.

4.  E.g. run `pytest` to verify that the test suite works with your
    installation.

## Further info

You can add your own packages to the environment as needed. E.g.
`jupyterlab`:

``` shell
conda install -n eis_toolkit -c conda-forge jupyterlab 
```
