# Development with `conda`

If you do not have docker, you can setup your local development environment
with `conda`. If you encounter issues where code that runs on `Docker` does not
run in the `conda` environment or produces different results, post such issues on GitHub.

## Prerequisites

A recent version of `conda` must be installed. See:

-   <https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html>

The `environment.yml` dependency specification is tested in `eis_toolkit` with
the `libmamba` solver instead of the default. **If you encounter installation
issues following this guide further**, especially on Windows, you can enable the
`libmamba` solver globally(!) as follows:

``` shell
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

See <https://conda.github.io/conda-libmamba-solver/getting-started/> for
further info.

## Set up a local `conda` environment

*Run all commands in the repository root unless instructed otherwise*

1.  Install dependencies and create a new `conda` environment using the
    provided `environment.yml` file. The environment name is defined in
    `environment.yml` (eis_toolkit).

``` shell
conda env create -f environment.yml
# You can overwrite an existing environment named eis_toolkit with the --force flag
conda env create -f environment.yml --force
```

2.  Activate the environment.

``` shell
conda activate eis_toolkit
```

3.  With the environment active, the package and all its dependencies
    should be available for execution.

4.  E.g. run `pytest` to verify that the test suite works with your
    installation. If not, you should first verify your installation and
    secondly make a GitHub issue if you cannot figure out the problem.

## Further info

You can add your own packages to the environment as needed. E.g.
`jupyterlab`:

``` shell
# -c conda-forge specifies the conda-forge channel, which is recommended
conda install -n eis_toolkit -c conda-forge jupyterlab 
```
