# Generating dependency license list

The generated list of dependency licenses exists at
`docs/dependency_licenses.md`. This has been generated with the
[`pip-licenses`](https://pypi.org/project/pip-licenses/) command-line tool.
This requires that it is installed in the same virtual environment as the
packages it is trying to find licenses of. The process of updating the file is
manual and should be done when new packages are added to `pyproject.toml` or
`poetry.lock` is updated. However, package updates rarely include license
changes.

Start by cleaning your local `poetry` environment:

``` bash
poetry env remove python
```

Add `pip-licenses` to `pyproject.toml` and `poetry.lock`:

``` bash
poetry add pip-licenses --lock
```

Then install only the main dependencies (now including `pip-licenses`) in
pyproject.toml (not dev dependencies):

``` bash
poetry install --with main
# Old poetry version: poetry install --no-dev
```

`pip-licenses` is now available in the poetry environment:

``` bash
poetry run pip-licenses --order=license --format=markdown > docs/dependency_licenses.md
```

Commit only the `docs/dependency_licenses.md` file. You can restore
`pyproject.toml` and `poetry.lock` files with `git restore pyproject.toml
poetry.lock` -command.

To clean your local `poetry` environment:

``` bash
# Remove poetry environment from the current project
poetry env remove python
# Install main and dev packages from pyproject.toml again
poetry install
```
