# Contributing

When contributing to this repository, please first discuss the changes you wish to make via an issue.
If an issue for the changes you intend to make does not exist, create one.

## General contributing workflow

1. Raise an issue for the changes you wish to make (or start working on a pre-existing issue).
2. Make a feature branch for your changes.

> Name the branch as <issue_number>-add-<name_of_the_function_to_be_added> or something as descriptive

3. Base your feature branch on the master branch.

> Remember to

```bash
git pull
```

before checking out to a new branch.

4.  Do all editing formatting and testing on the issue-specific branch.
    Commit only to that branch, do not edit the master branch directly.

5.  Once you have something working, make sure your commits are
    according to the desired coding style and that your branch contains
    appropriate documentation and tests.

6.  Create a pull request (PR) to merge your branch into the master. In
    it, summarize your changes. Assign a reviewer / reviewers for the
    PR.

## Terminology and general coding principles

1. Packages

The folders inside `./eis_toolkit` are called subpackages. Feel free to suggest
modifications to the current subpackage division via creating an issue for it!
Note that the subpackages can split up into more subpackages if needed.

2. Modules

Module names come from the names of the .py files containing function
declarations. You will need to create a new python file for each functionality.
The name of the file containing the function declaration(s) for providing the
functionality will be essentially the same as the function’s name but instead
of the basic form use –ing form if it makes sense.

-   Try to create modules in a way that each module contains only one
    functionality. Split this functionality into two function
    declarations: one for external use and one (the core functionality)
    for internal use. See e.g. implementation of [clipping
    functionality](./eis_toolkit/raster_processing/clipping.py) for
    reference.

1. Functions

Name each function according to what it is supposed to do. Try to
express the purpose as simplistic as possible. In principle, each
function should be created for executing one task. We prefer modular
structure and low hierarchy by trying to avoid nested function
declarations. It is highly recommended to call other functions for
executing sub tasks.

**Example (packages, modules & functions):**

Create a function which clips a raster file with polygon -\> name the
function as clip. Write this function declaration into a new python file
with name clipping.py inside of the
`eis_toolkit/raster_processing` folder.

4. Classes

A class can be defined inside of a module or a function. Class names
should begin with a capital letter and follow the CamelCase naming
convention: if a class name contains multiple words, the spaces are
simply ignored and each separate word begins with capital letters.

When implementing the toolkit functions, create classes only when they
are clearly beneficial.

> If you create new custom exception classes, add them directly into
> `eis_toolkit/exceptions.py` file.

5. Variables

Avoid using global variables. Name your variables clearly for code
maintainability and to avoid bugs.

6. Docstrings and code comments 

For creating docstrings, we rely on google convention (see section 3.8
in [link](https://google.github.io/styleguide/pyguide.html) for more
detailed instructions). Let's try to minimize the amount of code
comments. Well defined docstrings should do most of the job with clear
code structure.

## Naming policy

General guidelines about naming policy (applies to package, module,
function, class and variable names):

-   all names should be given in English

-   avoid too cryptic names by using complete words

-   if the name consists of multiple words, use snake_case so replace
    space with underscore character (\_) (CamelCase is used for classes
    as an exception to this rule)

-   do not include special characters, capital letters or numbers into
    the names unless in case of using numbers in variable names and
    there is an unavoidable need for it / using numbers significantly
    increases clarity

## Code style

### pre-commit

> Note that pre-commit was added as the primary style check tool later
> in the project and you need to install and enable it manually!

The repository contains a `.pre-commit-config.yaml` file that has configuration
to run a set of [`pre-commit`](https://pre-commit.com) hooks. As the name
implies, they run before committing code and reject commits that would include
code that is not formatted or contains linting errors. `pre-commit` must be
installed on your system **and** the hooks must be enabled within your local
copy of the repository to run.

To install `pre-commit` on Debian or Ubuntu -based systems with `apt` as
the package manager you should be able to run: 

```bash
apt update
apt install pre-commit
```

Alternatively, it can be installed with the system installation of `Python`:

```bash
pip install pre-commit
```

Visit the `pre-commit` website for more guidance on various system installation
methods (<https://pre-commit.com>).

To enable the hooks locally, enter the directory with your local
version of `eis_toolkit`, and run:

```bash
pre-commit install
```

Within this local repository, before any commits, the hooks should now run.
Note that the `black` formatting hook will modify files and consequently, the
edits by `pre-commit` will be unstaged. Stage the changes to add them back to
the commit. 

To disable the hooks and allow commits even with errors pointed out by
`pre-commit`, you can add the `--no-verify` option to the `git` command-line:

```bash
git commit -m "<message>" --no-verify
```

However, this is not recommended and you should instead fix any issues pointed
out by `pre-commit`.

You can also run the hooks without committing on all files. Make sure you save
any text changes as `pre-commit` can modify unformatted files:

```bash
pre-commit run --all-files
```

## Testing

Creating and executing tests improves code quality and helps to ensure
that nothing gets broken after merging the PR.

> **Please** note that creating and running tests is not optional!

Create a new python file into eis_toolkit/tests folder every time you wish to add a new functionality
into eis_toolkit. Name that file as <name_of_the_function_to_be_added>_test.py.

In this test file you can declare all test functions related to the new
function. Add a function at least for testing that

-   the new function works as expected (in this you can utilize other
    software for generating the reference solution)
-   custom exception class errors get fired when expected

You can utilize both local and remote test data in your tests. For more
information about creating and running tests, take a look at [test
instructions](./instructions/testing.md).

## Documentation

When adding (or editing) a module, function or class, **please** make
sure the documentation stays up-to-date! For more information, take a
look at [documentation
instructions](./instructions/generating_documentation.md).

## Creating a PR

Final step in your workflow is to create a PR from the feature
branch you have been developing. Note that in this repository we have configured a workflow
which executes pytest every time anyone creates a PR. Most often there is nothing you
need to do, the check begins automatically and produces an output for you
whether the tests got passed and the feature branch is ready to get merged or not. If errors
do emerge, take a look into the Details section to get a more informative
understanding of the problem.

If you act according to the workflow stated in this document, these PR checks
should always pass since you have already run pytest through before committing :)
The purpose of this automatic workflow is to double check that nothing gets broken by merge.

However, **IF** you make changes to the dependencies of the repository (i.e.
edit pyproject.toml file), you need to update `poetry.lock` and
`environment.yaml` files in order to the workflow tests to stay up-to-date. You
can update the `poetry.lock` file by running the following commands:

```bash
# Not required if you added a package with poetry add command
poetry lock --no-update
```

and committing the new version of the particular file into your feature branch.
Dependencies in `environment.yaml` need to be kept up to date manually by
including the same package, which was added to `pyproject.toml`, in
`environment.yaml`. Please note that this file is only used for GitHub
workflows, otherwise we utilize poetry for dependency handling.

## Recent changes

Some changes have been made to the style guide:

-   Use `numbers.Number` as the type when both floats and integers are
    accepted by functions:

```python
from numbers import Number

def func(int_or_float: Number):
    ...
- Write comments to exceptions:
```python
raise InvalidParameterValueException(f"Window size is too small: {height}, {width}.")
```

-   Use beartype's decorator for automatic function argument type
    checking and import types from `beartype.typing` if a warning is
    raised by beartype on imports from `typing`:

```python
from beartype import beartype
from beartype.typing import Sequence

@beartype
def my_function(parameter_1: float, parameter_2: bool, parameter_seq: Sequence):
- Don't put parameter types and return variable name into function docstring:
```python
# OLD
    """Description here.

    Args:
        parameter_1 (float): A parameter.
        parameter_2 (bool): A parameter.

    Returns:
        return_value (bool): The return value.
    """

# NEW
    """Description here.

    Args:
        parameter_1: A parameter.
        parameter_2: A parameter.

    Returns:
        The return value.
    """
```

## Developer's checklist

Here are some things to remember while implementing a new tool:

-   Create an issue **before or when you start** developing a
    functionality
-   Adhere to the style guide
    -   Look at existing implementations and copy the form
    -   Enable pre-commit and fix style/other issues according to the
        error messages
-   Remember to use typing hints
-   Write tests for your functions
-   Add a .md file for you functionality
-   If you think the tool you are developing could use a separate
    general utility function, make an issue about this need before
    starting to develop it on your own. Also check if a utility function
    exists already
-   Remember to implement only the minimum what is required for the
    tool! With data functions, you can usually assume file
    reading/writing, nodata handling and other such processes are done
    before/after executing your tool


## Additonal instructions

Here are some additional instructions related to the development of EIS toolkit:
- [Testing your changes](./instructions/testing.md)
- [Generating documentation](./instructions/generating_documentation.md)
- [Using jupyterlab](./instructions/using_jupyterlab.md)
