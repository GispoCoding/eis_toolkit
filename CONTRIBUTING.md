# Contributing

When contributing to this repository, please first discuss the changes you wish to make via an issue.
If an issue for the changes you intend to make does not exist, create one.

## General contributing workflow

1. Raise an issue for the changes you wish to make (or start working on a pre-existing issue).
2. Make a branch for your changes. Base it on the master branch.
3. Do all
- editing
- formatting

and

- testing

on the issue-specific branch. Commit only to that branch, do not edit the master branch directly.

4. Once you have something working, make sure it has the appropriate [documentation](./instructions/generating_documentation.md), and [tests](./instructions/testing.md)
5. 
6. Create a pull request to merge your branch into the master. In it, summarize your changes.
Assign a reviewer / reviewers for the pull request.

### Formatting

In order to guarantee consistent coding style, a bunch of different linters and formatters have been brought into use.
For more convenient user experience, running
- mypy
- flake8
- black

and

- isort

have been combined into one task. The task can be executed from container's command line with

``` console
invoke lint
```

Possible errors will be printed onto the command line. **Please** fix them before committing anything!

### Testing

Running tests defined in /eis_toolkit/tests folder can be done e.g. by executing

```console
pytest
```

in the container's command line.

**Please** add a new test file every time you wish to add new function to the toolkit!
