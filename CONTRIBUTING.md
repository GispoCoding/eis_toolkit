# Contributing

When contributing to this repository, please first discuss the changes you wish to make via an issue.
If an issue for the changes you intend to make does not exist, create one.

## General contributing workflow

1. Raise an issue for the changes you wish to make (or start working on a pre-existing issue).
2. Make a feature branch for your changes.
> Name the branch as add_<name_of_the_function_to_be_added>
3. Base your feature branch on the master branch.
> Remember to
``` console
git pull
```
before checking out to a new branch.

4. Do all
- editing
- formatting

and

- testing

on the issue-specific branch. Commit only to that branch, do not edit the master branch directly.

5. Once you have something working, make sure your commits are according to the desired coding style and that your branch contains appropriate documentation and tests.

6. Create a pull request to merge your branch into the master. In it, summarize your changes.
Assign a reviewer / reviewers for the pull request.

## Terminology and general coding principles

1. Packages

The folders at eis_toolkit/eis_toolkit are called packages. The initial division to packages already exist. Feel free to suggest modifications to the
current package division via creating an issue for it! Note that the packages can split up into sub packages if needed. Subpackages' names should also represent the main purpose of the modules belonging to the particular subpackage.

2. Modules

Module names come from the names of the .py files containing function declarations. You will need to create a new python file for each functionality. The name of the file containing the function declaration(s) for providing the functionality will be essentially the same as the function’s name but instead of the basic form we will be using –ing form.

- Try to create modules in a way that each module contains only one functionality. Split this functionality into two function declarations: one for external use and one (the core functionality) for internal use. See e.g. implementation of [clipping functionality](./eis_toolkit/raster_processing/clipping.py) for reference.

3. Functions
 
Name each function according to what it is supposed to do. Try to express the purpose as simplistic as possible. In principle, each function should be creted for executing one task. We prefer modular structure and low hierarchy by trying to avoid nested function declarations. It is highly recommended to call other functions for executing sub tasks.

**Example (packages, modules & functions):**

Create a function which clips a raster file with polygon -> name the function as clip. Write this function declaration into a new python file with name clipping.py inside of the eis_toolkit/eis_toolkit/raster_processing folder.

4. Classes

A class can be defined inside of a module or a function. Class names should begin with a capital letter and follow the CamelCase naming convention: if a class name contains multiple words, the spaces are simply ignored and each separate word begins with capital letters. 

> If you create new custom exception classes, add them directly into eis_toolkit/eis_toolkit/exceptions.py file.

5. Variables

Avoid using global variables.

6. Docstrings and code comments 

For creating docstrings, we rely on google convention (see section 3.8 in [link](https://google.github.io/styleguide/pyguide.html) for more detailed instructions). Let’s try to minimize the amount of code comments. Well defined docstrings should do most of the job with clear code structure.

## Naming policy

General guidelines about naming policy (applies to package, module, function, class and variable names):
- all names should be given in English
- avoid too cryptic names by using complete words
- if the name consists of multiple words, replace space with underscore character (_)
- do not include special characters, capital letters or numbers into the names unless in case of using numbers in variable names and there is an unavoidable need for it / using numbers significantly increases clarity

## Code style

In order to guarantee consistent coding style, a bunch of different linters and formatters have been brought into use.

> **Please** note that running code style checks is not optional!

For more convenient user experience, running
- mypy (checks type annotations)
- flake8 (checks the compliance to PEP8)
- black (formats the code)

and

- isort (sorts the import statements)

have been combined into one task. The task can be executed from container's command line with

``` console
invoke lint
```

Possible errors will be printed onto the command line.

**Please** fix them before committing anything!

- Note that sometimes the best way to "fix" an error is to ignore that particular error code for some spesific line of code. However, be conscious on when to use this approach!

## Testing

Creating and executing tests improves code quality and helps to ensure that nothing gets broken
after merging the pull request.

> **Please** note that creating and running tests is not optional!

Create a new python file into eis_toolkit/tests folder every time you wish to add a new functionality
into eis_toolkit. Name that file as <name_of_the_function_to_be_added>_test.py.

In this test file you can declare all test functions related to the new function. Add a function at least for testing that 
- the new function works as expected (in this you can utilize other software for generating the reference solution) 
- custom exception class errors get fired when expected

You can utilize both local and remote test data in your tests.
For more information about creating and running tests, take a look at [test instructions](./instructions/testing.md).

## Documentation

When adding (or editing) a module, function or class, **please** make sure the documentation stays up-to-date!
For more information, take a look at [documentation instructions](./instructions/generating_documentation.md).
