====
EIS Toolkit
====

Related to EIS Horizon EU project.

Testing
====

This repository contains dummy files and functions for testing purposes. If you want to
just test the installation of eis toolkit and use it, follow the **For users** section.
If you want to set up a local development environment for contributing, read also the
**For developers** section.

Instructions for testing of eis_toolkit's functionalities:

For users
----

1. Navigate to the Releases section and download tar.gz or .zip file
2. Find out the path where python packages QGIS uses are installed e.g. by opening QGIS
   and navigating to Python console and executing

.. code-block:: python

   import imp

   str.replace(imp.find_module('numpy')[1], '/numpy', '')

3. Open terminal and execute

.. code-block:: shell

   pip install --target=<path_found_in_step_2> -U <path_to_eis_toolkit-dummy_test.tar.gz>

or

.. code-block:: shell

   pip install --target=<path_found_in_step_2> -U <path_to_eis_toolkit-dummy_test.zip>

4. Go back to QGIS's python console and run e.g.

.. code-block:: python
 
   from eis_toolkit.dependency_test.dummy import test_function

   test_function(12,2)

or

.. code-block:: python

   from eis_toolkit.dependency_test.dummy_sklearn import sk_mean

In both cases, a result should appear into the QGIS's Python console's output window.

For developers
----

Prerequisites
^^^^

1. Install `poetry <https://python-poetry.org/>`_ as per your platform's `instructions <https://python-poetry.org/docs/#installation>`_
2. Get your local copy of the repository

.. code-block:: shell

   git clone https://github.com/GispoCoding/eis_toolkit.git

Setup local environment
^^^^

*Run all commands in the root of the repository unless otherwise directed*

1. Install dependencies and create a virtual environment 

.. code-block:: shell

   poetry install

2. To use the virtual environment you can either enter it:

.. code-block:: shell

   poetry shell

Or prefix your normal shell commands with:

.. code-block:: shell

   poetry run

Test your changes
^^^^

Without QGIS
""""

**From the command line**: You can run your code from the command-line with the virtual
environment (as shown above)

**With jupyter lab**: You can also use jupyterlab for more complicated testing (for
example if you need results stored in active memory). Launch jupyterlab with:

.. code-block:: shell

   poetry run jupyter lab

The notebooks are found in the `notebooks/` directory. You can import and use
eis_toolkit's functions in these notebooks as you normally would.

With QGIS
""""

1. Find out the path where python packages QGIS uses are installed e.g. by opening QGIS
   and navigating to Python console and executing

.. code-block:: python

   import imp

   str.replace(imp.find_module('numpy')[1], '/numpy', '')

2. Build eis_toolkit

.. code-block:: shell

   poetry build

3. Install eis_toolkit to the location found in step 1

.. code-block:: shell

   pip install --target=<path_found_in_step_1> -U <path_to_cloned_eis_toolkit_folder>

4. Now eis_toolkit is available to QGIS's python. You can, for example, Go back to
   QGIS's python console and run e.g.

.. code-block:: python

   from eis_toolkit.dependency_test.dummy import test_function

   test_function(12,2)

or

.. code-block:: python

 from eis_toolkit.dependency_test.dummy_sklearn import sk_mean

A result should appear into the QGIS's Python console's output window.

Documentation
====

In case you add a new class or function into the toolkit, please update the documentation site!

1. Modify mkgendocs.yml by adding a new page to pages section

- Give name to a new page, e.g. new_class.md
- Give path to the corresponding python file, e.g. eis_toolkit/new_class.py
- Give list of the functions to be documented

2. Navigate to the root directory level (the same level where mkgendocs.yml file is located)
   and run

.. code-block:: shell

    gendocs --config mkgendocs.yml

3. Run

.. code-block:: shell

    mkdocs serve

4. Go to http://127.0.0.1:8000/

If you **just** want to take a look at the documentation (not to modify it),
clone this repository and execute steps 3 and 4.
