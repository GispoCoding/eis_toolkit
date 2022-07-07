====
EIS Toolkit
====

Related to EIS Horizon EU project.

Testing
====

This repository contains dummy files and functions for testing purposes. If you want to
just test the installation of eis_toolkit and use it, follow the **For users** section.
If you want to set up a local development environment for contributing, read also the
**For developers** section.

Instructions for testing of eis_toolkit's functionalities:

For users
----

1. Navigate to the Releases section and download tar.gz or .zip file
2. Find out the path where python packages (which QGIS uses) are installed e.g. by opening QGIS
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

4. Go back to QGIS's Python console and run e.g.

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

1. Install `poetry <https://python-poetry.org/>`_ acoording to your platform's `instructions <https://python-poetry.org/docs/#installation>`_

2. Get your local copy of the repository

.. code-block:: shell

   git clone https://github.com/GispoCoding/eis_toolkit.git

Set up a local environment
^^^^

*Run all commands in the repository root unless instructed otherwise*

1. Install dependencies and create a virtual environment

.. code-block:: shell

   poetry install

2. To use the virtual environment you can either enter it with

.. code-block:: shell

   poetry shell

or prefix your normal shell commands with

.. code-block:: shell

   poetry run

Test the effect of your changes
^^^^

Without QGIS
""""

**From the command line**: You can run your code from the command line with the virtual
environment by

1. Running (inside of the VE)

.. code-block:: shell

   pip install eis_toolkit


2. Opening VE's python console with

.. code-block:: shell

   python

3. Running e.g.

.. code-block:: python

   from eis_toolkit.dependency_test.dummy import test_function

   test_function(12,2)

**With JupyterLab**: You can also use `JupyterLab <https://jupyterlab.readthedocs.io/en/stable/>`_ for testing of more complicated functionalities
(for example if you need to store intermediate results in active memory). Launch JupyterLab with

.. code-block:: shell

   poetry run jupyter lab

The notebooks are found in the `notebooks/` directory. You can import and use
eis_toolkit's functions in these notebooks in the same way as you normally would.

*NOTE.* Any of the functionalities utilizing GDAL functions will not work when testing
eis_toolkit outside of QGIS's Python console unless you have separately taken care of
installing GDAL library. In order to install GDAL it is necessary to have libgdal and
its development headers installed. This makes it impossible for us to handle GDAL dependency
automatically. Note also that GDAL's installation procedure varies a lot between different
operating systems.

With QGIS
""""

1. Find out the path where python packages (which QGIS uses) are installed e.g. by opening QGIS
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

4. Now eis_toolkit is available in QGIS's python. You can, for example, go back to
   QGIS's Python console and run

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

1. Modify mkgendocs.yml by adding a new page to pages section:

- Give name to a new page, e.g. new_class.md
- Give path to the corresponding python file, e.g. eis_toolkit/new_class.py
- Give list of the function names to be documented

2. Navigate to the root directory level (the same level where mkgendocs.yml file is located)
   and run

.. code-block:: shell

    gendocs --config mkgendocs.yml

3. Run

.. code-block:: shell

    mkdocs serve

4. Go to http://127.0.0.1:8000/

If you **just** want to take a look at the documentation (not to modify it),
act according to **For developers** section's Prerequisites and Set up of a local development
environment and execute steps 3 and 4.
