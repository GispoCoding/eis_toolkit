====
EIS Toolkit
====

Related to EIS Horizon EU project.

Initializations for developers
====

Install poetry
----

<Eemil>

Finally, clone this repository and start contributing!

Testing
====

This repository contains dummy files and functions for testing purposes. Instructions
for testing of eis_toolkit's functionalities:

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

This repository contains dummy files and functions for testing purposes. Instructions
for testing of eis_toolkit's functionalities:

1. Check out Initializations for developers section!

2. Run

.. code-block:: shell

 poetry install

 poetry shell


3. Find out the path where python packages QGIS uses are installed e.g. by opening QGIS
and navigating to Python console and executing

.. code-block:: python

 import imp

 str.replace(imp.find_module('numpy')[1], '/numpy', '')

3. Run

.. code-block:: shell

 poetry build

4. Open terminal and execute

.. code-block:: shell

 pip install --target=<path_found_in_step_2> -U <path_to_cloned_eis_toolkit_folder>

5. Go back to QGIS's python console and run e.g.

.. code-block:: python

 from eis_toolkit.dependency_test.dummy import test_function

 test_function(12,2)

or

.. code-block:: python

 from eis_toolkit.dependency_test.dummy_sklearn import sk_mean

A result should appear into the QGIS's Python console's output window.

For both
----

It is possible to test some functionalities of eis_toolkit also outside of the
QGIS environment.

 Note that any of the GDAL functions won't work in the notebooks, but everything else should work!

<Eemil: write here instructions for testing eis_toolkit with notebook file>

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
