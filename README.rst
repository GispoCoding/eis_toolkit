====
EIS Toolkit
====

Related to EIS Horizon EU project.

For developers
====

Testing
----

This repository contains dummy files and functions for testing purposes. Instructions
for testing of eis_toolkit's functionalities:

1. Navigate to the Releases section and download tar.gz or .zip file

2. Find out the path where python packages QGIS uses are installed e.g. by opening QGIS
and navigating to Python console and executing

 import imp

 str.replace(imp.find_module('numpy')[1], '/numpy', '')

3. Open terminal and execute
 pip install --target=<path_found_in_step_2> -U <path_to_eis_toolkit-dummy_test.tar.gz>

or

 pip install --target=<path_found_in_step_2> -U <path_to_eis_toolkit-dummy_test.zip>

4. Go back to QGIS's python console and run e.g.

 from eis_toolkit.dependency_test.dummy import test_function

 print(test_function(12,2))

or

 from eis_toolkit.dependency_test.dummy_sklearn import sk_mean

 x = np.array([[1., -1., 2.], [1., 1., 1.]])

 print(sk_mean(x))

In both cases, a result should appear into the QGIS's Python console's output window.


Documentation
----

In case you add a new class or function into the toolkit, please update the documentation site!

1. Modify mkgendocs.yml by adding a new page to pages section

- Give name to a new page, e.g. new_class.md
- Give path to the corresponding python file, e.g. eis_toolkit/new_class.py
- Give list of the functions to be documented

2. Navigate to the root directory level (the same level where mkgendocs.yml file is located)
and run

    gendocs --config mkgendocs.yml

3. Run

    mkdocs serve

4. Go to http://127.0.0.1:8000/

If you *just* want to take a look at the documentation (not to modify it),
clone this repository and execute steps 3 and 4.
