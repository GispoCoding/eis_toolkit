====
EIS Toolkit
====

Related to EIS Horizon EU project. This repository is in early
development stage.

Current contents

- a bunch of different configuration files
- one preprocessing tool (clip raster with polygon)
- dummy files and functions for testing purposes.

.. important::
    This repository only contains source code related to eis_toolkit python package.
    The user interface will be implemented into separate repository.


Testing
====

If you wish to just test the installation of eis_toolkit and use it,
follow the **For users** section. If you want to set up a local
development environment for contributing, read also the
**For developers** section.

Docker
----
A Dockerfile and a docker-compose.yml exist at project root. If you have docker,
you can skip all dependency setup with the following:

With docker compose
^^^^
Build and run the eis_toolkit container

.. code-block:: shell

    docker compose up -d

Attach to the running container

.. code-block:: shell

    docker attach eis_toolkit

You are now in your local development container. You can for eample run

.. code-block:: shell
 
    poetry shell

to get into the venv and

.. code-block:: shell
 
    jupyter-lab --ip=0.0.0.0 --no-browser --allow-root

to launch jupyter lab from the container. A jupyter session should be
available at http://127.0.0.1:8888/ (the second link jupyter prints to command line)

Without docker compose
^^^^
Everything is possible without docker compose too. You just have to manually
build the eis_toolkit container

.. code-block:: shell

    docker build . --tag eis

and then run it with the appropriate flags. For example:

.. code-block:: shell
 
    docker run -it -p 8888:8888 eis

For users
----

0. Make sure that GDAL's dependencies

- libgdal (3.5.1 or greater)
- header files (gdal-devel)

are satisfied. If not, install them.

1. Navigate to the Releases section and download latest tar.gz or
.zip file

2. Create a new virtual environment (VE) by navigating to the folder
you wish to create the VE in, and by executing

.. code-block:: shell

    python3 -m venv <name_of_your_virtualenv>

3. Activate the VE:

- UNIX / MacOS

.. code-block:: shell

    source <name_of_your_virtualenv>/bin/activate

- Windows

.. code-block:: shell

    <name_of_your_virtualenv>\Scripts\activate.bat

.. hint::
    You should see (*<name_of_your_virtualenv>*) appearing in front of the command prompt.

4. Install eis_toolkit by running

.. code-block:: shell

   pip install <path_to_eis_toolkit-X.Y.Z.tar.gz>

or

.. code-block:: shell

   pip install <path_to_eis_toolkit-X.Y.Z.zip>

5. Open Python console with

.. code-block:: shell

    python

and run e.g.

.. code-block:: python

   from eis_toolkit.dummy_tests.dummy import test_function

   test_function(12,2)

or

.. code-block:: python

   from eis_toolkit.dummy_tests.dummy_gdal import driver_cnt

   driver_cnt(1)

**Note.** By using VEs we make sure that installing eis_toolkit does not break down anything (e.g. QGIS).


Performing more complex tests
^^^^

In case you do not want to insert your test commands one by one into the
command line's python console, you can create a local test file and
execute it with

.. code-block:: shell

    python <name_of_your_test_file>.py

.. hint::
    Your .py test file can, for example, look like:

.. code-block:: python

    import rasterio as rio
    import numpy as np
    from matplotlib import pyplot
    from pathlib import Path

    output_path = Path('/home/pauliina/Downloads/eis_outputs/clip_result.tif')
    src = rio.open(output_path)
    arr = src.read(1)
    # Let's replace No data values with numpy NaN values in order to plot clipped raster
    # so that the colour changes are visible for human eye
    arr = np.where(arr<-100, np.nan, arr)

    pyplot.imshow(arr, cmap='gray')
    pyplot.show()


For developers
----

Prerequisites
^^^^

0. Make sure that GDAL's dependencies

- libgdal (3.5.1 or greater)
- header files (gdal-devel)

are satisfied. If not, install them.

1. Install `poetry <https://python-poetry.org/>`_ according to your platform's
`instructions <https://python-poetry.org/docs/#installation>`_

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

From command line
""""

You can run your code from the command line within the virtual environment created by poetry.

1. Run

.. code-block:: shell

   pip install eis_toolkit


2. Open python console with

.. code-block:: shell

   python

and run e.g.

.. code-block:: python

   from eis_toolkit.dummy_tests.dummy import test_function

   test_function(12,2)


With JupyterLab
""""

You can also use `JupyterLab <https://jupyterlab.readthedocs.io/en/stable/>`_ for testing purposes
for example in cases when you want to store intermediate results in active memory.

Launch JupyterLab with

.. code-block:: shell

   poetry run jupyter lab

The notebooks are found in this repository, under the `notebooks/` directory. You can import and use
eis_toolkit's functions in these notebooks in the same way as you normally would use any other python package.

.. hint::
    There exists three example notebook files. The first one contains general usage instructions for running
    and modifying JupyterLab notebooks. The second one has been created for testing that dependencies to other
    python packages work and the third one has been created for testing the functionality of the clip tool.


Documentation
====

In case you add a new class, module or function into the toolkit, please update the documentation site!

1. Modify mkgendocs.yml by adding a new page to pages section:

- Give name to a new page, e.g. geoprocess/clip.md
- Give path to the corresponding python file, e.g. eis_toolkit/geoprocess/clipping.py
- Give list of the function names to be documented, e.g. clip

2. Navigate to the root directory level (the same level where mkgendocs.yml file is located)
   and run

.. code-block:: shell

    gendocs --config mkgendocs.yml

.. important::
    Executing the command above automatically creates new (empty) version of the index.md file.
    However, this is not desired behaviuor since the index.md file already contains some general
    information about the eis_toolkit. Hence, please use Rollback or otherwise undo the modifications
    in index.md file before committing, or do not commit the index.md file at all.

3. Run

.. code-block:: shell

    mkdocs serve

4. Go to http://127.0.0.1:8000/

If you **just** want to take a look at the documentation (not to modify it),
act according to **For developers** section's Prerequisites and Set up of a local development
environment and execute steps 3 and 4.
