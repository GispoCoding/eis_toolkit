# EIS Toolkit
Related to EIS Horizon EU project. This repository is in early development stage.

Current contents
- a bunch of different configuration files
- one preprocessing tool (clip raster with polygon)
- dummy files and functions for testing purposes.

*This repository only contains source code related to eis_toolkit python package. The user interface will be implemented into separate repository.*


## Testing
If you wish to just test the installation of eis_toolkit and use it, follow the **For users** section. If you want to set up a local development environment for contributing, read  the **For developers** section.

### For developers
#### Prerequisites
All contributing developers need git, and a copy of the repository.

```shell
git clone https://github.com/GispoCoding/eis_toolkit.git
```

After this you have two options for setting up your local development environment.
1. Docker
2. Python venv

Docker is recommended as it containerizes the whole development environment, making sure it stays identical across different developers and operating systems. It also keeps your own computer clean of all dependencies.

#### Setting up a local development environment with docker (recommended)
Build and run the eis_toolkit container. Run this and every other command in the repository root unless otherwise directed.

```shell
docker compose up -d
```

#### Working with the container
Attach to the running container

```shell
docker attach eis_toolkit
``` 

You are now in your local development container, and all your commands in the current terminal window interact with the container.

**Note** that your local repository gets automatically mounted into the container. This means that:
- The repository in your computer's filesystem and in the container are exactly the same
- Changes from either one carry over to the other instantly, without any need for restarting the container

For your workflow this means that:
- You can edit all files like you normally would (on your own computer, inside your favourite text editor etc.)
- You must do all testing and running the code inside the container

Whether or not using docker we manage the dependencies with poetry. So, inside the
container, you can get into the venv like you normally would

```shell
poetry shell
```

and run your code and tests from the command line. For example:

```shell
python <path/to/your/file.py>
``` 

or

```shell
pytest .
```

If you want to use jupyterlab from the container, it is entrirely possible. Follow the instructions [here](./instructions/using_jupyterlab.md)

#### Setting up a local development environment without docker
See [setup without docker](./instructions/dev_setup_without_docker.md)

### For users
0. Make sure that GDAL's dependencies
  - libgdal (3.5.1 or greater)
  - header files (gdal-devel)

are satisfied. If not, install them.

1. Navigate to the Releases section and download latest tar.gz or
.zip file

2. Create a new virtual environment (VE) by navigating to the folder you wish to create the VE in, and by executing

```shell
python3 -m venv <name_of_your_virtualenv>
```

3. Activate the VE:

- UNIX / MacOS

```shell
source <name_of_your_virtualenv>/bin/activate
```

- Windows

```shell
<name_of_your_virtualenv>\Scripts\activate.bat
```

You should now see (*<name_of_your_virtualenv>*) appearing in front of the command prompt.

4. Install eis_toolkit by running

```shell
pip install <path_to_eis_toolkit-X.Y.Z.tar.gz>
```

or

```shell
pip install <path_to_eis_toolkit-X.Y.Z.zip>
```

5. Open Python console with

```shell
python
```

and run e.g.

```shell
from eis_toolkit.dummy_tests.dummy import test_function

test_function(12,2)
```

or

```shell
from eis_toolkit.dummy_tests.dummy_gdal import driver_cnt
driver_cnt(1)
```

**Note.** By using VEs we make sure that installing eis_toolkit does not break down anything (e.g. QGIS).
