# EIS Toolkit

Related to EIS Horizon EU project. This repository is in early development stage.

Current contents
- a bunch of different configuration files
- instructions on how to contribute to the repository
- implementation of one preprocessing tool (clip raster with polygon) to act as an example for further development
- dummy files and functions for executing the most tentative tests between different environments

*This repository only contains source code related to eis_toolkit python package. The user interface will be implemented into separate repository.*

Licensed under the EUPL-1.2-or-later.

## Contributing

If you are contributing by implementing new funcionalities, read the **For developers** section. It will guide you to set up a local development environment. If you wish to just test the installation of eis_toolkit, follow the **For users** section (note that the currently documented installation process is by no means final).

*For general contributing guidelines, see [CONTRIBUTING](./CONTRIBUTING.md).*

## For developers

### Prerequisites

All contributing developers need git, and a copy of the repository.

```console
git clone https://github.com/GispoCoding/eis_toolkit.git
```

After this you have two options for setting up your local development environment.
1. Docker
2. Python venv

Docker is recommended as it containerizes the whole development environment, making sure it stays identical across different developers and operating systems. Using a container also keeps your own computer clean of all dependencies.

### Setting up a local development environment with docker (recommended)
Build and run the eis_toolkit container. Run this and every other command in the repository root unless otherwise directed.

```console
docker compose up -d
```

If you need to rebuild already existing container (e.g. dependencies have been updated), run

```console
docker compose up -d --build
```

### Working with the container

Attach to the running container

```console
docker attach eis_toolkit
```

You are now in your local development container, and all your commands in the current terminal window interact with the container.

**Note** that your local repository gets automatically mounted into the container. This means that:
- The repository in your computer's filesystem and in the container are exactly the same
- Changes from either one carry over to the other instantly, without any need for restarting the container

For your workflow this means that:
- You can edit all files like you normally would (on your own computer, with your favourite text editor etc.)
- You must do all testing and running the code inside the container

### Python inside the container

Whether or not using docker we manage the python dependencies with poetry. This means that a python venv is found in the container too. Inside the container, you can get into the venv like you normally would

```console
poetry shell
```

and run your code and tests from the command line. For example:

```console
python <path/to/your/file.py>
```

or

```console
pytest
```

You can also run commands from outside the venv, just prefix them with poetry run. For example:

```console
poetry run pytest
```

### Additonal instructions

Here are some additional instructions related to the development of EIS toolkit:
- [Testing your changes](./instructions/testing.md)
- [Generating documentation](./instructions/generating_documentation.md)
- [Using jupyterlab](./instructions/using_jupyterlab.md)

If you want to set up the development environment without docker, see:
- [Setup without docker](./instructions/dev_setup_without_docker.md)
