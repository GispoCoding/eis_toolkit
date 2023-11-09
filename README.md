# EIS Toolkit

![tests](https://github.com/GispoCoding/eis_toolkit/workflows/Tests/badge.svg)
[![EUPL1.2 license](https://img.shields.io/badge/License-EUPL1.2-blue.svg)](http://perso.crans.org/besson/LICENSE.html)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

## Python library for mineral prospectivity mapping
EIS Toolkit will be a comprehensive Python library for mineral prospectivity mapping and analysis. EIS Toolkit is developed as part of [EIS Horizon EU project](https://eis-he.eu/), which aims to aid EU's efforts in the green transition by securing critical raw materials. EIS Toolkit will serve both as a standalone library that brings together and implements relevant tools for mineral prospectivity mapping and as a computational backend for [EIS QGIS Plugin](https://github.com/GispoCoding/eis_qgis_plugin).


## Repository status
The first beta release of EIS Toolkit is now out, but this repository is still in development. 

Current repository contents include
- implementations of basic tools to conduct MPM
- Jupyter notebooks showing usage and functionality of some of the implemented tools
- basic tests for implemented features
- instructions on how to contribute to the repository

To check the implementation status of the toolkit and planned tools, visit the [wiki page of EIS Toolkit](https://github.com/GispoCoding/eis_toolkit/wiki).

This repository contains source code related to eis_toolkit Python package, not source code of EIS QGIS Plugin.


## Installing
You can find the latest release of EIS Toolkit in the [releases page](https://github.com/GispoCoding/eis_toolkit/releases) as Python wheel. To install EIS Toolkit, simply download the wheel and install with pip

```console
pip install eis_toolkit-0.1.0-py3-none-any.whl
```

We recommend installing EIS Toolkit in a dedicated virtual environment as the library has a lot of dependencies.

Note that EIS Toolkit is not yet released in PyPi or Conda, but will be at a later stage.


## Using EIS Toolkit
EIS Toolkit can be used in Python scripts, Jupyter notebooks or via the CLI. At the moment, almost all tools have their own module and can be imported like this:
```python
# In general
from eis_toolkit.category.module import module_function

# Some examples
from eis_toolkit.raster_processing.reprojecting import reproject_raster
from eis_toolkit.exploratory_analyses.pca import compute_pca, plot_pca
```

To use the CLI, simply use the command
```console
eis
```

or

```console
eis --help
```

to get started. However, please note that the CLI has been primarily designed to communicate with other programs, such as QGIS.

Documentation of EIS Toolkit can be read [here](https://gispocoding.github.io/eis_toolkit/) (generated from docstrings).


## Contributing

If you are contributing by implementing new functionalities, read the **For developers** section. It will guide you to set up a local development environment. If you wish to just test the installation of eis_toolkit, follow the **For users** section (note that the currently documented installation process is by no means final).

*For general contributing guidelines, see [CONTRIBUTING](./CONTRIBUTING.md).*

## For developers

All contributing developers need git, and a copy of the repository.

```console
git clone https://github.com/GispoCoding/eis_toolkit.git
```

After this you have three options for setting up your local development environment.
1. Docker - [instructions](./instructions/dev_setup_with_docker.md)
2. Poetry - [instructions](./instructions/dev_setup_without_docker.md)
3. Conda - [instructions](./instructions/dev_setup_without_docker_with_conda.md)


### Additonal instructions

Here are some additional instructions related to the development of EIS toolkit:
- [Testing your changes](./instructions/testing.md)
- [Generating documentation](./instructions/generating_documentation.md)
- [Using jupyterlab](./instructions/using_jupyterlab.md)

## License

Licensed under the EUPL-1.2 or later.
