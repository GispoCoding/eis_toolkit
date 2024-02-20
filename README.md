<!-- logo -->
<p align="center">
  <img src="https://github.com/GispoCoding/eis_qgis_plugin/assets/113038549/6792ed06-f1f1-4a69-b9f6-1ca78eaeff4a" align="center"/>
</p>

<h1 align="center">EIS Toolkit</h2>
<p align="center">Python package for mineral prospectivity mapping</p>

<!-- badges -->
<p align="center">
  <img src="https://github.com/GispoCoding/eis_toolkit/workflows/Tests/badge.svg"/>
  <a href="https://github.com/GispoCoding/eis_qgis_plugin/actions/workflows/code-style.ym">
    <img src="https://github.com/GispoCoding/eis_qgis_plugin/actions/workflows/code-style.yml/badge.svg?branch=master"
  /></a>
  <a href="http://perso.crans.org/besson/LICENSE.html">
    <img src="https://img.shields.io/badge/License-EUPL1.2-blue.svg"
  /></a>
</p>

<!-- links to sections / TOC -->
<p align="center">
  <a href="#introduction">Introduction</a>
  ·
  <a href="#getting-started">Installation</a>
  ·
  <a href="#usage">Usage</a>
  ·
  <a href="#roadmap">Roadmap</a>
  ·
  <a href="#contributing">Contributing</a>
  ·
  <a href="#license">License</a>
</p>


EIS Toolkit is a comprehensive Python package for mineral prospectivity mapping and analysis. EIS Toolkit is developed as part of [EIS Horizon EU project](https://eis-he.eu/), which aims to aid EU's efforts in the green transition by securing critical raw materials. EIS Toolkit serves both as a standalone library that brings together and implements relevant tools for mineral prospectivity mapping and as a computational backend for [EIS QGIS Plugin](https://github.com/GispoCoding/eis_qgis_plugin).

> [!NOTE]  
> This repository is still in development. Check the [wiki page of EIS Toolkit](https://github.com/GispoCoding/eis_toolkit/wiki) for list of tools and [roadmap](#roadmap) for more details about the project.

## Repository contents (DELETE?)
- implementations of basic (geospatial) tools for MPM
- demos in form of Jupyter notebooks
- basic tests for included tools
- instructions on how to contribute to the repository
- installation instructions

This repository contains source code related to eis_toolkit Python package, not source code of EIS QGIS Plugin.

## Installation
You can find the latest release of EIS Toolkit in the [releases page](https://github.com/GispoCoding/eis_toolkit/releases) as Python wheel. To install EIS Toolkit, simply download the wheel and install with pip

```console
pip install eis_toolkit-0.1.0-py3-none-any.whl
```

We recommend installing EIS Toolkit in an empty virtual environment to ensure compatible package versions.

> [!NOTE]
> EIS Toolkit has not yet been released in PyPi or Conda, but will be at a later stage.


## Usage
EIS Toolkit can be used in Python scripts, Jupyter notebooks or via the CLI. At the moment, most tools have their own module and are imported like this:
```python
# In general
from eis_toolkit.category.module import module_function

# Some examples
from eis_toolkit.raster_processing.reprojecting import reproject_raster
from eis_toolkit.exploratory_analyses.pca import compute_pca
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

Documentation of EIS Toolkit can be read [here](https://gispocoding.github.io/eis_toolkit/).


## Contributing

If you are contributing by implementing new functionalities, read the **For developers** section. It will guide you to set up a local development environment. If you wish to only test EIS Toolkit, follow the **Usage** section (note that the currently documented installation process may be subject to change).

*For general contributing guidelines, see [CONTRIBUTING](./CONTRIBUTING.md).*

### For developers

All contributing developers need git, and a copy of the repository.

```console
git clone https://github.com/GispoCoding/eis_toolkit.git
```

After this you have three options for setting up your local development environment.
1. Docker - [instructions](./instructions/dev_setup_with_docker.md)
2. Poetry - [instructions](./instructions/dev_setup_without_docker.md)
3. Conda - [instructions](./instructions/dev_setup_without_docker_with_conda.md)

## License

Licensed under the EUPL-1.2 or later.
