<!-- logo -->
<p align="center">
  <img src="https://github.com/GispoCoding/eis_toolkit/assets/113038549/e25b6d40-2785-4e21-bf74-221e8b096c64" align="center"/>
</p>

<h1 align="center">EIS Toolkit</h2>
<p align="center">Python package for mineral prospectivity mapping</p>

<!-- badges -->
<p align="center">
  <img src="https://github.com/GispoCoding/eis_toolkit/workflows/Tests/badge.svg"/>
  <a href="https://github.com/GispoCoding/eis_toolkit/actions/workflows/pre-commit.yaml">
    <img src="https://github.com/GispoCoding/eis_toolkit/actions/workflows/pre-commit.yaml/badge.svg"
  /></a>
  <a href="http://perso.crans.org/besson/LICENSE.html">
    <img src="https://img.shields.io/badge/License-EUPL1.2-blue.svg"
  /></a>
</p>

<!-- links to sections / TOC -->
<p align="center">
  <a href="#installation">Installation</a>
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


## Installation
We recommend installing EIS Toolkit in an empty virtual environment to ensure compatibility between package versions. 

EIS Toolkit is available in conda-forge and PyPI and can be installed with one of the following commands.

```console
pip install eis_toolkit
```

```console
conda install -c conda-forge eis_toolkit
```

A Python wheel can be downloaded also from the [releases page](https://github.com/GispoCoding/eis_toolkit/releases) of this GitHub repository.

> [!TIP]
> GDAL needs to be installed separately on Windows when using pip / PyPI. If you have trouble installing EIS Toolkit due to GDAL, you can download a compatible GDAL wheel (for example from [this repository](https://github.com/cgohlke/geospatial-wheels/releases)), install it first, and then attempt to install EIS Toolkit again.


## Usage
EIS Toolkit can be used in Python scripts, Jupyter notebooks or via the CLI. At the moment, most tools have their own module and are imported like this:
```python
# In general
from eis_toolkit.category.module import module_function

# Some examples
from eis_toolkit.raster_processing.reprojecting import reproject_raster
from eis_toolkit.exploratory_analyses.pca import compute_pca
```

The documentation of EIS Toolkit can be read [here](https://gispocoding.github.io/eis_toolkit/). You can find several Jupyter notebooks in this repostiory that demonstrate how tools of EIS Toolkit can be used. 


### EIS QGIS Plugin
For those that prefer using tools of EIS Toolkit via a graphical user interface, check [EIS QGIS Plugin](https://github.com/GispoCoding/eis_qgis_plugin). The plugin includes the main GUI application called EIS Wizard and all individual EIS Toolkit tools as QGIS processing algorithms.

The plugin is developed by the same core team that develops EIS Toolkit.

### CLI
EIS Toolkit includes a [Typer](https://typer.tiangolo.com/) command-line interface that serves as a common interface for integrating the toolkit with external applications, such as QGIS. The CLI can be used directly too, for example

```console
eis resample-raster-cli --input-raster path/to/raster.tif --output-raster path/to/output.tif --resolution 50 --resampling-method bilinear
```

For general help, use

```console
eis --help
```

or help for a tool

```console
eis <tool-name> --help
```

> [!NOTE] 
> Please note that the CLI has been primarily designed to communicate with external programs and may be clunky in direct use.

## Roadmap

- Milestone 1: **Beta release 0.1** (November 2023). The toolkit should have the basic funtionalities required for a full MPM workflow. Official testing phase begins. The plugin will be still under active development.
- Milestone 2: **Release 1.0** (May 2024). Most features should be incorporated at this time and the toolkit useful for actual MPM work. Testing will continue, more advanced methods added and the user experience refined.

## Contributing

We welcome contributions to EIS Toolkit in various forms:
- ✨ Developing new tools
- 🐞 Fixing bugs in the code
- 📝 Bug and other reporting
- 💡 Feature suggestions

To contribute with code or documentation, you'll need a local development environment and a copy of the repository. Please refer to the **For developers** section below for detailed setup instructions. If you're interested in bug reporting or making feature suggestions, you can familiarize yourself with the toolkit and test it as described in the **Usage** section. When you encounter bugs or have ideas for new features, you can create an issue in this repository.

### For developers

All contributing developers need Git and a copy of the repository.

```console
git clone https://github.com/GispoCoding/eis_toolkit.git
```

After this you have three options for setting up your local development environment.
1. Docker - [instructions](./instructions/dev_setup_with_docker.md)
2. Poetry - [instructions](./instructions/dev_setup_without_docker.md)
3. Conda - [instructions](./instructions/dev_setup_without_docker_with_conda.md)

*For general contributing guidelines, see [CONTRIBUTING](./CONTRIBUTING.md).*

## License

Licensed under the EUPL-1.2 or later.
