# EIS Toolkit

![tests](https://github.com/GispoCoding/eis_toolkit/workflows/Tests/badge.svg)
[![EUPL1.2 license](https://img.shields.io/badge/License-EUPL1.2-blue.svg)](http://perso.crans.org/besson/LICENSE.html)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

## Python library for mineral prospectivity mapping
EIS Toolkit will be a comprehensive Python library for mineral prospectivity mapping and analysis. EIS Toolkit is developed as part of [EIS Horizon EU project](https://eis-he.eu/), which aims to aid EU's efforts in the green transition by securing critical raw materials. EIS Toolkit will serve both as a standalone library that brings together and implements relevant tools for mineral prospectivity mapping and as a computational backend for [EIS QGIS Plugin](https://github.com/GispoCoding/eis_qgis_plugin).


## Repository status
This repository is still in development. First release is planned for autumn 2023.

Current contents include
- implementations for most of basic preprocessing tools
- Jupyter notebooks showing usage and functionality of some of the implemented tools
- basic tests for implemented features
- instructions on how to contribute to the repository

This repository contains source code related to eis_toolkit python package, not source code of EIS QGIS Plugin.


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
2. Poetry - [instructions]((./instructions/dev_setup_without_docker.md))
3. Conda - [instructions](./instructions/dev_setup_without_docker_with_conda.md)


### Additonal instructions

Here are some additional instructions related to the development of EIS toolkit:
- [Testing your changes](./instructions/testing.md)
- [Generating documentation](./instructions/generating_documentation.md)
- [Using jupyterlab](./instructions/using_jupyterlab.md)

## For users
TBD when first release is out.

## License

Licensed under the EUPL-1.2 or later.
