from setuptools import find_packages, setup

setup(
    name="eis_toolkit",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # Add your package dependencies here
    ],
    entry_points={
        "console_scripts": [
            "eis=eis_toolkit.cli_click:cli",
        ],
    },
)
