# Using jupyter
We include [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/) as a development dependency for testing purposes. You can use it for example in cases when you want to store intermediate results in active memory or just to see your pretty plots in the same place you are experimenting.

The notebooks are found under the `notebooks/` directory. You can import and use eis_toolkit's functions in these notebooks in the same way as you normally would use any other python package.

*There exists three example notebook files. The first one contains general usage instructions for running and modifying JupyterLab notebooks. The second one has been created for testing that dependencies to other python packages work and the third one has been created for testing the functionality of the clip tool.*

## With docker
To start the server from your container run (inside the running container)

```shell
jupyter lab --ip=0.0.0.0 --no-browser --allow-root
```

A jupyter server should now be available. Access it with the last link jupyter prints
to the terminal (you can just click it to automatically open it in a browser)

## Without docker
Start the jupyter server with

```shell
jupyter lab
```

It should automatically open in a browser.
