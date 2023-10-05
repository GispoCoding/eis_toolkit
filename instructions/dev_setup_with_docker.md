### Development with Docker

Build and run the eis_toolkit container. Run this and every other command in the repository root unless otherwise directed.
The Docker environment uses conda.

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

Inside the container we manage the python dependencies with conda. You can activate the conda env like you normally would

```console
conda activate eis_toolkit
```

and run your code and tests from the command line. For example:

```console
python <path/to/your/file.py>
```

or

```console
pytest
```