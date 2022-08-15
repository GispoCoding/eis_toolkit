# Testing your changes
## Writing tests
### Automated tests with pytest
We use [pytest](https://docs.pytest.org/) for automated tests. Look to the documentation for examples and guidelines. The tests in this repository can also serve as a starting point.

All tests should be under the `tests/` directory. Structure them into modules based on what functionality they are testing. For example, `clip_test.py` to test clipping.

running tests is as simple as:

```console
pytest
```

All functionality that can reasonably be tested should have tests.

### Local python files for quick experiments
You can also of course experiment by writing your own local .py file that tests something and then run it.

```console
python <name_of_your_test_file>.py
```

**Note** Don't push these to the remote - write proper tests when experimentation turns into testing.

### Jupyterlab
Jupyterlab could also come in handy for testing. See more instructions [here](./using_jupyterlab.md).

## Adding data for testing
Under `tests/data/` you have two directories for storing data:
- `tests/data/local/` to host data that should not be pushed to remote. Git ignores everything in this directory, and there will be no trace of it's contents on the remote. Good for when you use closed data for testing, but use this by default for storing any data.
- `tests/data/remote/` to host data that should be pushed to remote. Don't add anything here unless you specifically need it on the remote. Also, filesizes matter here - favour small files.
