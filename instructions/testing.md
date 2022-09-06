# Testing your changes
## Writing tests with pytest
We use [pytest](https://docs.pytest.org/) for automated tests. Look to its documentation for examples and guidelines. The tests in this repository can also serve as a starting point.

All tests should be under the `tests/` directory. Structure them into modules based on what functionality they are testing. For example, `clip_test.py` to test clipping.

running tests is as simple as:

```console
pytest
```

All functionality that can reasonably be tested should have tests.

## Experimenting with jupyterlab
Jupyterlab could also come in handy if you want to experiment. See more instructions [here](./using_jupyterlab.md).

## Adding data for testing
See the instructions [here](./adding_data.md)
