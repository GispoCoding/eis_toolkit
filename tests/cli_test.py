"""Tests for eis_toolkit.cli."""
import json
from contextlib import nullcontext
from pathlib import Path
from traceback import print_tb
from typing import ContextManager

import pytest
import rasterio
from click.testing import CliRunner, Result

from eis_toolkit.cli import cli

CLICK_RUNNER = CliRunner()

RASTER_FILE_PATH = Path("tests/data/remote/small_raster.tif")
GEODATAFRAME_FILE_PATH = Path("tests/data/remote/small_area.shp")
CONFIG_FILE_PATH = Path("tests/data/remote/clip_raster_cli_test_config.json")


def click_error_print(result: Result):
    """Print click result traceback."""
    # If exit_code is zero then the cli execution succeeded
    # If it is not zero then the execution failed
    if result.exit_code == 0:
        return

    # Print the Python traceback related to failed execution
    assert result.exc_info is not None
    _, _, tb = result.exc_info
    print_tb(tb)

    # Print the stdout of the execution
    print(result.output)
    assert result.exception is not None

    # Raise the collected Python exception that was raised
    raise result.exception


@pytest.mark.parametrize(
    "raster_file_path,geodataframe_file_path,raises",
    [
        # Using the contextmanagers nullcontext and pytest.raises, the tests
        # can be conditionalized to both error and succeed based on the
        # parameter (nullcontext == succeed)
        (RASTER_FILE_PATH, GEODATAFRAME_FILE_PATH, nullcontext()),
        (Path("invalid/path/to/nowhere.tif"), Path("invalid/path/to/nowhere.shp"), pytest.raises(SystemExit)),
    ],
)
def test_cli_clip_raster(raster_file_path: Path, geodataframe_file_path: Path, raises: ContextManager, tmp_path: Path):
    """Test cli_clip_raster click entrypoint."""
    # tmp_path is a pytest fixture that is always a temporary directory path on
    # the file system if is put in the function signature
    output_raster_path = tmp_path / "clipped_raster.tif"

    # Define the arguments for the command-line execution
    args = [
        # The first arg defines which subcommand to use
        "cli-clip-raster",
        str(raster_file_path),
        str(geodataframe_file_path),
        f"--output-raster-file={output_raster_path}",
    ]
    # Run the eis_toolkit.cli:cli click entrypoint function
    result = CLICK_RUNNER.invoke(cli=cli, args=args)

    # If error is expected, raises will equal to pytest.raises(<exception>)
    # where the <exception> can e.g. be FileNotFoundError. For click executions
    # is is always/usually SystemExit
    with raises:
        # This function will error and print tracebacks and exceptions if the
        # execution fails
        click_error_print(result=result)

        # Check that the expected result occurred
        assert output_raster_path.exists()
        assert output_raster_path.is_file()

        with rasterio.open(output_raster_path, mode="r") as out_raster, rasterio.open(
            raster_file_path, mode="r"
        ) as src_raster:
            # Check that coordinate reference system is maintained
            assert out_raster.meta["crs"] == src_raster.meta["crs"]


@pytest.mark.parametrize("config_file", [CONFIG_FILE_PATH])
def test_cli_clip_raster_config(config_file: Path):
    """Test cli_clip_raster click entrypoint."""
    # Define the arguments for the command-line execution
    args = [
        # The first arg defines which subcommand to use
        "cli-clip-raster-config",
        str(config_file),
    ]
    # Run the eis_toolkit.cli:cli click entrypoint function
    result = CLICK_RUNNER.invoke(cli=cli, args=args)

    # This function will error and print tracebacks and exceptions if the
    # execution fails
    click_error_print(result=result)

    # Need to to parse the config here to
    output_raster_path = Path(json.loads(config_file.read_text())["output_raster_file"])

    # Check that the expected result occurred
    assert output_raster_path.exists()
    assert output_raster_path.is_file()
