"""The usual hodgepodge of helper utilities without a better home."""

import contextlib
import os
from pathlib import Path
from typing import Generator


def package_root() -> Path:
    """Return the absolute path of Python package root, i.e.
    PATH/TO/VMECPP/REPO/src/vmecpp.

    Useful e.g. to point tests to test files using paths relative to the repo root
    rather than paths relative to the test module.
    """
    return Path(__file__).parent


@contextlib.contextmanager
def change_working_directory_to(path: Path) -> Generator[None, None, None]:
    """Changes the working director within a context manager.

    Args:
        path: The path to change the working directory to.
    """
    origin = Path.cwd()

    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)


def get_vmec_configuration_name(vmec_file: Path) -> str:
    """Given a VMEC input file (case_name.json or input.case_name) or output file
    (wout_case_name.nc) extract the 'case_name' section of the name and return it."""
    filename = vmec_file.name

    if filename.endswith(".json"):
        case_name = filename[:-5]
    elif filename.startswith("input."):
        case_name = filename[6:]
    elif filename.startswith("wout_") and filename.endswith(".nc"):
        case_name = filename[5:-3]
    else:
        msg = f"This does not look like a VMEC input or output file: {filename}"
        raise ValueError(msg)

    return case_name
