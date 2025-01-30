"""The usual hodgepodge of helper utilities without a better home."""

import contextlib
import os
import pathlib
import shutil
import subprocess
import tempfile
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


def indata_to_json(
    filename: pathlib.Path, use_mgrid_file_absolute_path: bool = False
) -> pathlib.Path:
    """Convert a VMEC2000 INDATA file to a VMEC++ JSON input file.

    The new file is created in the current working directory. Given
    `input.name`, the corresponding JSON file will be called
    `name.json`.

    Args:
        filename: The path to the VMEC2000 INDATA file.
        use_mgrid_file_absolute_path: If True, the absolute path to
            the parent directory of `filename` will be prepended to
            the output mgrid_file path.

    Returns:
        The path to the newly created JSON file.
    """
    if not filename.exists():
        msg = f"{filename} does not exist."
        raise FileNotFoundError(msg)

    indata_to_json_exe = pathlib.Path(
        package_root(), "cpp", "third_party", "indata2json", "indata2json"
    )
    if not indata_to_json_exe.is_file():
        msg = f"{indata_to_json_exe} is not a file."
        raise FileNotFoundError(msg)
    if not os.access(indata_to_json_exe, os.X_OK):
        msg = f"Missing permission to execute {indata_to_json_exe}."
        raise PermissionError(msg)

    original_input_file = filename.absolute()
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as tmpdir, change_working_directory_to(
        Path(tmpdir)
    ):
        # The Fortran indata2json supports a limited length of the path to the input file.
        # We work in a temporary directory in which we copy the input so that paths are always short.
        local_input_file = original_input_file.name
        shutil.copyfile(original_input_file, local_input_file)

        if use_mgrid_file_absolute_path:
            command = [
                indata_to_json_exe,
                "--mgrid_folder",
                original_input_file.parent.absolute(),
                local_input_file,
            ]
        else:
            command = [indata_to_json_exe, local_input_file]
        result = subprocess.run(command, check=True)

        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, [indata_to_json_exe, local_input_file]
            )

        configuration_name = get_vmec_configuration_name(filename)
        output_file = pathlib.Path(f"{configuration_name}.json")

        if not output_file.is_file():
            msg = (
                "The indata2json command was executed with no errors but output file "
                f"{output_file} is missing. This should never happen!"
            )
            raise RuntimeError(msg)

        # copy output back
        shutil.copyfile(output_file, Path(original_cwd, output_file))

    return output_file
