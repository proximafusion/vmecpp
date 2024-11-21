import os
import pathlib
import shutil
import subprocess
import tempfile
from pathlib import Path

from vmecpp import _util


def indata_to_json(filename: pathlib.Path) -> pathlib.Path:
    """Convert a VMEC2000 INDATA file to a VMEC++ JSON input file.

    The new file is created in the current working directory. Given
    `input.name`, the corresponding JSON file will be called
    `name.json`.

    Args:
        filename: The path to the VMEC2000 INDATA file.

    Returns:
        The path to the newly created JSON file.
    """
    if not filename.exists():
        msg = f"{filename} does not exist."
        raise FileNotFoundError(msg)

    indata_to_json_exe = pathlib.Path(
        _util.package_root(), "cpp", "third_party", "indata2json", "indata2json"
    )
    if not indata_to_json_exe.is_file():
        msg = f"{indata_to_json_exe} is not a file."
        raise FileNotFoundError(msg)
    if not os.access(indata_to_json_exe, os.X_OK):
        msg = f"Missing permission to execute {indata_to_json_exe}."
        raise PermissionError(msg)

    original_input_file = filename.absolute()
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as tmpdir, _util.change_working_directory_to(
        Path(tmpdir)
    ):
        # The Fortran indata2json supports a limited length of the path to the input file.
        # We work in a temporary directory in which we copy the input so that paths are always short.
        local_input_file = original_input_file.name
        shutil.copyfile(original_input_file, local_input_file)

        result = subprocess.run([indata_to_json_exe, local_input_file], check=True)

        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, [indata_to_json_exe, local_input_file]
            )

        configuration_name = _util.get_vmec_configuration_name(filename)
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
