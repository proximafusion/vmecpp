import contextlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Generator

from vmecpp import _util

logger = logging.getLogger(__name__)


def is_vmec2000_input(input_file: Path) -> bool:
    """Returns true if the input file looks like a Fortran VMEC/VMEC2000 INDATA file."""
    # we peek at the first few characters in the file: if they are "&INDATA",
    # this is an INDATA file
    with open(input_file) as f:
        first_line = f.readline().strip()
    return first_line == "&INDATA"


@contextlib.contextmanager
def ensure_vmecpp_input(input_path: Path) -> Generator[Path, None, None]:
    """If input_path looks like a Fortran INDATA file, convert it to a VMEC++ JSON input
    and return the path to this new JSON file.

    Otherwise assume it is a VMEC++ json input: simply return the input_path unchanged.
    """
    if is_vmec2000_input(input_path):
        logger.debug(
            f"VMEC++ is being run with input file '{input_path}', which looks like "
            "a Fortran INDATA file. It will be converted to a VMEC++ JSON input "
            "on the fly. Please consider permanently converting the input to a "
            " VMEC++ input JSON using the //third_party/indata2json tool."
        )

        vmecpp_input_path = _util.indata_to_json(input_path)
        try:
            yield vmecpp_input_path
        finally:
            os.remove(vmecpp_input_path)
    else:
        # if the file is not a VMEC2000 indata file, we assume
        # it is a VMEC++ JSON input file
        yield input_path


@contextlib.contextmanager
def ensure_vmec2000_input(input_path: Path) -> Generator[Path, None, None]:
    """If input_path does not look like a VMEC2000 INDATA file, assume it is a VMEC++
    JSON input file, convert it to VMEC2000's format and return the path to the
    converted file.

    Otherwise simply return the input_path unchanged.

    Given a VMEC++ JSON input file with path 'path/to/[input.]NAME[.json]' the converted
    INDATA file will have path 'some/tmp/dir/input.NAME'.
    A temporary directory is used in order to avoid race conditions when calling this
    function multiple times on the same input concurrently; the `NAME` section of the
    file name is preserved as it is common to have logic that extracts it and re-uses
    it e.g. to decide how related files should be called.
    """

    if is_vmec2000_input(input_path):
        # nothing to do: must yield result on first generator call,
        # then exit (via a return)
        yield input_path
        return

    vmecpp_input_basename = input_path.name.removesuffix(".json").removeprefix("input.")
    indata_file = f"input.{vmecpp_input_basename}"

    with open(input_path) as vmecpp_json_f:
        vmecpp_json_dict = json.load(vmecpp_json_f)

    indata_contents = _vmecpp_json_to_indata(vmecpp_json_dict)

    # Otherwise we actually need to perform the JSON -> INDATA conversion.
    # We need the try/finally in order to correctly clean up after
    # ourselves even in case of errors raised from the body of the `with`
    # in user code.
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / indata_file
        with open(out_path, "w") as out_f:
            out_f.write(indata_contents)
        yield out_path


# adapted from https://github.com/jonathanschilling/indata2json/blob/4274976/json2indata
def _vmecpp_json_to_indata(vmecpp_json: dict[str, Any]) -> str:
    """Convert a dictionary with the contents of a VMEC++ JSON input file to the
    corresponding conents of a VMEC2000 INDATA file."""

    indata: str = "&INDATA\n"

    indata += "\n  ! numerical resolution, symmetry assumption\n"
    indata += _bool_to_namelist("lasym", vmecpp_json)
    for varname in ("nfp", "mpol", "ntor", "ntheta", "nzeta"):
        indata += _int_to_namelist(varname, vmecpp_json)

    indata += "\n  ! multi-grid steps\n"
    indata += _int_array_to_namelist("ns_array", vmecpp_json)
    indata += _float_array_to_namelist("ftol_array", vmecpp_json)
    indata += _int_array_to_namelist("niter_array", vmecpp_json)

    indata += "\n  ! solution method tweaking parameters\n"
    indata += _float_to_namelist("delt", vmecpp_json)
    indata += _float_to_namelist("tcon0", vmecpp_json)
    indata += _float_array_to_namelist("aphi", vmecpp_json)
    indata += _bool_to_namelist("lforbal", vmecpp_json)

    indata += "\n  ! printout interval\n"
    indata += _int_to_namelist("nstep", vmecpp_json)

    indata += "\n  ! total enclosed toroidal magnetic flux\n"
    indata += _float_to_namelist("phiedge", vmecpp_json)

    indata += "\n  ! mass / pressure profile\n"
    indata += _string_to_namelist("pmass_type", vmecpp_json)
    indata += _float_array_to_namelist("am", vmecpp_json)
    indata += _float_array_to_namelist("am_aux_s", vmecpp_json)
    indata += _float_array_to_namelist("am_aux_f", vmecpp_json)
    indata += _float_to_namelist("pres_scale", vmecpp_json)
    indata += _float_to_namelist("gamma", vmecpp_json)
    indata += _float_to_namelist("spres_ped", vmecpp_json)

    indata += "\n  ! select constraint on iota or enclosed toroidal current profiles\n"
    indata += _int_to_namelist("ncurr", vmecpp_json)

    indata += "\n  ! (initial guess for) iota profile\n"
    indata += _string_to_namelist("piota_type", vmecpp_json)
    indata += _float_array_to_namelist("ai", vmecpp_json)
    indata += _float_array_to_namelist("ai_aux_s", vmecpp_json)
    indata += _float_array_to_namelist("ai_aux_f", vmecpp_json)

    indata += "\n  ! enclosed toroidal current profile\n"
    indata += _string_to_namelist("pcurr_type", vmecpp_json)
    indata += _float_array_to_namelist("ac", vmecpp_json)
    indata += _float_array_to_namelist("ac_aux_s", vmecpp_json)
    indata += _float_array_to_namelist("ac_aux_f", vmecpp_json)
    indata += _float_to_namelist("curtor", vmecpp_json)
    indata += _float_to_namelist("bloat", vmecpp_json)

    indata += "\n  ! free-boundary parameters\n"
    indata += _bool_to_namelist("lfreeb", vmecpp_json)
    indata += _string_to_namelist("mgrid_file", vmecpp_json)
    indata += _float_array_to_namelist("extcur", vmecpp_json)
    indata += _int_to_namelist("nvacskip", vmecpp_json)

    indata += "\n  ! initial guess for magnetic axis\n"
    indata += _float_array_to_namelist("raxis_cc", vmecpp_json)
    indata += _float_array_to_namelist("zaxis_cs", vmecpp_json)
    indata += _float_array_to_namelist("raxis_cs", vmecpp_json)
    indata += _float_array_to_namelist("zaxis_cc", vmecpp_json)

    indata += "\n  ! (initial guess for) boundary shape\n"
    indata += _fourier_coefficients_to_namelist("rbc", vmecpp_json)
    indata += _fourier_coefficients_to_namelist("zbs", vmecpp_json)
    indata += _fourier_coefficients_to_namelist("rbs", vmecpp_json)
    indata += _fourier_coefficients_to_namelist("zbc", vmecpp_json)

    indata += "\n/\n&END\n"

    return indata


def _bool_to_namelist(varname: str, vmecpp_json: dict[str, Any]) -> str:
    if varname not in vmecpp_json:
        return ""

    return f"  {varname} = {'.true.' if vmecpp_json[varname] else '.false.'}\n"


def _string_to_namelist(varname: str, vmecpp_json: dict[str, Any]) -> str:
    if varname not in vmecpp_json:
        return ""

    return f"  {varname} = '{vmecpp_json[varname]}'\n"


def _int_to_namelist(varname: str, vmecpp_json: dict[str, Any]) -> str:
    if varname not in vmecpp_json:
        return ""

    return f"  {varname} = {vmecpp_json[varname]}\n"


def _float_to_namelist(varname: str, vmecpp_json: dict[str, Any]) -> str:
    if varname not in vmecpp_json:
        return ""

    return f"  {varname} = {vmecpp_json[varname]:.20e}\n"


def _int_array_to_namelist(varname: str, vmecpp_json: dict[str, Any]) -> str:
    if varname in vmecpp_json and len(vmecpp_json[varname]) > 0:
        elements = ", ".join(map(str, vmecpp_json[varname]))
        return f"  {varname} = {elements}\n"
    return ""


def _float_array_to_namelist(varname: str, vmecpp_json: dict[str, Any]) -> str:
    if varname in vmecpp_json and len(vmecpp_json[varname]) > 0:
        elements = ", ".join([f"{x:.20e}" for x in vmecpp_json[varname]])
        return f"  {varname} = {elements}\n"
    return ""


def _fourier_coefficients_to_namelist(varname: str, vmecpp_json: dict[str, Any]) -> str:
    if varname in vmecpp_json and len(vmecpp_json[varname]) > 0:
        out = ""
        for coefficient in vmecpp_json[varname]:
            m = coefficient["m"]
            n = coefficient["n"]
            value = coefficient["value"]
            out += f"  {varname}({n}, {m}) = {value:.20e}\n"
        return out
    return ""
