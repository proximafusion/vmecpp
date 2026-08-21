# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

import contextlib
import enum
import json
import logging
import os
import sys
import tempfile
import types
import typing
from collections.abc import Generator
from pathlib import Path

import jaxtyping as jt
import netCDF4
import numpy as np
import pydantic

from vmecpp import _util
from vmecpp._continuation import _run_fourier_continuation, interpolate_solution
from vmecpp._free_boundary import (
    MagneticFieldResponseTable,
    MakegridParameters,
)
from vmecpp._iteration import (
    IterationResult,
    IterationState,
    RestartReason,
    iterate,
    solve_equilibrium,
    solve_multigrid,
)
from vmecpp._pydantic_numpy import BaseModelWithNumpy
from vmecpp.cpp import _vmecpp  # type: ignore # bindings to the C++ core

logger = logging.getLogger(__name__)


_ArrayType = typing.TypeVar("_ArrayType")


def _print_progress_tip_once() -> None:
    global _progress_tip_shown  # noqa: PLW0603
    global _progress_tip_shown
    if not _progress_tip_shown:
        _progress_tip_shown = True
        print(  # noqa: T201
            "Tip: Use vmecpp.run(..., verbose=1) for classic table output."
        )



from vmecpp.cpp._vmecpp import OutputMode

def run(
    indata: _vmecpp.VmecINDATA,
    magnetic_response_table: _vmecpp.MagneticFieldResponseTable | None = None,
    *,
    initial_state: _vmecpp.HotRestartState | None = None,
    max_threads: int | None = None,
    verbose: bool | int | _vmecpp.OutputMode = _vmecpp.OutputMode.PROGRESS,
) -> _vmecpp.OutputQuantities:
    """Run VMEC++."""
    if max_threads is not None and max_threads <= 0:
        msg = (
            "The number of threads must be >=1. To automatically use all "
            "available threads, pass max_threads=None"
        )
        raise RuntimeError(msg)

    if _vmecpp.OutputMode(verbose) in (_vmecpp.OutputMode.PROGRESS, _vmecpp.OutputMode.PROGRESS_NON_TTY):
        _print_progress_tip_once()
    
    if magnetic_response_table is not None:
        return _vmecpp.run(
            indata, 
            magnetic_response_table, 
            initial_state, 
            max_threads, 
            _vmecpp.OutputMode(verbose)
        )
    else:
        return _vmecpp.run(
            indata, 
            initial_state, 
            max_threads, 
            _vmecpp.OutputMode(verbose)
        )

# Export _vmecpp types
VmecINDATA = _vmecpp.VmecINDATA
WOutFileContents = _vmecpp.WOutFileContents


def is_vmec2000_input(input_file: Path) -> bool:
    """Returns true if the input file looks like a Fortran VMEC/VMEC2000 INDATA file."""
    # we peek at the first few non-blank, non-comment lines in the file:
    # if one of them is "&INDATA", then this is an INDATA file
    with open(input_file) as f:
        for line in f:
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith("!"):
                continue
            return stripped_line == "&INDATA"
    return False


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

        # We also add the PID to the output file to ensure that the output file
        # is different for multiple processes that run indata_to_json
        # concurrently on the same input, as it happens e.g. when the SIMSOPT
        # wrapper is run under `mpirun`.
        configuration_name = _util.get_vmec_configuration_name(input_path)
        output_file = input_path.with_name(f"{configuration_name}.{os.getpid()}.json")

        vmecpp_input_path = _util.indata_to_json(
            input_path, output_override=output_file
        )
        assert vmecpp_input_path == output_file.resolve()
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

    indata_contents = _util.vmecpp_json_to_indata(vmecpp_json_dict)

    # Otherwise we actually need to perform the JSON -> INDATA conversion.
    # We need the try/finally in order to correctly clean up after
    # ourselves even in case of errors raised from the body of the `with`
    # in user code.
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / indata_file
        with open(out_path, "w") as out_f:
            out_f.write(indata_contents)
        yield out_path


def set_profile(
    vmec_input: _vmecpp.VmecINDATA,
    field: typing.Literal["pressure", "iota", "current"],
    f: typing.Callable[[np.ndarray], np.ndarray],
) -> _vmecpp.VmecINDATA:
    """Populate a line segment profile using callable ``f``.

    This allows users to set a precise pressure/iota/curent profile on flux
    surfaces without any precision loss by fitting the profile to e.g.
    experimental data first.

    The callable is evaluated on all unique ``s`` values required for the
    multi-grid steps (full and half grids). The resulting knots and values are
    stored in the auxiliary arrays for the chosen profile. Therefore you should
    populate the multigrid ``ns_array`` resolutions before calling this function.

    Args:
        vmec_input: The vmec input to be modified.
        field: The profile quantity to set.
        f: A callable taking an array of flux coordinates ``s`` and returning
            the value of the selected quantity (pressure/iota/current)
    Returns:
        A modified copy of the given ``VmecInput``.
    """
    s_values: set[float] = set()
    for ns in vmec_input.ns_array:
        full_grid = np.linspace(0.0, 1.0, ns)
        half_grid = full_grid - 0.5 * (full_grid[1] - full_grid[0])
        s_values.update(full_grid)
        s_values.update(half_grid)
    knots = np.array(np.sort(np.array(list(s_values))))
    values = np.array(f(knots))
    res = vmec_input.copy()
    if field == "pressure":
        res.pmass_type = "line_segment"
        res.am_aux_s = knots
        res.am_aux_f = values
        res.am = np.array([])
        return res
    if field == "iota":
        res.piota_type = "line_segment"
        res.ai_aux_s = knots
        res.ai_aux_f = values
        res.ai = np.array([])
        return res
    if field == "current":
        res.pcurr_type = "line_segment_i"
        res.ac_aux_s = knots
        res.ac_aux_f = values
        res.ac = np.array([])
        return res
    msg = "field must be one of 'pressure', 'iota', 'current'"
    raise ValueError(msg)


# Backwards compatible name
populate_raw_profile = set_profile


# Ordered this way to ensure run, VmecInput, and VmecOutput are the first three
# items in the generated documentation.
__all__ = [  # noqa: RUF022
    "run",
    "VmecINDATA",
    "WOutFileContents",
    "MagneticFieldResponseTable",
    "MakegridParameters",
    "set_profile",
    "iterate",
    "solve_equilibrium",
    "solve_multigrid",
    "IterationResult",
    "IterationState",
]
_progress_tip_shown = False
