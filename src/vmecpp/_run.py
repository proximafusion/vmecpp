# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
""":func:`run`, the main VMEC++ entry point, and the Fourier-continuation driver."""

from __future__ import annotations

import sys

import numpy as np

from vmecpp._continuation import _step_input, interpolate_solution
from vmecpp._free_boundary import MagneticFieldResponseTable
from vmecpp._indata import VmecInput
from vmecpp._output import VmecOutput
from vmecpp._types import OutputMode
from vmecpp._wout import (
    JxBOut,
    Mercier,
    Threed1AxisGeometry,
    Threed1Betas,
    Threed1FirstTable,
    Threed1GeometricAndMagneticQuantities,
    Threed1ShafranovIntegrals,
    Threed1Volumetrics,
    VmecWOut,
)
from vmecpp.cpp import _vmecpp  # type: ignore # bindings to the C++ core

_progress_tip_shown = False


def _print_progress_tip_once() -> None:
    global _progress_tip_shown  # noqa: PLW0603
    if not _progress_tip_shown:
        _progress_tip_shown = True
        print(  # noqa: T201
            "Tip: Use vmecpp.run(..., verbose=1) for classic table output."
        )


def run(
    input: VmecInput,
    magnetic_field: MagneticFieldResponseTable | None = None,
    *,
    max_threads: int | None = None,
    verbose: bool | int | OutputMode = OutputMode.PROGRESS,
    restart_from: VmecOutput | None = None,
) -> VmecOutput:
    """Run VMEC++ using the provided input. This is the main entrypoint for both fixed-
    and free-boundary calculations.

    Args:
        input: a VmecInput instance, corresponding to the contents of a classic VMEC input file
        magnetic_field: if present, VMEC++ will pass the magnetic field object in memory instead of reading
            it from an mgrid file (only relevant in free-boundary runs).
        max_threads: maximum number of threads that VMEC++ should spawn. The actual number might still
            be lower that this in case there are too few flux surfaces to keep these many threads
            busy. If None, a number of threads equal to the number of logical cores is used.
        verbose: controls the output format. Accepts bool for backward compatibility:
            0. silent, no logging (False)
            1. legacy table output (True)
            2. animated progress bar for TTY enabled terminals
            3. animated progress bar for non-TTY outputs (e.g. Jupyter)
        restart_from: if present, VMEC++ is initialized using the converged equilibrium from the
            provided VmecOutput. This can dramatically decrease the number of iterations to
            convergence when running VMEC++ on a configuration that is very similar to the `restart_from` equilibrium.
            If `input.mpol`/`input.ntor` is a sequence (see below), this is used to hot-restart
            only the first continuation step; later steps always hot-restart from the previous one.

    If `input.mpol` and/or `input.ntor` is a sequence rather than a plain int, `run` performs
    continuation in Fourier resolution: each entry pairs with the corresponding `input.ns_array`
    entry (a scalar mpol/ntor broadcasts to every step), and each step is solved in turn,
    hot-restarting from the previous step's solution interpolated to the new resolution (see
    `interpolate_solution`).

    Example:
        >>> import vmecpp
        >>> path = "examples/data/solovev.json"
        >>> vmec_input = vmecpp.VmecInput.from_file(path)
        >>> output = vmecpp.run(vmec_input, verbose=False, max_threads=1)
        >>> round(output.wout.b0, 10) # Exact value may differ by C library
        0.2033313711
    """
    input = VmecInput.model_validate(input)

    if not isinstance(input.mpol, int) or not isinstance(input.ntor, int):
        return _run_fourier_continuation(
            input,
            magnetic_field,
            max_threads=max_threads,
            verbose=verbose,
            restart_from=restart_from,
        )

    cpp_indata = input._to_cpp_vmecindata()

    if restart_from is None:
        initial_state = None
    else:
        initial_state = _vmecpp.HotRestartState(
            wout=restart_from.wout._to_cpp_wout(),
            indata=restart_from.input._to_cpp_vmecindata(),
        )

    if max_threads is not None and max_threads <= 0:
        msg = (
            "The number of threads must be >=1. To automatically use all "
            "available threads, pass max_threads=None"
        )
        raise RuntimeError(msg)

    _verbose = OutputMode(verbose)

    if _verbose == OutputMode.PROGRESS:
        # Rich printing has been requested, let's auto detect if the terminal
        # is TTY capable
        is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        if not is_tty:
            _verbose = OutputMode.PROGRESS_NON_TTY
    if _verbose in (OutputMode.PROGRESS, OutputMode.PROGRESS_NON_TTY):
        _print_progress_tip_once()

    if magnetic_field is None:
        cpp_output_quantities = _vmecpp.run(
            cpp_indata,
            initial_state=initial_state,
            max_threads=max_threads,
            verbose=_verbose.value,
        )
    else:
        # magnetic_response_table takes precedence anyway, but let's be explicit, to ensure
        # we don't silently use the mgrid file in input, instead of the magnetic_response_table object.
        cpp_indata.mgrid_file = "NONE"
        cpp_output_quantities = _vmecpp.run(
            cpp_indata,
            magnetic_response_table=magnetic_field._to_cpp_magnetic_field_response_table(),
            initial_state=initial_state,
            max_threads=max_threads,
            verbose=_verbose.value,
        )

    cpp_wout = cpp_output_quantities.wout
    wout = VmecWOut._from_cpp_wout(cpp_wout)
    jxbout = JxBOut._from_cpp_jxbout(cpp_output_quantities.jxbout)
    mercier = Mercier._from_cpp_mercier(cpp_output_quantities.mercier)
    threed1_volumetrics = Threed1Volumetrics._from_cpp_threed1volumetrics(
        cpp_output_quantities.threed1_volumetrics
    )
    threed1_first_table = Threed1FirstTable._from_cpp_threed1_first_table(
        cpp_output_quantities.threed1_first_table
    )
    threed1_geometric_magnetic = Threed1GeometricAndMagneticQuantities._from_cpp_threed1_geometric_and_magnetic_quantities(
        cpp_output_quantities.threed1_geometric_magnetic
    )
    threed1_axis = Threed1AxisGeometry._from_cpp_threed1_axis_geometry(
        cpp_output_quantities.threed1_axis
    )
    threed1_betas = Threed1Betas._from_cpp_threed1_betas(
        cpp_output_quantities.threed1_betas
    )
    threed1_shafranov_integrals = (
        Threed1ShafranovIntegrals._from_cpp_threed1_shafranov_integrals(
            cpp_output_quantities.threed1_shafranov_integrals
        )
    )
    return VmecOutput(
        input=input,
        wout=wout,
        jxbout=jxbout,
        mercier=mercier,
        threed1_volumetrics=threed1_volumetrics,
        threed1_first_table=threed1_first_table,
        threed1_geometric_magnetic=threed1_geometric_magnetic,
        threed1_axis=threed1_axis,
        threed1_betas=threed1_betas,
        threed1_shafranov_integrals=threed1_shafranov_integrals,
    )


def _run_fourier_continuation(
    input: VmecInput,
    magnetic_field: MagneticFieldResponseTable | None,
    *,
    max_threads: int | None,
    verbose: bool | int | OutputMode,
    restart_from: VmecOutput | None,
) -> VmecOutput:
    """Solves an equilibrium by continuation in Fourier resolution.

    Called by :func:`vmecpp.run` whenever ``input.mpol`` and/or ``input.ntor`` is a
    sequence rather than a plain int. Each entry pairs with the corresponding
    ``input.ns_array`` entry (a scalar ``mpol``/``ntor`` broadcasts to every step).
    Each step solves a single ``(ns, mpol, ntor)`` resolution and hot-restarts from
    the previous step's solution interpolated to the new resolution (see
    :func:`interpolate_solution`); if ``restart_from`` is given, it seeds the first
    step instead of a cold start.

    Args:
        input: the target configuration. Its boundary is the final-resolution
            boundary; each step truncates or zero-pads it to that step's resolution.
        magnetic_field, max_threads, verbose, restart_from: forwarded to
            :func:`vmecpp.run` for every step (``restart_from`` only seeds the first).

    Returns:
        The converged :class:`VmecOutput` at the final resolution, with ``input`` set
        to the original (full-schedule) ``input`` argument.
    """
    ns_schedule = [int(x) for x in input.ns_array]
    n_steps = len(ns_schedule)

    def _resolve(value: int | np.ndarray, name: str) -> list[int]:
        if isinstance(value, int):
            return [value] * n_steps
        resolved = [int(x) for x in value]
        if len(resolved) != n_steps:
            msg = (
                f"'{name}' has {len(resolved)} entries, but 'ns_array' has "
                f"{n_steps}; a Fourier-resolution continuation schedule must have "
                "one entry per ns_array step (or be a scalar, broadcast to every "
                "step)."
            )
            raise ValueError(msg)
        return resolved

    mpol_schedule = _resolve(input.mpol, "mpol")
    ntor_schedule = _resolve(input.ntor, "ntor")
    ftol_schedule = [float(x) for x in input.ftol_array]
    niter_schedule = [int(x) for x in input.niter_array]

    output = restart_from
    for i in range(n_steps):
        step_input = _step_input(
            input,
            ns_schedule[i],
            mpol_schedule[i],
            ntor_schedule[i],
            ftol_schedule[i],
            niter_schedule[i],
        )
        guess = None if output is None else interpolate_solution(output, step_input)
        output = run(
            step_input,
            magnetic_field,
            max_threads=max_threads,
            verbose=verbose,
            restart_from=guess,
        )

    assert output is not None  # n_steps >= 1, so the loop always assigns output
    return output.model_copy(update={"input": input})
