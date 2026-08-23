# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Generalized resolution interpolation and the Python-side continuation driver.

VMEC++ converges much more reliably when a hard equilibrium is approached through
a sequence of increasing resolutions (the classic ``ns_array`` multi-grid, and also
Fourier continuation via a sequence-valued ``VmecInput.mpol`` / ``.ntor``). Each step
solves a single resolution and hot-restarts from the previous step's solution,
interpolated to the new resolution by :func:`interpolate_solution`.

:func:`vmecpp.run` dispatches to :func:`_run_fourier_continuation` (this module)
whenever ``input.mpol`` and/or ``input.ntor`` is a sequence rather than a plain int;
:func:`interpolate_solution` itself is public API and can also be used directly, e.g.
to hand-roll a custom continuation schedule (see
``examples/fourier_resolution_increase.py``).

The interpolation is purely a Python operation on a converged :class:`VmecOutput`:
the flux-surface geometry is interpolated radially along the normalized toroidal
flux ``s`` and the Fourier spectrum is padded (modes new to the target are set to
zero) or truncated (modes the target drops are discarded). VMEC++ recomputes every
derived quantity from this geometry on restart, so only the geometry (``rmnc``,
``zmns``, ``lmns_full`` and their non-symmetric partners, plus the axis) needs to be
physically meaningful; all other arrays are carried over at consistent shapes only.
"""

from __future__ import annotations

import typing

import numpy as np

if typing.TYPE_CHECKING:
    from vmecpp import OutputMode, VmecInput, VmecOutput
    from vmecpp._free_boundary import MagneticFieldResponseTable

# State-vector geometry arrays, shape [mn_mode, n_surfaces]. These are the only
# quantities VMEC++ reads back when hot-restarting, so they must be interpolated.
_STATE_GEOMETRY_FIELDS = (
    "rmnc",
    "zmns",
    "lmns",
    "lmns_full",
    "rmns",
    "zmnc",
    "lmnc",
    "lmnc_full",
)

# Axis Fourier arrays, shape [ntor + 1]. The wout and the input use different field
# names for the magnetic axis (e.g. wout ``raxis_cc`` vs input ``raxis_c``).
_WOUT_AXIS_FIELDS = ("raxis_cc", "zaxis_cs", "raxis_cs", "zaxis_cc")
_INPUT_AXIS_FIELDS = ("raxis_c", "zaxis_s", "raxis_s", "zaxis_c")

# Handled explicitly below or regenerated for the target; excluded from the generic
# radial pass.
_FIELDS_HANDLED_EXPLICITLY = frozenset(
    _STATE_GEOMETRY_FIELDS + _WOUT_AXIS_FIELDS + ("xm", "xn", "xm_nyq", "xn_nyq")
)


def _state_mode_table(mpol: int, ntor: int, nfp: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(xm, xn)`` for the state-vector Fourier modes in VMEC ordering.

    ``m`` runs ``0 .. mpol - 1``; for ``m == 0`` the toroidal index ``n`` runs
    ``0 .. ntor`` and for ``m > 0`` it runs ``-ntor .. ntor``. ``xn`` is stored as
    ``n * nfp``, matching the convention used in the wout file.
    """
    xm: list[int] = []
    xn: list[int] = []
    for m in range(mpol):
        n_min = 0 if m == 0 else -ntor
        for n in range(n_min, ntor + 1):
            xm.append(m)
            xn.append(n * nfp)
    return np.asarray(xm, dtype=np.int64), np.asarray(xn, dtype=np.int64)


def _radial_interpolate(values: np.ndarray, ns_new: int) -> np.ndarray:
    """Linearly interpolate auxiliary radial arrays along the last axis.

    The radial coordinate is the normalized toroidal flux ``s`` on the full grid,
    ``s = linspace(0, 1, ns)``. This is used only for arrays VMEC++ recomputes on
    restart (profiles, Nyquist spectra); their exact values do not affect the
    restart, only their length must be consistent. The flux-surface geometry uses
    :func:`_radial_interpolate_geometry` instead.
    """
    values = np.asarray(values, dtype=float)
    ns_old = values.shape[-1]
    if ns_old == ns_new:
        return values.copy()
    s_old = np.linspace(0.0, 1.0, ns_old)
    s_new = np.linspace(0.0, 1.0, ns_new)
    if values.ndim == 1:
        return np.interp(s_new, s_old, values)
    flat = values.reshape(-1, ns_old)
    out = np.empty((flat.shape[0], ns_new), dtype=float)
    for i in range(flat.shape[0]):
        out[i] = np.interp(s_new, s_old, flat[i])
    return out.reshape((*values.shape[:-1], ns_new))


def _radial_interpolate_geometry(
    values: np.ndarray, xm: np.ndarray, ns_new: int
) -> np.ndarray:
    """Radially interpolate the state geometry ``[n_modes, ns_old]`` to ``ns_new``.

    This reproduces the radial interpolation VMEC++ performs internally between
    multi-grid steps. The flux-surface geometry is naturally a function of
    ``sqrt(s)``, so the interpolation is done on the ``sqrt(s)`` grid. Odd-``m``
    modes vanish like ``sqrt(s)`` toward the magnetic axis; they are divided by
    ``sqrt(s)`` before interpolation, linearly extrapolated to the axis from the two
    neighbouring surfaces, and reset to zero at the axis afterwards.

    Args:
        values: geometry coefficients ``[n_modes, ns_old]`` in the target mode order.
        xm: poloidal mode number per row, ``[n_modes]`` (selects the odd-``m`` path).
        ns_new: number of full-grid flux surfaces in the target resolution.
    """
    values = np.asarray(values, dtype=float)
    ns_old = values.shape[-1]
    if ns_old == ns_new:
        return values.copy()

    old_sqrt_s = np.sqrt(np.linspace(0.0, 1.0, ns_old))
    new_sqrt_s = np.sqrt(np.linspace(0.0, 1.0, ns_new))
    # 1 / sqrt(s), with the axis value copied from the first half-grid point.
    old_scale = np.empty(ns_old)
    old_scale[1:] = 1.0 / old_sqrt_s[1:]
    old_scale[0] = old_scale[1]
    new_scale = np.empty(ns_new)
    new_scale[1:] = 1.0 / new_sqrt_s[1:]
    new_scale[0] = new_scale[1]

    out = np.empty((values.shape[0], ns_new), dtype=float)
    for i in range(values.shape[0]):
        radial = values[i].copy()
        odd_m = int(xm[i]) % 2 == 1
        if odd_m:
            radial = radial * old_scale
            if ns_old >= 3:
                radial[0] = 2.0 * radial[1] - radial[2]
        interpolated = np.interp(new_sqrt_s, old_sqrt_s, radial)
        if odd_m:
            interpolated = interpolated / new_scale
            interpolated[0] = 0.0
        out[i] = interpolated
    return out


def _remap_modes(
    values: np.ndarray,
    src_xm: np.ndarray,
    src_xn: np.ndarray,
    dst_xm: np.ndarray,
    dst_xn: np.ndarray,
) -> np.ndarray:
    """Remap an array indexed by Fourier mode (axis 0) onto a new mode table.

    Modes shared by both tables are carried over, modes new to the target are set to
    zero, and modes the target drops are discarded.
    """
    values = np.asarray(values, dtype=float)
    src_index = {
        (int(m), int(n)): i for i, (m, n) in enumerate(zip(src_xm, src_xn, strict=True))
    }
    out = np.zeros((len(dst_xm), *values.shape[1:]), dtype=float)
    for i, (m, n) in enumerate(zip(dst_xm, dst_xn, strict=True)):
        j = src_index.get((int(m), int(n)))
        if j is not None:
            out[i] = values[j]
    return out


def _remap_axis(values: np.ndarray, ntor_new: int) -> np.ndarray:
    """Truncate or zero-pad an axis Fourier array ``[ntor + 1]`` to ``ntor_new``."""
    from vmecpp import _util  # noqa: PLC0415  (lazy import avoids a circular import)

    # Slice handles the truncation (dropping modes above ntor_new); right_pad
    # handles the zero-extension. right_pad alone cannot shrink the array.
    values = np.asarray(values, dtype=float)
    return _util.right_pad(values[: ntor_new + 1], ntor_new + 1)


def interpolate_solution(source: VmecOutput, target_input: VmecInput) -> VmecOutput:
    """Interpolate a converged solution onto the resolution of ``target_input``.

    Args:
        source: a converged :class:`VmecOutput`.
        target_input: the input for the next continuation step. Its ``mpol``,
            ``ntor`` and (single-element) ``ns_array`` define the target resolution,
            and its ``nfp`` must match the source.

    Returns:
        A :class:`VmecOutput` whose ``wout`` holds the source geometry interpolated
        to the target resolution, suitable as ``restart_from`` for
        :func:`vmecpp.run` with ``target_input``.
    """
    import vmecpp  # noqa: PLC0415  (lazy import avoids a circular import)

    wout = source.wout
    nfp = int(wout.nfp)
    ns_new = int(target_input.ns_array[-1])
    mpol_new = int(target_input.mpol)
    ntor_new = int(target_input.ntor)

    src_xm = np.asarray(wout.xm, dtype=np.int64)
    src_xn = np.asarray(wout.xn, dtype=np.int64)
    dst_xm, dst_xn = _state_mode_table(mpol_new, ntor_new, nfp)

    new_wout = wout.model_copy(deep=True)

    # Geometry: remap the Fourier spectrum, then interpolate radially in sqrt(s).
    for name in _STATE_GEOMETRY_FIELDS:
        val = getattr(wout, name)
        if val is None:
            continue
        remapped = _remap_modes(val, src_xm, src_xn, dst_xm, dst_xn)
        setattr(new_wout, name, _radial_interpolate_geometry(remapped, dst_xm, ns_new))

    # Magnetic axis: truncate or zero-pad in n.
    for name in _WOUT_AXIS_FIELDS:
        val = getattr(wout, name)
        if val is None:
            continue
        setattr(new_wout, name, _remap_axis(val, ntor_new))

    # Everything else that lives on the radial grid is recomputed on restart, so it
    # only needs a consistent length; interpolate it so the wout stays self-coherent.
    ns_old = int(wout.ns)
    for name in type(wout).model_fields:
        if name in _FIELDS_HANDLED_EXPLICITLY:
            continue
        val = getattr(wout, name)
        if not isinstance(val, np.ndarray) or val.size == 0:
            continue
        if val.shape[-1] == ns_old:
            setattr(new_wout, name, _radial_interpolate(val, ns_new))

    new_wout.ns = ns_new
    new_wout.mpol = mpol_new
    new_wout.ntor = ntor_new
    new_wout.mnmax = len(dst_xm)
    new_wout.xm = dst_xm
    new_wout.xn = dst_xn

    return vmecpp.VmecOutput(
        input=target_input,
        wout=new_wout,
        jxbout=source.jxbout,
        mercier=source.mercier,
        threed1_volumetrics=source.threed1_volumetrics,
        threed1_first_table=source.threed1_first_table,
        threed1_geometric_magnetic=source.threed1_geometric_magnetic,
        threed1_axis=source.threed1_axis,
        threed1_betas=source.threed1_betas,
        threed1_shafranov_integrals=source.threed1_shafranov_integrals,
    )


def _step_input(
    base_input: VmecInput, ns: int, mpol: int, ntor: int, ftol: float, niter: int
) -> VmecInput:
    """Build a single-resolution VmecInput for one continuation step.

    The boundary coefficients are truncated or zero-padded to ``(mpol, ntor)`` (the
    ``VmecInput`` validator handles the 2D boundary; the axis is resized here), and the
    schedule arrays are collapsed to the single value for this step.
    """
    import vmecpp  # noqa: PLC0415  (lazy import avoids a circular import)

    step = base_input.model_copy(deep=True)
    step.mpol = int(mpol)
    step.ntor = int(ntor)
    step.ns_array = np.asarray([ns], dtype=np.int64)
    step.ftol_array = np.asarray([ftol], dtype=float)
    step.niter_array = np.asarray([niter], dtype=np.int64)
    # Truncate or zero-pad the 2D boundary coefficients to the new (mpol, ntor).
    for name in ("rbc", "zbs", "rbs", "zbc"):
        val = getattr(step, name)
        if val is not None:
            setattr(
                step,
                name,
                vmecpp.VmecInput.resize_2d_coeff(
                    np.asarray(val, dtype=float), mpol_new=mpol, ntor_new=ntor
                ),
            )
    # Truncate or zero-pad the axis arrays to the new ntor.
    for name in _INPUT_AXIS_FIELDS:
        val = getattr(step, name)
        if val is not None:
            setattr(step, name, _remap_axis(np.asarray(val, dtype=float), ntor))
    return step


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
    import vmecpp  # noqa: PLC0415  (lazy import avoids a circular import)

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
        output = vmecpp.run(
            step_input,
            magnetic_field,
            max_threads=max_threads,
            verbose=verbose,
            restart_from=guess,
        )

    assert output is not None  # n_steps >= 1, so the loop always assigns output
    return output.model_copy(update={"input": input})
