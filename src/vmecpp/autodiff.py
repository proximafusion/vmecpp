"""JAX access to an in-memory VMEC++ solve and its implicit VJP.

The solver is deliberately kept outside the JAX trace. A forward call runs
VMEC++ through the C++ model, while the reverse callback reruns the same model
and solves the transposed interior force system. This is the usual implicit
layer for a differentiable code: JAX differentiates the consumer objective,
and VMEC++ supplies the producer's residual transpose.

The solve pins the m=1 poloidal-origin gauge (``VmecInput.always_fix_m1_gauge``):
the native iteration lets that gauge drift under its force until ``fsqz`` drops
below 1e-6 and freezes it wherever it is, so the converged state would depend on
the iteration history and the fixed-gauge force Jacobian would not describe it.
With the gauge pinned, the gauge entries of the state are a linear function of
the boundary, and the VJP pulls them back to the boundary like the boundary
entries themselves.

The public parameterization is the fixed-boundary case, with either a
prescribed iota or a prescribed toroidal current profile (``ncurr``). The
differentiable parameter is one dense array with rows ``rbc`` and ``zbs``, and
for ``lasym`` also ``rbs`` and ``zbc``, of shape ``(rows, mpol, 2 * ntor + 1)``.
An asymmetric boundary is rotated in the poloidal angle by an angle that
depends on its m = 1, n = 0 coefficients before the solve; the VJP includes
that dependence. Power-series profile parameters are differentiable as well;
free-boundary parameters are not.
"""

from __future__ import annotations

import collections
import itertools
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg

from vmecpp import autodiff_wout, geometry
from vmecpp.cpp import _vmecpp  # type: ignore

_GEOMETRY_COEFFICIENTS = (
    "r_cc",
    "r_ss",
    "r_sc",
    "r_cs",
    "z_sc",
    "z_cs",
    "z_cc",
    "z_ss",
    "lambda_sc",
    "lambda_cs",
    "lambda_cc",
    "lambda_ss",
)


def _cpp_geometry_flat(value, ns: int, mpol: int, ntor: int) -> np.ndarray:
    shape = (ns, mpol, ntor + 1)
    arrays = [
        np.asarray(value.toroidal_flux, dtype=np.float64).reshape(ns),
        np.asarray(value.poloidal_flux, dtype=np.float64).reshape(ns),
    ]
    coefficients = value.coefficients
    for name in _GEOMETRY_COEFFICIENTS:
        raw = np.asarray(getattr(coefficients, name), dtype=np.float64)
        raw = np.zeros(shape, dtype=np.float64) if raw.size == 0 else raw.reshape(shape)
        arrays.append(raw.ravel())
    return np.concatenate(arrays)


def _geometry_from_flat(flat, vmec_input) -> tuple[geometry.Geometry, Any]:
    """The geometry at the front of a flat solver output, and the entries after it."""
    ns = int(np.asarray(vmec_input.ns_array)[-1])
    mpol = vmec_input.mpol
    ntor = vmec_input.ntor
    modes = ns * mpol * (ntor + 1)
    arrays = [flat[:ns], flat[ns : 2 * ns]]
    offset = 2 * ns
    for _ in _GEOMETRY_COEFFICIENTS:
        arrays.append(flat[offset : offset + modes].reshape(ns, mpol, ntor + 1))
        offset += modes
    return geometry.Geometry(*arrays, nfp=vmec_input.nfp), flat[offset:]


def _make_indata(template, boundary: np.ndarray, profiles=None):
    indata = template.copy()
    indata.rbc[...] = boundary[0]
    indata.zbs[...] = boundary[1]
    if indata.lasym:
        indata.rbs[...] = boundary[2]
        indata.zbc[...] = boundary[3]
    for name, value in (profiles or {}).items():
        setattr(indata, name, float(value) if np.ndim(value) == 0 else value)
    return indata


def _key(solver_id: int, boundary: np.ndarray, profiles: dict) -> tuple[int, bytes]:
    values = [
        (name, np.shape(value), np.asarray(value).tobytes())
        for name, value in sorted(profiles.items())
    ]
    return solver_id, repr((np.asarray(boundary).tobytes(), values)).encode()


# VmecModel.status's integer value for vmecpp::VmecStatus::SUCCESSFUL_TERMINATION (see
# common/util/util.h); every other status, including NORMAL_TERMINATION (no fatal
# error, but the iteration budget was exhausted before ftol was met), means the step
# did not converge.
_VMEC_STATUS_SUCCESSFUL_TERMINATION = 11

# Recent forward solves, keyed by solver and parameters: the solved state, the
# forward output and, from the latest VJP until a solve or VJP at other
# parameters, the LU factors of the transposed interior system. NumPy data only,
# since JAX retains callback closures.
_FORWARD_SOLVES: collections.OrderedDict[tuple[int, bytes], dict[str, Any]] = (
    collections.OrderedDict()
)
_FORWARD_SOLVES_SIZE = 2
# Unique per solver, so solvers of different inputs never share a cache entry.
_SOLVER_IDS = itertools.count()

_MU_0 = 4.0e-7 * np.pi


def _solve_model(template, boundary: np.ndarray, profiles=None):
    """Run all requested VMEC++ resolutions and return the final model.

    Each entry of ``ns_array`` converges to its own ``ftol_array`` entry.
    Coarse, non-final steps are allowed to exhaust their iteration budget
    without reaching ``ftol``: their only job is to hand a good initial guess
    to the next, finer step, exactly as ``vmecpp.run`` treats them. The final
    step is the one that must actually converge; a schedule truncated to its
    first steps for a cheap solve is only valid if its new last step is
    standalone-convergent.
    """
    indata = _make_indata(template, boundary, profiles)
    indata.always_fix_m1_gauge = True
    resolutions = [int(value) for value in np.asarray(indata.ns_array)]
    model = None
    for ns in resolutions:
        if ns < 3:
            continue
        if model is None:
            model = _vmecpp.VmecModel.create(indata, ns)
        else:
            model.refine_to(ns)
        model.solve()
    if model is None:
        error_message = "VMEC input has no resolution with ns >= 3"
        raise ValueError(error_message)
    if model.status != _VMEC_STATUS_SUCCESSFUL_TERMINATION:
        error_message = (
            f"VMEC++ did not converge at the final multi-grid step (ns = {model.ns}): "
            f"status {model.status}, ftol = {model.ftolv:.3e}, final force residuals "
            f"fsqr = {model.fsqr:.3e}, fsqz = {model.fsqz:.3e}, fsql = {model.fsql:.3e}. "
            "If ns_array was truncated to its first steps, its new last entry must be "
            "standalone-convergent at its own ftol_array/niter_array entry."
        )
        raise RuntimeError(error_message)
    return model


def _span_slices(model) -> dict[str, slice]:
    """Return slices in VmecModel's canonical active-state ordering."""
    names: list[str] = ["r_cc"]
    if model.lthreed:
        names.append("r_ss")
    if model.lasym:
        names.append("r_sc")
    if model.lasym and model.lthreed:
        names.append("r_cs")
    names.append("z_sc")
    if model.lthreed:
        names.append("z_cs")
    if model.lasym:
        names.append("z_cc")
    if model.lasym and model.lthreed:
        names.append("z_ss")
    names.append("lambda_sc")
    if model.lthreed:
        names.append("lambda_cs")
    if model.lasym:
        names.append("lambda_cc")
    if model.lasym and model.lthreed:
        names.append("lambda_ss")
    span_size = model.ns * model.mpol * (model.ntor + 1)
    return {
        name: slice(index * span_size, (index + 1) * span_size)
        for index, name in enumerate(names)
    }


def _interior_and_boundary(model) -> tuple[np.ndarray, np.ndarray]:
    slices = _span_slices(model)
    state_size = int(np.asarray(model.get_state()).size)
    boundary: list[int] = []
    modes_per_surface = model.mpol * (model.ntor + 1)
    edge_start = (model.ns - 1) * modes_per_surface
    for name in (
        "r_cc",
        "r_ss",
        "r_sc",
        "r_cs",
        "z_sc",
        "z_cs",
        "z_cc",
        "z_ss",
    ):
        if name in slices:
            span = slices[name]
            boundary.extend(range(span.start + edge_start, span.stop))
    boundary_array = np.asarray(sorted(boundary), dtype=np.int64)
    interior_array = np.setdiff1d(np.arange(state_size), boundary_array)
    return interior_array, boundary_array


def _gauge_spans(model) -> list[str]:
    """The spans whose ``m = 1`` slot holds a pinned gauge combination."""
    if model.mpol_geometry <= 1:
        return []
    return (["z_cs"] if model.lthreed else []) + (
        ["z_cc"] if model.lasym and model.lthreed else []
    )


def _gauge_entries(model) -> np.ndarray:
    """State entries holding the m=1 poloidal-origin gauge on interior surfaces.

    ``FourierCoeffs::m1Constraint`` stores ``(r_ss - z_cs) / 2`` for ``m = 1``
    in the ``z_cs`` slot, and for ``lasym`` ``(r_sc - z_cc) / 2`` in the ``z_cc``
    slot. Their forces are zeroed when the gauge is fixed, so with
    ``always_fix_m1_gauge`` the solve leaves these entries at the initial guess:
    the boundary value scaled by ``sqrt(s)`` on every surface. The ``n = 0``
    entries are identically zero, the asymmetric one by the poloidal-origin
    shift of the boundary, and are left to the structural deflation. Modes
    excluded by the geometry resolution are zeroed during force evaluation.
    """
    slices = _span_slices(model)
    modes_per_surface = model.mpol * (model.ntor + 1)
    entries = [
        slices[name].start + j * modes_per_surface + 1 * (model.ntor + 1) + n
        for name in _gauge_spans(model)
        for j in range(1, model.ns - 1)
        for n in range(1, model.ntor_geometry + 1)
    ]
    return np.asarray(entries, dtype=np.int64)


def _gauge_span_starts(model, gauge: np.ndarray) -> np.ndarray:
    starts = np.asarray(
        [_span_slices(model)[name].start for name in _gauge_spans(model)]
    )
    if starts.size == 0:
        return np.zeros_like(gauge)
    span_size = model.ns * model.mpol * (model.ntor + 1)
    index = np.searchsorted(starts + span_size, gauge, side="right")
    return starts[index]


def _gauge_radial_weight(model, gauge: np.ndarray) -> np.ndarray:
    """``sqrt(s_j)`` for every gauge entry: the factor mapping the boundary gauge
    to surface ``j`` in ``FourierGeometry::interpFromBoundaryAndAxis``."""
    modes_per_surface = model.mpol * (model.ntor + 1)
    surface = (gauge - _gauge_span_starts(model, gauge)) // modes_per_surface
    return np.sqrt(surface / (model.ns - 1.0))


def _pull_gauge_back_to_boundary(
    model, state_bar: np.ndarray, gauge: np.ndarray
) -> np.ndarray:
    """Fold the gauge cotangents onto the boundary gauge entries they derive from."""
    result = state_bar.copy()
    modes_per_surface = model.mpol * (model.ntor + 1)
    span_start = _gauge_span_starts(model, gauge)
    mode = (gauge - span_start) % modes_per_surface
    edge = span_start + (model.ns - 1) * modes_per_surface + mode
    np.add.at(result, edge, _gauge_radial_weight(model, gauge) * state_bar[gauge])
    result[gauge] = 0.0
    return result


def _poloidal_origin_shift(model, boundary: np.ndarray) -> tuple[float, float]:
    """The angle ``delta`` by which ``Boundaries::parseToInternalArrays`` rotates an
    asymmetric boundary, and the sign ``sigma`` of the gauge it targets."""
    ntor = model.ntor
    handedness = boundary[0, 1].sum() * boundary[1, 1].sum() - (
        boundary[2, 1].sum() * boundary[3, 1].sum()
    )
    sign_of_jacobian = -1.0
    will_flip_theta = handedness * sign_of_jacobian > 0.0
    sigma = -sign_of_jacobian * (-1.0 if will_flip_theta else 1.0)
    delta = np.arctan2(
        boundary[2, 1, ntor] - sigma * boundary[3, 1, ntor],
        boundary[0, 1, ntor] + sigma * boundary[1, 1, ntor],
    )
    return float(delta), sigma


def _boundary_from_state_vjp(
    model, state_bar: np.ndarray, boundary: np.ndarray | None = None
) -> np.ndarray:
    """Transpose the fixed-boundary parser; an asymmetric input needs ``boundary``."""
    slices = _span_slices(model)
    mpol = model.mpol
    ntor = model.ntor
    modes_per_surface = mpol * (ntor + 1)
    edge = model.ns - 1
    rows = 4 if model.lasym else 2
    result = np.zeros((rows, mpol, 2 * ntor + 1), dtype=np.float64)

    def edge_value(name: str, m: int, n: int) -> float:
        if name not in slices:
            return 0.0
        values = state_bar[slices[name]]
        return float(values[edge * modes_per_surface + m * (ntor + 1) + n])

    # Undo the state scaling and the m=1 gauge transform used by
    # Boundaries::ensureM1Constrained. The boundary parser itself is the
    # transpose of the positive/negative-toroidal-mode accumulation below.
    names = ["r_cc", "z_sc", "r_ss", "z_cs", "r_sc", "z_cc", "r_cs", "z_ss"]
    bar = {name: np.zeros((mpol, ntor + 1)) for name in names}
    for m in range(mpol):
        for n in range(ntor + 1):
            if m >= model.mpol_geometry or n > model.ntor_geometry:
                continue
            scale = (1.0 if m == 0 else np.sqrt(2.0)) * (
                1.0 if n == 0 else np.sqrt(2.0)
            )
            for name in names:
                bar[name][m, n] = edge_value(name, m, n) / scale
    for r_name, z_name in (("r_ss", "z_cs"), ("r_sc", "z_cc")):
        if r_name in slices and z_name in slices and mpol > 1:
            r_bar = bar[r_name][1].copy()
            z_bar = bar[z_name][1].copy()
            bar[r_name][1] = 0.5 * (r_bar + z_bar)
            bar[z_name][1] = 0.5 * (r_bar - z_bar)

    if model.have_to_flip_theta:
        for m in range(1, mpol):
            parity = 1.0 if m % 2 == 0 else -1.0
            for name, factor in (
                ("r_cc", 1),
                ("z_sc", -1),
                ("r_ss", -1),
                ("z_cs", 1),
                ("r_sc", -1),
                ("z_cc", 1),
                ("r_cs", 1),
                ("z_ss", -1),
            ):
                bar[name][m] *= factor * parity

    # cotangents of the (rotated) rbc, zbs, rbs, zbc the parser sorts
    for m in range(mpol):
        for signed_n in range(-ntor, ntor + 1):
            source = ntor + signed_n
            target = abs(signed_n)
            sign = 1.0 if signed_n > 0 else -1.0 if signed_n < 0 else 0.0
            result[0, m, source] += bar["r_cc"][m, target]
            if m > 0:
                result[0, m, source] += sign * bar["r_ss"][m, target]
                result[1, m, source] += bar["z_sc"][m, target]
            result[1, m, source] -= sign * bar["z_cs"][m, target]
            if model.lasym:
                if m > 0:
                    result[2, m, source] += bar["r_sc"][m, target]
                    result[3, m, source] += sign * bar["z_ss"][m, target]
                result[2, m, source] -= sign * bar["r_cs"][m, target]
                result[3, m, source] += bar["z_cc"][m, target]
    if not model.lasym:
        return result
    if boundary is None:
        error_message = "the asymmetric boundary parser transpose needs the boundary"
        raise ValueError(error_message)

    # Transpose of the rotation by \delta(rbc, zbs, rbs, zbc at m = 1, n = 0):
    # rbc' = rbc \cos(m \delta) + rbs \sin(m \delta), zbs' = zbs \cos - zbc \sin,
    # rbs' = rbs \cos - rbc \sin, zbc' = zbc \cos + zbs \sin.
    delta, sigma = _poloidal_origin_shift(model, boundary)
    m_index = np.arange(mpol)[:, None]
    cos, sin = np.cos(m_index * delta), np.sin(m_index * delta)
    rbc, zbs, rbs, zbc = (np.asarray(row, dtype=np.float64) for row in boundary)
    rotated = (
        rbc * cos + rbs * sin,
        zbs * cos - zbc * sin,
        rbs * cos - rbc * sin,
        zbc * cos + zbs * sin,
    )
    delta_bar = float(
        np.sum(
            m_index
            * (
                result[0] * rotated[2]
                - result[1] * rotated[3]
                - result[2] * rotated[0]
                + result[3] * rotated[1]
            )
        )
    )
    unrotated = np.stack(
        [
            result[0] * cos - result[2] * sin,
            result[1] * cos + result[3] * sin,
            result[0] * sin + result[2] * cos,
            -result[1] * sin + result[3] * cos,
        ]
    )
    x = rbc[1, ntor] + sigma * zbs[1, ntor]
    y = rbs[1, ntor] - sigma * zbc[1, ntor]
    radius2 = x * x + y * y
    unrotated[0, 1, ntor] -= delta_bar * y / radius2
    unrotated[1, 1, ntor] -= delta_bar * sigma * y / radius2
    unrotated[2, 1, ntor] += delta_bar * x / radius2
    unrotated[3, 1, ntor] -= delta_bar * sigma * x / radius2
    return unrotated


def _structural_nullfree_interior(
    model, interior: np.ndarray, n_probe: int = 6, tol: float = 1.0e-9, seed: int = 0
) -> np.ndarray:
    """Interior DOFs that actually enter the force.

    The augmented Hessian has a structural null space: state-independent gauge
    and parity modes that no force depends on and that produce no force. They
    make the transposed interior system singular, and the objective cotangent
    generally has a component outside its range, so the adjoint solve is
    inconsistent and stagnates rather than converging. In two dimensions the
    surviving null directions happen to stay orthogonal to the cotangent; in
    three dimensions the extra ``r_ss``, ``z_cs`` and ``lambda_cs`` blocks bring
    in modes that do not, which is why this deflation is not optional there.

    A DOF is kept when both its Hessian column and its row are nonzero. Column
    ``i`` is zero iff ``(H^T v)[i] = 0`` for random ``v``, and row ``i`` is zero
    iff ``(H v)[i] = 0``, so a handful of probes finds every structural zero. The
    set depends only on the mode structure, not on the state, so it is detected
    once per model and reused across adjoint solves.
    """
    state_size = int(np.asarray(model.get_state()).size)
    generator = np.random.default_rng(seed)
    column = np.zeros(state_size)
    row = np.zeros(state_size)
    for _ in range(n_probe):
        probe = np.ascontiguousarray(generator.standard_normal(state_size))
        column = np.maximum(
            column,
            np.abs(
                np.asarray(
                    model.exact_hessian_vector_product_transpose(probe),
                    dtype=np.float64,
                )
            ),
        )
        row = np.maximum(
            row,
            np.abs(
                np.asarray(model.exact_hessian_vector_product(probe), dtype=np.float64)
            ),
        )
    threshold = tol * max(column.max(), row.max(), 1.0)
    keep = [i for i in interior if column[i] > threshold and row[i] > threshold]
    if not keep:
        error_message = (
            "VMEC++ adjoint: the interior force operator is entirely structurally "
            "null; the model is not in a differentiable state"
        )
        raise RuntimeError(error_message)
    return np.asarray(keep, dtype=np.int64)


class _BlockTridiagonal:
    """A matrix that is block tridiagonal in the surface index, solved by block LU.

    ``surface[k]`` is the surface of unknown ``k``. Unknowns on the same surface
    form one diagonal block; ``lower[j]`` and ``upper[j]`` couple surface ``j`` to
    ``j - 1`` and ``j + 1``. Blocks are stored dense and padded to the largest
    surface.
    """

    def __init__(self, surface: np.ndarray) -> None:
        self.surface = surface
        self.ns = int(surface.max()) + 1
        self.members = [np.nonzero(surface == j)[0] for j in range(self.ns)]
        self.local = np.zeros(surface.size, dtype=np.int64)
        for members in self.members:
            self.local[members] = np.arange(members.size)
        width = max(members.size for members in self.members)
        self.diagonal = np.zeros((self.ns, width, width))
        self.lower = np.zeros((self.ns, width, width))
        self.upper = np.zeros((self.ns, width, width))
        self.pivots: list[np.ndarray] = []

    def set_entries(self, rows: np.ndarray, columns: np.ndarray, values) -> None:
        offset = self.surface[columns] - self.surface[rows]
        if np.any(np.abs(offset) > 1):
            error_message = "entry outside the block tridiagonal band"
            raise ValueError(error_message)
        for band, blocks in ((-1, self.lower), (0, self.diagonal), (1, self.upper)):
            mask = offset == band
            blocks[
                self.surface[rows[mask]],
                self.local[rows[mask]],
                self.local[columns[mask]],
            ] = values[mask]

    def _block(self, blocks: np.ndarray, j: int, k: int) -> np.ndarray:
        return blocks[j, : self.members[j].size, : self.members[k].size]

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        result = np.zeros_like(vector)
        for j, members in enumerate(self.members):
            result[members] = self._block(self.diagonal, j, j) @ vector[members]
            if j > 0:
                result[members] += (
                    self._block(self.lower, j, j - 1) @ vector[self.members[j - 1]]
                )
            if j + 1 < self.ns:
                result[members] += (
                    self._block(self.upper, j, j + 1) @ vector[self.members[j + 1]]
                )
        return result

    def factorize(self) -> None:
        """Block Thomas elimination: LU-factor each Schur complement in place."""
        self.pivots = []
        for j in range(self.ns):
            schur = self._block(self.diagonal, j, j)
            if j > 0:
                # upper[j-1] <- D'[j-1]^{-1} upper[j-1], then D'[j] = D[j] - L[j] U'[j-1]
                coupling = scipy.linalg.lu_solve(
                    (self._block(self.diagonal, j - 1, j - 1), self.pivots[j - 1]),
                    self._block(self.upper, j - 1, j),
                )
                self._block(self.upper, j - 1, j)[...] = coupling
                schur -= self._block(self.lower, j, j - 1) @ coupling
            factor, pivots = scipy.linalg.lu_factor(schur, check_finite=False)
            self._block(self.diagonal, j, j)[...] = factor
            self.pivots.append(pivots)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        forward = np.zeros_like(rhs)
        for j, members in enumerate(self.members):
            forward[members] = rhs[members]
            if j > 0:
                forward[members] -= self._block(
                    self.lower, j, j - 1
                ) @ scipy.linalg.lu_solve(
                    (self._block(self.diagonal, j - 1, j - 1), self.pivots[j - 1]),
                    forward[self.members[j - 1]],
                )
        result = np.zeros_like(rhs)
        for j in range(self.ns - 1, -1, -1):
            members = self.members[j]
            result[members] = scipy.linalg.lu_solve(
                (self._block(self.diagonal, j, j), self.pivots[j]), forward[members]
            )
            if j + 1 < self.ns:
                result[members] -= (
                    self._block(self.upper, j, j + 1) @ result[self.members[j + 1]]
                )
        return result


def _assemble_block_tridiagonal(model, solved: np.ndarray) -> _BlockTridiagonal:
    """``H^T`` restricted to ``solved``, exactly, from coloured Hessian products.

    The force on surface ``j`` depends only on the state on surfaces ``j - 1``,
    ``j`` and ``j + 1``, so the operator is block tridiagonal in the surface index.
    One product ``H p`` with ``p`` the indicator of every solved entry of one
    mode on the surfaces ``j % 3 == c`` recovers all of those columns at once:
    rows on surface ``i`` can only come from the one probed surface among
    ``i - 1, i, i + 1``. That is ``3 * (modes per surface)`` products in total,
    independent of ``ns``. ``H`` rather than ``H^T`` is probed because the
    forward product is several times cheaper; the transpose is taken on storage.
    """
    state_size = int(np.asarray(model.get_state()).size)
    modes_per_surface = model.mpol * (model.ntor + 1)
    surface = np.zeros(state_size, dtype=np.int64)
    mode = np.zeros(state_size, dtype=np.int64)
    for index, span in enumerate(_span_slices(model).values()):
        local = np.arange(span.stop - span.start)
        surface[span] = local // modes_per_surface
        mode[span] = index * modes_per_surface + local % modes_per_surface
    solved_surface = surface[solved]
    matrix = _BlockTridiagonal(solved_surface)
    # One probe per (color, mode): group the solved entries once by that key.
    n_modes = int(mode.max()) + 1
    group_key = (solved_surface % 3) * n_modes + mode[solved]
    order = np.argsort(group_key, kind="stable")
    boundaries = np.flatnonzero(np.diff(group_key[order])) + 1
    # probed_position[j + 1]: position in solved of the probed entry on surface j,
    # padded so the neighbours j = -1 and j = ns of the edge surfaces read -1.
    probed_position = np.full(matrix.ns + 2, -1, dtype=np.int64)
    for group in np.split(order, boundaries):
        color = int(group_key[group[0]]) // n_modes
        probe = np.zeros(state_size)
        probe[solved[group]] = 1.0
        product = np.asarray(
            model.exact_hessian_vector_product(np.ascontiguousarray(probe)),
            dtype=np.float64,
        )[solved]
        hit = np.flatnonzero(product)
        hit_surface = solved_surface[hit]
        column_surface = hit_surface - ((hit_surface - color + 1) % 3 - 1)
        probed_position[solved_surface[group] + 1] = group
        column = probed_position[column_surface + 1]
        probed_position[solved_surface[group] + 1] = -1
        found = column >= 0
        # (H^T)[column, row] = H[row, column]
        matrix.set_entries(column[found], hit[found], product[hit[found]])
    return matrix


@dataclass(frozen=True)
class _Factorization:
    """The transposed interior force system of one solved state, LU-factored."""

    interior: np.ndarray
    prescribed: np.ndarray
    gauge: np.ndarray
    matrix: _BlockTridiagonal


def _factorize(model) -> _Factorization:
    """Assemble and factor ``H^T`` on the interior of the model's evaluated state."""
    interior, boundary = _interior_and_boundary(model)
    # The pinned gauge entries are prescribed by the boundary, like the boundary
    # entries themselves: they leave the interior system and are pulled back.
    gauge = _gauge_entries(model)
    interior = np.setdiff1d(interior, gauge)
    prescribed = np.concatenate([boundary, gauge])
    # Deflate the structural null space; without this the transposed
    # interior system is singular and inconsistent in 3D.
    interior = _structural_nullfree_interior(model, interior)
    matrix = _assemble_block_tridiagonal(model, interior)
    matrix.factorize()
    return _Factorization(interior, prescribed, gauge, matrix)


def _implicit_vjps(
    model,
    geometry_bars: np.ndarray,
    with_profiles: bool = False,
    factorization: _Factorization | None = None,
    boundary: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, _Factorization | None]:
    """Boundary and half-grid profile cotangents of each row of ``geometry_bars``, and
    the factorization of the transposed interior system that serves them all.

    ``boundary`` is the differentiated parameter, needed for an asymmetric input.
    """
    if not getattr(model, "has_exact_force_jacobian", False):
        error_message = (
            "This VMEC++ build has no exact residual transpose. Rebuild with "
            "VMECPP_ENABLE_ENZYME to differentiate a solved equilibrium. "
            "No finite-difference derivative is used."
        )
        raise RuntimeError(error_message)
    if not model.always_fix_m1_gauge:
        error_message = (
            "VMEC++ adjoint: the model was solved with a free m=1 gauge; the exact "
            "force Jacobian describes the solve only with always_fix_m1_gauge"
        )
        raise RuntimeError(error_message)
    geometry_bars = np.asarray(geometry_bars, dtype=np.float64).reshape(
        -1, np.shape(geometry_bars)[-1]
    )
    ns = model.ns
    state = np.ascontiguousarray(np.asarray(model.get_state(), dtype=np.float64))
    boundary_bars = []
    profile_bars = []
    for geometry_bar in geometry_bars:
        # The VJP products use the model's state buffers as scratch, so each
        # cotangent starts from the solved state.
        model.set_state(state)
        state_bar = np.asarray(
            model.geometry_state_vjp(geometry_bar[2 * ns :], geometry_bar[ns : 2 * ns]),
            dtype=np.float64,
        )
        model.set_state(state)
        model.evaluate(2, 2, True)
        if factorization is None:
            factorization = _factorize(model)
        interior = factorization.interior
        prescribed = factorization.prescribed
        embedded = np.zeros(state.size)
        embedded[interior] = factorization.matrix.solve(state_bar[interior])
        transposed = np.asarray(
            model.exact_hessian_vector_product_transpose(
                np.ascontiguousarray(embedded)
            ),
            dtype=np.float64,
        )
        # The interior rows of the same product are the residual of the direct
        # solve: a free check that the assembly and the factorization are sound.
        residual = float(np.linalg.norm(transposed[interior] - state_bar[interior]))
        residual_scale = max(float(np.linalg.norm(state_bar[interior])), 1.0e-300)
        if residual > 1.0e-6 * residual_scale:
            error_message = (
                "VMEC++ implicit adjoint solve failed: relative residual "
                f"{residual / residual_scale:.3e}"
            )
            raise RuntimeError(error_message)
        full_state_bar = np.zeros(state.size)
        full_state_bar[prescribed] = state_bar[prescribed] - transposed[prescribed]
        full_state_bar = _pull_gauge_back_to_boundary(
            model, full_state_bar, factorization.gauge
        )
        profile_bar = np.zeros((3, ns - 1))
        if with_profiles:
            # dJ/dp = J_p - adjoint^T F_p, J_p through the poloidal flux
            poloidal_flux_bar = geometry_bar[ns : 2 * ns]
            profile_bar = np.stack(model.profile_vjp(-embedded, poloidal_flux_bar))
            profile_bar[1] *= -1.0 if model.have_to_flip_theta else 1.0
        boundary_bars.append(_boundary_from_state_vjp(model, full_state_bar, boundary))
        profile_bars.append(profile_bar)
    return np.stack(boundary_bars), np.stack(profile_bars), factorization


def _implicit_vjp(
    model,
    geometry_bar: np.ndarray,
    with_profiles: bool = False,
    boundary: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The boundary cotangent and that of :func:`autodiff_wout.half_grid_profiles`."""
    boundary_bars, profile_bars, _ = _implicit_vjps(
        model, np.asarray(geometry_bar)[None], with_profiles, boundary=boundary
    )
    return boundary_bars[0], profile_bars[0]


@dataclass(frozen=True)
class DifferentiableVmec:
    """A callable JAX view of one fixed-boundary VMEC++ input.

    The exact VJP covers the boundary coefficients and the power-series profile
    parameters passed as ``profiles`` (see
    :data:`autodiff_wout.PROFILE_PARAMETERS`).
    """

    vmec_input: Any
    _solver_id: int = field(
        default_factory=lambda: next(_SOLVER_IDS),
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.vmec_input.lfreeb:
            error_message = "DifferentiableVmec currently requires lfreeb=false"
            raise ValueError(error_message)
        if not isinstance(self.vmec_input.mpol, int) or not isinstance(
            self.vmec_input.ntor, int
        ):
            error_message = "DifferentiableVmec currently requires scalar mpol and ntor"
            raise ValueError(error_message)
        resolutions = np.asarray(self.vmec_input.ns_array)
        if resolutions.size == 0 or resolutions[-1] < 3:
            error_message = "DifferentiableVmec requires an ns_array entry >= 3"
            raise ValueError(error_message)

    @property
    def parameter_shape(self) -> tuple[int, int, int]:
        """Rows ``rbc``, ``zbs`` and, for ``lasym``, ``rbs``, ``zbc``."""
        rows = 4 if self.vmec_input.lasym else 2
        return (rows, self.vmec_input.mpol, 2 * self.vmec_input.ntor + 1)

    @property
    def ns(self) -> int:
        return int(np.asarray(self.vmec_input.ns_array)[-1])

    @property
    def geometry_size(self) -> int:
        modes = self.vmec_input.mpol * (self.vmec_input.ntor + 1)
        return 2 * self.ns + len(_GEOMETRY_COEFFICIENTS) * self.ns * modes

    @property
    def extra_size(self) -> int:
        """Size of the non-differentiated outputs that follow the geometry."""
        return 0

    @property
    def output_shape(self) -> tuple[int]:
        return (self.geometry_size + self.extra_size,)

    def _remember(self, boundary: np.ndarray, profiles: dict, solve: dict) -> None:
        # A solve at other parameters releases the held factorization.
        for other in _FORWARD_SOLVES.values():
            other.pop("factorization", None)
        _FORWARD_SOLVES[_key(self._solver_id, boundary, profiles)] = solve
        while len(_FORWARD_SOLVES) > _FORWARD_SOLVES_SIZE:
            _FORWARD_SOLVES.popitem(last=False)

    def _cached(self, boundary: np.ndarray, profiles: dict) -> dict | None:
        key = _key(self._solver_id, boundary, profiles)
        solve = _FORWARD_SOLVES.get(key)
        if solve is not None:
            _FORWARD_SOLVES.move_to_end(key)
        return solve

    def _forward_callback(self, boundary: np.ndarray, profiles=None) -> np.ndarray:
        profiles = profiles or {}
        solve = self._cached(boundary, profiles)
        if solve is not None and "output" in solve:
            return solve["output"]
        model = _solve_model(self.vmec_input._to_cpp_vmecindata(), boundary, profiles)
        output = _cpp_geometry_flat(
            model.get_geometry(), model.ns, model.mpol, model.ntor
        )
        self._remember(
            boundary, profiles, {"state": np.array(model.get_state()), "output": output}
        )
        return output

    def _solved_state(self, boundary: np.ndarray, profiles: dict) -> np.ndarray:
        """The state vector of a fresh forward solve, for a cache miss."""
        model = _solve_model(self.vmec_input._to_cpp_vmecindata(), boundary, profiles)
        return np.array(model.get_state())

    def _solve_for_vjp(self, boundary: np.ndarray, profiles: dict) -> dict:
        """The cache entry of the forward solve, solving on a miss."""
        solve = self._cached(boundary, profiles)
        if solve is None:
            solve = {"state": self._solved_state(boundary, profiles)}
            self._remember(boundary, profiles, solve)
        return solve

    def _solved_model(self, boundary: np.ndarray, profiles: dict | None = None):
        """A model at the forward solve's state, for the VJP.

        A model created cold and set to a solved state reproduces the solved model's
        forces, Hessian products and preconditioner exactly.
        """
        profiles = profiles or {}
        state = self._solve_for_vjp(boundary, profiles)["state"]
        indata = _make_indata(self.vmec_input._to_cpp_vmecindata(), boundary, profiles)
        indata.always_fix_m1_gauge = True
        model = _vmecpp.VmecModel.create(indata, self.ns)
        model.set_state(np.ascontiguousarray(state))
        return model

    def _backward_callback(
        self, boundary: np.ndarray, geometry_bars: np.ndarray, profiles=None
    ) -> tuple[np.ndarray, np.ndarray]:
        """The boundary and profile cotangents of each row of ``geometry_bars``."""
        profiles = profiles or {}
        model = self._solved_model(boundary, profiles)
        solve = self._solve_for_vjp(boundary, profiles)
        # One factorization is held at a time: another point's is released before
        # this one is built.
        for other in _FORWARD_SOLVES.values():
            if other is not solve:
                other.pop("factorization", None)
        boundary_bars, profile_bars, solve["factorization"] = _implicit_vjps(
            model,
            geometry_bars,
            with_profiles=bool(profiles),
            factorization=solve.get("factorization"),
            boundary=boundary,
        )
        return boundary_bars, profile_bars

    def __call__(self, boundary, profiles=None) -> geometry.Geometry:
        """The geometry for ``boundary`` and optional ``profiles`` values."""
        return self._solve(boundary, profiles)[0]

    def _solve(self, boundary, profiles=None) -> tuple[geometry.Geometry, jax.Array]:
        """The geometry and the non-differentiated outputs of :attr:`extra_size`."""
        boundary = jnp.asarray(boundary, dtype=jnp.float64)
        if boundary.shape != self.parameter_shape:
            error_message = (
                f"boundary has shape {boundary.shape}, expected {self.parameter_shape}"
            )
            raise ValueError(error_message)
        ns = self.ns
        profiles = profiles or {}
        # the solve sees the coefficients; its VJP is taken with respect to the
        # half-grid profiles, from which JAX differentiates the coefficients
        values = jax.lax.stop_gradient(profiles)
        half = autodiff_wout.half_grid_profiles(self.vmec_input, ns, profiles)
        output_spec = jax.ShapeDtypeStruct(self.output_shape, boundary.dtype)
        cotangent_spec = (
            jax.ShapeDtypeStruct(boundary.shape, boundary.dtype),
            jax.ShapeDtypeStruct(half.shape, boundary.dtype),
        )

        def concrete(values):
            return {name: np.asarray(value) for name, value in values.items()}

        def forward_callback(value, values):
            flat = self._forward_callback(np.asarray(value), concrete(values))
            return flat.astype(np.asarray(value).dtype, copy=False)

        def backward_callback(value, values, cotangent):
            # Batch axes lead every argument, with size one where unbatched.
            # Cotangents at one parameter point share one factorization.
            value = np.asarray(value)
            values = concrete(values)
            cotangent = np.asarray(cotangent)
            lead = cotangent.ndim - 1
            parameter_batches = [value.shape[:lead]] + [
                array.shape[:lead] for array in values.values()
            ]
            batch = np.broadcast_shapes(cotangent.shape[:lead], *parameter_batches)
            geometry_bars = np.broadcast_to(
                cotangent[..., : self.geometry_size], (*batch, self.geometry_size)
            ).reshape(-1, self.geometry_size)
            if all(size == 1 for shape in parameter_batches for size in shape):
                bars = self._backward_callback(
                    value.reshape(value.shape[lead:]),
                    geometry_bars,
                    {
                        name: array.reshape(array.shape[lead:])
                        for name, array in values.items()
                    },
                )
            else:
                points = np.broadcast_to(value, (*batch, *value.shape[lead:])).reshape(
                    -1, *value.shape[lead:]
                )
                point_values = {
                    name: np.broadcast_to(array, (*batch, *array.shape[lead:])).reshape(
                        -1, *array.shape[lead:]
                    )
                    for name, array in values.items()
                }
                per_point = [
                    self._backward_callback(
                        points[k],
                        geometry_bars[k : k + 1],
                        {name: array[k] for name, array in point_values.items()},
                    )
                    for k in range(points.shape[0])
                ]
                bars = tuple(
                    np.concatenate([bar[i] for bar in per_point]) for i in range(2)
                )
            return tuple(
                bar.reshape(*batch, *bar.shape[1:]).astype(value.dtype) for bar in bars
            )

        def forward(value, values):
            return jax.pure_callback(
                forward_callback, output_spec, value, values, vmap_method="sequential"
            )

        @jax.custom_vjp
        def solve_flat(value, half, values):
            del half
            return forward(value, values)

        def solve_fwd(value, half, values):
            del half
            return forward(value, values), (value, values)

        def solve_bwd(residuals, cotangent):
            value, values = residuals
            bars = jax.pure_callback(
                backward_callback,
                cotangent_spec,
                value,
                values,
                cotangent,
                vmap_method="expand_dims",
            )
            return (*bars, jax.tree.map(jnp.zeros_like, values))

        solve_flat.defvjp(solve_fwd, solve_bwd)
        return _geometry_from_flat(solve_flat(boundary, half, values), self.vmec_input)


def make_solver(vmec_input) -> DifferentiableVmec:
    """Return a JAX-compatible callable that runs VMEC++ for vmec_input.

    Example:

        from vmecpp import autodiff, simsopt_compat

        solver = autodiff.make_solver(input)
        objective = lambda boundary: simsopt_compat.quasisymmetry_total(
            solver(boundary), [0.6]
        )
        value, gradient = jax.value_and_grad(objective)(boundary)

    Forward execution and the VJP both invoke VMEC++ in memory. The VJP is
    available only in an Enzyme-enabled build, because it requires the exact
    transpose of the force residual. No finite-difference derivative is used.
    """
    if not _vmecpp.VMECPP_ENABLE_ENZYME:
        error_message = (
            "vmecpp.autodiff.make_solver requires a build with the exact force "
            "Jacobian; rebuild with the CMake option VMECPP_ENABLE_ENZYME=ON. "
            "No finite-difference derivative is used."
        )
        raise RuntimeError(error_message)
    return DifferentiableVmec(vmec_input)


@dataclass(frozen=True)
class _RunSolver(DifferentiableVmec):
    """:class:`DifferentiableVmec` whose forward solve is a pinned-gauge C++ run.

    The C++ output of the forward solve supplies what the JAX output stage does
    not compute (mercier, threed1, the solver diagnostics) and, as the
    outputs after the geometry, the half-grid mass, iota and buco profiles and the
    sign the solver gives iota. The VJP
    linearizes at the state of a ``VmecModel`` hot-restarted from the forward
    solve's wout, which reproduces it to roundoff.
    """

    max_threads: int | None = None
    verbose: int = 0

    @property
    def extra_size(self) -> int:
        return 3 * (self.ns - 1) + 1

    def _run(self, boundary: np.ndarray, profiles: dict):
        indata = _make_indata(self.vmec_input._to_cpp_vmecindata(), boundary, profiles)
        indata.always_fix_m1_gauge = True
        output = _vmecpp.run(
            indata,
            max_threads=self.max_threads,
            verbose=_vmecpp.OutputMode(self.verbose),
        )
        initial_state = _vmecpp.HotRestartState(wout=output.wout, indata=indata)
        model = _vmecpp.VmecModel.create(indata, self.ns, initial_state=initial_state)
        flip = -1.0 if model.have_to_flip_theta else 1.0
        return output, np.array(model.get_state()), flip

    def _forward_callback(self, boundary: np.ndarray, profiles=None) -> np.ndarray:
        import vmecpp  # noqa: PLC0415

        profiles = profiles or {}
        solve = self._cached(boundary, profiles)
        if solve is not None and "output" in solve:
            return solve["output"]
        output, state, flip = self._run(boundary, profiles)
        wout = vmecpp.VmecWOut._from_cpp_wout(output.wout)
        geometry_flat = _cpp_geometry_flat(
            _vmecpp.make_geometry(output),
            self.ns,
            self.vmec_input.mpol,
            self.vmec_input.ntor,
        )
        mass_half = np.asarray(wout.mass, dtype=np.float64)[1:] * _MU_0
        iota_half = np.asarray(wout.iotas, dtype=np.float64)[1:]
        buco_half = np.asarray(wout.buco, dtype=np.float64)[1:]
        extra = [mass_half, iota_half, buco_half, [flip]]
        flat = np.concatenate([geometry_flat, *extra])
        self._remember(
            boundary,
            profiles,
            {
                "state": state,
                "wout_fields": dict(wout.__dict__),
                "tables": vmecpp._output_tables_from_cpp(output),
                "output": flat,
            },
        )
        return flat

    def _solved_state(self, boundary: np.ndarray, profiles: dict) -> np.ndarray:
        return self._run(boundary, profiles)[1]

    def forward_outputs(self) -> dict[str, Any] | None:
        """The C++ outputs of this solver's latest forward solve, if it has run."""
        for (solver_id, _), solve in reversed(_FORWARD_SOLVES.items()):
            if solver_id == self._solver_id:
                return solve
        return None


__all__ = ["DifferentiableVmec", "make_solver"]
