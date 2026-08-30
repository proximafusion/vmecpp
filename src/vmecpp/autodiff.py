"""JAX access to an in-memory VMEC++ solve and its implicit VJP.

The solver is deliberately kept outside the JAX trace. A forward call runs
VMEC++ through the C++ model, while the reverse callback reruns the same model
and solves the transposed interior force system. This is the usual implicit
layer for a differentiable code: JAX differentiates the consumer objective,
and VMEC++ supplies the producer's residual transpose.

The first public parameterization is the fixed-boundary, prescribed-iota case.
The differentiable parameter is one dense array with rows ``rbc`` and ``zbs``
and shape ``(2, mpol, 2 * ntor + 1)``. This first solver wrapper deliberately
supports the stellarator-symmetric fixed-boundary, prescribed-iota case. The
geometry API itself already supports asymmetric snapshots; profile and
free-boundary parameter VJPs remain explicit unsupported cases until their
residual dependence is exposed by the exact C++ derivative path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from vmecpp import geometry
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


def _make_indata(template, boundary: np.ndarray):
    indata = template.copy()
    indata.rbc[...] = boundary[0]
    indata.zbs[...] = boundary[1]
    return indata


def _solve_model(template, boundary: np.ndarray):
    """Run all requested VMEC++ resolutions and return the final model."""
    indata = _make_indata(template, boundary)
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


def _boundary_from_state_vjp(model, state_bar: np.ndarray) -> np.ndarray:
    """Transpose the fixed-boundary parser for a symmetric input."""
    if model.lasym:
        error_message = (
            "DifferentiableVmec currently requires lasym=false; the asymmetric "
            "poloidal-origin shift is not a differentiable parameterization"
        )
        raise RuntimeError(error_message)
    slices = _span_slices(model)
    mpol = model.mpol
    ntor = model.ntor
    modes_per_surface = mpol * (ntor + 1)
    edge = model.ns - 1
    result = np.zeros((2, mpol, 2 * ntor + 1), dtype=np.float64)

    def edge_value(name: str, m: int, n: int) -> float:
        if name not in slices:
            return 0.0
        values = state_bar[slices[name]]
        return float(values[edge * modes_per_surface + m * (ntor + 1) + n])

    # Undo the state scaling and the m=1 gauge transform used by
    # Boundaries::ensureM1Constrained. The boundary parser itself is the
    # transpose of the positive/negative-toroidal-mode accumulation below.
    rbcc_bar = np.zeros((mpol, ntor + 1))
    zbsc_bar = np.zeros((mpol, ntor + 1))
    rbss_bar = np.zeros((mpol, ntor + 1))
    zbcs_bar = np.zeros((mpol, ntor + 1))
    for m in range(mpol):
        for n in range(ntor + 1):
            scale = (1.0 if m == 0 else np.sqrt(2.0)) * (
                1.0 if n == 0 else np.sqrt(2.0)
            )
            rbcc_bar[m, n] = edge_value("r_cc", m, n) / scale
            zbsc_bar[m, n] = edge_value("z_sc", m, n) / scale
            if model.lthreed:
                rbss_bar[m, n] = edge_value("r_ss", m, n) / scale
                zbcs_bar[m, n] = edge_value("z_cs", m, n) / scale
    if model.lthreed and mpol > 1:
        for n in range(ntor + 1):
            r_bar = rbss_bar[1, n]
            z_bar = zbcs_bar[1, n]
            rbss_bar[1, n] = 0.5 * (r_bar + z_bar)
            zbcs_bar[1, n] = 0.5 * (r_bar - z_bar)

    if model.have_to_flip_theta:
        for m in range(1, mpol):
            parity = 1.0 if m % 2 == 0 else -1.0
            rbcc_bar[m] *= parity
            zbsc_bar[m] *= -parity
            if model.lthreed:
                rbss_bar[m] *= -parity
                zbcs_bar[m] *= parity

    for m in range(mpol):
        for signed_n in range(-ntor, ntor + 1):
            source = ntor + signed_n
            target = abs(signed_n)
            sign = 1.0 if signed_n > 0 else -1.0 if signed_n < 0 else 0.0
            result[0, m, source] += rbcc_bar[m, target]
            if model.lthreed and m > 0:
                result[0, m, source] += sign * rbss_bar[m, target]
            if m > 0:
                result[1, m, source] += zbsc_bar[m, target]
            if model.lthreed:
                result[1, m, source] -= sign * zbcs_bar[m, target]
    return result


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


def _implicit_boundary_vjp(model, geometry_bar: np.ndarray) -> np.ndarray:
    if not getattr(model, "has_exact_force_jacobian", False):
        error_message = (
            "This VMEC++ build has no exact residual transpose. Rebuild with "
            "VMECPP_ENABLE_ENZYME to differentiate a solved equilibrium. "
            "No finite-difference derivative is used."
        )
        raise RuntimeError(error_message)
    coefficient_bar = np.asarray(geometry_bar[2 * model.ns :], dtype=np.float64)
    state_bar = np.asarray(model.geometry_state_vjp(coefficient_bar), dtype=np.float64)
    state = np.asarray(model.get_state(), dtype=np.float64)
    interior, boundary = _interior_and_boundary(model)
    try:
        model.set_state(np.ascontiguousarray(state))
        model.set_freeze_constraint_multiplier(True)
        model.evaluate(2, 2, True)
        state_size = state.size
        # Deflate the structural null space; without this the transposed
        # interior system is singular and inconsistent in 3D.
        interior = _structural_nullfree_interior(model, interior)

        def transpose(value: np.ndarray) -> np.ndarray:
            return np.asarray(
                model.exact_hessian_vector_product_transpose(
                    np.ascontiguousarray(value)
                ),
                dtype=np.float64,
            )

        def matvec(value: np.ndarray) -> np.ndarray:
            embedded = np.zeros(state_size)
            embedded[interior] = value
            return transpose(embedded)[interior]

        def precondition(value: np.ndarray) -> np.ndarray:
            embedded = np.zeros(state_size)
            embedded[interior] = value
            return np.asarray(
                model.apply_preconditioner(np.ascontiguousarray(embedded)),
                dtype=np.float64,
            )[interior]

        operator_factory: Any = LinearOperator
        operator = operator_factory(
            (interior.size, interior.size), matvec=matvec, dtype=np.float64
        )
        preconditioner = operator_factory(
            (interior.size, interior.size), matvec=precondition, dtype=np.float64
        )
        adjoint, info = gmres(
            operator,
            state_bar[interior],
            M=preconditioner,
            rtol=1.0e-8,
            restart=200,
            maxiter=400,
        )
        if info != 0:
            error_message = f"VMEC++ implicit adjoint solve failed with info={info}"
            raise RuntimeError(error_message)
        embedded = np.zeros(state_size)
        embedded[interior] = adjoint
        internal_boundary_bar = state_bar[boundary] - transpose(embedded)[boundary]
        full_state_bar = np.zeros(state_size)
        full_state_bar[boundary] = internal_boundary_bar
        return _boundary_from_state_vjp(model, full_state_bar)
    finally:
        model.set_freeze_constraint_multiplier(False)


@dataclass(frozen=True)
class DifferentiableVmec:
    """A callable JAX view of one fixed-boundary VMEC++ input.

    The current exact VJP covers the boundary coefficients. Pressure, iota,
    current, and flux parameters are intentionally not accepted as hidden
    constants: exposing them requires their residual derivatives in the C++
    contract, rather than a finite-difference fallback.
    """

    vmec_input: Any

    def __post_init__(self) -> None:
        if self.vmec_input.lfreeb:
            error_message = "DifferentiableVmec currently requires lfreeb=false"
            raise ValueError(error_message)
        if self.vmec_input.ncurr != 0:
            error_message = "DifferentiableVmec currently requires ncurr=0"
            raise ValueError(error_message)
        if self.vmec_input.lasym:
            error_message = (
                "DifferentiableVmec currently requires lasym=false; use the "
                "product-basis geometry API for asymmetric snapshots"
            )
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
        return (2, self.vmec_input.mpol, 2 * self.vmec_input.ntor + 1)

    @property
    def output_shape(self) -> tuple[int]:
        ns = int(np.asarray(self.vmec_input.ns_array)[-1])
        modes = self.vmec_input.mpol * (self.vmec_input.ntor + 1)
        return (2 * ns + len(_GEOMETRY_COEFFICIENTS) * ns * modes,)

    def _forward_callback(self, boundary: np.ndarray) -> np.ndarray:
        model = _solve_model(self.vmec_input._to_cpp_vmecindata(), boundary)
        return _cpp_geometry_flat(
            model.get_geometry(), model.ns, model.mpol, model.ntor
        )

    def _backward_callback(
        self, boundary: np.ndarray, geometry_bar: np.ndarray
    ) -> np.ndarray:
        model = _solve_model(self.vmec_input._to_cpp_vmecindata(), boundary)
        return _implicit_boundary_vjp(model, geometry_bar)

    def __call__(self, boundary) -> geometry.Geometry:
        boundary = jnp.asarray(boundary, dtype=jnp.float64)
        if boundary.shape != self.parameter_shape:
            error_message = (
                f"boundary has shape {boundary.shape}, expected {self.parameter_shape}"
            )
            raise ValueError(error_message)
        output_spec = jax.ShapeDtypeStruct(self.output_shape, boundary.dtype)
        parameter_spec = jax.ShapeDtypeStruct(boundary.shape, boundary.dtype)

        def forward_callback(value):
            return self._forward_callback(np.asarray(value)).astype(
                np.asarray(value).dtype, copy=False
            )

        def backward_callback(value, cotangent):
            return self._backward_callback(
                np.asarray(value), np.asarray(cotangent)
            ).astype(np.asarray(value).dtype, copy=False)

        @jax.custom_vjp
        def solve_flat(value):
            return jax.pure_callback(
                forward_callback, output_spec, value, vmap_method="sequential"
            )

        def solve_fwd(value):
            result = jax.pure_callback(
                forward_callback, output_spec, value, vmap_method="sequential"
            )
            return result, value

        def solve_bwd(value, cotangent):
            result = jax.pure_callback(
                backward_callback,
                parameter_spec,
                value,
                cotangent,
                vmap_method="sequential",
            )
            return (result,)

        solve_flat.defvjp(solve_fwd, solve_bwd)
        flat = solve_flat(boundary)
        ns = int(np.asarray(self.vmec_input.ns_array)[-1])
        mpol = self.vmec_input.mpol
        ntor = self.vmec_input.ntor
        modes = ns * mpol * (ntor + 1)
        arrays = [flat[:ns], flat[ns : 2 * ns]]
        offset = 2 * ns
        for _ in _GEOMETRY_COEFFICIENTS:
            arrays.append(flat[offset : offset + modes].reshape(ns, mpol, ntor + 1))
            offset += modes
        return geometry.Geometry(*arrays, nfp=self.vmec_input.nfp)


def make_solver(vmec_input) -> DifferentiableVmec:
    """Return a JAX-compatible callable that runs VMEC++ for vmec_input.

    Example:

        from vmecpp import autodiff

        solver = autodiff.make_solver(input)
        objective = lambda boundary: qs.quasisymmetry_residual(
            solver(boundary), 0.6
        )
        value, gradient = jax.value_and_grad(objective)(boundary)

    Forward execution and the VJP both invoke VMEC++ in memory. The VJP is
    available only in an Enzyme-enabled build, because it requires the exact
    transpose of the force residual. No finite-difference derivative is used.
    """
    return DifferentiableVmec(vmec_input)


__all__ = ["DifferentiableVmec", "make_solver"]
