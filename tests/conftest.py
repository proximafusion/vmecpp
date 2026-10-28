"""Fixtures shared by the test files."""

from typing import Any

import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator, gmres

from vmecpp import autodiff
from vmecpp.cpp import _vmecpp  # type: ignore

_BOUNDARY_ROWS = ("rbc", "zbs", "rbs", "zbc")


def _parser_state_tangent(indata, boundary, direction):
    """The state tangent of the fixed-boundary parser, by a central difference."""
    ns = int(np.asarray(indata.ns_array)[-1])
    step = 1.0e-6

    def state(value):
        rows = _BOUNDARY_ROWS[: len(value)]
        perturbed = indata.model_copy(update=dict(zip(rows, value, strict=True)))
        model = _vmecpp.VmecModel.create(perturbed._to_cpp_vmecindata(), ns)
        return np.asarray(model.get_state(), dtype=np.float64)

    return (state(boundary + step * direction) - state(boundary - step * direction)) / (
        2.0 * step
    )


def _geometry_tangent_through_the_solve(indata, boundary, seed_state):
    """The geometry tangent of the solve, solved with H where the adjoint uses H^T."""
    model = autodiff._solve_model(indata._to_cpp_vmecindata(), boundary)
    state = np.asarray(model.get_state(), dtype=np.float64)
    interior, edge = autodiff._interior_and_boundary(model)
    # The pinned m=1 gauge entries are prescribed by the boundary, like the
    # edge entries; the parser tangent carries their sqrt(s) profile.
    gauge = autodiff._gauge_entries(model)
    edge = np.concatenate([edge, gauge])
    interior = np.setdiff1d(interior, gauge)
    model.set_state(np.ascontiguousarray(state))
    model.evaluate(2, 2, True)
    keep = autodiff._structural_nullfree_interior(model, interior)
    size = state.size

    def forward(value):
        return np.asarray(
            model.exact_hessian_vector_product(np.ascontiguousarray(value)),
            dtype=np.float64,
        )

    def restricted(value):
        embedded = np.zeros(size)
        embedded[keep] = value
        return forward(embedded)[keep]

    def precondition(value):
        embedded = np.zeros(size)
        embedded[keep] = value
        return np.asarray(
            model.apply_preconditioner(np.ascontiguousarray(embedded)),
            dtype=np.float64,
        )[keep]

    factory: Any = LinearOperator
    operator = factory((keep.size, keep.size), matvec=restricted, dtype=np.float64)
    preconditioner = factory(
        (keep.size, keep.size), matvec=precondition, dtype=np.float64
    )
    seeded = np.zeros(size)
    seeded[edge] = seed_state[edge]
    tangent, info = gmres(
        operator,
        -forward(seeded)[keep],
        M=preconditioner,
        rtol=1.0e-12,
        restart=200,
        maxiter=400,
    )
    assert info == 0
    state_tangent = np.zeros(size)
    state_tangent[keep] = tangent
    state_tangent[edge] = seed_state[edge]

    # with ncurr = 1 the evaluation updates the poloidal flux from the state
    step = 1.0e-6

    def flat(value):
        model.set_state(np.ascontiguousarray(value))
        model.evaluate(2, 2, False)
        return autodiff._cpp_geometry_flat(
            model.get_geometry(), model.ns, model.mpol, model.ntor
        )

    return (flat(state + step * state_tangent) - flat(state - step * state_tangent)) / (
        2.0 * step
    )


@pytest.fixture
def forward_sensitivity():
    """The flat geometry tangent of the solve along a boundary direction."""

    def sensitivity(indata, boundary, direction):
        boundary = np.asarray(boundary, dtype=np.float64)
        seed_state = _parser_state_tangent(indata, boundary, direction)
        return _geometry_tangent_through_the_solve(indata, boundary, seed_state)

    return sensitivity
