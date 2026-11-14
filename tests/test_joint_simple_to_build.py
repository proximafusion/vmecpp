# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""examples/joint_simple_to_build.py against ConStellaration, the gradient of an
analytic field and differences of re-converged equilibria."""

import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import vmecpp
from vmecpp.joint import CONSTRAINT_TOLERANCE, boundary_modes, optimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from joint_quasi_isodynamic import (  # type: ignore
    BoozerTables,
    boozer_strength,
    converged_wout,
    near_axis_start,
)
from joint_simple_to_build import (  # type: ignore
    _WOUT_QUANTITIES,
    SimpleToBuildProblem,
    gradient_scale_lengths,
    score,
    shuffled_well_residuals,
)

jax.config.update("jax_enable_x64", True)

MAJOR, MINOR, NFP = 2.0, 0.5, 2


def _quantities(wout) -> dict:
    return {name: np.asarray(getattr(wout, name)) for name in _WOUT_QUANTITIES}


@pytest.fixture(scope="module")
def wout():
    """The near-axis start converged on 25 surfaces."""
    vmec_input = near_axis_start(5, 5, 25)
    vmec_input.ftol_array = np.asarray([1.0e-14])
    return vmecpp.run(vmec_input, verbose=False, max_threads=1).wout


@pytest.fixture(scope="module")
def spectrum(wout):
    """``a`` and ``d`` of the field strength of the outermost half-grid surface of the
    wout in Boozer coordinates, its rotational transform and whether the benchmark
    starts its field lines half a period on."""
    tables = BoozerTables.of(wout, 6 * wout.mpol, 2 * wout.ntor - 1)
    columns = [
        np.asarray(getattr(wout, name))[:, wout.ns - 1]
        for name in ("bmnc", "bsubumnc", "bsubvmnc", "lmns")
    ]
    iota = float(wout.iotas[wout.ns - 1])
    a, d = boozer_strength(tables, *columns, iota)
    return a, d, iota, bool(np.asarray(wout.bmnc)[1, 2] < 0)


def test_the_gradient_scale_length_matches_constellaration(wout) -> None:
    """ConStellaration's ``normalized_magnetic_gradient_scale_length`` times the number
    of field periods for the same wout, on its own grid for these modes."""
    lengths = np.asarray(
        gradient_scale_lengths(_quantities(wout), wout, 3, n_theta=16, n_phi=14)
    )
    np.testing.assert_allclose(
        [lengths.min(), lengths.max(), lengths[3, 5], lengths.mean()],
        [10.1591276525, 28.7419796110, 13.2816263201, 15.0038912695],
        rtol=1.0e-5,
    )


def test_the_residuals_of_quasi_isodynamicity_match_constellaration(spectrum) -> None:
    """ConStellaration's ``quasi_isodynamicity_residual`` for the same wout with the
    Boozer transformation of booz_xform: the sum of the squares of its 150,000
    residuals, which is the ``qi`` of the benchmark, the largest of them and two of
    them."""
    residuals = np.asarray(shuffled_well_residuals(*spectrum[:3], 3, spectrum[3]))
    assert residuals.shape == (150000,)
    np.testing.assert_allclose(
        [residuals @ residuals, np.abs(residuals).max()],
        [5.012176609184e-03, 4.4367471125e-04],
        rtol=1.0e-5,
    )
    np.testing.assert_allclose(
        residuals[[1000, 77777]], [-4.3533712212e-04, 2.3092219457e-04], rtol=1.0e-4
    )


def test_wells_of_one_shape_keep_their_points() -> None:
    """Without rotational transform, ``1 + 0.2 cos(nfp zeta) + 0.05 cos(theta)`` has the
    same well on every field line, raised by the line.

    Stretched to span 0 to 1, the well is ``(1 + cos(nfp zeta)) / 2`` on each line, so
    no point of a level moves, and the residuals are that well less the scaled field
    strength, within the straight lines between the points of 401 levels.
    """
    lines, points = 9, 400
    a, d = np.zeros((2, 2)), np.zeros((2, 2))
    a[0, 0], a[0, 1], a[1, 0] = 1.0, 0.2, 0.05
    residuals = np.asarray(
        shuffled_well_residuals(
            a, d, 0.0, 3, False, field_lines=lines, output_points=points
        )
    ).reshape(lines, points)
    theta = np.linspace(0.0, 2.0 * np.pi, lines)[:, None]
    zeta = np.linspace(0.0, 2.0 * np.pi / 3.0, points)[None, :]
    np.testing.assert_allclose(
        residuals * np.sqrt(lines * points),
        0.1 * (np.cos(3.0 * zeta) - np.cos(theta)),
        atol=1.0e-3,
    )


def test_the_residuals_of_quasi_isodynamicity_have_the_derivative_of_their_differences(
    spectrum,
) -> None:
    """The sum of their squares with respect to the mode (1, 1) of the field strength,
    through the wells, the points of their levels and their order."""
    a, d, iota, half_period = spectrum

    def total(amplitude):
        shifted = jnp.asarray(a).at[1, 1].add(amplitude)
        residuals = shuffled_well_residuals(shifted, d, iota, 3, half_period)
        return residuals @ residuals

    step = 1.0e-6
    difference = (total(step) - total(-step)) / (2.0 * step)
    np.testing.assert_allclose(jax.grad(total)(0.0), difference, rtol=1.0e-5)


def _components(s):
    """The components of a field along the poloidal and the toroidal angle of the
    circular surfaces of a torus by mode ``(m, n)``, on the surface ``s``."""
    return (
        {(0, 0): 0.3 + 0.1 * s, (1, 0): 0.1 * s, (2, 1): 0.05 * s**2},
        {(0, 0): 1.0 - 0.2 * s, (1, 0): 0.1 * s, (1, 1): 0.05 * s},
    )


def _series(part, theta, phi):
    return sum(
        value * jnp.cos(m * theta - n * NFP * phi) for (m, n), value in part.items()
    )


def _field(point):
    """The field of :func:`_components` at a point of space."""
    x, y, z = point
    radius, phi = jnp.hypot(x, y), jnp.arctan2(y, x)
    rho, theta = jnp.hypot(radius - MAJOR, z), jnp.arctan2(z, radius - MAJOR)
    poloidal, toroidal = (
        _series(part, theta, phi) for part in _components((rho / MINOR) ** 2)
    )
    along_theta = jnp.stack(
        [
            -rho * jnp.sin(theta) * jnp.cos(phi),
            -rho * jnp.sin(theta) * jnp.sin(phi),
            rho * jnp.cos(theta),
        ]
    )
    along_phi = jnp.stack([-radius * jnp.sin(phi), radius * jnp.cos(phi), 0.0 * z])
    return poloidal * along_theta + toroidal * along_phi


def _analytic_wout(ns: int, m_nyq: int = 14, n_nyq: int = 5):
    """The field of :func:`_components` and its surfaces as the quantities of a wout on
    ``ns`` surfaces, with their modes."""
    full = np.linspace(0.0, 1.0, ns)
    half = np.concatenate([[0.0], 0.5 * (full[1:] + full[:-1])])
    modes = [(0, n) for n in range(n_nyq + 1)] + [
        (m, n) for m in range(1, m_nyq + 1) for n in range(-n_nyq, n_nyq + 1)
    ]
    table = {mode: index for index, mode in enumerate(modes)}
    wout = {name: np.zeros((len(modes), ns)) for name in _WOUT_QUANTITIES[2:6]}
    for name, part in zip(("bsupumnc", "bsupvmnc"), _components(half), strict=True):
        for mode, value in part.items():
            wout[name][table[mode]] = value
    # the Jacobian of a circular cross-section of radius MINOR sqrt(s)
    wout["gmnc"][table[(0, 0)]] = -0.5 * MINOR**2 * MAJOR
    wout["gmnc"][table[(1, 0)]] = -0.5 * MINOR**3 * np.sqrt(half)
    # the field strength by its Fourier coefficients on a grid finer than its modes
    theta, phi = np.meshgrid(
        2.0 * np.pi * np.arange(4 * m_nyq) / (4 * m_nyq),
        2.0 * np.pi * np.arange(4 * n_nyq) / (4 * n_nyq * NFP),
        indexing="ij",
    )
    rho = MINOR * np.sqrt(half)[:, None, None]
    poloidal, toroidal = (
        np.asarray(_series(part, theta, phi))
        for part in _components(half[:, None, None])
    )
    strength = np.hypot(poloidal * rho, toroidal * (MAJOR + rho * np.cos(theta)))
    for (m, n), index in table.items():
        wave = np.cos(m * theta - n * NFP * phi)
        wout["bmnc"][index] = (1.0 if (m, n) == (0, 0) else 2.0) * (
            strength * wave
        ).mean(axis=(1, 2))
    wout["rmnc"] = np.stack([np.full(ns, MAJOR), MINOR * np.sqrt(full)])
    wout["zmns"] = np.stack([np.zeros(ns), MINOR * np.sqrt(full)])
    wout["Aminor_p"] = MINOR
    return wout, SimpleNamespace(
        xm=np.asarray([0.0, 1.0]),
        xn=np.asarray([0.0, 0.0]),
        xm_nyq=np.asarray([float(m) for m, _ in modes]),
        xn_nyq=np.asarray([float(n * NFP) for _, n in modes]),
    )


def test_the_gradient_scale_length_is_that_of_the_gradient_of_a_field() -> None:
    """A field tangent to the circular surfaces of a torus, whose components vary with
    the surface and both angles, against ``B sqrt(2 / (grad B : grad B))`` of its
    derivatives in space.

    The extrapolation to the boundary leaves a difference that falls with the square of
    the distance between the surfaces.
    """
    theta, phi = np.meshgrid(
        np.linspace(0.0, 2.0 * np.pi, 9),
        np.linspace(0.0, np.pi / NFP, 7),
        indexing="ij",
    )
    radius = MAJOR + MINOR * np.cos(theta)
    points = np.stack(
        [radius * np.cos(phi), radius * np.sin(phi), MINOR * np.sin(theta)], axis=-1
    ).reshape(-1, 3)
    strength = jnp.linalg.norm(jax.vmap(_field)(points), axis=1)
    gradient = jax.vmap(jax.jacfwd(_field))(points)
    expected = np.asarray(
        strength * jnp.sqrt(2.0 / (gradient**2).sum(axis=(1, 2))) / MINOR * NFP
    ).reshape(theta.shape)
    differences = {}
    for ns in (51, 101):
        found = gradient_scale_lengths(*_analytic_wout(ns), NFP, n_theta=9, n_phi=7)
        differences[ns] = np.abs(np.asarray(found) / expected - 1.0).max()
    assert differences[101] < 1.0e-5
    np.testing.assert_allclose(differences[51] / differences[101], 4.0, rtol=0.1)


@pytest.fixture(scope="module")
def problem() -> SimpleToBuildProblem:
    return SimpleToBuildProblem(near_axis_start(4, 4, 13), max_mode=1, length=12.0)


def test_the_quantities_of_the_start_are_those_of_its_wout(problem) -> None:
    """The residuals and the constraints of a state in force balance against the
    quantities of a cold run of its boundary on the same modes and surfaces:
    quasi-isodynamicity, the aspect ratio, the rotational transform at the edge per
    field period, the mirror ratio, the elongations and the gradient scale lengths,
    which hold the bound at each point below it."""
    model = problem.relax(problem.boundary0, ftol=1.0e-14)
    evaluation = problem.evaluate(model, problem.boundary0)
    cold = problem.vmec_input.model_copy(update={"ftol_array": np.asarray([1.0e-14])})
    wout = vmecpp.run(cold, verbose=False, max_threads=1).wout
    found = score(problem, wout)
    values = evaluation.constraints
    first = 2 + problem.planes
    np.testing.assert_allclose(
        evaluation.parts["quasi_isodynamicity"],
        found["quasi_isodynamicity"],
        rtol=1.0e-6,
    )
    np.testing.assert_allclose(
        [values[0], values[1], values[2], values[3:first].max()],
        [found["aspect"], found["edge_iota"], found["mirror"], found["elongation"]],
        rtol=1.0e-6,
    )
    lengths = np.asarray(problem.lengths(_quantities(wout), wout))
    np.testing.assert_allclose(values[first:], lengths[:-1].ravel(), rtol=1.0e-6)
    np.testing.assert_allclose(lengths[-1], lengths[0], rtol=1.0e-9)
    np.testing.assert_allclose(
        found["gradient_scale_length"], values[first:].min(), rtol=1.0e-6
    )
    weights = problem.shifts.weights[first:]
    np.testing.assert_allclose(
        evaluation.residuals[-values[first:].size :],
        weights * np.minimum(values[first:] - problem.length, 0.0),
        rtol=1.0e-9,
        atol=1.0e-12,
    )
    assert (values[first:] < problem.length).any()
    assert (values[first:] > problem.length).any()


def test_the_derivatives_match_converged_solves(problem) -> None:
    """The total derivatives against central differences of re-converged equilibria:
    the residuals of quasi-isodynamicity, which have kinks, the aspect ratio and the
    rotational transform at the edge, the mirror ratio, whose derivative is that of its
    weighted means, the elongations and the gradient scale lengths."""
    keys = boundary_modes(1, problem.ntor)
    state = np.array(problem.relax(problem.boundary0, ftol=1.0e-14).get_state())
    model = problem.create_model(problem.boundary0)
    tangents = problem.state_tangents(model, state, keys)
    _, jacobian = problem.residual_jacobian(
        model, state, problem.boundary0, keys, tangents
    )
    rows = jacobian.shape[0] - problem.lower.size
    first = rows + 2 + problem.planes
    step = 1.0e-6
    for index in (0, 3, 6):
        sides = []
        for sign in (1.0, -1.0):
            boundary = problem.boundary0.copy()
            boundary[keys[index]] += sign * step
            solved = problem.relax(
                boundary, state + sign * step * tangents[:, index], ftol=1.0e-15
            )
            evaluation = problem.evaluate(solved, boundary)
            sides.append(
                np.concatenate([evaluation.residuals[:rows], evaluation.constraints])
            )
        difference = (sides[0] - sides[1]) / (2.0 * step)
        for block, tolerance in (
            (slice(0, rows), 2.0e-2),
            (slice(rows, rows + 2), 5.0e-4),
            (slice(rows + 2, rows + 3), 5.0e-2),
            (slice(rows + 3, first), 5.0e-4),
            (slice(first, None), 5.0e-4),
        ):
            np.testing.assert_allclose(
                jacobian[block, index],
                difference[block],
                rtol=0.0,
                atol=tolerance * np.linalg.norm(difference[block]),
            )


START = {
    (0, 0): (0.999517891875, 0.0),
    (0, 1): (-0.010898549833, -0.207419661702),
    (0, 2): (-0.027496853097, 0.025709987017),
    (1, -2): (0.003290434413, -0.005046658870),
    (1, -1): (-0.059674015877, -0.087178136548),
    (1, 0): (0.121222022305, -0.118338129256),
    (1, 1): (-0.016585104367, 0.031346644792),
    (1, 2): (-0.004308148911, 0.000290336826),
    (2, -2): (-0.002162680812, -0.003545779715),
    (2, -1): (0.015175643523, -0.016385024223),
    (2, 0): (-0.010849979721, -0.012500885473),
    (2, 1): (-0.001781282057, -0.004779636190),
    (2, 2): (0.002227121311, 0.001675400717),
}
"""A boundary that ``QuasiIsodynamicProblem`` reaches from the near-axis one within its
limits on modes up to 2 and 13 surfaces, as rbc and zbs by (m, n)."""


@pytest.fixture(scope="module")
def raised():
    """The problem from START with the gradient scale length held 5% above the smallest
    of that boundary."""
    vmec_input = near_axis_start(4, 4, 13)
    rbc, zbs = np.zeros((4, 9)), np.zeros((4, 9))
    for (m, n), (r, z) in START.items():
        rbc[m, 4 + n], zbs[m, 4 + n] = r, z
    vmec_input.rbc, vmec_input.zbs = rbc, zbs
    before = SimpleToBuildProblem(vmec_input, max_mode=2)
    values = before.evaluate(
        before.relax(before.boundary0, ftol=1.0e-12), before.boundary0
    ).constraints
    length = 1.05 * values[2 + before.planes :].min()
    problem = SimpleToBuildProblem(vmec_input, max_mode=2, length=length)
    return problem, values, optimize(problem, (2,))


def test_a_run_raises_the_gradient_scale_length_within_the_limits(raised) -> None:
    """The run ends with the gradient scale length of every point above the new bound
    and the other constraints within theirs, and a cold converged run of the final
    boundary on more modes stays within the limits of the problem."""
    problem, before, result = raised
    first = 2 + problem.planes
    values = result.evaluation.constraints
    assert before[first:].min() < 0.96 * problem.length
    assert result.evaluation.parts["violation"] <= CONSTRAINT_TOLERANCE
    assert values[first:].min() >= (1.0 - CONSTRAINT_TOLERANCE) * problem.length
    assert values[0] <= (1.0 + CONSTRAINT_TOLERANCE) * problem.aspect
    assert values[1] >= (1.0 - CONSTRAINT_TOLERANCE) * problem.edge_iota
    assert values[2] <= (1.0 + CONSTRAINT_TOLERANCE) * 0.97 * problem.mirror
    assert values[3:first].max() <= (1.0 + CONSTRAINT_TOLERANCE) * problem.elongation
    found = score(problem, converged_wout(problem, result.boundary, 6))
    assert found["gradient_scale_length"] > 0.95 * problem.length
    assert found["aspect"] < 1.01 * problem.aspect
    assert found["edge_iota"] > 0.99 * problem.edge_iota
    assert found["mirror"] < problem.mirror
    assert found["elongation"] < 1.01 * problem.elongation
