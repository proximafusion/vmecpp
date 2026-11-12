# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""examples/joint_mhd_stable.py against SIMSOPT, ConStellaration and central differences
of re-converged equilibria."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt.mhd import Vmec, vmec_fieldlines

import vmecpp
from vmecpp.joint import CONSTRAINT_TOLERANCE, boundary_modes, optimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from joint_mhd_stable import (  # type: ignore
    _WOUT_QUANTITIES,
    MHDStableProblem,
    field_line_geometry,
    flux_compression,
    magnetic_well,
    score,
    unfavourable_part,
)
from joint_quasi_isodynamic import converged_wout, near_axis_start  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_joint_simple_to_build import START  # type: ignore

jax.config.update("jax_enable_x64", True)

SURFACE, FIELD_LINES, POINTS = 0.49, 101, 64
"""The surface and the grid of the flux compression of the benchmark."""


def _quantities(wout) -> dict:
    return {name: np.asarray(getattr(wout, name)) for name in _WOUT_QUANTITIES}


@pytest.fixture(scope="module")
def wout():
    """The near-axis start at a pressure, converged on 25 surfaces."""
    vmec_input = near_axis_start(5, 5, 25)
    vmec_input.pres_scale = 3000.0
    vmec_input.am = np.asarray([1.0, -1.0, 0.0])
    vmec_input.ftol_array = np.asarray([1.0e-14])
    return vmecpp.run(vmec_input, verbose=False, max_threads=1).wout


@pytest.fixture(scope="module")
def simsopt_lines(wout, tmp_path_factory):
    """SIMSOPT's magnetic well of the wout and its field lines on the surface and the
    grid of the benchmark."""
    path = tmp_path_factory.mktemp("wout") / "wout_near_axis.nc"
    wout.save(path)
    vmec = Vmec(str(path))
    alpha = np.linspace(
        -np.pi / wout.nfp, np.pi / wout.nfp, FIELD_LINES, endpoint=False
    )
    theta = np.linspace(-np.pi, np.pi, POINTS, endpoint=False)
    return vmec.vacuum_well(), vmec_fieldlines(
        vmec, s=[SURFACE], alpha=alpha, theta1d=theta
    )


def test_the_magnetic_well_matches_simsopt_and_constellaration(
    wout, simsopt_lines
) -> None:
    """SIMSOPT's ``vacuum_well`` of the wout, and ConStellaration's for the same
    wout."""
    well = magnetic_well(np.asarray(wout.vp))
    np.testing.assert_allclose(well, simsopt_lines[0], rtol=1.0e-10)
    np.testing.assert_allclose(well, 0.222749371352, rtol=1.0e-5)


def test_the_field_line_geometry_matches_simsopt(wout, simsopt_lines) -> None:
    """The poloidal angles of the points, the field along the straight-field-line
    poloidal angle, the squared gradient of the flux label and the curvature drift
    across the field lines."""
    found = field_line_geometry(
        _quantities(wout),
        wout,
        surface=SURFACE,
        field_lines=FIELD_LINES,
        points=POINTS,
    )
    for name in (
        "theta_vmec",
        "B_sup_theta_pest",
        "grad_s_dot_grad_s",
        "B_cross_kappa_dot_grad_alpha",
    ):
        expected = np.asarray(getattr(simsopt_lines[1], name))[0]
        np.testing.assert_allclose(
            found[name], expected, rtol=0.0, atol=1.0e-10 * np.abs(expected).max()
        )
    np.testing.assert_allclose(
        found["edge_toroidal_flux_over_2pi"],
        simsopt_lines[1].edge_toroidal_flux_over_2pi,
        rtol=1.0e-12,
    )


def test_the_flux_compression_is_that_of_constellaration(wout, simsopt_lines) -> None:
    """With whole cells, ConStellaration's
    ``compute_flux_compression_in_regions_of_bad_curvature`` for the same wout and its
    formula on SIMSOPT's field lines, and within 3% of them by the parts of the
    cells."""
    lines = simsopt_lines[1]
    weight = np.abs(lines.edge_toroidal_flux_over_2pi / lines.B_sup_theta_pest)
    compression = wout.Aminor_p**2 / (4.0 * SURFACE) * lines.grad_s_dot_grad_s
    drift = lines.B_cross_kappa_dot_grad_alpha * np.sign(wout.phipf[-1])
    expected = np.mean(np.heaviside(drift, 0.0) * compression * weight) / weight.mean()
    quantities = _quantities(wout)
    whole = flux_compression(quantities, wout, whole_cells=True)
    np.testing.assert_allclose(whole, expected, rtol=1.0e-10)
    np.testing.assert_allclose(whole, 1.639994694815, rtol=1.0e-5)
    np.testing.assert_allclose(
        flux_compression(quantities, wout), expected, rtol=3.0e-2
    )


@pytest.mark.parametrize(("name", "table"), [("lmns", ""), ("bmnc", "_nyq")])
def test_the_flux_compression_has_the_derivative_of_its_differences(
    wout, name, table
) -> None:
    """With respect to the mode (1, 1) on every surface of lambda, through the poloidal
    angles of the points, and of the field strength, through the curvature drift and the
    edge of the region where it is unfavourable."""
    quantities = _quantities(wout)
    poloidal = np.asarray(getattr(wout, "xm" + table))
    toroidal = np.asarray(getattr(wout, "xn" + table))
    mode = int(np.flatnonzero((poloidal == 1) & (toroidal == wout.nfp))[0])

    def value(amplitude):
        shifted = jnp.asarray(quantities[name]).at[mode].add(amplitude)
        return flux_compression({**quantities, name: shifted}, wout)

    step = 1.0e-6
    difference = (value(step) - value(-step)) / (2.0 * step)
    np.testing.assert_allclose(jax.grad(value)(0.0), difference, rtol=1.0e-5)


def test_the_parts_of_the_cells_of_a_line_add_up_to_its_unfavourable_length() -> None:
    """A drift that rises along one line and falls along another, each through zero
    between two points, and two lines without a change of sign."""
    points = np.arange(16.0)
    drift = np.stack([points - 5.3, 9.75 - points, points + 1.0, -points - 1.0])
    parts = np.asarray(unfavourable_part(drift))
    # the cells of the points reach from -0.5 to 15.5
    np.testing.assert_allclose(
        parts.sum(axis=1), [15.5 - 5.3, 9.75 + 0.5, 16.0, 0.0], rtol=1.0e-12
    )
    assert ((parts >= 0.0) & (parts <= 1.0)).all()


@pytest.fixture(scope="module")
def problem() -> MHDStableProblem:
    return MHDStableProblem(near_axis_start(4, 4, 13), max_mode=1, length=12.0)


def test_the_stability_of_the_start_is_that_of_its_wout(problem) -> None:
    """The last two constraints of a state in force balance against one less the
    magnetic well and the flux compression of a cold run of its boundary on the same
    modes and surfaces."""
    model = problem.relax(problem.boundary0, ftol=1.0e-14)
    values = problem.evaluate(model, problem.boundary0).constraints
    cold = problem.vmec_input.model_copy(update={"ftol_array": np.asarray([1.0e-14])})
    wout = vmecpp.run(cold, verbose=False, max_threads=1).wout
    assert values.shape == problem.lower.shape
    np.testing.assert_allclose(
        values[-2:],
        [
            1.0 - magnetic_well(np.asarray(wout.vp)),
            flux_compression(_quantities(wout), wout),
        ],
        rtol=1.0e-6,
    )


def test_the_derivatives_of_the_stability_match_converged_solves(problem) -> None:
    """The total derivatives of the magnetic well and of the flux compression, a mean
    over a region whose boundary moves, against central differences of re-converged
    equilibria."""
    keys = boundary_modes(1, problem.ntor)
    state = np.array(problem.relax(problem.boundary0, ftol=1.0e-14).get_state())
    model = problem.create_model(problem.boundary0)
    tangents = problem.state_tangents(model, state, keys)
    _, jacobian = problem.residual_jacobian(
        model, state, problem.boundary0, keys, tangents
    )
    step = 1.0e-6
    for index in (0, 3, 6):
        sides = []
        for sign in (1.0, -1.0):
            boundary = problem.boundary0.copy()
            boundary[keys[index]] += sign * step
            solved = problem.relax(
                boundary, state + sign * step * tangents[:, index], ftol=1.0e-15
            )
            sides.append(problem.evaluate(solved, boundary).constraints[-2:])
        difference = (sides[0] - sides[1]) / (2.0 * step)
        np.testing.assert_allclose(jacobian[-2, index], difference[0], rtol=1.0e-3)
        np.testing.assert_allclose(jacobian[-1, index], difference[1], rtol=3.0e-2)


@pytest.fixture(scope="module")
def stabilized():
    """The problem from the boundary START of the simple-to-build tests on coarser
    grids, with the magnetic well held 0.01 above that of the boundary and the flux
    compression 3% below."""
    vmec_input = near_axis_start(4, 4, 13)
    rbc, zbs = np.zeros((4, 9)), np.zeros((4, 9))
    for (m, n), (r, z) in START.items():
        rbc[m, 4 + n], zbs[m, 4 + n] = r, z
    vmec_input.rbc, vmec_input.zbs = rbc, zbs
    sizes = {
        "max_mode": 2,
        "mirror": 0.2,
        "elongation": 5.0,
        "field_lines": 15,
        "toroidal_points": 151,
        "levels": 101,
        "output_points": 300,
        "grid": (14, 10),
        "compression_grid": (41, 32),
    }
    before = MHDStableProblem(vmec_input, **sizes)
    values = before.evaluate(
        before.relax(before.boundary0, ftol=1.0e-12), before.boundary0
    ).constraints
    problem = MHDStableProblem(
        vmec_input,
        length=0.95 * values[2 + before.planes : -2].min(),
        well=1.0 - values[-2] + 0.01,
        compression=0.97 * values[-1],
        **sizes,
    )
    return problem, optimize(problem, (2,))


def test_a_run_brings_the_well_and_the_flux_compression_within_their_bounds(
    stabilized,
) -> None:
    """The run ends with both on their bounds and the other constraints within theirs,
    and a cold converged run of the final boundary on more modes and surfaces has the
    magnetic well and ConStellaration's own flux compression within the bounds."""
    problem, result = stabilized
    values = result.evaluation.constraints
    assert result.evaluation.parts["violation"] <= CONSTRAINT_TOLERANCE
    np.testing.assert_allclose(
        values[-2:],
        [1.0 - problem.well, problem.compression],
        rtol=2.0 * CONSTRAINT_TOLERANCE,
    )
    found = score(problem, converged_wout(problem, result.boundary, 6))
    assert found["well"] > problem.well - 1.0e-3
    assert found["flux_compression"] < 1.01 * problem.compression
    assert found["aspect"] < 1.01 * problem.aspect
    assert found["edge_iota"] > 0.99 * problem.edge_iota
    assert found["mirror"] < problem.mirror
    assert found["elongation"] < 1.01 * problem.elongation
