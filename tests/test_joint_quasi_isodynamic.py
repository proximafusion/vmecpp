# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""examples/joint_quasi_isodynamic.py against booz_xform, ConStellaration, analytic
fields and shapes, and differences of re-converged equilibria."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import vmecpp
from vmecpp.joint import boundary_modes, optimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from joint_quasi_isodynamic import (  # type: ignore
    BoozerTables,
    QuasiIsodynamicProblem,
    boozer_angles,
    boozer_strength,
    converged_wout,
    elongations,
    measures,
    mirror_ratio,
    near_axis_start,
    quasi_isodynamic_residuals,
)

jax.config.update("jax_enable_x64", True)

NFP = 3


@pytest.fixture(scope="module")
def wout():
    """The near-axis start converged on 25 surfaces."""
    vmec_input = near_axis_start(5, 5, 25)
    vmec_input.ftol_array = np.asarray([1.0e-14])
    return vmecpp.run(vmec_input, verbose=False, max_threads=1).wout


@pytest.fixture(scope="module")
def edge(wout):
    """The Boozer tables of the wout and the Fourier coefficients and the iota of its
    outermost half-grid surface."""
    tables = BoozerTables.of(wout, 6 * wout.mpol, 2 * wout.ntor - 1)
    columns = [
        np.asarray(getattr(wout, name))[:, wout.ns - 1]
        for name in ("bmnc", "bsubumnc", "bsubvmnc", "lmns")
    ]
    return tables, columns, float(wout.iotas[wout.ns - 1])


def _field(modes, shape=(3, 3)):
    """``a`` and ``d`` of a field strength from its modes by (m, n)."""
    a, d = np.zeros(shape), np.zeros(shape)
    for (m, n), (cosine, sine) in modes.items():
        a[m, n], d[m, n] = cosine, sine
    return a, d


def test_the_boozer_spectrum_matches_booz_xform(edge) -> None:
    """The largest modes of booz_xform 0.1.0 for the same wout, as ``a`` and ``d``.

    One iteration of the run at its end moves them by 5e-8.
    """
    tables, columns, iota = edge
    a, d = boozer_strength(tables, *columns, iota)
    for table, reference in (
        (
            a,
            {
                (0, 0): 0.8891003506522859,
                (0, 1): 0.39158807632753817,
                (1, 0): 0.049097388016531095,
                (0, 2): 0.03603145235609151,
                (2, 2): -0.02307920360000088,
                (2, 1): -0.014330086659640126,
            },
        ),
        (
            d,
            {
                (1, 1): 0.08563920466080686,
                (2, 1): 0.015712310264240994,
                (1, 3): -0.014013054441917062,
                (1, 4): 0.009338114277438666,
            },
        ),
    ):
        np.testing.assert_allclose(
            [table[mode] for mode in reference],
            list(reference.values()),
            rtol=0.0,
            atol=1.0e-6,
        )


def test_the_boozer_spectrum_returns_the_field_strength_at_the_boozer_angles(
    edge,
) -> None:
    """The series in the Boozer angles of each grid point against the field strength of
    the wout there, within the modes the series leaves out."""
    tables, columns, iota = edge
    theta, zeta, _ = boozer_angles(tables, *columns[1:], iota)
    a, d = boozer_strength(tables, *columns, iota)
    m = np.arange(a.shape[0])[:, None, None]
    n = (np.arange(a.shape[1]) * tables.nfp)[None, :, None]
    series = np.sum(
        np.asarray(a)[:, :, None] * np.cos(m * theta) * np.cos(n * zeta)
        + np.asarray(d)[:, :, None] * np.sin(m * theta) * np.sin(n * zeta),
        axis=(0, 1),
    )
    np.testing.assert_allclose(series, columns[0] @ tables.cosines_nyq, atol=1.0e-3)


@pytest.mark.parametrize(
    "mirror", [0.2, -0.2], ids=["maximum at 0", "at half a period"]
)
def test_a_mirror_field_has_no_residuals(mirror: float) -> None:
    """A field strength that depends on the toroidal angle alone is quasi-isodynamic."""
    a, d = _field({(0, 0): (1.0, 0.0), (0, 1): (mirror, 0.0)})
    residuals = quasi_isodynamic_residuals(a, d, 0.8, NFP)
    np.testing.assert_allclose(residuals, 0.0, atol=1.0e-12)


def test_a_second_well_adds_its_depth_to_the_variation() -> None:
    """``cos(x) + cos(2 x) / 2`` on every field line has a second well a ninth as deep
    as the span of the field, and no other residual."""
    a, d = _field({(0, 0): (3.0, 0.0), (0, 1): (1.0, 0.0), (0, 2): (0.5, 0.0)})
    residuals = np.asarray(
        quasi_isodynamic_residuals(a, d, 0.8, NFP, field_lines=7, levels=11)
    )
    np.testing.assert_allclose(residuals[: 7 * 11], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        residuals[7 * 11 : 7 * 12], 1.0 / (9.0 * np.sqrt(7.0)), rtol=1.0e-3
    )
    np.testing.assert_allclose(residuals[7 * 12 :], 0.0, atol=1.0e-12)


def test_wells_of_unequal_depth_lie_below_a_level_on_unequal_fractions() -> None:
    """Without rotational transform, ``(0.2 + 0.05 cos(theta)) cos(nfp zeta)`` is a
    cosine of its own depth on every field line."""
    lines, levels = 8, 21
    a, d = _field({(0, 0): (1.0, 0.0), (0, 1): (0.2, 0.0), (1, 1): (0.05, 0.0)})
    residuals = np.asarray(
        quasi_isodynamic_residuals(
            a, d, 0.0, NFP, field_lines=lines, toroidal_points=2001, levels=levels
        )
    )
    depth = (0.2 + 0.05 * np.cos(2.0 * np.pi * np.arange(lines) / lines)) / 0.25
    level = (np.arange(levels) + 0.5) / levels
    below = (
        1.0
        - np.arccos(np.clip((2.0 * level - 1.0) / depth[:, None], -1.0, 1.0)) / np.pi
    )
    np.testing.assert_allclose(
        residuals[: lines * levels],
        (below - below.mean(axis=0)).ravel() / np.sqrt(lines * levels),
        atol=1.0e-5,
    )
    wells, ends, lowest = residuals[lines * levels :].reshape(3, lines)
    np.testing.assert_allclose(wells, 0.0, atol=1.0e-12)
    # each cosine ends as far below the maximum as it stays above the minimum
    for part in (ends, lowest):
        np.testing.assert_allclose(
            part, 0.5 * (1.0 - depth) / np.sqrt(lines), atol=1.0e-12
        )


def test_the_mirror_ratio_of_a_cosine_is_its_amplitude() -> None:
    """The smooth extremes lie inside the true ones by less than 1 / SHARPNESS of the
    mean."""
    strength = 1.0 + 0.2 * np.cos(np.linspace(0.0, 2.0 * np.pi, 400, endpoint=False))
    assert 0.2 - 5.0e-3 < float(mirror_ratio(strength)) < 0.2
    np.testing.assert_allclose(mirror_ratio(strength, 1.0e5), 0.2, atol=1.0e-4)


def _ellipse_torus(elongation, turns: int):
    """The modes of a torus of major radius 1 around a circular axis, with ``xm``,
    ``xn``, ``rmnc``, ``zmns`` and the axis as :func:`elongations` takes them.

    Its cross-section is an ellipse of semi-axes 0.1 and 0.1 times an elongation, which
    makes ``turns`` half turns per field period.
    """
    mean, half = 0.05 * (1.0 + elongation), 0.05 * (1.0 - elongation)
    return (
        np.asarray([0.0, 1.0, 1.0]),
        np.asarray([0.0, 0.0, turns * NFP]),
        jnp.stack([1.0, mean, half]),
        jnp.stack([0.0, mean, -half]),
        np.asarray([1.0]),
        np.asarray([0.0]),
    )


@pytest.mark.parametrize("turns", [0, 1], ids=["axisymmetric", "rotating"])
def test_an_ellipse_has_its_elongation_in_every_plane(turns: int) -> None:
    values = elongations(*_ellipse_torus(2.5, turns), NFP, planes=5, points=401)
    np.testing.assert_allclose(values, 2.5, rtol=1.0e-4)


def test_the_elongation_matches_constellaration(wout, edge) -> None:
    """ConStellaration's ``max_elongation`` of the same wout, on its own planes and
    points for these modes."""
    tables = edge[0]
    values = elongations(
        tables.xm,
        tables.xn,
        np.asarray(wout.rmnc)[:, -1],
        np.asarray(wout.zmns)[:, -1],
        np.asarray(wout.raxis_cc),
        np.asarray(wout.zaxis_cs),
        NFP,
        planes=14,
        points=16,
    )
    np.testing.assert_allclose(values.max(), 2.918489, rtol=1.0e-5)


def test_the_elongation_has_the_derivative_of_its_differences() -> None:
    """With respect to the elongation of a rotating ellipse, through the Newton steps
    into the planes and to the ellipse of each polygon."""

    def largest(elongation):
        return elongations(
            *_ellipse_torus(elongation, 1), NFP, planes=5, points=17
        ).max()

    step = 1.0e-6
    difference = (largest(2.5 + step) - largest(2.5 - step)) / (2.0 * step)
    np.testing.assert_allclose(jax.grad(largest)(2.5), difference, rtol=1.0e-7)


@pytest.fixture(scope="module")
def problem() -> QuasiIsodynamicProblem:
    return QuasiIsodynamicProblem(near_axis_start(4, 4, 13), max_mode=1)


def test_the_residuals_of_the_start_are_those_of_its_wout(problem) -> None:
    """The residuals of a state in force balance against the measures of a cold run of
    its boundary on the same modes and surfaces."""
    model = problem.relax(problem.boundary0, ftol=1.0e-14)
    parts = problem.evaluate(model, problem.boundary0).parts
    cold = problem.vmec_input.model_copy(update={"ftol_array": np.asarray([1.0e-14])})
    found = measures(problem, vmecpp.run(cold, verbose=False, max_threads=1).wout)
    np.testing.assert_allclose(parts["aspect"], found["aspect"], rtol=1.0e-8)
    np.testing.assert_allclose(parts["mean_iota"], found["mean_iota"], rtol=1.0e-7)
    np.testing.assert_allclose(
        parts["quasi_isodynamicity"], found["quasi_isodynamicity"], rtol=1.0e-6
    )
    # the smooth extremes of the field strength lie inside those of the grid
    np.testing.assert_allclose(
        parts["mirror_excess"], found["mirror"] - problem.mirror, atol=5.0e-3
    )
    assert found["elongation"] < problem.elongation


def test_the_residual_derivatives_match_converged_solves(problem) -> None:
    """The total derivatives against central differences of re-converged equilibria:
    the targets, the limits and the residuals of quasi-isodynamicity, whose fractions
    below a level have kinks."""
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
            sides.append(problem.evaluate(solved, boundary).residuals)
        difference = (sides[0] - sides[1]) / (2.0 * step)
        for block, tolerance in ((slice(0, 4), 5.0e-4), (slice(4, None), 2.0e-2)):
            np.testing.assert_allclose(
                jacobian[block, index],
                difference[block],
                rtol=0.0,
                atol=tolerance * np.linalg.norm(difference[block]),
            )


@pytest.fixture(scope="module")
def nested_and_joint():
    results = {}
    for name in ("nested", "joint"):
        problem = QuasiIsodynamicProblem(near_axis_start(4, 4, 13), max_mode=2)
        results[name] = (problem, optimize(problem, (1, 2), nested=name == "nested"))
    return results


@pytest.mark.parametrize("name", ["nested", "joint"])
def test_both_methods_reach_the_targets_within_the_limits(
    nested_and_joint, name
) -> None:
    """A cold converged run of the final boundary on more modes.

    The start has a mirror ratio of 0.50 and residuals of quasi-isodynamicity of 1.8e-2,
    and the two stages end with the mirror ratio at its limit.
    """
    problem, result = nested_and_joint[name]
    found = measures(problem, converged_wout(problem, result.boundary, 6))
    np.testing.assert_allclose(found["aspect"], problem.aspect, rtol=1.0e-2)
    np.testing.assert_allclose(found["mean_iota"], problem.mean_iota, rtol=1.0e-2)
    assert found["mirror"] < 1.05 * problem.mirror
    assert found["elongation"] < problem.elongation
    assert found["quasi_isodynamicity"] < 1.5e-3


def test_joint_uses_fewer_force_evaluations(nested_and_joint) -> None:
    nested_problem, _ = nested_and_joint["nested"]
    joint_problem, _ = nested_and_joint["joint"]
    assert (
        joint_problem.costs.force_evaluations
        < 0.75 * nested_problem.costs.force_evaluations
    )
