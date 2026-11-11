# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""examples/joint_free_boundary_coils.py against the Biot-Savart field of moved coils
and differences of converged free boundaries."""

import sys
from pathlib import Path

import numpy as np
import pytest
from simsopt import configs
from simsopt.field import BiotSavart, Current, coils_via_symmetries

import vmecpp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import joint_free_boundary  # type: ignore
import joint_free_boundary_coils  # type: ignore


@pytest.fixture(scope="module")
def case():
    """The fixed-boundary input inside a flux surface of the coils, their field with the
    currents and the positions of the base coils as controls, and those controls."""
    vmec_input, field = joint_free_boundary_coils.giuliani(5, 5, 15, 0)
    return vmec_input, field, field.controls()


@pytest.fixture(scope="module")
def problem(case):
    vmec_input, field, design = case
    wout = vmecpp.run(vmec_input, max_threads=1, verbose=False).wout
    return joint_free_boundary.FreeBoundaryProblem(
        vmec_input,
        aspect=float(wout.aspect),
        mean_iota=float(np.mean(np.asarray(wout.iotas)[1:])),
        ntheta=16,
        nphi=16,
        field=field,
        scales=field.scales(design, 0.05, 5.0e-3),
        modes=joint_free_boundary.shape_modes(4, 5),
        surfaces=(0.5, 1.0),
    )


@pytest.fixture(scope="module")
def free_boundary(problem, case):
    """The converged free boundary of the coils as they are."""
    return joint_free_boundary.converge(
        problem, np.zeros(len(problem.modes)), case[2], None
    )


def test_the_field_of_moved_controls_is_that_of_the_moved_coils(case) -> None:
    """The field of controls with a changed current and a changed position against the
    Biot-Savart field of coils built with them."""
    _, field, design = case
    count = len(field.currents)
    controls = design.copy()
    controls[1] *= 1.02
    controls[count + 2] += 4.0e-3
    r = np.array([1.0, 1.1, 0.9])
    phi = np.array([0.1, 0.7, 1.9])
    z = np.array([0.0, 0.05, -0.04])
    components = field(controls)[0](r, phi, z)
    curves, currents, _, nfp = configs.get_data("giuliani")[:4]
    values = [float(current.get_value()) for current in currents]
    values[1] *= 1.02
    coefficients = curves[0].x.copy()
    coefficients[field.free[0][2]] += 4.0e-3
    curves[0].x = coefficients
    reference = BiotSavart(
        coils_via_symmetries(curves, [Current(value) for value in values], nfp, True)
    )
    reference.set_points(np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=-1))
    b = reference.B()
    expected = (
        b[:, 0] * np.cos(phi) + b[:, 1] * np.sin(phi),
        -b[:, 0] * np.sin(phi) + b[:, 1] * np.cos(phi),
        b[:, 2],
    )
    for component, value in zip(components, expected, strict=True):
        np.testing.assert_allclose(component, value, rtol=1.0e-12)


@pytest.mark.parametrize("index", [0, 6])
def test_control_derivatives_match_converged_free_boundaries(
    problem, free_boundary, index
) -> None:
    """The derivatives of the objective residuals and of the shape with respect to a
    current and to the position of a coil, in units of their scales, against central
    differences of free boundaries converged at changed controls."""
    relative = 5.0e-2
    sides = []
    for sign in (1.0, -1.0):
        controls = free_boundary["controls"].copy()
        controls[index] += sign * relative * problem.scales[index]
        change = sign * relative * free_boundary["sensitivity"][:, index]
        sides.append(
            joint_free_boundary.converge(
                problem,
                free_boundary["shape"] + change,
                controls,
                problem.carried(
                    free_boundary["state"], change, free_boundary["tangents"]
                ),
            )
        )
    for derivative, name in (("reduced", "residuals"), ("sensitivity", "shape")):
        difference = (sides[0][name] - sides[1][name]) / (2.0 * relative)
        np.testing.assert_allclose(
            free_boundary[derivative][:, index],
            difference,
            rtol=0.0,
            atol=2.0e-4 * np.linalg.norm(difference),
            err_msg=name,
        )
