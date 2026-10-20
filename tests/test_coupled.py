# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Several free-boundary plasmas in one vacuum field.

The plasmas are copies of the Solov'ev free-boundary case with lasym=True, in the field
of the coils of coils.solovev made exactly up-down symmetric: a straight wire on the z
axis and circular loops around each plasma.
"""

from pathlib import Path

import numpy as np
import pydantic
import pytest
from scipy.special import ellipe, ellipk

import vmecpp
from vmecpp import _coupled

TEST_DATA = Path(__file__).resolve().parents[1] / "src" / "vmecpp" / "cpp" / "vmecpp"
TEST_DATA = TEST_DATA / "test_data"
MU0 = 4.0e-7 * np.pi

# The loops of coils.solovev, (radius, height), which carry extcur[1:]; pairs of
# loops mirror each other in z
LOOPS = np.array(
    [
        [0.867845, 1.86006],
        [4.49628, 3.69583],
        [8.63810, -2.46839],
        [8.63847, 2.46836],
        [4.49620, -3.69589],
        [0.867842, -1.86005],
        [1.19984, 3.34106],
        [2.62980, 5.24924],
        [9.65488, -3.29570],
        [9.65501, 3.29541],
        [2.62966, -5.24925],
        [1.19984, -3.34102],
    ]
)
PAIRS = [(0, 5), (1, 4), (2, 3), (6, 11), (7, 10), (8, 9)]


@pytest.fixture(scope="module")
def solovev_input() -> vmecpp.VmecInput:
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA / "solovev_free_bdy.json")
    zeros = np.zeros_like(np.asarray(vmec_input.rbc))
    return vmec_input.model_copy(
        update={
            "lasym": True,
            "rbs": zeros.copy(),
            "zbc": zeros.copy(),
            "raxis_s": np.zeros(1),
            "zaxis_c": np.zeros(1),
            "ftol_array": np.array([1.0e-10, 1.0e-18]),
            "niter_array": np.array([10000, 100000]),
        }
    )


def _coil_field(vmec_input: vmecpp.VmecInput, heights: list[float]):
    """The field of the wire and of a symmetric copy of the loops around each of heights
    on the z axis."""
    currents = np.asarray(vmec_input.extcur)[1:]
    loops, loop_currents = [], []
    for i, j in PAIRS:
        radius = 0.5 * (LOOPS[i, 0] + LOOPS[j, 0])
        height = 0.5 * (abs(LOOPS[i, 1]) + abs(LOOPS[j, 1]))
        current = 0.5 * (currents[i] + currents[j])
        loops += [(radius, height), (radius, -height)]
        loop_currents += [current, current]
    loops, loop_currents = np.array(loops), np.array(loop_currents)
    wire = float(vmec_input.extcur[0])

    def field(r, _phi, z):
        b_r = np.zeros_like(r)
        b_z = np.zeros_like(r)
        a = loops[:, 0, None]
        for height in heights:
            dz = z - (loops[:, 1, None] + height)
            alpha2 = (a - r) ** 2 + dz**2
            beta2 = (a + r) ** 2 + dz**2
            m = 1.0 - alpha2 / beta2
            scale = MU0 * loop_currents[:, None] / (2 * np.pi * alpha2 * np.sqrt(beta2))
            e, k = ellipe(m), ellipk(m)
            b_r += np.sum(scale * dz / r * ((a**2 + r**2 + dz**2) * e - alpha2 * k), 0)
            b_z += np.sum(scale * ((a**2 - r**2 - dz**2) * e + alpha2 * k), 0)
        return b_r, MU0 * wire / (2 * np.pi * r), b_z

    return field


@pytest.fixture(scope="module")
def alone(solovev_input) -> vmecpp.VmecOutput:
    field = _coil_field(solovev_input, [0.0])
    return vmecpp.run(solovev_input, field, max_threads=2, verbose=False)


def _boundary(output: vmecpp.VmecOutput) -> np.ndarray:
    wout = output.wout
    return np.concatenate(
        [np.asarray(getattr(wout, n))[:, -1] for n in ("rmnc", "zmns", "rmns", "zmnc")]
    )


def _relative_deviation(output: vmecpp.VmecOutput, reference: vmecpp.VmecOutput):
    return (
        np.abs(_boundary(output) - _boundary(reference)).max()
        / np.abs(_boundary(reference)).max()
    )


def _surface_points(output: vmecpp.VmecOutput, js: int, height: float) -> np.ndarray:
    """Points of flux surface js at eight poloidal angles, shifted up by height."""
    wout = output.wout
    xm = np.asarray(wout.xm)
    theta = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    angle = xm[:, None] * theta[None, :]
    r = np.sum(
        np.asarray(wout.rmnc)[:, js, None] * np.cos(angle)
        + np.asarray(wout.rmns)[:, js, None] * np.sin(angle),
        axis=0,
    )
    z = np.sum(
        np.asarray(wout.zmns)[:, js, None] * np.sin(angle)
        + np.asarray(wout.zmnc)[:, js, None] * np.cos(angle),
        axis=0,
    )
    phi = np.linspace(0.0, 1.0, len(theta))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z + height], axis=1)


def _cartesian_field(field, points: np.ndarray) -> np.ndarray:
    phi = np.arctan2(points[:, 1], points[:, 0])
    b_r, b_phi, b_z = field(np.hypot(points[:, 0], points[:, 1]), phi, points[:, 2])
    return _coupled._vectors_to_cartesian(b_r, b_phi, b_z, phi)


def test_free_boundary_table(alone):
    """The vacuum field on the boundary has the vacuum pressure of the run, which Nestor
    takes from the flux-coordinate components of the same field."""
    fb = alone.threed1_free_boundary
    assert np.shape(fb.rb) == np.shape(fb.brv)
    np.testing.assert_allclose(
        fb.bsqvacf,
        0.5 * (fb.brv**2 + fb.bphiv**2 + fb.bzv**2),
        rtol=1.0e-10,
        atol=0.0,
    )

    fixed = vmecpp.run(
        vmecpp.VmecInput.from_file(TEST_DATA / "solovev.json"),
        max_threads=1,
        verbose=False,
    )
    assert not np.any(fixed.threed1_free_boundary.brv)


def test_virtual_casing_field_cancels_the_external_field_inside(solovev_input, alone):
    field = _coupled._plasma_field(
        alone, vmecpp.Placement(), axisymmetric_rings=False, distance=0.5
    )
    external = _coil_field(solovev_input, [0.0])
    # measured 2.5e-6 on the axis and 3.2e-5 halfway out
    for js in (0, alone.wout.ns // 2):
        points = _surface_points(alone, js, 0.0)
        b_external = _cartesian_field(external, points)
        residual = field.field(points) + b_external
        assert np.abs(residual).max() < 1.0e-4 * np.abs(b_external).max()


def test_rings_reproduce_the_surface_quadrature_outside(alone):
    """Outside an axisymmetric plasma, its field from rings of toroidal current equals
    the quadrature over the whole torus."""
    surface = alone.threed1_free_boundary
    r, z = np.asarray(surface.rb)[0], np.asarray(surface.zb)[0]
    r_axis = float(np.asarray(alone.wout.raxis_cc)[0])
    outward = np.stack([r - r_axis, z], axis=1)
    outward /= np.linalg.norm(outward, axis=1)[:, None]
    phi = 0.3
    points = np.stack(
        [
            (r + outward[:, 0]) * np.cos(phi),
            (r + outward[:, 0]) * np.sin(phi),
            z + outward[:, 1],
        ],
        axis=1,
    )
    rings = _coupled._plasma_field(
        alone, vmecpp.Placement(), axisymmetric_rings=True, distance=1.0
    ).field(points)
    quadrature = _coupled._plasma_field(
        alone, vmecpp.Placement(), axisymmetric_rings=False, distance=1.0
    ).field(points)
    # measured 5e-10, the trapezoidal rule's error at the spacing the rounds use
    assert np.abs(rings - quadrature).max() < 1.0e-8 * np.abs(quadrature).max()


def test_one_plasma_is_the_run_in_the_external_field(solovev_input, alone):
    field = _coil_field(solovev_input, [0.0])
    (output,) = vmecpp.run_coupled(
        [solovev_input], [vmecpp.Placement()], field, max_threads=2
    )
    np.testing.assert_array_equal(_boundary(output), _boundary(alone))


def test_copies_far_apart_reproduce_the_plasma_alone(solovev_input, alone):
    heights = [5.0e3, -5.0e3]
    outputs = vmecpp.run_coupled(
        [solovev_input, solovev_input],
        [vmecpp.Placement(translation=np.array([0.0, 0.0, h])) for h in heights],
        _coil_field(solovev_input, heights),
        max_threads=2,
    )
    # measured 1e-10, from the round-off of coordinates 5 km from the origin
    for output in outputs:
        assert _relative_deviation(output, alone) < 1.0e-9


def test_coaxial_pair_responds_to_the_dipole_field_of_the_other(solovev_input):
    """Two copies on one axis, the lower the mirror image of the upper.

    Against the plasma solved in both coil sets without the other plasma, each differs
    by its response to the other's dipole field, which falls as the distance cubed and,
    as it opposes the vertical field that holds the plasma, lets it expand outward.
    """
    deviations = []
    for distance in (120.0, 240.0):
        heights = [distance / 2, -distance / 2]
        # the upper plasma in its own frame: its coils at 0, the lower plasma's at -D
        reference = vmecpp.run(
            solovev_input,
            _coil_field(solovev_input, [0.0, -distance]),
            max_threads=2,
            verbose=False,
        )
        upper, lower = vmecpp.run_coupled(
            [solovev_input, solovev_input],
            [vmecpp.Placement(translation=np.array([0.0, 0.0, h])) for h in heights],
            _coil_field(solovev_input, heights),
            max_threads=2,
        )
        mirrored = np.concatenate(
            [
                np.asarray(lower.wout.rmnc)[:, -1],
                np.asarray(lower.wout.zmns)[:, -1],
                -np.asarray(lower.wout.rmns)[:, -1],
                -np.asarray(lower.wout.zmnc)[:, -1],
            ]
        )
        scale = np.abs(_boundary(reference)).max()
        # measured 6e-10 at 120 m
        assert np.abs(mirrored - _boundary(upper)).max() < 1.0e-7 * scale
        # measured +1.1e-4 m at 120 m
        expansion = (
            np.asarray(upper.wout.rmnc)[0, -1] - np.asarray(reference.wout.rmnc)[0, -1]
        )
        assert expansion > 0.0
        deviations.append(_relative_deviation(upper, reference))
    # measured 7.91
    assert 7.0 < deviations[0] / deviations[1] < 9.0


@pytest.mark.parametrize("pressure", [1.0, 1.0e3])
def test_close_pair_shares_one_vacuum_field(solovev_input, pressure):
    """Inside each plasma, its virtual casing field cancels the external field together
    with the field of the other plasma; at a thousand times the pressure, beta = 0.5
    percent, as well."""
    vmec_input = solovev_input.model_copy(
        update={"pres_scale": solovev_input.pres_scale * pressure}
    )
    heights = [15.0, -15.0]
    placements = [
        vmecpp.Placement(translation=np.array([0.0, 0.0, h])) for h in heights
    ]
    external = _coil_field(vmec_input, heights)
    upper, lower = vmecpp.run_coupled(
        [vmec_input, vmec_input], placements, external, max_threads=2
    )
    own = _coupled._plasma_field(
        upper, placements[0], axisymmetric_rings=False, distance=0.5
    )
    other = _coupled._plasma_field(
        lower, placements[1], axisymmetric_rings=True, distance=5.0
    )
    points = _surface_points(upper, 0, heights[0])
    b_external = _cartesian_field(external, points)
    scale = np.abs(b_external).max()
    # measured 2.5e-6 with the other plasma's field and 1.7e-4 without it
    shared = own.field(points) + b_external + other.field(points)
    assert np.abs(shared).max() < 1.0e-5 * scale
    assert np.abs(own.field(points) + b_external).max() > 5.0e-5 * scale


def test_run_coupled_checks_its_inputs(solovev_input):
    field = _coil_field(solovev_input, [0.0])
    two = [vmecpp.Placement(), vmecpp.Placement(translation=np.array([0, 0, 20.0]))]
    with pytest.raises(ValueError, match="one placement per input"):
        vmecpp.run_coupled([solovev_input], two, field)
    symmetric = solovev_input.model_copy(
        update={
            "lasym": False,
            "rbs": None,
            "zbc": None,
            "raxis_s": None,
            "zaxis_c": None,
        }
    )
    with pytest.raises(ValueError, match="lasym=True"):
        vmecpp.run_coupled([symmetric, solovev_input], two, field)
    fixed = solovev_input.model_copy(update={"lfreeb": False})
    with pytest.raises(ValueError, match="free-boundary"):
        vmecpp.run_coupled([fixed, solovev_input], two, field)
    side_by_side = [
        vmecpp.Placement(),
        vmecpp.Placement(translation=np.array([20.0, 0, 0])),
    ]
    with pytest.raises(ValueError, match="z axis"):
        vmecpp.run_coupled([solovev_input, solovev_input], side_by_side, field)


def _three_dimensional(vmec_input: vmecpp.VmecInput) -> vmecpp.VmecInput:
    """The input with toroidal resolution, as plasmas side by side need."""
    vmec_input = vmec_input.model_copy(deep=True)
    vmec_input.resize(8, 2)
    vmec_input.nzeta = 12
    vmec_input.ns_array = np.array([16])
    vmec_input.ftol_array = np.array([1.0e-12])
    vmec_input.niter_array = np.array([20000])
    return vmec_input


def _coil_sets_field(vmec_input: vmecpp.VmecInput, centers: np.ndarray):
    """The field of a coil set around each vertical line (x, y) in centers, in the
    cylindrical coordinates of the common frame.

    Each wire's toroidal field is confined to a torus around its own plasma, as the
    field of toroidal field coils is.
    """
    one_set = _coil_field(vmec_input, [0.0])

    def field(r, phi, z):
        x, y = r * np.cos(phi), r * np.sin(phi)
        b = np.zeros((np.size(r), 3))
        for x0, y0 in centers:
            dx, dy = x - x0, y - y0
            r_c, phi_c = np.hypot(dx, dy), np.arctan2(dy, dx)
            b_r, b_phi, b_z = one_set(r_c, phi_c, z)
            b_phi = np.where((r_c < 9.0) & (np.abs(z) < 7.0), b_phi, 0.0)
            b += _coupled._vectors_to_cartesian(b_r, b_phi, b_z, phi_c)
        return _coupled._vectors_to_cylindrical(b, phi)

    return field


def _surface_points_3d(output: vmecpp.VmecOutput, js: int) -> np.ndarray:
    """Points of flux surface js of a run with toroidal resolution, at eight poloidal
    angles spread over the toroidal angle."""
    wout = output.wout
    xm, xn = np.asarray(wout.xm), np.asarray(wout.xn)
    theta = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    phi = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False) + 0.1
    angle = xm[:, None] * theta[None, :] - xn[:, None] * phi[None, :]
    r = np.sum(
        np.asarray(wout.rmnc)[:, js, None] * np.cos(angle)
        + np.asarray(wout.rmns)[:, js, None] * np.sin(angle),
        axis=0,
    )
    z = np.sum(
        np.asarray(wout.zmns)[:, js, None] * np.sin(angle)
        + np.asarray(wout.zmnc)[:, js, None] * np.cos(angle),
        axis=0,
    )
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def test_virtual_casing_quadrature_of_a_three_dimensional_run(solovev_input):
    """The surface current integrated over the toroidal grid of a run with toroidal
    resolution cancels the external field inside like the axisymmetric one's."""
    vmec_input = _three_dimensional(solovev_input)
    vmec_input.nzeta = 24
    external = _coil_field(solovev_input, [0.0])
    output = vmecpp.run(vmec_input, external, max_threads=2, verbose=False)
    field = _coupled._plasma_field(
        output, vmecpp.Placement(), axisymmetric_rings=False, distance=0.25
    )
    # measured 7e-5 halfway out
    for js in (0, output.wout.ns // 2):
        points = _surface_points_3d(output, js)
        b_external = _cartesian_field(external, points)
        residual = field.field(points) + b_external
        assert np.abs(residual).max() < 2.0e-4 * np.abs(b_external).max()


def test_side_by_side_pair_with_toroidal_resolution(solovev_input):
    """Two plasmas side by side 25 m apart, the second placed as the first turned by pi,
    each in its own coil set: they come out the same in their own frames, and the field
    of the other makes each non-axisymmetric."""
    vmec_input = _three_dimensional(solovev_input)
    half = 12.5
    alone = vmecpp.run(
        vmec_input,
        _coil_sets_field(solovev_input, np.array([[0.0, 0.0]])),
        max_threads=2,
        verbose=False,
    )
    placements = [
        vmecpp.Placement(translation=np.array([half, 0.0, 0.0])),
        vmecpp.Placement(
            rotation=np.diag([-1.0, -1.0, 1.0]),
            translation=np.array([-half, 0.0, 0.0]),
        ),
    ]
    centers = np.array([[half, 0.0], [-half, 0.0]])
    a, b = vmecpp.run_coupled(
        [vmec_input, vmec_input],
        placements,
        _coil_sets_field(solovev_input, centers),
        max_threads=2,
    )
    scale = np.abs(_boundary(alone)).max()
    # measured 1.3e-6
    assert np.abs(_boundary(a) - _boundary(b)).max() < 1.0e-5 * scale
    toroidal = np.asarray(a.wout.xn) != 0
    assert np.abs(np.asarray(alone.wout.rmnc)[toroidal, -1]).max() < 1.0e-12
    non_axisymmetric = max(
        np.abs(np.asarray(getattr(a.wout, name))[toroidal, -1]).max()
        for name in ("rmnc", "zmns", "rmns", "zmnc")
    )
    # measured 8.0e-4 m at 25 m and 3.2e-5 m at 50 m
    assert 2.0e-4 < non_axisymmetric < 2.0e-3
    # measured 3.4e-3, 8.2 times the deviation at 50 m
    assert 1.5e-3 < _relative_deviation(a, alone) < 6.0e-3


def test_placement_takes_a_proper_rotation():
    with pytest.raises(pydantic.ValidationError, match="proper rotation"):
        vmecpp.Placement(rotation=np.diag([1.0, 1.0, -1.0]))
    with pytest.raises(pydantic.ValidationError, match="proper rotation"):
        vmecpp.Placement(rotation=2.0 * np.eye(3))
    assert vmecpp.Placement(rotation=np.diag([-1.0, -1.0, 1.0])).is_coaxial()
