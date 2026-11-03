# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
from scipy import constants
from scipy.sparse.linalg import eigsh
from scipy.special import jnp_zeros

import vmecpp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
import vessel_modes as vm  # noqa: E402

WALL = 2.0  # minor radius of the wall of the round vessel


def round_torus(radius):
    """A round plasma of minor radius 1 m."""
    rbc, zbs = np.zeros((4, 1)), np.zeros((4, 1))
    rbc[0, 0], rbc[1, 0], zbs[1, 0] = radius, 1.0, 1.0
    return vmecpp.VmecInput(
        nfp=1,
        mpol=4,
        ntor=0,
        ns_array=np.array([25]),
        ftol_array=np.array([1e-10]),
        niter_array=np.array([5000]),
        phiedge=1.0,
        ncurr=0,
        ai=np.array([0.1]),
        pmass_type="power_series",
        am=np.array([0.0]),
        pres_scale=0.0,
        rbc=rbc,
        zbs=zbs,
        raxis_c=np.array([radius]),
        zaxis_s=np.array([0.0]),
    )


@pytest.fixture(scope="module")
def torus():
    return vm.solve_vessel(round_torus(10.0), gap=1.0)


@pytest.fixture(scope="module")
def stellarator_input():
    return vmecpp.VmecInput.from_file(
        ROOT / "examples" / "data" / "cth_like_fixed_bdy.json"
    )


@pytest.fixture(scope="module")
def stellarator(stellarator_input):
    return vm.solve_vessel(
        stellarator_input, gap=0.3, volume=69.0, field=2.3, kept=(4, 2)
    )


def filled(vessel):
    """The vessel with the plasma edge far past its wall, a uniform density."""
    uniform = copy.copy(vessel)
    uniform.plasma_edge = 1.0e6
    return uniform


def natural_units(vessel, f, temperature, seconds):
    r"""Density, mass, volume and noise power of the haloscope equation with \hbar = c =
    1, in eV^4, eV, 1/eV^3 and eV^2."""
    e, hbar = constants.e, constants.hbar
    hbarc = hbar * constants.c / e
    noise = constants.k * temperature * np.sqrt(f / vm.Q_HALO / seconds)
    return (
        vm.RHO / e * hbarc**3,
        2.0 * np.pi * hbar * f / e,
        vessel.volume / hbarc**3,
        noise / e * hbar / e,
    )


def exact_axisymmetric_mode(aspect, cells=320):
    r"""The E_\phi = u / \sqrt{R} mode of a round wall of radius 1, -\nabla^2 u + (3 /
    4) u / R^2 = k^2 u by finite differences: k b and the form factor in a field B ~ 1 /
    R."""
    h = 2.0 / cells
    axis = aspect - 1.0 + h * np.arange(cells + 1)
    height = -1.0 + h * np.arange(cells + 1)
    r, z = np.meshgrid(axis, height, indexing="ij")
    inside = (r - aspect) ** 2 + z**2 < 1.0
    index = -np.ones(inside.shape, dtype=int)
    index[inside] = np.arange(inside.sum())
    rows, columns = [], []
    for step in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        neighbour = np.roll(index, step, axis=(0, 1))
        both = inside & (neighbour >= 0)
        rows.append(index[both])
        columns.append(neighbour[both])
    rows, columns = np.concatenate(rows), np.concatenate(columns)
    size = int(inside.sum())
    radius = r[inside]
    matrix = sp.csc_matrix(
        (-np.ones(rows.size) / h**2, (rows, columns)), shape=(size, size)
    ) + sp.diags(4.0 / h**2 + 0.75 / radius**2, format="csc")
    values, vectors = eigsh(matrix, k=1, sigma=0.0)
    u = np.abs(vectors[:, 0])
    volume = np.sum(radius)
    mean_field = np.sum(aspect / radius * radius) / volume
    overlap = np.sum(u / np.sqrt(radius) * aspect / radius * radius)
    return np.sqrt(values[0]), overlap**2 / (mean_field**2 * volume * np.sum(u**2))


def test_vessel_grids_reproduce_the_solve(stellarator):
    r"""The grids integrate to the volume of the solve, and B^\theta e_\theta + B^\zeta
    e_\zeta has the magnitude of |B|."""
    np.testing.assert_allclose(
        4.0 * np.pi**2 * stellarator.measure, stellarator.volume, rtol=1e-3
    )
    for grid in stellarator.grids:
        np.testing.assert_allclose(np.linalg.norm(grid["unit"], axis=0), 1.0, rtol=2e-2)


def test_curl_curl_form_annihilates_gradients(stellarator):
    """The discrete gradient of any potential that vanishes on the wall has no curl, in
    a three-dimensional vessel."""
    matrices, _ = vm.assemble_cavity(stellarator, 0)
    cells = stellarator.cells
    forms = vm.Forms(stellarator, 0, 3 * cells + 2)
    vm.close_edges(forms, cells)
    m, d_t, d_v, size = forms.m, forms.d_t, forms.d_v, forms.size
    rng = np.random.default_rng(0)
    potential = rng.normal(size=(cells + 1, size)) + 1j * rng.normal(
        size=(cells + 1, size)
    )
    potential[0, m != 0] = 0.0
    potential[-1] = 0.0
    field = np.zeros((3 * cells + 2, size), dtype=complex)
    for node in range(cells + 1):
        field[3 * node] = d_t * potential[node]
        field[3 * node + 1] = d_v * potential[node]
    for cell in range(cells):
        field[3 * cell + 2] = (potential[cell + 1] - potential[cell]) / stellarator.dx
    vector = field.ravel()[forms.keep]
    residual = matrices["a"] @ vector
    assert (
        np.abs(residual).max()
        < 1.0e-9 * np.abs(matrices["a"]).max() * np.abs(vector).max()
    )


def test_cavity_mode_of_a_round_torus_is_the_bent_cylinder(torus):
    """A thin torus of wall radius b has k b = 2.405 and form factor 4 / 2.405^2, and
    the wall limits Q to Z_0 k b / (2 R_s)."""
    row = vm.cavity_mode(torus, margin=1.0)[0]
    np.testing.assert_allclose(row["k"] * WALL, 2.405, rtol=0.01)
    np.testing.assert_allclose(row["form"], 4.0 / 2.405**2, rtol=0.02)
    expected = vm.ETA0 * row["k"] * WALL / (2.0 * vm.wall_resistance(row["f"]))
    np.testing.assert_allclose(row["q"], expected, rtol=0.03)


@pytest.mark.parametrize("aspect", [3.0, 1.5])
def test_cavity_mode_is_the_exact_axisymmetric_mode(aspect):
    """At low aspect ratio the vector mode keeps the curvature of the field."""
    vessel = vm.solve_vessel(round_torus(aspect * WALL), gap=1.0, cells=96)
    row = vm.cavity_mode(vessel)[0]
    k, form = exact_axisymmetric_mode(aspect)
    np.testing.assert_allclose(row["k"] * WALL, k, rtol=5e-3)
    np.testing.assert_allclose(row["form"], form, rtol=0.02)


def test_uniform_plasma_adds_its_frequency_in_quadrature(torus):
    r"""With E along the field and a uniform density, k^2 grows by \omega_{pe}^2 /
    c^2."""
    vessel = filled(torus)
    empty, plasma = vm.cavity_mode(vessel)[0], vm.cavity_mode(vessel, 1.0e14)[0]
    expected = np.sqrt(empty["k"] ** 2 + 1.0e14 * vm.PE2_PER_DENSITY)
    np.testing.assert_allclose(plasma["k"], expected, rtol=1e-3)


def test_density_shifts_the_mode_by_its_parallel_share(stellarator):
    r"""To first order in three dimensions, k^2 grows by \omega_{pe}^2 / c^2 times the
    share of the density-weighted E along the field."""
    shift = 1.02 * vm.tracked_mode(stellarator, 0.0)[0]
    empty, plasma = (
        vm.cavity_modes(stellarator, 0, density, shift) for density in (0.0, 1.0e12)
    )
    i = int(np.argmax(empty["overlaps"]["b"]))
    j = int(np.argmin(np.abs(plasma["k"] - empty["k"][i])))
    np.testing.assert_allclose(
        (plasma["k"][j] ** 2 - empty["k"][i] ** 2) / (1.0e12 * vm.PE2_PER_DENSITY),
        empty["parallel"][i],
        rtol=1e-2,
    )


def test_asymmetric_paths_reproduce_a_symmetric_vessel(stellarator_input):
    """A stellarator-symmetric boundary run through lasym gives the same mode."""
    symmetric = vm.solve_vessel(stellarator_input, gap=0.3, kept=(4, 2))
    plasma = stellarator_input.model_copy(deep=True)
    plasma.lasym = True
    plasma.rbs = np.zeros_like(plasma.rbc)
    plasma.zbc = np.zeros_like(plasma.zbs)
    plasma.raxis_s = np.zeros_like(plasma.raxis_c)
    plasma.zaxis_c = np.zeros_like(plasma.zaxis_s)
    asymmetric = vm.solve_vessel(plasma, gap=0.3, kept=(4, 2))
    expected, actual = vm.cavity_mode(symmetric)[0], vm.cavity_mode(asymmetric)[0]
    np.testing.assert_allclose(actual["f"], expected["f"], rtol=1e-4)
    np.testing.assert_allclose(actual["form"], expected["form"], rtol=1e-3)


def test_flipping_an_asymmetric_vessel_keeps_its_frequency():
    """An up-down asymmetric vessel and its mirror image have one spectrum."""
    frequencies = []
    for sign in (1.0, -1.0):
        plasma = round_torus(10.0)
        plasma.lasym = True
        # a triangular shift of the top against the bottom
        rbs = np.zeros_like(plasma.rbc)
        rbs[2, 0] = sign * 0.2
        plasma.rbs = rbs
        plasma.zbc = np.zeros_like(plasma.zbs)
        plasma.raxis_s = np.zeros_like(plasma.raxis_c)
        plasma.zaxis_c = np.zeros_like(plasma.zaxis_s)
        frequencies.append(vm.cavity_mode(vm.solve_vessel(plasma, gap=1.0))[0]["f"])
    np.testing.assert_allclose(frequencies[0], frequencies[1], rtol=1e-4)


@pytest.mark.parametrize(("seconds", "margin"), [(1800.0, 0.5), (86400.0, 1.0)])
def test_reach_of_the_cavity_mode_is_the_haloscope_equation(torus, seconds, margin):
    r"""The reach solves k T \sqrt{\Delta\nu / t} = g^2 (\rho / m) B^2 V C Q / 4.

    A dark photon of random polarization replaces the right side by \epsilon^2 \rho m V
    C' Q / 4, with C' = C_z / 3 = 2 / (3 (x^2 - 1)) on the TE_{11} mode of k b = x along
    z, and C' = C / 6 on the mode of E along the field along x, from the mean of
    \sin^2 \phi.
    """
    row, along_z, along_x = vm.cavity_mode(torus, seconds=seconds, margin=margin)
    rho, mass, volume, noise = natural_units(
        torus, row["f"], row["temperature"], seconds
    )
    hbarc = constants.hbar * constants.c / constants.e
    field2 = torus.mean_field**2 / constants.mu_0 / constants.e * hbarc**3
    g = np.sqrt(4.0 * noise * mass / (rho * field2 * volume * row["form"] * row["q"]))
    np.testing.assert_allclose(row["reach"], g * 1.0e9, rtol=1e-3)
    x = jnp_zeros(1, 1)[0]
    np.testing.assert_allclose(along_z["k"] * WALL, x, rtol=0.01)
    rho, mass, volume, noise = natural_units(
        torus, along_z["f"], along_z["temperature"], seconds
    )
    form = 2.0 / (x**2 - 1.0)
    eps = np.sqrt(12.0 * noise / (rho * mass * volume * form * along_z["q"]))
    np.testing.assert_allclose(along_z["reach"], eps, rtol=0.03)
    rho, mass, volume, noise = natural_units(
        torus, along_x["f"], along_x["temperature"], seconds
    )
    eps = np.sqrt(
        4.0 * noise / (rho * mass * volume * row["form"] / 6.0 * along_x["q"])
    )
    np.testing.assert_allclose(along_x["reach"], eps, rtol=0.06)


def test_fast_wave_of_a_filled_torus_is_the_first_magnetic_mode(torus):
    r"""Plasma out to the wall makes the lowest mode across the field k b = x at v_A,
    with J_1'(x) = 0.

    The drive along z is \epsilon^{-1} \oint \Psi n_R dl along the torus, with
    \epsilon^{-1} = (k b / x)^2 at unit norm of \Psi.
    """
    vessel = filled(torus)
    q, temperature, seconds = 100.0, 1000.0 * vm.E / vm.K_B, 1800.0
    row = vm.fast_wave(vessel, "z", 1.0e20, q, temperature, seconds)
    x = jnp_zeros(1, 1)[0]
    speed = 1.0 / np.sqrt(
        1.0 + vm.C**2 * vm.MU0 * 1.0e20 * vm.MI / vessel.mean_field**2
    )
    np.testing.assert_allclose(row["k"] * WALL / speed, x, rtol=0.05)
    assert row["across"] > 0.8
    drive = (row["k"] * WALL / x) ** 2 * np.sqrt(
        2.0 * vessel.volume / WALL**2 / (1.0 - 1.0 / x**2)
    )
    # the noise of unit norm in the volume is 2 pi that in the volume over 4 pi^2
    noise = 2.0 * np.pi * vm.noise_field(row["f"], q, temperature, seconds)
    eps = noise * row["k"] / (drive * vm.B_HALO * vm.POLARIZATION * q)
    np.testing.assert_allclose(row["eps"], eps, rtol=0.1)


def test_fast_wave_frequency_follows_the_alfven_speed(stellarator):
    """In a three-dimensional vessel the frequency falls as the square root of the
    density."""
    vessel = filled(stellarator)
    dense, thin = vm.fast_wave(vessel, "z", 4.0e20), vm.fast_wave(vessel, "z", 1.0e20)
    np.testing.assert_allclose(thin["lowest"] / dense["lowest"], 2.0, rtol=0.02)
