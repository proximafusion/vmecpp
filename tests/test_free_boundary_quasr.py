# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Long-running free-boundary integration tests driven by QUASR configurations.

The QUASR database (https://quasr.flatironinstitute.org) ships stellarator and
tokamak configurations as SIMSOPT serialisations that bundle a set of nested
flux surfaces together with the coil set that produces them. Each serial file
can be loaded with::

    from simsopt._core import load
    surfaces, coils = load("serial0167381.json")

These tests use those configurations as a physically self-consistent test bed
for the VMEC++ free-boundary solver:

* the outermost surface provides the initial plasma boundary,
* the coils provide the external magnetic field. The mgrid magnetic-field
  response table is computed by VMEC++ itself
  (``vmecpp.MagneticFieldResponseTable``); SIMSOPT is used only to load the
  configuration and to evaluate the vacuum field for ``phiedge``,
* the enclosed vacuum toroidal flux provides ``phiedge``.

For every configuration three physics regimes are exercised:

* ``vacuum``          -- zero pressure, zero net toroidal current,
* ``beta1``          -- ~1% volume-averaged beta, zero net current,
* ``beta2_current``  -- ~2% volume-averaged beta with a net toroidal current.

Convergence is *not* guaranteed: free-boundary VMEC++ frequently plateaus on
these shaped equilibria, and a non-converging run raises and is reported as a
test failure on purpose. That is the intended signal -- the suite doubles as a
convergence-diagnostic bed, so an algorithmic improvement that suddenly makes a
previously-failing case converge shows up as a newly-passing test.

The runs are expensive (``ns = [8, 24, 71]``, ``mpol = ntor = 10``), so the
whole module is marked ``slow`` and is skipped unless network access, SIMSOPT
and matplotlib are available. Resolution and the set of configurations can be
overridden through environment variables (see ``_int_env`` / ``_id_env``) to run
a cheaper smoke tier locally or in CI.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import vmecpp

pytestmark = pytest.mark.slow

# Third-party imports needed only for these integration tests. Missing any of
# them (or having no network access) simply skips the module rather than failing
# the whole suite.
simsopt_load = pytest.importorskip(
    "simsopt._core", reason="SIMSOPT is required for the QUASR free-boundary tests"
).load
BiotSavart = pytest.importorskip(
    "simsopt.field", reason="SIMSOPT is required for the QUASR free-boundary tests"
).BiotSavart
MplPath = pytest.importorskip(
    "matplotlib.path",
    reason="matplotlib is required to integrate the enclosed toroidal flux",
).Path

VACUUM_PERMEABILITY = 4.0e-7 * np.pi  # mu0 [T m / A]

# QUASR serial files are checked into the repository via Git LFS so the suite
# runs without network access (robust CI). Any ID not found locally is fetched
# from the QUASR database as a fallback.
LOCAL_DATA_DIR = Path(__file__).parent / "data" / "quasr"

# QUASR IDs exercised by default: the full set of checked-in configurations.
# Override with VMECPP_QUASR_IDS (comma separated) to run a subset -- the whole
# sweep is slow (one multigrid free-boundary solve per configuration x profile).
DEFAULT_QUASR_IDS = (
    954,
    957,
    9914,
    19493,
    19609,
    19940,
    29346,
    50136,
    65579,
    112718,
    165868,
    167381,
    336902,
)


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


def _ns_array_env() -> np.ndarray:
    """Multigrid ladder, overridable via VMECPP_QUASR_NS (comma separated)."""
    raw = os.environ.get("VMECPP_QUASR_NS")
    if raw:
        return np.array([int(x) for x in raw.split(",")], dtype=np.int64)
    return np.array([8, 24, 71], dtype=np.int64)


def _id_env() -> tuple[int, ...]:
    raw = os.environ.get("VMECPP_QUASR_IDS")
    if raw:
        return tuple(int(x) for x in raw.split(","))
    return DEFAULT_QUASR_IDS


# Resolution defaults follow the task specification; overridable for a fast tier.
MPOL = _int_env("VMECPP_QUASR_MPOL", 10)
NTOR = _int_env("VMECPP_QUASR_NTOR", 10)
# VMEC needs the number of toroidal grid points per field period to resolve n up
# to NTOR (nzeta > 2*NTOR); the mgrid grid must use the same count.
NZETA = _int_env("VMECPP_QUASR_NZETA", 36)
FTOL_FINAL = float(os.environ.get("VMECPP_QUASR_FTOL", "1e-9"))
NITER_CAP = _int_env("VMECPP_QUASR_NITER", 2000)


@dataclass(frozen=True)
class Profile:
    """A physics regime to impose on top of a QUASR configuration."""

    name: str
    target_beta: float
    # Net toroidal current as a fraction of a characteristic current
    # I_char = 2*pi*<R>*<B_phi>/mu0. 0.0 means a current-free equilibrium.
    current_fraction: float = 0.0
    # power_series coefficients for the (unscaled) pressure shape p(s) ~ (1-s).
    pressure_shape: tuple[float, ...] = (1.0, -1.0)
    # power_series coefficients for the toroidal current density shape.
    current_shape: tuple[float, ...] = (1.0, -1.0)


PROFILES = [
    Profile(name="vacuum", target_beta=0.0),
    Profile(name="beta1", target_beta=0.01),
    Profile(name="beta2_current", target_beta=0.02, current_fraction=0.02),
]


@dataclass
class QuasrConfig:
    """A QUASR configuration reduced to what VMEC++ free boundary needs.

    Everything here is profile-independent and therefore built once per
    configuration (see the ``quasr_configs`` fixture) and reused across all
    three physics regimes. In particular the mgrid ``response`` table -- the
    expensive part -- is generated on the fly by VMEC++ a single time.
    """

    config_id: int
    nfp: int
    stellsym: bool
    boundary: object  # simsopt SurfaceRZFourier at full torus resolution
    extcur: np.ndarray  # per-coil currents [A], one circuit per coil
    response: vmecpp.MagneticFieldResponseTable  # VMEC++ mgrid response table
    phiedge: float  # enclosed vacuum toroidal flux [Wb]
    r_char: float  # characteristic major radius <R>
    b_char: float  # characteristic |B_phi| in the plasma region [T]


# ---------------------------------------------------------------------------
# Configuration loading and geometry/field conversion
# ---------------------------------------------------------------------------


def _quasr_url(config_id: int) -> str:
    id_str = f"{config_id:07d}"
    return (
        "https://quasr.flatironinstitute.org/simsopt_serials/"
        f"{id_str[0:4]}/serial{id_str}.json"
    )


def _resolve_config_file(config_id: int, cache_dir: Path) -> Path:
    """Return a QUASR serial file, preferring the checked-in LFS copy.

    The configurations are committed under ``tests/data/quasr`` via Git LFS, so
    a normal checkout with LFS pulled needs no network access. If the file is
    absent (ID not checked in, or LFS not pulled) it is downloaded to
    ``cache_dir`` as a fallback; the test is skipped if that also fails.
    """
    local = LOCAL_DATA_DIR / f"serial{config_id:07d}.json"
    # A non-pulled LFS file is a small pointer stub; treat only real payloads as
    # present (the serial files are hundreds of kB).
    if local.exists() and local.stat().st_size > 4096:
        return local

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"serial{config_id:07d}.json"
    if path.exists() and path.stat().st_size > 0:
        return path
    request = urllib.request.Request(
        _quasr_url(config_id), headers={"User-Agent": "vmecpp-tests"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            path.write_bytes(response.read())
    except (urllib.error.URLError, TimeoutError) as exc:
        pytest.skip(f"cannot reach QUASR database for ID {config_id}: {exc}")
    return path


def _write_makegrid_coils_file(path: Path, coils: list, nfp: int = 1) -> None:
    """Write coils in MAKEGRID ``coils.*`` format for VMEC++ to read.

    Each coil is emitted as its own current group so that VMEC++'s response
    table has one circuit per coil, scaled by ``extcur``. This is the same
    on-disk format that SIMSOPT's ``coils_to_makegrid`` produces, written here
    directly so the mgrid response computation stays entirely inside VMEC++.
    """
    with open(path, "w") as wfile:
        wfile.write(f"periods {nfp:3d} \n")
        wfile.write("begin filament \n")
        wfile.write("mirror NIL \n")
        for icoil, coil in enumerate(coils):
            gamma = coil.curve.gamma()
            current = coil.current.get_value()
            wfile.writelines(
                f"{x:23.15E} {y:23.15E} {z:23.15E} {current:23.15E}\n"
                for x, y, z in gamma
            )
            # Close the loop: repeat the first point with zero current, then the
            # circuit group index and a label.
            x0, y0, z0 = gamma[0]
            label = getattr(coil.curve, "name", f"coil{icoil + 1}")
            wfile.write(
                f"{x0:23.15E} {y0:23.15E} {z0:23.15E} {0.0:23.15E}"
                f" {icoil + 1} {label:10} \n"
            )
        wfile.write("end \n")


def _cross_section_rz(surface, phi_index: int = 0) -> tuple[np.ndarray, np.ndarray]:
    xyz = surface.gamma()[phi_index]  # (ntheta, 3) at a fixed toroidal angle
    return np.hypot(xyz[:, 0], xyz[:, 1]), xyz[:, 2]


def _enclosed_toroidal_flux(coils, surface, n_grid: int = 64) -> tuple[float, float]:
    """Vacuum toroidal flux through the boundary cross-section at phi = 0.

    Returns (phiedge, characteristic |B_phi|). The flux is the surface integral of B_phi
    over the enclosed (R, Z) area, evaluated with the coils' Biot-Savart field on a
    masked Cartesian grid at phi = 0 (where phi_hat = +y).
    """
    biot_savart = BiotSavart(coils)
    r_boundary, z_boundary = _cross_section_rz(surface)
    polygon = MplPath(np.column_stack([r_boundary, z_boundary]))

    r_grid = np.linspace(r_boundary.min(), r_boundary.max(), n_grid)
    z_grid = np.linspace(z_boundary.min(), z_boundary.max(), n_grid)
    rr, zz = np.meshgrid(r_grid, z_grid)
    inside = polygon.contains_points(np.column_stack([rr.ravel(), zz.ravel()])).reshape(
        rr.shape
    )

    points = np.column_stack([rr.ravel(), np.zeros(rr.size), zz.ravel()])
    biot_savart.set_points(points)
    b_phi = biot_savart.B().reshape(n_grid, n_grid, 3)[:, :, 1]

    cell_area = (r_grid[1] - r_grid[0]) * (z_grid[1] - z_grid[0])
    phiedge = float(np.sum(b_phi[inside]) * cell_area)
    b_char = float(np.abs(b_phi[inside]).mean())
    return phiedge, b_char


def _boundary_coefficients(
    surface, mpol: int, ntor: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """VMEC++ rbc/zbs arrays of shape (mpol, 2*ntor+1) from a SIMSOPT surface."""
    rz = surface.to_RZFourier()
    # SurfaceRZFourier uses m up to mpol inclusive, unlike VMEC++.
    resized = rz.change_resolution(mpol - 1, ntor)
    rz = rz if resized is None else resized

    rbc = np.zeros((mpol, 2 * ntor + 1))
    zbs = np.zeros((mpol, 2 * ntor + 1))
    for m in range(mpol):
        for n in range(2 * ntor + 1):
            rbc[m, n] = rz.get_rc(m, n - ntor)
            zbs[m, n] = rz.get_zs(m, n - ntor)
    return rbc, zbs, float(rz.get_rc(0, 0))


def _build_response_table(
    boundary, coils_file: Path, nfp: int, nzeta: int
) -> vmecpp.MagneticFieldResponseTable:
    r_boundary, z_boundary = _cross_section_rz(boundary)
    # A generous margin around the plasma so the vacuum field is well resolved.
    makegrid_parameters = vmecpp.MakegridParameters(
        normalize_by_currents=True,
        assume_stellarator_symmetry=False,
        number_of_field_periods=nfp,
        r_grid_minimum=float(r_boundary.min() * 0.7),
        r_grid_maximum=float(r_boundary.max() * 1.2),
        number_of_r_grid_points=48,
        z_grid_minimum=float(z_boundary.min() * 1.2),
        z_grid_maximum=float(z_boundary.max() * 1.2),
        number_of_z_grid_points=48,
        number_of_phi_grid_points=nzeta,
    )
    # VMEC++ computes the mgrid response table from the coils file.
    return vmecpp.MagneticFieldResponseTable.from_coils_file(
        coils_file, makegrid_parameters
    )


def _load_config(config_id: int, cache_dir: Path) -> QuasrConfig:
    path = _resolve_config_file(config_id, cache_dir)
    surfaces, coils = simsopt_load(str(path))
    boundary = surfaces[-1]
    nfp = int(boundary.nfp)

    coils_file = cache_dir / f"coils.quasr{config_id:07d}"
    _write_makegrid_coils_file(coils_file, coils, nfp=1)
    extcur = np.array([coil.current.get_value() for coil in coils], dtype=float)

    phiedge, b_char = _enclosed_toroidal_flux(coils, boundary)
    rbc, _, _ = _boundary_coefficients(boundary, MPOL, NTOR)
    # Built once here and reused for every profile of this configuration.
    response = _build_response_table(boundary, coils_file, nfp, NZETA)

    return QuasrConfig(
        config_id=config_id,
        nfp=nfp,
        stellsym=bool(boundary.stellsym),
        boundary=boundary,
        extcur=extcur,
        response=response,
        phiedge=phiedge,
        r_char=float(rbc[0, NTOR]),
        b_char=b_char,
    )


# ---------------------------------------------------------------------------
# VmecInput assembly
# ---------------------------------------------------------------------------


def _pressure_scale(config: QuasrConfig, profile: Profile) -> float:
    """Analytic pressure amplitude targeting profile.target_beta.

    For p(s) = pres_scale * (1 - s), the volume-averaged beta is approximately
    betatotal ~ 2 mu0 <p> / <B^2> ~ mu0 * pres_scale / b_char^2 (using
    <(1 - s)> ~ 0.5 and B ~ b_char). This is only a first guess; the resulting
    beta is checked with a wide tolerance and the exact value is not the point of
    the test.
    """
    if profile.target_beta == 0.0:
        return 0.0
    return profile.target_beta * config.b_char**2 / VACUUM_PERMEABILITY


def _ftol_array(n_grids: int) -> np.ndarray:
    """Per-grid force tolerance: loose on coarse grids, FTOL_FINAL on the last."""
    if n_grids == 1:
        return np.array([FTOL_FINAL])
    return np.geomspace(1e-6, FTOL_FINAL, n_grids)


def _make_input(
    config: QuasrConfig,
    profile: Profile,
    ns_array: np.ndarray,
) -> vmecpp.VmecInput:
    rbc, zbs, r_axis_guess = _boundary_coefficients(config.boundary, MPOL, NTOR)

    vmec_input = vmecpp.VmecInput.default()
    vmec_input.lasym = False
    vmec_input.nfp = config.nfp
    vmec_input.mpol = MPOL
    vmec_input.ntor = NTOR
    vmec_input.ntheta = 0
    vmec_input.nzeta = NZETA
    vmec_input.ns_array = np.asarray(ns_array, dtype=np.int64)
    vmec_input.ftol_array = _ftol_array(len(ns_array))
    vmec_input.niter_array = np.full(len(ns_array), NITER_CAP, dtype=np.int64)
    vmec_input.delt = 0.7
    vmec_input.tcon0 = 1.0
    vmec_input.nstep = 200
    vmec_input.phiedge = config.phiedge
    vmec_input.gamma = 0.0
    vmec_input.bloat = 1.0

    # Pressure profile: p(s) = pres_scale * sum(am_i s^i).
    vmec_input.pmass_type = "power_series"
    vmec_input.am = np.array(profile.pressure_shape, dtype=float)
    vmec_input.pres_scale = _pressure_scale(config, profile)

    # Current profile (ncurr=1 drives a prescribed net toroidal current curtor).
    vmec_input.ncurr = 1
    vmec_input.pcurr_type = "power_series"
    vmec_input.ac = np.array(profile.current_shape, dtype=float)
    i_char = 2.0 * np.pi * config.r_char * config.b_char / VACUUM_PERMEABILITY
    vmec_input.curtor = profile.current_fraction * i_char

    vmec_input.lfreeb = True
    vmec_input.extcur = config.extcur
    vmec_input.nvacskip = 6

    vmec_input.raxis_c = np.zeros(NTOR + 1)
    vmec_input.zaxis_s = np.zeros(NTOR + 1)
    vmec_input.raxis_c[0] = r_axis_guess
    vmec_input.rbc = rbc
    vmec_input.zbs = zbs
    return vmec_input


# ---------------------------------------------------------------------------
# Fixtures and tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def quasr_cache_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("quasr_cache")


@pytest.fixture(scope="module")
def quasr_configs(quasr_cache_dir) -> dict[int, QuasrConfig]:
    return {
        config_id: _load_config(config_id, quasr_cache_dir) for config_id in _id_env()
    }


@pytest.mark.parametrize("config_id", _id_env())
@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.name)
def test_free_boundary_quasr(
    config_id: int, profile: Profile, quasr_configs: dict[int, QuasrConfig]
) -> None:
    config = quasr_configs[config_id]
    ns_array = _ns_array_env()

    vmec_input = _make_input(config, profile, ns_array)

    # A non-converging run raises RuntimeError; that is a deliberate, informative
    # failure for this convergence-diagnostic suite (see the module docstring).
    output = vmecpp.run(vmec_input, magnetic_field=config.response, verbose=False)
    wout = output.wout

    # Lenient physical sanity checks so that convergence is what the test keys on.
    assert wout.ier_flag == 0
    assert wout.volume > 0.0
    assert np.isfinite(wout.aspect)
    assert wout.aspect > 1.0
    assert np.isfinite(wout.betatotal)

    if profile.target_beta == 0.0:
        # Vacuum: beta and net toroidal current are both essentially zero.
        assert wout.betatotal == pytest.approx(0.0, abs=1e-4)
    else:
        # The analytic pressure scale should land within a factor of ~2 of the
        # requested beta; the exact value is not the focus of the test.
        assert 0.4 * profile.target_beta < wout.betatotal < 2.5 * profile.target_beta

    if profile.current_fraction != 0.0:
        # The prescribed net toroidal current should be reflected in the output.
        assert wout.ctor == pytest.approx(vmec_input.curtor, rel=0.1)
