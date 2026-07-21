"""Build a cached free-boundary config from a QUASR serial.

Writes into <cachedir>:
  meta.json      -- nfp, phiedge, r_char, b_char, raxis_guess, rbc/zbs (lists), mpol, ntor, nzeta
  response.json  -- vmecpp.MagneticFieldResponseTable serialized
  coils.quasr<id> -- makegrid coils file
"""

import json
import sys
from pathlib import Path

import numpy as np
from simsopt._core import load as simsopt_load
from simsopt.field import BiotSavart, coils_to_makegrid

import vmecpp

VACUUM_PERMEABILITY = 4.0e-7 * np.pi


def boundary_extent(surface):
    xyz = surface.gamma()
    r = np.hypot(xyz[..., 0], xyz[..., 1])
    z = xyz[..., 2]
    return float(r.min()), float(r.max()), float(z.min()), float(z.max())


def enclosed_toroidal_flux(coils, surface, n_theta=1000):
    bs = BiotSavart(coils)
    loop = surface.cross_section(0.0, thetas=n_theta)
    bs.set_points(loop)
    a_field = bs.A()
    b_char = float(np.mean(np.linalg.norm(bs.B(), axis=1)))
    dl = np.roll(loop, -1, axis=0) - loop
    flux = float(np.sum(0.5 * (a_field + np.roll(a_field, -1, axis=0)) * dl))
    r_axis_guess = float(np.hypot(*loop[:, :2].mean(axis=0)))
    bs.set_points(np.array([[r_axis_guess, 0.0, 0.0]]))
    sign = float(np.sign(bs.B()[0, 1])) or 1.0
    return abs(flux) * sign, b_char


def boundary_coefficients(surface, mpol, ntor):
    rz = surface.to_RZFourier()
    resized = rz.change_resolution(mpol - 1, ntor)
    rz = rz if resized is None else resized
    return rz.rc.copy(), rz.zs.copy(), float(rz.get_rc(0, 0))


def main():
    serial = Path(sys.argv[1])
    cachedir = Path(sys.argv[2])
    mpol = int(sys.argv[3])
    ntor = int(sys.argv[4])
    nzeta = int(sys.argv[5])
    mgrid_points = int(sys.argv[6])
    mgrid_margin = float(sys.argv[7])
    cachedir.mkdir(parents=True, exist_ok=True)
    config_id = int("".join(c for c in serial.stem if c.isdigit()))

    surfaces, coils = simsopt_load(str(serial))
    boundary = surfaces[-1]
    nfp = int(boundary.nfp)
    assert boundary.stellsym
    n_base = len(coils) // (2 * nfp)
    base_coils = coils[:n_base]

    coils_file = cachedir / f"coils.quasr{config_id:07d}"
    coils_to_makegrid(
        coils_file,
        [c.curve for c in base_coils],
        [c.current for c in base_coils],
        groups=[1] * len(coils),
        nfp=nfp,
        stellsym=True,
    )
    extcur = float(coils[0].current.get_value())

    phiedge, b_char = enclosed_toroidal_flux(coils, boundary)
    rbc, zbs, raxis_guess = boundary_coefficients(boundary, mpol, ntor)

    r_min, r_max, z_min, z_max = boundary_extent(boundary)
    dr, dz = r_max - r_min, z_max - z_min
    mg = vmecpp.MakegridParameters(
        normalize_by_currents=True,
        assume_stellarator_symmetry=True,
        number_of_field_periods=nfp,
        r_grid_minimum=r_min - mgrid_margin * dr,
        r_grid_maximum=r_max + mgrid_margin * dr,
        number_of_r_grid_points=mgrid_points,
        z_grid_minimum=z_min - mgrid_margin * dz,
        z_grid_maximum=z_max + mgrid_margin * dz,
        number_of_z_grid_points=mgrid_points,
        number_of_phi_grid_points=nzeta,
    )
    response = vmecpp.MagneticFieldResponseTable.from_coils_file(coils_file, mg)
    (cachedir / "response.json").write_text(response.model_dump_json())
    (cachedir / "makegrid.json").write_text(mg.model_dump_json())

    meta = {
        "config_id": config_id,
        "nfp": nfp,
        "phiedge": phiedge,
        "r_char": float(rbc[0, ntor]),
        "b_char": b_char,
        "raxis_guess": raxis_guess,
        "extcur": extcur,
        "mpol": mpol,
        "ntor": ntor,
        "nzeta": nzeta,
        "rbc": rbc.tolist(),
        "zbs": zbs.tolist(),
    }
    (cachedir / "meta.json").write_text(json.dumps(meta))
    # aspect ratio / shaping proxy printed for ranking
    print(
        f"built {config_id}: nfp={nfp} phiedge={phiedge:.4f} "
        f"R={rbc[0, ntor]:.3f} Rext=[{r_min:.2f},{r_max:.2f}] Zext=[{z_min:.2f},{z_max:.2f}]"
    )


if __name__ == "__main__":
    main()
