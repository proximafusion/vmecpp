# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Build the NCSX vacuum-field mgrid used by the high-beta free-boundary case.

Evaluates the Biot-Savart field of the NCSX modular coils (simsopt's built-in
configuration) on a cylindrical (R, phi, Z) grid over one field period and writes
it as an mgrid NetCDF file (mgrid_ncsx.nc) next to this script. Requires simsopt.

Run once before ncsx_high_beta.py.
"""

from pathlib import Path

import numpy as np
from simsopt.configs import get_ncsx_data
from simsopt.field import BiotSavart, MGrid, coils_via_symmetries

nfp = 3
curves, currents, ma = get_ncsx_data()
coils = coils_via_symmetries(curves, currents, nfp, True)
bs = BiotSavart(coils)

nr, nz, nphi = 121, 121, 48
rmin, rmax, zmin, zmax = 0.7, 2.3, -0.75, 0.75
Rv = np.linspace(rmin, rmax, nr)
Zv = np.linspace(zmin, zmax, nz)
Pv = np.linspace(0.0, 2 * np.pi / nfp, nphi, endpoint=False)

br = np.zeros((nphi, nz, nr))
bp = np.zeros_like(br)
bz = np.zeros_like(br)
Rg, Zg = np.meshgrid(Rv, Zv)  # (nz, nr)
for ip, ph in enumerate(Pv):
    c, s = np.cos(ph), np.sin(ph)
    pts = np.stack([Rg * c, Rg * s, Zg], -1).reshape(-1, 3)
    bs.set_points(pts)
    B = bs.B().reshape(nz, nr, 3)
    bx, by, bzc = B[..., 0], B[..., 1], B[..., 2]
    br[ip] = bx * c + by * s
    bp[ip] = -bx * s + by * c
    bz[ip] = bzc

ir = int(np.argmin(np.abs(Rv - 1.47)))
iz = int(np.argmin(np.abs(Zv - 0.0)))
bmag = np.hypot(br[0, iz, ir], np.hypot(bp[0, iz, ir], bz[0, iz, ir]))
print(
    f"# on-axis-ish (R=1.47,phi=0,Z=0): br={br[0, iz, ir]:.4f} bp={bp[0, iz, ir]:.4f} "
    f"bz={bz[0, iz, ir]:.4f} |B|={bmag:.4f} T",
    flush=True,
)

out = str(Path(__file__).resolve().parent / "mgrid_ncsx.nc")
mgrid = MGrid(nr=nr, nz=nz, nphi=nphi, nfp=nfp, rmin=rmin, rmax=rmax, zmin=zmin, zmax=zmax)
mgrid.add_field_cylindrical(br, bp, bz, name="ncsx")
mgrid.write(out)
print(f"# wrote {out} (nr={nr} nz={nz} nphi={nphi} nfp={nfp})", flush=True)
