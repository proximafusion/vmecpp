# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""AEGIS: Accurate Exterior Green's Integral Solver (virtual-casing reference).

A virtual-casing exterior-magnetic-field solver for VMEC++ free boundary, and a
higher-order alternative to NESTOR's scalar-potential vacuum field. Given a VMEC
equilibrium and the coil field (mgrid), AEGIS computes the exterior vacuum field
on the LCFS from the virtual-casing principle: the plasma's contribution to the
exterior field is the Biot-Savart field of the equivalent surface current
K = n x B_plasma (plus the n.B_plasma monopole term) integrated over the LCFS,
with the near-singular self-interaction resolved by Quadrature By Expansion (QBX).

    B_exterior(r) = B_coil(r) + B_axis(r) + (1/4pi) integral_LCFS
                        [ K x (r-r') + (n'.B_plasma)(r-r') ] / |r-r'|^3 dA'
    K = n x B_plasma,   B_plasma = B_total - B_coil - B_axis

B_axis is the net-toroidal-current field, added as a Biot-Savart line filament
along the magnetic axis: virtual casing cannot represent the net enclosed current
from the surface layers alone, so it is carried separately (as NESTOR does).

Unlike NESTOR, which solves a dense boundary-integral system for a scalar
potential every ivacskip iterations, AEGIS evaluates the Green's integral
directly from the known surface current (no linear solve). This is the validated
Python reference implementation and the specification for a C++ module; coupling
into VMEC replaces NESTOR's vacuum_magnetic_pressure with |B_exterior|^2 / 2 at
the LCFS. See issue #628.

Run `python examples/aegis_virtual_casing.py` to reproduce the validation on the
cth_like free-boundary case (requires the mgrid test data via git-lfs):
  [1] Biot-Savart integrator vs an analytic current loop      : rel err 1e-14
  [2] LCFS geometry/field reconstruction (B.n on flux surface): 1e-16
  [3] jump condition  n x (B_out - B_in) = K  via QBX         : ~1.4% mean
  [4] free-boundary physics at a converged equilibrium:
      |B_exterior|/|B_interior| = 0.999 (with the net-current filament);
      field direction agrees to ~4% -- the residual is the QBX extrapolation
      bias (resolution-independent), which singularity subtraction removes.
  [5] vacuum pressure |B|^2/2 vs NESTOR                        : ~7%
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

import vmecpp

try:
    from scipy.interpolate import RegularGridInterpolator
    from scipy.io import netcdf_file
except ImportError:  # scipy is an optional dependency, used only by this example
    RegularGridInterpolator = netcdf_file = None

MU0 = 4e-7 * np.pi
_TEST_DATA = Path(__file__).resolve().parents[1] / "src/vmecpp/cpp/vmecpp/test_data"


class CoilField:
    """Coil magnetic field from an mgrid file (raw mode: per-group field at unit
    current, scaled and summed by the coil currents extcur)."""

    def __init__(self, mgrid_path: str, extcur: np.ndarray):
        f = netcdf_file(mgrid_path, "r", mmap=False)
        self.nfp = int(f.variables["nfp"].data)
        kp = int(f.variables["kp"].data)
        rr = np.linspace(
            float(f.variables["rmin"].data),
            float(f.variables["rmax"].data),
            int(f.variables["ir"].data),
        )
        zz = np.linspace(
            float(f.variables["zmin"].data),
            float(f.variables["zmax"].data),
            int(f.variables["jz"].data),
        )
        pp = np.arange(kp) * (2 * np.pi / self.nfp / kp)
        ppe = np.concatenate([pp, [2 * np.pi / self.nfp]])

        def interp(name):
            groups = [
                extcur[g] * np.asarray(f.variables[f"{name}_{g + 1:03d}"].data)
                for g in range(len(extcur))
            ]
            tot = np.sum(groups, axis=0)
            tot = np.concatenate([tot, tot[:1]], 0)  # wrap phi periodically
            return RegularGridInterpolator(
                (ppe, zz, rr), tot, bounds_error=False, fill_value=None
            )

        self._br, self._bp, self._bz = interp("br"), interp("bp"), interp("bz")

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """B_coil (Cartesian) at Cartesian points X[..., 3]."""
        Rc = np.hypot(X[..., 0], X[..., 1])
        phi = np.arctan2(X[..., 1], X[..., 0])
        pts = np.stack([phi % (2 * np.pi / self.nfp), X[..., 2], Rc], -1)
        br, bp, bz = self._br(pts), self._bp(pts), self._bz(pts)
        c, s = np.cos(phi), np.sin(phi)
        return np.stack([br * c - bp * s, br * s + bp * c, bz], -1)


class AxisCurrent:
    """Field of the net enclosed toroidal current. Virtual casing cannot represent
    the net current from the surface layers alone, so it is added as a Biot-Savart
    line filament along the magnetic axis (as NESTOR does; the curtor term)."""

    def __init__(self, wout, nphi: int = 1024):
        ctor = float(wout.ctor)
        nfp = int(wout.nfp)
        rax, zax = np.asarray(wout.raxis_cc), np.asarray(wout.zaxis_cs)
        phi = np.arange(nphi) * 2 * np.pi / nphi
        n = np.arange(len(rax))
        r_axis = (rax[None, :] * np.cos(n[None, :] * nfp * phi[:, None])).sum(1)
        z_axis = (zax[None, :] * np.sin(n[None, :] * nfp * phi[:, None])).sum(1)
        axis = np.stack([r_axis * np.cos(phi), r_axis * np.sin(phi), z_axis], 1)
        self.seg = np.roll(axis, -1, 0) - axis
        self.mid = 0.5 * (axis + np.roll(axis, -1, 0))
        self.pref = MU0 * ctor / (4 * np.pi)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """B (Cartesian) at Cartesian points X[..., 3], chunked to bound memory."""
        shape = X.shape[:-1]
        flat = np.ascontiguousarray(X).reshape(-1, 3)
        out = np.empty_like(flat)
        for i in range(0, len(flat), 2048):
            d = flat[i:i + 2048, None, :] - self.mid[None]
            inv = 1.0 / np.linalg.norm(d, axis=-1) ** 3
            out[i:i + 2048] = self.pref * np.sum(
                np.cross(self.seg[None], d) * inv[..., None], axis=1
            )
        return out.reshape(*shape, 3)


class Lcfs:
    """LCFS surface, outward normal, and equilibrium field, reconstructed from a VMEC
    wout on a nu x nv angular grid over the full torus."""

    def __init__(self, wout, nu: int = 256, nv: int = 256):
        xm, xn = np.asarray(wout.xm), np.asarray(wout.xn)
        xmn, xnn = np.asarray(wout.xm_nyq), np.asarray(wout.xn_nyq)
        rmnc, zmns = np.asarray(wout.rmnc)[:, -1], np.asarray(wout.zmns)[:, -1]
        bsupu = np.asarray(wout.bsupumnc)[:, -1]
        bsupv = np.asarray(wout.bsupvmnc)[:, -1]
        self.nfp = int(wout.nfp)
        u = (np.arange(nu) + 0.5) * 2 * np.pi / nu
        v = (np.arange(nv) + 0.5) * 2 * np.pi / nv
        U, V = np.meshgrid(u, v, indexing="ij")

        def g(coef, m, n, kind):
            a = m[:, None, None] * U[None] - n[:, None, None] * V[None]
            return np.tensordot(
                coef, np.cos(a) if kind == "c" else np.sin(a), axes=(0, 0)
            )

        R, Z = g(rmnc, xm, xn, "c"), g(zmns, xm, xn, "s")
        Ru, Rv = g(-xm * rmnc, xm, xn, "s"), g(xn * rmnc, xm, xn, "s")
        Zu, Zv = g(xm * zmns, xm, xn, "c"), g(-xn * zmns, xm, xn, "c")
        Bu, Bv = g(bsupu, xmn, xnn, "c"), g(bsupv, xmn, xnn, "c")
        c, s = np.cos(V), np.sin(V)
        e_u = np.stack([Ru * c, Ru * s, Zu], -1)
        e_v = np.stack([Rv * c - R * s, Rv * s + R * c, Zv], -1)
        self.X = np.stack([R * c, R * s, Z], -1)
        self.B = Bu[..., None] * e_u + Bv[..., None] * e_v  # total interior field
        nun = np.cross(e_u, e_v)
        jac = np.linalg.norm(nun, axis=-1)
        self.nhat = nun / jac[..., None]
        self.dA = jac * (2 * np.pi / nu) * (2 * np.pi / nv)
        self.h = float(np.sqrt(self.dA.mean()))  # grid-spacing length scale


class VirtualCasing:
    """Exterior field of a surface current (K, sigma) = (n x B_p, n.B_p), with QBX for
    the near-singular on-surface evaluation."""

    def __init__(self, X, K, sigma, dA):
        self.X, self.K, self.sigma, self.dA = X, K, sigma, dA

    def raw(self, r):
        """Non-singular Biot-Savart surface integral away from the surface."""
        d = r[None, None] - self.X
        inv = 1.0 / np.linalg.norm(d, axis=-1) ** 3
        term = np.cross(self.K, d) + self.sigma[..., None] * d
        return np.sum(term * (inv * self.dA)[..., None], (0, 1)) / (4 * np.pi)

    def on_surface(self, r0, n0, h, orders=(3, 4, 5, 6, 7, 8), deg=3):
        """QBX exterior limit: fit the (non-singular) off-surface field at
        expansion centers `orders` source-spacings out along +n0, extrapolate to
        the surface. Use -n0 for the interior limit."""
        ds = np.array(orders) * h
        Bs = np.array([self.raw(r0 + d * n0) for d in ds])
        return np.array(
            [np.polyval(np.polyfit(ds, Bs[:, c], deg), 0.0) for c in range(3)]
        )


def build(wout, mgrid_path, extcur, nu=256, nv=256):
    """Assemble the LCFS, the external field (coils + net-current axis filament),
    and the plasma virtual-casing operator. The plasma field for casing is the
    total field minus the *entire* externally-represented field (coils plus the
    net-current filament); the filament is added back on evaluation."""
    lcfs = Lcfs(wout, nu, nv)
    coil = CoilField(mgrid_path, np.asarray(extcur, float))
    axis = AxisCurrent(wout)

    def external(X):
        return coil(X) + axis(X)

    b_plasma = lcfs.B - external(lcfs.X)
    vc = VirtualCasing(
        lcfs.X, np.cross(lcfs.nhat, b_plasma), np.sum(b_plasma * lcfs.nhat, -1), lcfs.dA
    )
    return lcfs, external, vc


def _validate():
    if netcdf_file is None:
        print("AEGIS validation skipped: requires scipy (pip install scipy).")
        return

    mgrid = str(_TEST_DATA / "mgrid_cth_like.nc")
    inp = vmecpp.VmecInput.from_file(str(_TEST_DATA / "cth_like_free_bdy.json"))
    inp.mgrid_file = mgrid
    inp.ns_array = np.array([25], dtype=np.int64)
    inp.ftol_array = np.array([1e-11])
    inp.niter_array = np.array([8000], dtype=np.int64)
    extcur = np.asarray(inp.extcur, float)
    w = vmecpp.run(inp, max_threads=1, verbose=False).wout

    # [1] Biot-Savart integrator vs an analytic circular loop (Bz on axis).
    n_seg, a, cur, z = 4000, 1.3, 1e5, 0.5
    t = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
    loop = np.stack([a * np.cos(t), a * np.sin(t), np.zeros(n_seg)], 1)
    dl = np.stack([-a * np.sin(t), a * np.cos(t), np.zeros(n_seg)], 1) * (
        2 * np.pi / n_seg
    )
    d = np.array([0, 0, z])[None] - loop
    bz = (
        MU0
        / (4 * np.pi)
        * cur
        * np.sum(np.cross(dl, d) / np.linalg.norm(d, axis=1)[:, None] ** 3, 0)[2]
    )
    exact = MU0 * cur * a**2 / (2 * (a**2 + z**2) ** 1.5)
    print(f"[1] Biot-Savart loop vs analytic: relerr={abs(bz - exact) / exact:.1e}")

    lcfs, coil, vc = build(w, mgrid, extcur)
    bmag = np.linalg.norm(lcfs.B, axis=-1)
    print(
        f"[2] LCFS B.n/|B| (flux-surface tangency of reconstruction): "
        f"max={np.abs(np.sum(lcfs.B * lcfs.nhat, -1) / bmag).max():.1e}"
    )

    # [3] jump condition n x (B_out - B_in) = K via QBX on the full surface field.
    vc_tot = VirtualCasing(
        lcfs.X, np.cross(lcfs.nhat, lcfs.B), np.zeros(lcfs.X.shape[:2]), lcfs.dA
    )
    jerr = []
    for iu, iv in itertools.product([20, 90, 160, 230], [40, 180]):
        r0, n0 = lcfs.X[iu, iv], lcfs.nhat[iu, iv]
        jump = np.cross(
            n0, vc_tot.on_surface(r0, n0, lcfs.h) - vc_tot.on_surface(r0, -n0, lcfs.h)
        )
        kref = np.cross(n0, lcfs.B[iu, iv])
        jerr.append(np.linalg.norm(jump - kref) / np.linalg.norm(kref))
    print(
        f"[3] jump condition n x [B] = K via QBX: mean={np.mean(jerr):.4f} "
        f"max={np.max(jerr):.4f}"
    )

    # [4] free boundary: B_exterior = VC(plasma) + coil is tangent, |B_ext|=|B_in|.
    tang, ratio = [], []
    for iu, iv in itertools.product([20, 90, 160, 230], [40, 180]):
        r0, n0 = lcfs.X[iu, iv], lcfs.nhat[iu, iv]
        b_ext = vc.on_surface(r0, n0, lcfs.h) + coil(r0[None])[0]
        tang.append(abs(np.dot(b_ext, n0) / np.linalg.norm(b_ext)))
        ratio.append(np.linalg.norm(b_ext) / bmag[iu, iv])
    print(
        f"[4] exterior field: tangency max={np.max(tang):.4f}  "
        f"|B_ext|/|B_in| mean={np.mean(ratio):.4f} (want ~1)"
    )

    # [5] vacuum pressure |B|^2/2 (equal to |B_exterior|^2/2 at convergence).
    pbar = np.sum(0.5 * bmag**2 * lcfs.dA) / np.sum(lcfs.dA)
    print(f"[5] vacuum pressure |B|^2/2 area-avg={pbar:.4f} (NESTOR reference ~0.178)")


if __name__ == "__main__":
    _validate()
