# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Generate the figures for the AEGIS free-boundary verification report from the
tabulated sweep results (produced by expA_shafranov.py, expC_resolution.py,
iota_conv.py, and vc_compare.py in this directory).

Writes PNGs into ./plots.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "plots"
OUT.mkdir(exist_ok=True)

# Section 1: converged delbsq vs mpol (cth_like free boundary).
mpol = [5, 7, 9, 12, 16]
nestor_v = [1.26e-3, 6.0e-4, 1.02e-3, 1.44e-3, 1.46e-3]
aegis_v = [7.0e-4, 5.4e-4, 3.4e-4, 3.0e-4, 3.1e-4]
mpol_b = [5, 7, 9, 12]
nestor_b = [1.88e-3, 1.24e-3, 1.91e-3, 2.29e-3]
aegis_b = [9.2e-4, 5.7e-4, 4.9e-4, None]

fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.semilogy(mpol, nestor_v, "o-", color="C3", label="NESTOR, vacuum-level beta")
ax.semilogy(mpol, aegis_v, "o-", color="C0", label="AEGIS, vacuum-level beta")
ax.semilogy(mpol_b, nestor_b, "s--", color="C3", alpha=0.6, label="NESTOR, beta ~ 1%")
xb = [m for m, a in zip(mpol_b, aegis_b, strict=False) if a is not None]
yb = [a for a in aegis_b if a is not None]
ax.semilogy(xb, yb, "s--", color="C0", alpha=0.6, label="AEGIS, beta ~ 1%")
ax.set_xlabel("mpol")
ax.set_ylabel("converged delbsq")
ax.set_title("cth_like free boundary: boundary residual vs resolution")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "delbsq_vs_mpol.png", dpi=130)
plt.close(fig)

# Section 2: Shafranov sweep (cth_like, mpol 8).
beta = [0.19, 0.48, 0.99]
raxis_n = [0.77373, 0.77825, 0.78615]
raxis_a = [0.76685, 0.77314, 0.78370]
delbsq_n = [6.2e-4, 9.2e-4, 1.5e-3]
delbsq_a = [3.2e-4, 3.6e-4, 5.9e-4]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
a1.plot(beta, raxis_n, "o-", color="C3", label="NESTOR")
a1.plot(beta, raxis_a, "o-", color="C0", label="AEGIS")
a1.set_xlabel("total beta (%)")
a1.set_ylabel("magnetic-axis major radius")
a1.set_title("Shafranov shift")
a1.legend(fontsize=8)
a1.grid(True, alpha=0.3)
a2.semilogy(beta, delbsq_n, "o-", color="C3", label="NESTOR")
a2.semilogy(beta, delbsq_a, "o-", color="C0", label="AEGIS")
a2.set_xlabel("total beta (%)")
a2.set_ylabel("converged delbsq")
a2.set_title("boundary residual vs beta")
a2.legend(fontsize=8)
a2.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "shafranov_sweep.png", dpi=130)
plt.close(fig)

# Section 5: virtual-casing quadrature error vs resolution (analytic dipole).
npol = [48, 72, 96]
cos = [0.998, 0.999, 0.999]
ratio = [0.907, 0.939, 0.955]
rms = [1.14e-1, 8.11e-2, 6.41e-2]

fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(npol, cos, "o-", color="C0", label="direction cosine (want 1)")
ax.plot(npol, ratio, "s-", color="C2", label="magnitude ratio (want 1)")
ax.plot(npol, rms, "^-", color="C1", label="rms relative error")
ax.axhline(1.0, color="k", lw=0.7, alpha=0.5)
ax.set_xlabel("poloidal grid points")
ax.set_ylabel("AEGIS vs analytic exterior field")
ax.set_title("on-surface virtual-casing accuracy (interior dipole, cth_like)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "vc_isolation_accuracy.png", dpi=130)
plt.close(fig)

# Section 6: high-beta convergence on Solovev (beta-scaled edge damping).
hb_beta = [0.51, 2.0, 4.3, 5.9]  # percent
hb_delbsq = [2.9e-4, 1.2e-3, 3.1e-3, 4.6e-3]
hb_iters = [2288, 2246, 3675, 4905]
unscaled_ceiling = 1.3  # percent; unscaled AEGIS diverges above this

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
a1.semilogy(hb_beta, hb_delbsq, "o-", color="C0", label="AEGIS (beta-scaled A)")
a1.axvline(unscaled_ceiling, color="C3", ls="--", lw=1, label="unscaled AEGIS ceiling")
a1.set_xlabel("total beta (%)")
a1.set_ylabel("converged delbsq")
a1.set_title("Solovev high-beta: convergence")
a1.legend(fontsize=8)
a1.grid(True, which="both", alpha=0.3)
a2.plot(hb_beta, hb_iters, "o-", color="C0")
a2.axvline(unscaled_ceiling, color="C3", ls="--", lw=1)
a2.set_xlabel("total beta (%)")
a2.set_ylabel("iterations to ftol 1e-11")
a2.set_title("iterations vs beta")
a2.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "highbeta_convergence.png", dpi=130)
plt.close(fig)

print("wrote:", *[p.name for p in sorted(OUT.glob("*.png"))])
