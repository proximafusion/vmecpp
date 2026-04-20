# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""TEM stability diagnostic: |B| and bad curvature on a flux surface.

Reconstructs |B|(theta, zeta) and the bad-curvature proxy kappa_n(theta, zeta)
from VMEC++ output, then plots them side by side over one toroidal field period.
The bad-curvature proxy is defined as:

    kappa_n(theta, zeta) = -d(ln|B|)/ds

where s is the normalized toroidal flux and the radial derivative is evaluated via
centered finite differences between adjacent half-grid flux surfaces.  A positive
kappa_n value indicates locally unfavorable (bad) curvature: the field-line radius of
curvature points away from the plasma, so a perturbation displaced outward sees a
weaker restoring force.

Note: this script requires matplotlib as an additional dependency.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import vmecpp


def reconstruct_b_on_surface(
    wout: vmecpp.VmecWOut,
    surface_index: int,
    grid_theta: np.ndarray,
    grid_phi: np.ndarray,
) -> np.ndarray:
    """Reconstruct |B|(theta, phi) on a half-grid flux surface.

    Parameters
    ----------
    wout:
        VMEC++ wout output object.
    surface_index:
        Half-grid surface index in the range [1, wout.ns - 1].
    grid_theta:
        1-D array of poloidal angles in radians.
    grid_phi:
        1-D array of toroidal angles in radians covering one field period.

    Returns
    -------
    ndarray of shape (n_theta, n_phi)
        Magnetic field strength |B| at each grid point on the surface.
    """
    xm_nyq = wout.xm_nyq  # (mn_mode_nyq,)
    xn_nyq = wout.xn_nyq  # (mn_mode_nyq,)
    bmnc_j = wout.bmnc[:, surface_index]  # (mn_mode_nyq,)

    # Fourier argument: shape (mn_mode_nyq, n_theta, n_phi)
    kernel = (
        xm_nyq[:, None, None] * grid_theta[None, :, None]
        - xn_nyq[:, None, None] * grid_phi[None, None, :]
    )
    # Sum over Fourier modes: b has shape (n_theta, n_phi)
    b = np.tensordot(bmnc_j, np.cos(kernel), axes=([0], [0]))
    return b


def compute_bad_curvature(
    wout: vmecpp.VmecWOut,
    surface_index: int,
    grid_theta: np.ndarray,
    grid_phi: np.ndarray,
) -> np.ndarray:
    """Compute the bad-curvature proxy kappa_n = -d(ln|B|)/ds on a flux surface.

    Uses a centered finite difference in the normalized toroidal flux coordinate s
    between the two neighboring half-grid surfaces.  A positive value at a point
    (theta, phi) means the magnetic field decreases radially outward there, which
    corresponds to locally unfavorable (bad) curvature.

    Parameters
    ----------
    wout:
        VMEC++ wout output object.
    surface_index:
        Half-grid surface index for the target surface; must satisfy
        1 <= surface_index <= wout.ns - 2 so that neighbors are available.
    grid_theta:
        1-D array of poloidal angles in radians.
    grid_phi:
        1-D array of toroidal angles in radians covering one field period.

    Returns
    -------
    ndarray of shape (n_theta, n_phi)
        Bad-curvature proxy kappa_n at each grid point.  Positive means
        locally bad (unfavorable) curvature.
    """
    ns = wout.ns
    j_lo = max(1, surface_index - 1)
    j_hi = min(ns - 1, surface_index + 1)
    # Spacing in s between the two neighboring half-grid surfaces
    ds = (j_hi - j_lo) / (ns - 1)

    b_lo = reconstruct_b_on_surface(wout, j_lo, grid_theta, grid_phi)
    b_hi = reconstruct_b_on_surface(wout, j_hi, grid_theta, grid_phi)
    b_mid = reconstruct_b_on_surface(wout, surface_index, grid_theta, grid_phi)

    # kappa_n = -d(ln B)/ds = -(dB/ds) / B
    kappa_n = -(b_hi - b_lo) / (2.0 * ds * b_mid)
    return kappa_n


if __name__ == "__main__":
    input_file = Path(__file__).parent / "data" / "input.w7x"
    vmec_input = vmecpp.VmecInput.from_file(input_file)
    vmec_output = vmecpp.run(vmec_input)
    wout = vmec_output.wout

    ns = wout.ns
    nfp = wout.nfp

    # Mid-radius half-grid surface (skip index 0 which is unused on the half-grid)
    j = ns // 2
    s_value = (j - 0.5) / (ns - 1)

    num_theta = 128
    num_phi = 128

    grid_theta = np.linspace(0.0, 2.0 * np.pi, num_theta, endpoint=False)
    # One toroidal field period
    grid_phi = np.linspace(0.0, 2.0 * np.pi / nfp, num_phi, endpoint=False)

    b = reconstruct_b_on_surface(wout, j, grid_theta, grid_phi)
    kappa_n = compute_bad_curvature(wout, j, grid_theta, grid_phi)

    theta_deg = np.degrees(grid_theta)
    phi_deg = np.degrees(grid_phi)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Panel 1: |B| ---
    im0 = axes[0].pcolormesh(phi_deg, theta_deg, b, cmap="plasma", shading="auto")
    fig.colorbar(im0, ax=axes[0], label="|B| / T")
    b_levels = np.linspace(b.min(), b.max(), 10)
    axes[0].contour(
        phi_deg,
        theta_deg,
        b,
        levels=b_levels,
        colors="white",
        linewidths=0.6,
        alpha=0.6,
    )
    axes[0].set_xlabel("zeta / deg  (one field period)")
    axes[0].set_ylabel("theta / deg")
    axes[0].set_title("|B|  (magnetic field strength)")

    # --- Panel 2: bad curvature, with |B| contours overlaid ---
    vmax = np.max(np.abs(kappa_n))
    im1 = axes[1].pcolormesh(
        phi_deg,
        theta_deg,
        kappa_n,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        shading="auto",
    )
    fig.colorbar(im1, ax=axes[1], label="kappa_n = -d(ln|B|)/ds  (bad > 0)")
    # Overlay |B| contours to reveal alignment with kappa_n
    axes[1].contour(
        phi_deg,
        theta_deg,
        b,
        levels=b_levels,
        colors="black",
        linewidths=0.8,
        alpha=0.7,
    )
    axes[1].set_xlabel("zeta / deg  (one field period)")
    axes[1].set_ylabel("theta / deg")
    axes[1].set_title(
        "Bad curvature kappa_n  (kappa_n > 0 = unfavorable)\nblack contours: |B|"
    )

    fig.suptitle(
        f"TEM stability diagnostic  |  s = {s_value:.2f}  |  "
        "for TEM stability, bad curvature and magnetic wells should coincide\n"
        "(valley of |B| in phase with peak of kappa_n)"
    )

    plt.tight_layout()
    plt.show()
