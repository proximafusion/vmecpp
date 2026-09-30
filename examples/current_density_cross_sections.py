# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
r"""Plot the current density of a W7-X equilibrium in four toroidal cross sections.

The current density follows from Ampere's law in flux coordinates,

    \sqrt{g} J^\theta = (\partial_\zeta B_s - \partial_s B_\zeta) / \mu_0,
    \sqrt{g} J^\zeta  = (\partial_s B_\theta - \partial_\theta B_s) / \mu_0.

With ``lbsubs = True`` VMEC++ re-derives the full-grid :math:`B_s` from the radial
force balance instead of interpolating the half-grid :math:`B_s` onto the full grid,
which removes the grid-scale noise from the angular derivatives of :math:`B_s`.
The resulting :math:`J^\theta, J^\zeta` are available in real space on the full radial
grid in ``vmec_output.jxbout``.

Care is needed at the magnetic axis:

* ``jxbout`` only holds the interior full-grid surfaces ``1 .. ns-2``; the axis and
  boundary rows are zero.
* :math:`J^\theta` diverges like :math:`1/\sqrt{s}` towards the axis, while
  :math:`\partial R / \partial \theta` and :math:`\partial Z / \partial \theta`
  vanish like :math:`\sqrt{s}`, so the contravariant components cannot be
  extrapolated to the axis.
* The cylindrical components :math:`J_R, J_\phi, J_Z` are regular. Near the axis a
  regular function is :math:`f_0 + \sqrt{s} f_1(\theta) + s f_2(\theta)` with
  :math:`f_1` a pure :math:`m = 1` term, so its poloidal average
  :math:`\langle f \rangle(s) = f_0 + s \langle f_2 \rangle` is linear in :math:`s`
  and the single-valued axis value is :math:`2 \langle f \rangle_1 - \langle f \rangle_2`.

The boundary row is extrapolated linearly in :math:`s` at each grid point.

Note, that this script requires matplotlib as an additional dependency.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import vmecpp


def current_density_in_plane(vmec_output: vmecpp.VmecOutput, k: int) -> dict:
    """Evaluate geometry, B and J in the toroidal plane ``phi = 2 pi k / (nfp nzeta)``.

    Returns arrays of shape ``(ns, n_theta + 1)`` with the poloidal angle closed,
    holding the cylindrical coordinates ``R, Z``, the cylindrical components of the
    current density ``j_r, j_phi, j_z`` in A/m^2 and of the magnetic field
    ``b_r, b_phi, b_z`` in T.
    """
    wout = vmec_output.wout
    jxbout = vmec_output.jxbout
    if wout.lasym:
        msg = "This example assumes stellarator symmetry (lasym = False)."
        raise ValueError(msg)

    ns = wout.ns
    nfp = wout.nfp
    nzeta = vmec_output.input.nzeta
    n_theta_reduced = jxbout.jsupu3.shape[1] // nzeta
    n_theta = 2 * (n_theta_reduced - 1)

    def on_full_theta_range(field: np.ndarray) -> np.ndarray:
        # Real-space arrays cover theta in [0, pi]; a stellarator-symmetric scalar
        # satisfies f(2 pi - theta, -zeta) = f(theta, zeta).
        field = field.reshape(ns, nzeta, n_theta_reduced)
        upper = field[:, k, :]
        lower = field[:, (-k) % nzeta, n_theta_reduced - 2 : 0 : -1]
        return np.concatenate([upper, lower], axis=1)

    # Contravariant components on the interior full-grid surfaces 1 .. ns-2.
    interior = slice(1, ns - 1)
    sqrtg = on_full_theta_range(jxbout.sqrtg3)[interior]
    j_sup_u = on_full_theta_range(jxbout.jsupu3)[interior] / sqrtg
    j_sup_v = on_full_theta_range(jxbout.jsupv3)[interior] / sqrtg
    b_sup_u = on_full_theta_range(jxbout.bsupu3)[interior]
    b_sup_v = on_full_theta_range(jxbout.bsupv3)[interior]

    # Geometry and its tangential derivatives from the Fourier coefficients.
    theta = 2.0 * np.pi * np.arange(n_theta) / n_theta
    phi = 2.0 * np.pi * k / (nfp * nzeta)
    kernel = np.outer(wout.xm, theta) - wout.xn[:, np.newaxis] * phi
    cosk = np.cos(kernel)
    sink = np.sin(kernel)
    xm = wout.xm[:, np.newaxis]
    xn = wout.xn[:, np.newaxis]
    r = wout.rmnc.T @ cosk
    z = wout.zmns.T @ sink
    r_u = (wout.rmnc.T @ (-xm * sink))[interior]
    r_v = (wout.rmnc.T @ (xn * sink))[interior]
    z_u = (wout.zmns.T @ (xm * cosk))[interior]
    z_v = (wout.zmns.T @ (-xn * cosk))[interior]
    r_interior = r[interior]

    def to_cylindrical(sup_u, sup_v):
        return (
            sup_u * r_u + sup_v * r_v,
            r_interior * sup_v,
            sup_u * z_u + sup_v * z_v,
        )

    def with_axis_and_boundary(interior_values: np.ndarray) -> np.ndarray:
        full = np.empty((ns, n_theta))
        full[1 : ns - 1] = interior_values
        # Rows 1 and 2 are the surfaces s = 1/(ns-1) and s = 2/(ns-1).
        full[0] = 2.0 * full[1].mean() - full[2].mean()
        full[ns - 1] = 2.0 * full[ns - 2] - full[ns - 3]
        return full

    def closed(field: np.ndarray) -> np.ndarray:
        return np.concatenate([field, field[:, :1]], axis=1)

    result = {"R": closed(r), "Z": closed(z)}
    for prefix, sup_u, sup_v in (("j", j_sup_u, j_sup_v), ("b", b_sup_u, b_sup_v)):
        for name, values in zip(
            ("r", "phi", "z"), to_cylindrical(sup_u, sup_v), strict=True
        ):
            result[f"{prefix}_{name}"] = closed(with_axis_and_boundary(values))
    result["phi"] = phi
    return result


def toroidal_current_through_plane(plane: dict) -> float:
    """Integrate j_phi over the cross section with the midpoint rule on each cell."""
    r = plane["R"]
    z = plane["Z"]
    j_phi = plane["j_phi"]
    # Cell area from the cross product of the cell diagonals.
    d1_r = r[1:, 1:] - r[:-1, :-1]
    d1_z = z[1:, 1:] - z[:-1, :-1]
    d2_r = r[1:, :-1] - r[:-1, 1:]
    d2_z = z[1:, :-1] - z[:-1, 1:]
    area = 0.5 * np.abs(d1_r * d2_z - d1_z * d2_r)
    j_cell = 0.25 * (j_phi[1:, 1:] + j_phi[:-1, :-1] + j_phi[1:, :-1] + j_phi[:-1, 1:])
    return float(np.sum(area * j_cell))


def plot_current_density_cross_sections(
    lbsubs: bool = True, output_file: Path | None = None
) -> None:
    input_file = Path(__file__).parent / "data" / "input.w7x"
    vmec_input = vmecpp.VmecInput.from_file(input_file)
    # Full-grid B_s from the radial force balance.
    vmec_input.lbsubs = lbsubs
    # Real-space poloidal grid of 64 points instead of the default 2 * mpol + 6 = 30.
    vmec_input.ntheta = 64
    vmec_output = vmecpp.run(vmec_input, verbose=False)

    nfp = vmec_output.wout.nfp
    nzeta = vmec_input.nzeta
    # Toroidal planes at 0, 1/6, 1/3 and 1/2 of a field period.
    plane_indices = [0, nzeta // 6, nzeta // 3, nzeta // 2]
    planes = [current_density_in_plane(vmec_output, k) for k in plane_indices]

    # Parallel current density j . B / |B| in kA/m^2.
    j_parallel = []
    for plane in planes:
        j_dot_b = (
            plane["j_r"] * plane["b_r"]
            + plane["j_phi"] * plane["b_phi"]
            + plane["j_z"] * plane["b_z"]
        )
        mod_b = np.sqrt(plane["b_r"] ** 2 + plane["b_phi"] ** 2 + plane["b_z"] ** 2)
        j_parallel.append(j_dot_b / mod_b / 1.0e3)
    # Isolated surfaces with large |j_par| saturate the color scale.
    j_max = np.percentile(np.abs(np.concatenate(j_parallel, axis=None)), 99.0)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    ns = vmec_output.wout.ns
    meshes = []
    for ax, plane, j_par in zip(axes.flat, planes, j_parallel, strict=True):
        mesh = ax.pcolormesh(
            plane["R"],
            plane["Z"],
            j_par,
            shading="gouraud",
            cmap="RdBu_r",
            vmin=-j_max,
            vmax=j_max,
        )
        meshes.append(mesh)
        for j in np.linspace(0, ns - 1, 6).astype(int)[1:]:
            ax.plot(plane["R"][j], plane["Z"][j], color="k", lw=0.5)
        ax.plot(plane["R"][0, 0], plane["Z"][0, 0], "k+")
        field_period_fraction = plane["phi"] * nfp / (2.0 * np.pi)
        ax.set_title(
            rf"$\phi = {np.degrees(plane['phi']):.0f}^\circ$"
            f" ({field_period_fraction:.3g} field period)"
        )
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_aspect("equal")

        current = toroidal_current_through_plane(plane)
        print(
            f"phi = {np.degrees(plane['phi']):4.0f} deg: "
            f"toroidal current through the plane = {current:.3e} A, "
            f"max |j_par| = {np.abs(j_par).max():.1f} kA/m^2"
        )

    fig.colorbar(
        meshes[0],
        ax=axes,
        label=r"$j_\parallel = \mathbf{j} \cdot \mathbf{B} / |B|$ [kA/m$^2$]",
        shrink=0.8,
        extend="both",
    )
    fig.suptitle(f"W7-X parallel current density (lbsubs = {lbsubs})")

    if output_file is not None:
        fig.savefig(output_file, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    import sys

    plot_current_density_cross_sections(
        output_file=Path(sys.argv[1]) if len(sys.argv) > 1 else None
    )
