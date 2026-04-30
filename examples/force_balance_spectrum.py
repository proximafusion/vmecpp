# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Visualize force-residual Fourier coefficients and real space contributions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import vmecpp

input_file = Path() / "examples/data/input.w7x"
vmec_input = vmecpp.VmecInput.from_file(input_file)

output = vmecpp.run(vmec_input, max_threads=4)


def _compute_vmec_sizes() -> tuple[int, int, int, int, int]:
    n_zeta = (
        vmec_input.nzeta if vmec_input.nzeta > 0 else max(1, 2 * vmec_input.ntor + 4)
    )
    n_theta_even = 2 * (vmec_input.ntheta // 2)
    n_theta_reduced = n_theta_even // 2 + 1
    max_m = n_theta_even // 2
    max_n = n_zeta // 2
    return n_zeta, n_theta_even, n_theta_reduced, max_m, max_n


def _mode_scales(max_m: int, max_n: int) -> tuple[np.ndarray, np.ndarray]:
    mscale = np.ones(max_m + 1)
    if max_m > 0:
        mscale[1:] = np.sqrt(2.0)

    nscale = np.ones(max_n + 1)
    if max_n > 0:
        nscale[1:] = np.sqrt(2.0)

    return mscale, nscale


def _build_vmec_basis(
    max_m: int, max_n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_zeta, n_theta_even, n_theta_reduced, _, _ = _compute_vmec_sizes()
    mscale, nscale = _mode_scales(max_m, max_n)

    theta = 2.0 * np.pi * np.arange(n_theta_reduced) / n_theta_even
    zeta = 2.0 * np.pi * np.arange(n_zeta) / n_zeta

    cosmu = np.cos(np.outer(theta, np.arange(max_m + 1))) * mscale
    sinmu = np.sin(np.outer(theta, np.arange(max_m + 1))) * mscale
    cosnv = np.cos(np.outer(np.arange(max_n + 1), zeta)) * nscale[:, None]
    sinnv = np.sin(np.outer(np.arange(max_n + 1), zeta)) * nscale[:, None]

    int_norm = 1.0 / (n_zeta * (n_theta_reduced - 1))
    endpoint_weights = np.ones(n_theta_reduced)
    endpoint_weights[0] = 0.5
    endpoint_weights[-1] = 0.5

    cosmui = cosmu * (int_norm * endpoint_weights[:, None])
    sinmui = sinmu * (int_norm * endpoint_weights[:, None])
    return cosmui, sinmui, cosnv, sinnv


def _combined_mode_numbers(max_m: int, max_n: int) -> tuple[np.ndarray, np.ndarray]:
    mode_m = []
    mode_n = []

    for n in range(max_n + 1):
        mode_m.append(0)
        mode_n.append(n)

    for m in range(1, max_m + 1):
        for n in range(-max_n, max_n + 1):
            mode_m.append(m)
            mode_n.append(n)

    return np.array(mode_m, dtype=int), np.array(mode_n, dtype=int)


def _cc_ss_to_cos(
    fc_cc: np.ndarray, fc_ss: np.ndarray, max_m: int, max_n: int
) -> np.ndarray:
    mode_m, mode_n = _combined_mode_numbers(max_m, max_n)
    mscale, nscale = _mode_scales(max_m, max_n)

    fc_cos = np.zeros(mode_m.shape[0])
    for idx, (m, n) in enumerate(zip(mode_m, mode_n, strict=False)):
        abs_n = abs(n)
        basis_norm = 1.0 / (mscale[m] * nscale[abs_n])
        if m == 0 or abs_n == 0:
            fc_cos[idx] = fc_cc[abs_n, m] / basis_norm
        else:
            fc_cos[idx] = (
                0.5 * (fc_cc[abs_n, m] + np.sign(n) * fc_ss[abs_n, m]) / basis_norm
            )
    return fc_cos


def _sc_cs_to_sin(
    fc_sc: np.ndarray, fc_cs: np.ndarray, max_m: int, max_n: int
) -> np.ndarray:
    mode_m, mode_n = _combined_mode_numbers(max_m, max_n)
    mscale, nscale = _mode_scales(max_m, max_n)

    fc_sin = np.zeros(mode_m.shape[0])
    for idx, (m, n) in enumerate(zip(mode_m, mode_n, strict=False)):
        abs_n = abs(n)
        basis_norm = 1.0 / (mscale[m] * nscale[abs_n])
        if idx == 0:
            fc_sin[idx] = 0.0
        elif m == 0:
            fc_sin[idx] = -fc_cs[abs_n, m] / basis_norm
        elif abs_n == 0:
            fc_sin[idx] = fc_sc[abs_n, m] / basis_norm
        else:
            fc_sin[idx] = (
                0.5 * (fc_sc[abs_n, m] - np.sign(n) * fc_cs[abs_n, m]) / basis_norm
            )
    return fc_sin


def _vmec_forward_dft(
    reduced_zeta_theta_grid: np.ndarray, max_m: int, max_n: int
) -> tuple[np.ndarray, np.ndarray]:
    cosmui, sinmui, cosnv, sinnv = _build_vmec_basis(max_m, max_n)

    realspace_evn = reduced_zeta_theta_grid.T
    realspace_odd = np.zeros_like(realspace_evn)

    fc_cc = np.zeros((max_n + 1, max_m + 1))
    fc_ss = np.zeros((max_n + 1, max_m + 1))
    fc_sc = np.zeros((max_n + 1, max_m + 1))
    fc_cs = np.zeros((max_n + 1, max_m + 1))

    for n in range(max_n + 1):
        rnkcc = realspace_evn @ cosnv[n]
        rnkss = realspace_evn @ sinnv[n]
        rnksc = realspace_odd @ cosnv[n]
        rnkcs = realspace_odd @ sinnv[n]

        for m in range(max_m + 1):
            fc_cc[n, m] = np.sum(rnkcc * cosmui[:, m])
            fc_ss[n, m] = np.sum(rnkss * sinmui[:, m])
            fc_sc[n, m] = np.sum(rnksc * sinmui[:, m])
            fc_cs[n, m] = np.sum(rnkcs * cosmui[:, m])

    return _cc_ss_to_cos(fc_cc, fc_ss, max_m, max_n), _sc_cs_to_sin(
        fc_sc, fc_cs, max_m, max_n
    )


def _coefficients_to_grid(
    coefficients: np.ndarray, max_m: int, max_n: int
) -> np.ndarray:
    mode_m, mode_n = _combined_mode_numbers(max_m, max_n)
    coefficient_grid = np.full((max_m + 1, 2 * max_n + 1), np.nan)

    for coefficient, m, n in zip(coefficients, mode_m, mode_n, strict=False):
        coefficient_grid[m, n + max_n] = coefficient

    return coefficient_grid


def _plot_mode_grid(
    ax,
    coefficient_grid: np.ndarray,
    radial_surface_index: int,
    component_label: str,
    color_limit: float,
):
    max_m = coefficient_grid.shape[0] - 1
    max_n = (coefficient_grid.shape[1] - 1) // 2
    image = ax.imshow(
        coefficient_grid,
        origin="lower",
        aspect="auto",
        extent=[-max_n - 0.5, max_n + 0.5, -0.5, max_m + 0.5],
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-color_limit,
        vmax=color_limit,
    )

    rectangle = Rectangle(
        (-vmec_input.ntor - 0.5, -0.5),
        2 * vmec_input.mpol + 1,
        vmec_input.mpol,
        fill=False,
        edgecolor="black",
        linewidth=1.5,
        linestyle="--",
    )
    ax.add_patch(rectangle)
    ax.set_xlabel("Toroidal mode number n")
    ax.set_ylabel("Poloidal mode number m")
    ax.set_title(f"{component_label}, surface {radial_surface_index}")
    return image


n_zeta, _, n_theta_reduced, transform_max_m, transform_max_n = _compute_vmec_sizes()
jxb_gradp = output.jxbout.jxb_gradp.reshape(
    (output.jxbout.jxb_gradp.shape[0], n_zeta, n_theta_reduced)
)

surface_indices = np.linspace(10, jxb_gradp.shape[0] - 2, num=12, dtype=int)  # type: ignore
cosine_grids = []
sine_grids = []

for radial_surface_index in surface_indices:
    fc_cos, fc_sin = _vmec_forward_dft(
        jxb_gradp[radial_surface_index], transform_max_m, transform_max_n
    )
    cosine_grids.append(_coefficients_to_grid(fc_cos, transform_max_m, transform_max_n))
    sine_grids.append(_coefficients_to_grid(fc_sin, transform_max_m, transform_max_n))

cosine_color_limit = max(np.nanmax(np.abs(grid)) for grid in cosine_grids)
sine_color_limit = max(np.nanmax(np.abs(grid)) for grid in sine_grids)

cosine_fig, cosine_axes = plt.subplots(3, 4, figsize=(16, 10), constrained_layout=True)
cosine_image = None
for ax, radial_surface_index, grid in zip(
    cosine_axes.flat, surface_indices, cosine_grids, strict=False
):
    cosine_image = _plot_mode_grid(
        ax,
        grid,
        radial_surface_index=radial_surface_index,
        component_label="cos(m theta - n zeta)",
        color_limit=cosine_color_limit,
    )
if cosine_image is not None:
    cosine_fig.colorbar(
        cosine_image,
        ax=cosine_axes,
        label=r"$F^{\cos}_{m,n}$",
        shrink=0.9,
    )

realspace_fig, realspace_axes = plt.subplots(
    3, 4, figsize=(16, 10), constrained_layout=True
)
im = None
for ax, radial_surface_index in zip(realspace_axes.flat, surface_indices, strict=False):
    im = ax.imshow(
        jxb_gradp[radial_surface_index].T,
        origin="lower",
        aspect="auto",
        extent=[0, 2 * np.pi, 0, 2 * np.pi],
        interpolation="nearest",
        cmap="RdBu_r",
    )
    ax.set_xlabel(r"$\zeta$ (toroidal angle)")
    ax.set_ylabel(r"$\theta$ (poloidal angle)")
    ax.set_title(f"Real space, surface {radial_surface_index}")
if im is not None:
    realspace_fig.colorbar(
        im, ax=realspace_axes, label=r"$F(\theta, \zeta)$", shrink=0.9
    )


plt.show()
