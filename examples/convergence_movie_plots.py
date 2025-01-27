# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Plot snapshots of VMEC++ taken along the run."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import vmecpp

cache_folder = Path("/home/jons/results/vmec_w7x/movie_cache")
if not Path.exists(cache_folder):
    raise RuntimeError(
        "Output cache folder "
        + cache_folder
        + " does not exist. Run convergence_movie_make_runs.py to generate it."
    )

plots_folder = cache_folder / "plots"
Path.mkdir(plots_folder, parents=True, exist_ok=True)

saved_steps = np.loadtxt(cache_folder / "saved_steps.dat", dtype=int)


def flux_surfaces_rz(
    oq: vmecpp._vmecpp.OutputQuantities, phi: float = 0.0, num_theta: int = 101
) -> np.ndarray:
    """Evaluate the flux-surface geometry at a fixed toroidal angle.

    Returned shape: [ns][2: R, Z][num_theta]
    """
    rz = np.zeros([oq.wout.ns, 2, num_theta])

    theta_grid = np.linspace(0.0, 2.0 * np.pi, num_theta, endpoint=True)
    phi_grid = np.zeros([num_theta]) + phi
    kernel = np.outer(oq.wout.xm, theta_grid) - np.outer(oq.wout.xn, phi_grid)
    for js in range(oq.wout.ns):
        rz[js, 0, :] = oq.wout.rmnc[js, :] @ np.cos(kernel)
        rz[js, 1, :] = oq.wout.zmns[js, :] @ np.sin(kernel)

    return rz


# plt.figure(figsize=(4, 6))
plt.figure()
for saved_step in saved_steps:
    # print(saved_step, "/", saved_steps[-1])

    vmecpp_out_filename = cache_folder / f"vmecpp_w7x_{saved_step:04d}.h5"
    if not Path.exists(vmecpp_out_filename):
        raise RuntimeError(
            "VMEC++ output file "
            + str(vmecpp_out_filename)
            + " does not exist. Run convergence_movie_make_runs.py to generate it."
        )

    oq = vmecpp._vmecpp.OutputQuantities.load(vmecpp_out_filename)

    # phi = 0.0
    # num_theta = 101
    # rz = flux_surfaces_rz(oq=oq, phi=phi, num_theta=num_theta)
    # for js in range(oq.wout.ns):
    #     plt.plot(rz[js, 0, :], rz[js, 1, :], color="k", lw=0.5)
    # plt.axis("equal")
    # plt.grid(True)
    # plt.savefig(plots_folder / f"flux_surfaces_{saved_step:04d}.png")

    ns = oq.wout.ns
    ntheta = oq.indata.ntheta // 2 + 1
    nzeta = oq.indata.nzeta
    jxb_gradp = np.reshape(oq.jxbout.jxb_gradp, [ns, ntheta * nzeta])

    plt.clf()
    plt.semilogy(np.abs(np.average(jxb_gradp, axis=-1)[3:-1]))
    plt.title(f"{saved_step} / {saved_steps[-1]}")
    plt.savefig(plots_folder / f"jxb_gradp_fsa_{saved_step:04d}.png")
