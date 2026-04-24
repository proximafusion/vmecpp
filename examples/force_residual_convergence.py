# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Visualizing the convergence of force residuals using VMEC output quantities."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import vmecpp

# Enable the spectral-truncation diagnostic so the wout fields
# `force_discarded_{r,z,lambda}` are populated. Without this env var they stay
# NaN (the vectors still exist but hold no signal), matching the zero-cost
# default for production runs.
os.environ["VMECPP_SPECTRAL_DIAGNOSTIC"] = "1"

input_file = Path(__file__).parent / "data" / "input.w7x"
input = vmecpp.VmecInput.from_file(input_file)

# Run the VMEC solver to compute equilibrium
output = vmecpp.run(input)


def plot_force_residuals(ax):
    ax.plot(output.wout.force_residual_r, label="Force residual ($R$)")
    ax.plot(output.wout.force_residual_z, label="Force residual ($Z$)")
    ax.plot(output.wout.force_residual_lambda, label=r"Force residual ($\lambda$)")
    ax.plot(output.wout.fsqt, "k", label="Force residual (total)")
    if output.wout.lfreeb:
        ax.plot(output.wout.delbsq, label=r"$\Delta B^2$")
    ax.axhline(y=output.wout.ftolv, color="red", linestyle="dashed", label="Tolerance")
    for i, reason in output.wout.restart_reasons:
        ax.axvline(
            i,
            ymin=output.wout.ftolv,
            ymax=output.wout.fsqt.max(),
            color="gray",
            label="Restart: " + reason.name,
        )
    ax.set_yscale("log")


fig, (ax, ax_discard) = plt.subplots(
    2, 1, figsize=(9, 9), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
)

plot_force_residuals(ax)
ax.set_ylabel("Force residual")
# Let small late-iteration residuals be visible instead of clipping at
# matplotlib's default ~1e-6 floor.
ax.set_ylim(bottom=1e-16)
ax.legend()
ax.set_title("Force Residual Convergence")

# Create an inset plot in the bottom left corner
axins = ax.inset_axes((0.25, 0.5, 0.25, 0.4))
plot_force_residuals(axins)
axins.set_xlim(0, 30)
axins.set_ylim(
    min(
        output.wout.force_residual_r[:30].min(),
        output.wout.force_residual_z[:30].min(),
        output.wout.force_residual_lambda[:30].min(),
    ),
    output.wout.fsqt[:30].max() * 1.1,
)
axins.set_xticklabels([])
ax.indicate_inset_zoom(axins)

# Spectral-truncation diagnostic, tracked per-iteration the same way
# force_residual_{r,z,lambda} are tracked. Each value is the fraction of
# real-space force L2 energy that lies outside the retained (mpol, ntor)
# Fourier band, maxed across radial surfaces. A tiny value means the Fourier
# basis is resolving the force adequately; tens of percent means VMEC++ is
# minimizing against signal its own basis cannot represent, and convergence
# metrics should be read with skepticism.
discard_r = np.asarray(output.wout.force_discarded_r)
discard_z = np.asarray(output.wout.force_discarded_z)
discard_lambda = np.asarray(output.wout.force_discarded_lambda)

if discard_r.size and np.isfinite(discard_r).any():
    ax_discard.plot(discard_r, label="R force ($armn$) discarded")
    ax_discard.plot(discard_z, label="Z force ($azmn$) discarded")
    ax_discard.plot(discard_lambda, label=r"$\lambda$ force ($blmn$) discarded")
    ax_discard.set_yscale("log")
    ax_discard.set_ylim(bottom=1e-16, top=1.1)
    ax_discard.set_ylabel(r"$E_\mathrm{discarded} / E_\mathrm{total}$")
    ax_discard.legend()
    ax_discard.set_title(
        "Fraction of force L2 energy outside retained (mpol, ntor) Fourier band"
    )
else:
    ax_discard.text(
        0.5,
        0.5,
        "spectral-truncation diagnostic disabled\n(set VMECPP_SPECTRAL_DIAGNOSTIC=1)",
        ha="center",
        va="center",
        transform=ax_discard.transAxes,
    )

ax_discard.set_xlabel("Iteration")
plt.tight_layout()
plt.show()
