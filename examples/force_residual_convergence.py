# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Visualizing the convergence of force residuals using VMEC output quantities."""

from pathlib import Path

import matplotlib.pyplot as plt

import vmecpp

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


ax = plt.gca()
plot_force_residuals(ax)
ax.set_xlabel("Iteration")
ax.set_ylabel("Force residual")
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

plt.show()
