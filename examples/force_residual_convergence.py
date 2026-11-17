# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Visualizing the convergence of force residuals using VMEC output quantities."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import vmecpp

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "input_file",
    nargs="?",
    type=Path,
    default=Path(__file__).parent / "data" / "input.w7x",
    help="VMEC++ input file (defaults to the fixed-boundary W7-X case)",
)
parser.add_argument("--output", type=Path, help="Save the plot to this image file")
args = parser.parse_args()

input_file = args.input_file
vmec_input = vmecpp.VmecInput.from_file(input_file)

# Run the VMEC solver to compute equilibrium
wout = vmecpp.run(vmec_input).wout


def plot_force_residuals(ax):
    ax.plot(wout.force_residual_r, label="Force residual ($R$)")
    ax.plot(wout.force_residual_z, label="Force residual ($Z$)")
    ax.plot(wout.force_residual_lambda, label=r"Force residual ($\lambda$)")
    ax.plot(wout.fsqt, "k", label="Force residual (total)")
    if wout.lfreeb:
        ax.plot(wout.delbsq, label=r"$\Delta B^2$")
        ax.plot(wout.lcfs_force, label="LCFS force residual")
    ax.axhline(y=wout.ftolv, color="red", linestyle="dashed", label="Tolerance")
    for i, reason in wout.restart_reasons:
        ax.axvline(
            i,
            ymin=wout.ftolv,
            ymax=wout.fsqt.max(),
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
inset_traces = [
    wout.force_residual_r[:30],
    wout.force_residual_z[:30],
    wout.force_residual_lambda[:30],
    wout.fsqt[:30],
]
if wout.lfreeb:
    inset_traces.extend([wout.delbsq[:30], wout.lcfs_force[:30]])
positive_values = [trace[trace > 0] for trace in inset_traces]
axins.set_ylim(
    min(values.min() for values in positive_values if values.size),
    max(trace.max() for trace in inset_traces) * 1.1,
)
axins.set_xticklabels([])
ax.indicate_inset_zoom(axins)

if args.output:
    plt.savefig(args.output)

plt.show()
