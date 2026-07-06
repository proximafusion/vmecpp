# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Trace and plot the VMEC++ iteration dynamics from Python.

The force-balance iteration is owned in Python (``vmecpp._iteration``), so we can
observe *every* convergence and flow-control quantity at every step -- not just
the three force residuals stored in the wout. ``vmecpp.solve_equilibrium`` (and
``vmecpp.iterate``) accept a ``callback`` invoked once per force iteration with an
``IterationState`` snapshot; here we collect those snapshots and plot them as a
grid of subplots.

This is a debugging aid for understanding *why* the convergence dynamics change
qualitatively partway through a run. The three time-step-control styles
(``vmec_8_52`` / ``parvmec`` / ``robust``) share one forward model and differ only
in how they react to a growing residual, so running all three on the same hard
single-resolution case (``cma`` at ns=72) and overlaying the traces makes the
divergence concrete: VMEC 8.52's tight 100x residual leash reverts ~20 times once
the residual plateaus (each revert cuts the time step and bumps ``ijacob`` toward
the give-up escalation), while PARVMEC's permissive 1e4 leash rides straight
through and the ``robust`` scheme reverts only a couple of times.
"""

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import vmecpp
from vmecpp.cpp import _vmecpp  # type: ignore[import]

# cma at ns=72 with a tight tolerance: all three styles converge, but with very
# different flow-control dynamics -- the point of this example.
INPUT_FILE = (
    Path(__file__).parent.parent
    / "src"
    / "vmecpp"
    / "cpp"
    / "vmecpp"
    / "test_data"
    / "cma.json"
)
NS = 72
FTOL = 1.0e-11
NITER = 20000
STYLES = ("vmec_8_52", "parvmec", "robust")

vmec_input = vmecpp.VmecInput.from_file(INPUT_FILE)
cpp_indata = vmec_input._to_cpp_vmecindata()
cpp_indata.ns_array = np.array([NS], dtype=np.int64)
cpp_indata.ftol_array = np.array([FTOL])
cpp_indata.niter_array = np.array([NITER], dtype=np.int64)

# Run each style on its own freshly-created (identical) model, collecting a full
# IterationState snapshot every iteration via the callback.
histories: dict[str, list[vmecpp.IterationState]] = {}
ftolv = FTOL
for style in STYLES:
    history: list[vmecpp.IterationState] = []
    model = _vmecpp.VmecModel.create(cpp_indata, NS, None)
    result = vmecpp.solve_equilibrium(model, style=style, callback=history.append)
    histories[style] = history
    ftolv = model.ftolv
    print(
        f"{style:10s} converged={result.converged} "
        f"iters={result.num_iterations:5d} restarts={result.restarts:3d} "
        f"fsqr={result.fsqr:.2e}"
    )


def column(style: str, name: str) -> np.ndarray:
    return np.array([getattr(s, name) for s in histories[style]])


# Skip the initial-condition transient when autoscaling the log panels, otherwise
# the early spike pins the axis and flattens the dynamics we want to see.
def warmup(style: str) -> int:
    return min(50, len(histories[style]) // 100)


COLORS = {"vmec_8_52": "tab:blue", "parvmec": "tab:orange", "robust": "tab:green"}


def _autoscale_log(ax, per_style_series: dict[str, np.ndarray]) -> None:
    vals = np.concatenate(
        [np.asarray(s)[warmup(style) :] for style, s in per_style_series.items()]
    )
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size:
        ax.set_ylim(vals.min() * 0.5, vals.max() * 2.0)


def _overlay(ax, name: str, *, log: bool = False, autoscale: bool = False):
    """Plot `name` for every style on the same axis, colored per style."""
    series = {}
    for style in STYLES:
        y = column(style, name)
        series[style] = y
        ax.plot(column(style, "iteration"), y, color=COLORS[style], lw=0.9, label=style)
    if log:
        ax.set_yscale("log")
    if autoscale:
        _autoscale_log(ax, series)
    return series


def plot_invariant_sum(ax):
    _overlay(ax, "fsq_invariant", log=True, autoscale=True)
    ax.axhline(ftolv, color="red", ls="--", lw=0.8, label="ftolv")
    ax.legend(fontsize=7)


def plot_preconditioned_sum(ax):
    _overlay(ax, "fsq_preconditioned", log=True, autoscale=True)
    ax.legend(fontsize=7)


def plot_leash(ax):
    # The preconditioned residual against its running minimum res0: the gap is
    # what each style's leash watches. VMEC 8.52 reverts at 100x, hence its
    # repeated spikes-and-cuts; PARVMEC tolerates 1e4x.
    for style in STYLES:
        it = column(style, "iteration")
        ax.plot(it, column(style, "fsq_preconditioned"), color=COLORS[style], lw=0.7)
        ax.plot(it, column(style, "res0"), color=COLORS[style], lw=0.7, ls=":")
    ax.set_yscale("log")
    _autoscale_log(ax, {s: column(s, "fsq_preconditioned") for s in STYLES})
    ax.plot([], [], "k-", lw=0.7, label="fsq (precon)")
    ax.plot([], [], "k:", lw=0.7, label="res0 (running min)")
    ax.legend(fontsize=7)


def plot_timestep(ax):
    _overlay(ax, "delt", log=True)
    ax.legend(fontsize=7)


def plot_damping(ax):
    _overlay(ax, "otav")
    ax.legend(fontsize=7)


def plot_mhd_energy(ax):
    series = _overlay(ax, "mhd_energy")
    # The energy span is tiny next to the initial value; zoom to the converged
    # region so the per-style approach is visible.
    vals = np.concatenate([s[warmup(st) :] for st, s in series.items()])
    if vals.size:
        pad = 0.02 * (vals.max() - vals.min() + 1e-30)
        ax.set_ylim(vals.min() - pad, vals.max() + pad)
    ax.legend(fontsize=7)


def plot_ijacob(ax):
    # ijacob escalates on every VMEC 8.52 revert (toward the give-up at 75); the
    # permissive styles barely move it.
    _overlay(ax, "ijacob")
    ax.legend(fontsize=7)


def plot_segment_age(ax):
    # iter2 - iter1: steps since the last revert. Saw-tooth resets mark reverts;
    # a long monotone ramp means the style is riding through without reverting.
    for style in STYLES:
        ax.plot(
            column(style, "iteration"),
            column(style, "iter2") - column(style, "iter1"),
            color=COLORS[style],
            lw=0.9,
            label=style,
        )
    ax.legend(fontsize=7)


def plot_restart_counts(ax):
    _overlay(ax, "n_restarts")
    ax.legend(fontsize=7)


def plot_events(ax):
    # When each style reverts (one row per style).
    offsets, rows, colors = [], [], []
    for i, style in enumerate(STYLES):
        it = column(style, "iteration")
        rows.append(it[column(style, "restarted")])
        offsets.append(i)
        colors.append(COLORS[style])
    ax.eventplot(rows, lineoffsets=offsets, colors=colors, linelengths=0.8)
    ax.set_yticks(range(len(STYLES)))
    ax.set_yticklabels(STYLES, fontsize=7)
    ax.set_ylim(-0.6, len(STYLES) - 0.4)


plots: list[tuple[str, str, Callable]] = [
    ("Invariant residual sum (convergence test)", "fsq", plot_invariant_sum),
    ("Preconditioned residual sum (drives control)", "fsq1", plot_preconditioned_sum),
    ("Residual vs running minimum (the leash)", "residual", plot_leash),
    ("Time step", "delt", plot_timestep),
    ("Damping (mean 1/tau)", "otav", plot_damping),
    ("MHD energy", "W", plot_mhd_energy),
    ("Bad-Jacobian / give-up escalation", "ijacob", plot_ijacob),
    ("Time-step segment age (iter2 - iter1)", "steps", plot_segment_age),
    ("Cumulative reverts", "count", plot_restart_counts),
    ("Revert events", "", plot_events),
]

ncols = 2
nrows = (len(plots) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.6 * nrows), sharex=True)
axes = axes.ravel()

for ax, (title, ylabel, plotter) in zip(axes, plots, strict=False):
    plotter(ax)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, which="both", lw=0.3, alpha=0.4)

for ax in axes[len(plots) :]:
    ax.set_visible(False)
for ax in axes[len(plots) - ncols : len(plots)]:
    ax.set_xlabel("Iteration")

summary = ", ".join(
    f"{style}: {len(histories[style])} it / "
    f"{int(column(style, 'n_restarts')[-1])} reverts"
    for style in STYLES
)
fig.suptitle(
    f"VMEC++ iteration dynamics (cma, ns={NS}, ftol={FTOL:g})\n{summary}",
    fontsize=10,
)
fig.tight_layout(rect=(0, 0, 1, 0.96))
plt.show()
