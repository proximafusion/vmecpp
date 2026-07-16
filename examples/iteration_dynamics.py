# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Trace and plot the VMEC++ iteration dynamics from Python.

The force-balance iteration is owned in Python (``vmecpp._iteration``), so we can
observe *every* convergence and flow-control quantity at every step -- not just
the three force residuals stored in the wout. ``vmecpp.solve_multigrid`` (and
``vmecpp.iterate``) accept a ``callback`` invoked once per force iteration with an
``IterationState`` snapshot; here we collect those snapshots and plot them as a
grid of subplots.

This is a debugging aid for understanding *why* the convergence dynamics change
qualitatively partway through a run. The time-step-control styles
(``vmec_8_52`` / ``parvmec`` / ``robust`` / ``delt_recovery``) share one forward
model and differ only in how they react to a growing residual, so running them
on the same multigrid ladder (``cma`` over ns=[51, 99]) and overlaying the
traces makes the divergence concrete: VMEC 8.52's tight 100x residual leash
reverts on the stage-entry transient (each revert cuts the time step
*permanently* and bumps ``ijacob`` toward the give-up escalation), while
PARVMEC's permissive 1e4 leash rides straight through. ``delt_recovery`` (the
scheme prototyped in C++ on the improve-convergence branch as
``VMECPP_DELT_RECOVERY`` / ``VMECPP_DELT_START``) keeps 8.52's tight leash but
enters continuation stages at half the user step and grows it back on
progress, bounded by a learned stability ceiling (dotted in the time-step
panel): reverts still happen, but the stage tail recovers to the largest
stable step instead of crawling at the accumulated 0.9^n reduction. It only
arms on interpolation-seeded continuation stages (here: the ns=99 stage) and
is bit-identical to ``vmec_8_52`` on cold starts (here: the ns=51 stage, where
the blue trace hides exactly under the red one), so it never regresses the
easy cases.

On top of the styles, two orthogonal improvements are compared as variants
(see ``VARIANTS``): the lambda-preconditioner correction at the first evolved
surface (``solve_multigrid(lambda_preconditioner_boost=True)``) and an
initial time step above the historical delt <= 1 bound, which the recovery
control walks down to the learned stability ceiling instead of diverging.
"""

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import vmecpp

# cma over a [51, 99] multigrid ladder: all styles converge, but with very
# different flow-control dynamics at the stage transition -- the point of this
# example.
INPUT_FILE = (
    Path(__file__).parent.parent
    / "src"
    / "vmecpp"
    / "cpp"
    / "vmecpp"
    / "test_data"
    / "cma.json"
)
INPUT_FILE = Path(__file__).parent.parent / "examples" / "data" / "w7x.json"

# Variants to compare: an iteration style plus the orthogonal improvements
# validated on the improve-convergence branch (docs/convergence_study.md).
# - "delt_recovery": flow control only (ramp-in + step recovery + learned
#   stability ceiling + stagnation guard).
# - "+ lambda_boost": additionally correct the lambda preconditioner at the
#   first evolved surface (Findings 10-12: faclam overestimates the m<=1
#   lambda stiffness there ~5-10x; -23% tail iterations on w7x, physics
#   invariant). Orthogonal to the style.
# - "+ delt=2": additionally start from an initial time step above 1 (the
#   input bound is now ]0,10]); the recovery control walks an over-large
#   step down to the stability ceiling instead of diverging, so delt > 1
#   probes how much of the tail is step-size-limited.
VARIANTS: dict[str, dict] = {
    "vmec_8_52": {"style": "vmec_8_52"},
    # "parvmec": {"style": "parvmec"},
    # "robust": {"style": "robust"},
    "delt_recovery": {"style": "delt_recovery"},
    "delt_recovery + lambda_boost": {"style": "delt_recovery", "lambda_boost": True},
    "delt_recovery + lambda_boost + delt=2": {
        "style": "delt_recovery",
        "lambda_boost": True,
        "delt": 2.0,
    },
}
STYLES = tuple(VARIANTS)
# Radial interpolant for the stage transfers: "linear" (2-point, VMEC 8.52),
# "cubic" or "cubic_rho" (4-point Lagrange). The higher-order interpolants
# reduce the interpolation-added error of the stage seed down to the coarse
# grid's own truncation error (~100x lower entry residual).
INTERPOLATION = "cubic"
vmec_input = vmecpp.VmecInput.from_file(INPUT_FILE)
vmec_input.ns_array = np.array([51, 199])
vmec_input.ftol_array = np.array([1.0e-9, 1e-11])
vmec_input.niter_array = np.array([3000, 3000])
vmec_input.save(Path("vmec_input.json"), indent=4)
ftolv = float(vmec_input.ftol_array[-1])

# Run each style through the full multigrid ramp, collecting an IterationState
# snapshot every force iteration via the callback (across all stages).
# solve_multigrid arms delt_recovery's ramp-in (0.5x entry step) on the
# continuation stages automatically; the cold ns=51 stage runs plain 8.52.
histories: dict[str, list[vmecpp.IterationState]] = {}
stage_results: dict[str, list[vmecpp.IterationResult]] = {}
for label, cfg in VARIANTS.items():
    variant_input = vmec_input.model_copy(deep=True)
    if "delt" in cfg:
        variant_input.delt = cfg["delt"]
    history: list[vmecpp.IterationState] = []
    _, results = vmecpp.solve_multigrid(
        variant_input,
        iteration_style=cfg["style"],
        interpolation=INTERPOLATION,
        lambda_preconditioner_boost=cfg.get("lambda_boost", False),
        callback=history.append,
    )
    histories[label] = history
    stage_results[label] = results
    per_stage = [(r.num_iterations, r.restarts) for r in results]
    print(
        f"{label:38s} converged={all(r.converged for r in results)} "
        f"iters={sum(r.num_iterations for r in results):5d} "
        f"restarts={sum(r.restarts for r in results):3d} "
        f"per-stage (iters, restarts)={per_stage} fsqr={results[-1].fsqr:.2e}"
    )


def column(style: str, name: str) -> np.ndarray:
    return np.array([getattr(s, name) for s in histories[style]])


def xs(style: str) -> np.ndarray:
    """Global x-axis: force iterations counted across all multigrid stages (the
    per-state ``iteration`` field restarts at each stage)."""
    return np.arange(1, len(histories[style]) + 1)


def stage_starts(style: str) -> list[int]:
    """Global iteration indices at which a new multigrid stage begins."""
    it = column(style, "iteration")
    return [i + 1 for i in range(1, len(it)) if it[i] < it[i - 1]]


# Skip the initial-condition transient when autoscaling the log panels, otherwise
# the early spike pins the axis and flattens the dynamics we want to see.
def warmup(style: str) -> int:
    return min(50, len(histories[style]) // 100)


COLORS = {
    "vmec_8_52": "tab:blue",
    # "parvmec": "tab:orange",
    # "robust": "tab:green",
    "delt_recovery": "tab:red",
    "delt_recovery + lambda_boost": "tab:purple",
    "delt_recovery + lambda_boost + delt=2": "tab:orange",
}


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
        ax.plot(xs(style), y, color=COLORS[style], lw=0.9, label=style)
        for boundary in stage_starts(style):
            ax.axvline(boundary, color=COLORS[style], lw=0.5, ls="--", alpha=0.4)
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
        it = xs(style)
        ax.plot(it, column(style, "fsq_preconditioned"), color=COLORS[style], lw=0.7)
        ax.plot(it, column(style, "res0"), color=COLORS[style], lw=0.7, ls=":")
    ax.set_yscale("log")
    _autoscale_log(ax, {s: column(s, "fsq_preconditioned") for s in STYLES})
    ax.plot([], [], "k-", lw=0.7, label="fsq (precon)")
    ax.plot([], [], "k:", lw=0.7, label="res0 (running min)")
    ax.legend(fontsize=7)


def plot_timestep(ax):
    _overlay(ax, "delt", log=True)
    # the learned stability ceiling of the delt_recovery control: reverts
    # ratchet it down, the recovered step rides just below it
    for label, cfg in VARIANTS.items():
        if cfg["style"] != "delt_recovery":
            continue
        ax.plot(
            xs(label),
            column(label, "delt_ceiling"),
            color=COLORS[label],
            lw=0.8,
            ls=":",
        )
    ax.plot([], [], "k:", lw=0.8, label="recovery ceiling")
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
            xs(style),
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
        it = xs(style)
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
    f"{sum(r.restarts for r in stage_results[style])} reverts"
    for style in STYLES
)
ns_list = [int(ns) for ns in vmec_input.ns_array]
fig.suptitle(
    f"VMEC++ iteration dynamics ({INPUT_FILE.stem}, multigrid ns={ns_list})\n{summary}",
    fontsize=10,
)
fig.tight_layout(rect=(0, 0, 1, 0.96))
plt.show()
