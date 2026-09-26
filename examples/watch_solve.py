# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Watch VMEC++ converge.

The ``iteration_callback`` of ``vmecpp.run`` receives every force iteration with its
residuals and the geometry of the state. This example draws the flux surfaces at a
few toroidal angles above the force residuals while the solve runs, in a window or
into an animation file. Closing the window stops the solve.

    python examples/watch_solve.py examples/data/w7x.json --save w7x.gif
"""

from __future__ import annotations

import argparse
import typing
from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl
import numpy as np

import vmecpp

# RestartReason::NO_RESTART of the C++ solver
NO_RESTART = 1
# VmecStatus::MORE_ITERATIONS_NEEDED, the run stopped by the callback
STOPPED = 2

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
GOOD = "#0ca30c"
# the force residuals of R, Z and lambda
SERIES = {"R": "#2a78d6", "Z": "#eb6834", "λ": "#1baf7a"}
# the inner flux surfaces, light to dark with the radius
RAMP = (
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
)

STYLE: dict[typing.Any, typing.Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9.5,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK_SECONDARY,
    "axes.titlesize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK_MUTED,
    "ytick.labelcolor": INK_MUTED,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "legend.labelcolor": INK_SECONDARY,
}


def surface_curves(
    geometry,
    surface_indices: Sequence[int] | np.ndarray,
    theta: np.ndarray,
    zeta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """R and Z of full-grid surfaces at the cylindrical angle ``zeta``, one row per
    entry of ``surface_indices`` and one column per ``theta``.

    ``geometry`` is the ``Geometry`` of an ``IterationSnapshot``.
    """
    dimensions = geometry.dimensions
    c = _coefficient_arrays(geometry)
    cos_m = np.cos(np.outer(theta, np.arange(dimensions.mpol)))
    sin_m = np.sin(np.outer(theta, np.arange(dimensions.mpol)))
    toroidal = np.arange(dimensions.ntor + 1) * dimensions.nfp
    cos_n = np.cos(toroidal * zeta)
    sin_n = np.sin(toroidal * zeta)
    rows = np.asarray(surface_indices, dtype=int)

    def series(cc, ss, sc, cs):
        return (
            np.einsum("jmn,tm,n->jt", cc[rows], cos_m, cos_n)
            + np.einsum("jmn,tm,n->jt", ss[rows], sin_m, sin_n)
            + np.einsum("jmn,tm,n->jt", sc[rows], sin_m, cos_n)
            + np.einsum("jmn,tm,n->jt", cs[rows], cos_m, sin_n)
        )

    return (
        series(c["r_cc"], c["r_ss"], c["r_sc"], c["r_cs"]),
        series(c["z_cc"], c["z_ss"], c["z_sc"], c["z_cs"]),
    )


def magnetic_axis(geometry, zeta: float) -> tuple[float, float]:
    """R and Z of the magnetic axis at the cylindrical angle ``zeta``: the m = 0
    coefficients of the innermost surface."""
    dimensions = geometry.dimensions
    c = _coefficient_arrays(geometry)
    toroidal = np.arange(dimensions.ntor + 1) * dimensions.nfp
    cos_n = np.cos(toroidal * zeta)
    sin_n = np.sin(toroidal * zeta)
    r = c["r_cc"][0, 0] @ cos_n + c["r_cs"][0, 0] @ sin_n
    z = c["z_cc"][0, 0] @ cos_n + c["z_cs"][0, 0] @ sin_n
    return float(r), float(z)


def surface_indices(ns: int, count: int) -> np.ndarray:
    """Indices of ``count`` full-grid surfaces spaced evenly in sqrt(s), ending at the
    boundary."""
    rho = np.arange(1, count + 1) / count
    return np.unique(np.rint((ns - 1) * rho**2).astype(int).clip(1, ns - 1))


def _coefficient_arrays(geometry) -> dict[str, np.ndarray]:
    dimensions = geometry.dimensions
    shape = (dimensions.ns, dimensions.mpol, dimensions.ntor + 1)

    def block(values):
        array = np.asarray(values, dtype=float)
        return array.reshape(shape) if array.size else np.zeros(shape)

    names = ("r_cc", "r_ss", "r_sc", "r_cs", "z_sc", "z_cs", "z_cc", "z_ss")
    return {name: block(getattr(geometry.coefficients, name)) for name in names}


def backend_is_interactive() -> bool:
    from matplotlib.backends.registry import (  # noqa: PLC0415
        BackendFilter,
        backend_registry,
    )

    interactive = backend_registry.list_builtin(BackendFilter.INTERACTIVE)
    return mpl.get_backend().lower() in {name.lower() for name in interactive}


class SolveView:
    """The flux surfaces at a few toroidal angles above the force residuals."""

    def __init__(
        self,
        *,
        title: str,
        nfp: int,
        planes: int,
        surfaces: int,
        every: int,
        show: bool,
        save: str | Path | None,
        fps: int,
    ) -> None:
        import matplotlib.pyplot as plt  # noqa: PLC0415
        from matplotlib import transforms  # noqa: PLC0415

        self.plt = plt
        self.zetas = [float(z) for z in np.linspace(0.0, np.pi / nfp, planes)]
        self.surfaces = surfaces
        self.every = every
        self.show = show
        self.theta = np.linspace(0.0, 2.0 * np.pi, 181)

        self.count = 0
        self.iterations: list[float] = []
        self.residuals: dict[str, list[float]] = {name: [] for name in SERIES}
        self.restarts: list[int] = []
        self.stage: int | None = None
        self.ftol = 1.0
        self.vacuum_active = False
        self.last = None
        self.drawn_at = 0
        self.limits: tuple[float, float, float, float] | None = None

        self.fig = plt.figure(figsize=(max(6.4, 0.9 + 3.95 * planes), 6.6), dpi=100)
        self.fig.text(0.075, 0.955, title, fontsize=13, fontweight="bold", color=INK)
        self.status = self.fig.text(0.075, 0.918, "", color=INK_SECONDARY)
        self.badge = None

        top = self.fig.add_gridspec(
            1, planes, left=0.075, right=0.975, top=0.87, bottom=0.45, wspace=0.08
        )
        self.surface_axes = []
        for k, zeta in enumerate(self.zetas):
            ax = self.fig.add_subplot(top[0, k])
            ax.set_aspect("equal")
            ax.set_title(f"φ = {np.degrees(zeta):.0f}°", loc="left")
            ax.set_xlabel("R (m)")
            if k == 0:
                ax.set_ylabel("Z (m)")
            else:
                ax.tick_params(labelleft=False)
            self.surface_axes.append(ax)
        ramp = np.linspace(0, len(RAMP) - 1, max(surfaces - 1, 1)).round().astype(int)
        colors = [RAMP[i] for i in ramp]
        self.surface_lines = [
            [ax.plot([], [], color=color, lw=1.0)[0] for color in colors]
            for ax in self.surface_axes
        ]
        self.boundary_lines = [
            ax.plot([], [], color=INK, lw=1.6)[0] for ax in self.surface_axes
        ]
        self.axis_markers = [
            ax.plot([], [], "o", ms=6, color=INK, mec=SURFACE, mew=1.5)[0]
            for ax in self.surface_axes
        ]

        bottom = self.fig.add_gridspec(
            1, 1, left=0.075, right=0.855, top=0.33, bottom=0.085
        )
        ax = self.residual_axes = self.fig.add_subplot(bottom[0, 0])
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("force residual")
        ax.grid(axis="y", which="major", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", which="minor", left=False)
        self.top_edge = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        self.residual_lines = {
            name: ax.plot(
                [], [], color=color, lw=1.4, solid_capstyle="round", label=name
            )[0]
            for name, color in SERIES.items()
        }
        self.ftol_line = ax.axhline(
            1.0, color=INK_MUTED, lw=0.8, ls=(0, (4, 3)), label="ftol"
        )
        self.restart_marks = ax.plot(
            [],
            [],
            "v",
            ms=6,
            color=INK_SECONDARY,
            mec=SURFACE,
            mew=1.0,
            transform=self.top_edge,
            clip_on=False,
            label="restart",
        )[0]
        self._legend()

        # the animation file is opened with its first frame
        self.save = save
        self.fps = fps
        self.writer = None

    def on_iteration(self, snapshot) -> bool:
        """Record the iteration, draw it when due, and stop the run once the window is
        closed."""
        self.count += 1
        self.last = snapshot
        if snapshot.multigrid_step != self.stage:
            if self.stage is not None:
                # break the residual lines between multigrid stages
                self.iterations.append(self.count - 0.5)
                for values in self.residuals.values():
                    values.append(np.nan)
            self.stage = snapshot.multigrid_step
            self._event(f"ns {snapshot.ns}")
        self.iterations.append(self.count)
        for name, value in zip(
            SERIES, (snapshot.fsqr, snapshot.fsqz, snapshot.fsql), strict=True
        ):
            self.residuals[name].append(value)
        if snapshot.restart_reason != NO_RESTART:
            self.restarts.append(self.count)
        if snapshot.vacuum_pressure_active and not self.vacuum_active:
            self.vacuum_active = True
            self._event("vacuum on")
        self.ftol = snapshot.ftol
        if self.count % self.every == 0:
            self.draw(snapshot)
        return self.window_open()

    def finish(self, output) -> None:
        """Draw the state the solve ended on and mark how it ended."""
        if self.last is None or not self.window_open():
            return
        from matplotlib.artist import Artist  # noqa: PLC0415
        from matplotlib.offsetbox import (  # noqa: PLC0415
            AnchoredOffsetbox,
            HPacker,
            TextArea,
        )

        parts: list[Artist]
        if output.wout.ier_flag == 0:
            parts = [
                TextArea(
                    "✓",
                    textprops={
                        "color": GOOD,
                        "fontsize": 11,
                        "fontfamily": "DejaVu Sans",
                    },
                ),
                TextArea("converged", textprops={"color": INK_SECONDARY}),
            ]
        elif output.wout.ier_flag == STOPPED:
            parts = [TextArea("stopped", textprops={"color": INK_MUTED})]
        else:
            parts = [TextArea(output.wout.reason, textprops={"color": INK_MUTED})]
        self.badge = AnchoredOffsetbox(
            loc="upper right",
            child=HPacker(children=parts, sep=4, align="baseline"),
            bbox_to_anchor=(0.975, 0.975),
            bbox_transform=self.fig.transFigure,
            frameon=False,
            pad=0.0,
            borderpad=0.0,
        )
        self.fig.add_artist(self.badge)
        self.draw(self.last)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.finish()
            self.writer = None

    def window_open(self) -> bool:
        return not self.show or self.plt.fignum_exists(self.fig.number)

    def draw(self, snapshot) -> None:
        geometry = snapshot.geometry
        rows = surface_indices(geometry.dimensions.ns, self.surfaces)
        for k, zeta in enumerate(self.zetas):
            r, z = surface_curves(geometry, rows, self.theta, zeta)
            for line, ri, zi in zip(
                self.surface_lines[k], r[:-1], z[:-1], strict=False
            ):
                line.set_data(ri, zi)
            for line in self.surface_lines[k][len(rows) - 1 :]:
                line.set_data([], [])
            self.boundary_lines[k].set_data(r[-1], z[-1])
            r_axis, z_axis = magnetic_axis(geometry, zeta)
            self.axis_markers[k].set_data([r_axis], [z_axis])
            self._widen_limits(r[-1], z[-1])
        assert self.limits is not None
        for ax in self.surface_axes:
            ax.set_xlim(self.limits[0], self.limits[1])
            ax.set_ylim(self.limits[2], self.limits[3])

        for name, line in self.residual_lines.items():
            line.set_data(self.iterations, self.residuals[name])
        self.restart_marks.set_data(self.restarts, [1.0] * len(self.restarts))
        values = np.concatenate([self.residuals[name] for name in SERIES])
        values = values[np.isfinite(values) & (values > 0.0)]
        low, high = values.min(), values.max()
        # show the tolerance unless it lies far below everything drawn so far
        if self.ftol >= 1.0e-3 * low:
            low = min(low, self.ftol)
        self.ftol_line.set_ydata([self.ftol, self.ftol])
        self.residual_axes.set_ylim(low / 3.0, high * 3.0)
        self.residual_axes.set_xlim(0, max(self.count, 10) * 1.02)
        if self.restarts and self.restart_marks not in self.legend_handles:
            self._legend()

        fsq = max(snapshot.fsqr, snapshot.fsqz, snapshot.fsql)
        self.status.set_text(
            f"ns {snapshot.ns}  ·  iteration {self.count:,}  ·  "
            f"largest residual {fsq:.2e}  ·  ftol {snapshot.ftol:.0e}"
        )
        if self.show and self.window_open():
            self.plt.pause(1.0e-3)
        if self.save is not None:
            if self.writer is None:
                from matplotlib import animation  # noqa: PLC0415

                if Path(self.save).suffix.lower() == ".gif":
                    self.writer = animation.PillowWriter(fps=self.fps)
                else:
                    self.writer = animation.FFMpegWriter(fps=self.fps)
                self.writer.setup(self.fig, str(self.save), dpi=100)
            self.writer.grab_frame()
        elif not self.show:
            self.fig.canvas.draw()
        self.drawn_at = self.count

    def _legend(self) -> None:
        handles = [*self.residual_lines.values(), self.ftol_line]
        if self.restarts:
            handles.append(self.restart_marks)
        self.legend_handles = handles
        self.residual_axes.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            handlelength=1.8,
            borderaxespad=0.0,
        )

    def _event(self, label: str) -> None:
        """Mark the iteration a multigrid stage starts or the vacuum pressure enters."""
        ax = self.residual_axes
        if self.count > 1:
            ax.axvline(self.count, color=AXIS, lw=0.8, zorder=0)
        ax.text(
            self.count,
            1.07,
            label,
            transform=self.top_edge,
            color=INK_MUTED,
            fontsize=8,
            va="bottom",
        )

    def _widen_limits(self, r: np.ndarray, z: np.ndarray) -> None:
        """Widen the shared cross-section limits to this boundary, never shrinking
        them."""
        margin = 0.06 * max(r.max() - r.min(), z.max() - z.min())
        bounds = (
            r.min() - margin,
            r.max() + margin,
            z.min() - margin,
            z.max() + margin,
        )
        if self.limits is not None:
            bounds = (
                min(bounds[0], self.limits[0]),
                max(bounds[1], self.limits[1]),
                min(bounds[2], self.limits[2]),
                max(bounds[3], self.limits[3]),
            )
        self.limits = bounds


def watch(
    vmec_input: vmecpp.VmecInput,
    magnetic_field: vmecpp.MagneticFieldResponseTable | None = None,
    *,
    title: str = "",
    planes: int | None = None,
    surfaces: int = 8,
    every: int = 5,
    save: str | Path | None = None,
    fps: int = 20,
    show: bool | None = None,
    max_threads: int | None = None,
) -> vmecpp.VmecOutput:
    """Run VMEC++ on ``vmec_input`` and draw it converging.

    ``planes`` toroidal cross-sections are spread from phi = 0 to half a field
    period, one for an axisymmetric input by default; ``surfaces`` flux surfaces
    are drawn, spaced evenly in sqrt(s). A frame is drawn every ``every``
    iterations, into a window when the matplotlib backend is interactive and into
    ``save``, a ``.gif`` or a video file for ffmpeg, when it is given. Returns the
    ``VmecOutput`` of the run; closing the window stops the solve, with
    ``wout.ier_flag`` 2.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if show is None:
        show = backend_is_interactive()
    if not show and save is None:
        msg = "the matplotlib backend is not interactive: pass save='solve.gif'"
        raise ValueError(msg)
    if planes is None:
        planes = 1 if np.max(np.atleast_1d(vmec_input.ntor)) == 0 else 2

    with mpl.rc_context(STYLE):
        view = SolveView(
            title=title,
            nfp=int(vmec_input.nfp),
            planes=planes,
            surfaces=surfaces,
            every=every,
            show=show,
            save=save,
            fps=fps,
        )
        try:
            output = vmecpp.run(
                vmec_input,
                magnetic_field,
                max_threads=max_threads,
                verbose=False,
                iteration_callback=view.on_iteration,
            )
            view.finish(output)
        finally:
            view.close()
        if show and plt.fignum_exists(view.fig.number):
            plt.show()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        default=Path(__file__).parent / "data" / "solovev.json",
        type=Path,
        help="a VMEC++ JSON or INDATA file (default: examples/data/solovev.json)",
    )
    parser.add_argument(
        "--save", type=Path, help="record the solve to a .gif or video file"
    )
    parser.add_argument("--every", type=int, default=5, help="draw every N iterations")
    parser.add_argument("--title", help="the title above the figure")
    args = parser.parse_args()

    save = args.save
    if save is None and not backend_is_interactive():
        save = Path(f"{args.input.stem}.gif")
        print(f"no display: recording the solve to {save}")
    vmec_input = vmecpp.VmecInput.from_file(args.input)
    output = watch(
        vmec_input, title=args.title or args.input.stem, every=args.every, save=save
    )
    print(output.wout.reason)


if __name__ == "__main__":
    main()
