# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Live view of a VMEC++ solve.

:func:`watch` runs an equilibrium exactly as :func:`vmecpp.run` does and draws the
flux surfaces and the force residuals of every iteration while it runs, into a
window, an animation file, or both.
"""

from __future__ import annotations

import typing
from collections.abc import Sequence
from pathlib import Path

import numpy as np

if typing.TYPE_CHECKING:
    from vmecpp import (
        IterationSnapshot,
        MagneticFieldResponseTable,
        VmecInput,
        VmecOutput,
    )

# RestartReason::NO_RESTART of the C++ solver.
_NO_RESTART = 1

_VIDEO_SUFFIXES = (".mp4", ".mov", ".webm", ".mkv", ".avi")


def watch(
    input: VmecInput,
    magnetic_field: MagneticFieldResponseTable | None = None,
    *,
    planes: int | Sequence[float] = 2,
    surfaces: int = 8,
    every: int = 1,
    save: str | Path | None = None,
    fps: int = 20,
    show: bool | None = None,
    block: bool = True,
    max_threads: int | None = None,
    restart_from: VmecOutput | None = None,
) -> VmecOutput:
    """Run VMEC++ on ``input`` and watch the flux surfaces converge.

    The solve is the one :func:`vmecpp.run` performs; the figure shows the flux
    surfaces at a few toroidal angles above the force residuals of every
    iteration, with the multigrid stages and the time-step restarts marked.

    Args:
        input: the configuration to solve, as for :func:`vmecpp.run`.
        magnetic_field: an in-memory vacuum field for a free-boundary run, as for
            :func:`vmecpp.run`.
        planes: the toroidal angles of the cross-sections. An integer spreads that
            many angles evenly over half a field period, from phi = 0 to the
            half-period plane; a sequence gives the angles in radians.
        surfaces: how many flux surfaces to draw, spaced evenly in sqrt(s); the
            boundary is always one of them.
        every: draw every this many iterations. Every iteration is recorded in
            the residual trace regardless.
        save: write the animation to this file, ``.gif`` through Pillow or a
            video suffix through ffmpeg.
        fps: frame rate of the saved animation.
        show: open an interactive window. ``None`` opens one when the matplotlib
            backend is interactive.
        block: keep the window open after the solve until it is closed.
        max_threads: as for :func:`vmecpp.run`.
        restart_from: as for :func:`vmecpp.run`.

    Returns:
        the :class:`VmecOutput` of the solve. Closing the window stops the solve,
        and the output then holds the state reached, with ``wout.ier_flag``
        reporting that it did not converge.

    Example::

        import vmecpp

        vmec_input = vmecpp.VmecInput.from_file("examples/data/solovev.json")
        output = vmecpp.watch(vmec_input, save="solve.gif", every=5)
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    import vmecpp  # noqa: PLC0415

    input = vmecpp.VmecInput.model_validate(input)
    if every < 1:
        msg = f"watch: every must be at least 1, got {every}"
        raise ValueError(msg)
    if surfaces < 1:
        msg = f"watch: surfaces must be at least 1, got {surfaces}"
        raise ValueError(msg)
    if show is None:
        show = _backend_is_interactive()
    if not show and save is None:
        msg = (
            "watch: nothing to show; the matplotlib backend is not interactive, "
            "so pass save='solve.gif' (or a video suffix) to record the solve"
        )
        raise ValueError(msg)

    view = _SolveView(
        nfp=int(input.nfp),
        zetas=_plane_angles(planes, int(input.nfp)),
        surfaces=surfaces,
        every=every,
        show=show,
        save=save,
        fps=fps,
    )
    try:
        output = vmecpp.run(
            input,
            magnetic_field,
            max_threads=max_threads,
            verbose=False,
            restart_from=restart_from,
            iteration_callback=view.on_iteration,
        )
        view.finish()
    finally:
        view.close()
    if show and block and plt.fignum_exists(view.fig.number):
        plt.show()
    return output


def surface_curves(
    geometry,
    surface_indices: Sequence[int],
    theta: np.ndarray,
    zeta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """R and Z of full-grid surfaces at the cylindrical angle ``zeta``.

    ``geometry`` is a C++ ``Geometry`` as carried by :class:`vmecpp.IterationSnapshot`
    or returned by ``make_geometry``. The result has one row per entry of
    ``surface_indices`` and one column per ``theta``.
    """
    dimensions = geometry.dimensions
    coefficients = _coefficient_arrays(geometry)
    poloidal = np.arange(dimensions.mpol)
    toroidal = np.arange(dimensions.ntor + 1) * dimensions.nfp
    cos_m = np.cos(np.outer(theta, poloidal))
    sin_m = np.sin(np.outer(theta, poloidal))
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

    c = coefficients
    return (
        series(c["r_cc"], c["r_ss"], c["r_sc"], c["r_cs"]),
        series(c["z_cc"], c["z_ss"], c["z_sc"], c["z_cs"]),
    )


def magnetic_axis(geometry, zeta: float) -> tuple[float, float]:
    """R and Z of the magnetic axis at the cylindrical angle ``zeta``.

    Only the m = 0 coefficients of the innermost surface describe the axis; the m >= 1
    entries stored there are the axis extrapolation of the solver.
    """
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
    coefficients = geometry.coefficients

    def block(values):
        array = np.asarray(values, dtype=float)
        return array.reshape(shape) if array.size else np.zeros(shape)

    names = ("r_cc", "r_ss", "r_sc", "r_cs", "z_sc", "z_cs", "z_cc", "z_ss")
    return {name: block(getattr(coefficients, name)) for name in names}


def _plane_angles(planes: int | Sequence[float], nfp: int) -> list[float]:
    if isinstance(planes, int):
        if planes < 1:
            msg = f"watch: planes must be at least 1, got {planes}"
            raise ValueError(msg)
        return [float(z) for z in np.linspace(0.0, np.pi / nfp, planes)]
    angles = [float(z) for z in planes]
    if not angles:
        msg = "watch: planes must name at least one toroidal angle"
        raise ValueError(msg)
    return angles


def _plane_label(zeta: float, nfp: int) -> str:
    periods = zeta * nfp / (2.0 * np.pi)
    if periods == 0.0:
        return "phi = 0"
    return f"phi = {periods:g} of a field period"


def _backend_is_interactive() -> bool:
    import matplotlib as mpl  # noqa: PLC0415

    backend = mpl.get_backend().lower()
    try:
        from matplotlib.backends import (  # noqa: PLC0415
            BackendFilter,
            backend_registry,
        )

        interactive = backend_registry.list_builtin(BackendFilter.INTERACTIVE)
    except ImportError:
        interactive = mpl.rcsetup.interactive_bk  # type: ignore[attr-defined]
    return backend in {name.lower() for name in interactive}


def _make_writer(save: str | Path, fps: int):
    from matplotlib import animation  # noqa: PLC0415

    suffix = Path(save).suffix.lower()
    if suffix == ".gif":
        return animation.PillowWriter(fps=fps)
    if suffix in _VIDEO_SUFFIXES:
        return animation.FFMpegWriter(fps=fps)
    msg = (
        f"watch: cannot write '{save}'; use a .gif or one of "
        f"{', '.join(_VIDEO_SUFFIXES)}"
    )
    raise ValueError(msg)


class _SolveView:
    """The figure: flux surfaces at a few toroidal angles above the residual trace."""

    def __init__(
        self,
        *,
        nfp: int,
        zetas: Sequence[float],
        surfaces: int,
        every: int,
        show: bool,
        save: str | Path | None,
        fps: int,
    ) -> None:
        import matplotlib.pyplot as plt  # noqa: PLC0415

        self._plt = plt
        self.nfp = nfp
        self.zetas = list(zetas)
        self.surfaces = surfaces
        self.every = every
        self.show = show
        self.theta = np.linspace(0.0, 2.0 * np.pi, 181)

        self.count = 0
        self.iterations: list[int] = []
        self.residuals = {"fsqr": [], "fsqz": [], "fsql": []}
        self.restarts: list[tuple[int, float]] = []
        self.stage: int | None = None
        self.ftol: float | None = None
        self.last: IterationSnapshot | None = None
        self.drawn_at = 0

        width = 3.6 * len(self.zetas) + 1.2
        self.fig = plt.figure(figsize=(max(width, 6.4), 7.2))
        grid = self.fig.add_gridspec(
            2,
            len(self.zetas),
            height_ratios=[3, 2],
            hspace=0.35,
            wspace=0.3,
            left=0.09,
            right=0.98,
            top=0.9,
            bottom=0.08,
        )
        self.surface_axes = [
            self.fig.add_subplot(grid[0, k]) for k in range(len(self.zetas))
        ]
        self.residual_axes = self.fig.add_subplot(grid[1, :])

        import matplotlib as mpl  # noqa: PLC0415

        colors = mpl.colormaps["Blues"](np.linspace(0.35, 0.9, max(surfaces - 1, 1)))
        self.limits: tuple[float, float, float, float] | None = None
        self.vacuum_active = False
        self.surface_lines = []
        self.boundary_lines = []
        self.axis_markers = []
        for ax, zeta in zip(self.surface_axes, self.zetas, strict=True):
            ax.set_aspect("equal")
            ax.set_title(_plane_label(zeta, nfp))
            ax.set_xlabel("R")
            ax.set_ylabel("Z")
            self.surface_lines.append(
                [
                    ax.plot([], [], color=colors[k], lw=0.9)[0]
                    for k in range(surfaces - 1)
                ]
            )
            self.boundary_lines.append(ax.plot([], [], color="black", lw=1.5)[0])
            self.axis_markers.append(ax.plot([], [], "o", color="tab:red", ms=4)[0])

        ax = self.residual_axes
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("force residual")
        self.residual_lines = {
            name: ax.plot([], [], lw=1.0, label=name)[0]
            for name in ("fsqr", "fsqz", "fsql")
        }
        self.ftol_line = ax.axhline(1.0, color="tab:red", ls="--", lw=0.8, label="ftol")
        self.restart_marks = ax.plot(
            [], [], "x", color="tab:red", ms=6, label="restart"
        )[0]
        ax.legend(loc="upper right", fontsize=8)

        self.writer = None
        if save is not None:
            self.writer = _make_writer(save, fps)
            self.writer.setup(self.fig, str(save), dpi=100)

    def on_iteration(self, snapshot: IterationSnapshot) -> bool:
        """Record the iteration, draw it when due, and stop the run if the window is
        gone."""
        self.count += 1
        self.last = snapshot
        self.iterations.append(self.count)
        self.residuals["fsqr"].append(snapshot.fsqr)
        self.residuals["fsqz"].append(snapshot.fsqz)
        self.residuals["fsql"].append(snapshot.fsql)
        if snapshot.restart_reason != _NO_RESTART:
            self.restarts.append(
                (self.count, snapshot.fsqr + snapshot.fsqz + snapshot.fsql)
            )
        if snapshot.multigrid_step != self.stage:
            self.stage = snapshot.multigrid_step
            self.residual_axes.axvline(self.count, color="gray", ls=":", lw=0.8)
            self.residual_axes.text(
                self.count,
                1.0,
                f" ns = {snapshot.ns}",
                transform=self._stage_transform(),
                fontsize=8,
                va="top",
                color="gray",
            )
        if snapshot.ftol != self.ftol:
            self.ftol = snapshot.ftol
            self.ftol_line.set_ydata([snapshot.ftol, snapshot.ftol])
        if snapshot.vacuum_pressure_active and not self.vacuum_active:
            self.vacuum_active = True
            self.residual_axes.axvline(self.count, color="tab:purple", ls="-.", lw=0.8)
            self.residual_axes.text(
                self.count,
                0.02,
                " vacuum pressure on",
                transform=self._stage_transform(),
                fontsize=8,
                va="bottom",
                color="tab:purple",
            )
        if self.count % self.every == 0:
            self.draw(snapshot)
        return self.window_open()

    def finish(self) -> None:
        """Draw the state the solve ended on, marked converged when its residuals are
        below ftol."""
        if self.last is None:
            return
        converged = (
            max(self.last.fsqr, self.last.fsqz, self.last.fsql) <= self.last.ftol
        )
        if self.drawn_at != self.count:
            self.draw(self.last, converged=converged)
        elif converged:
            self._set_title(self.last, converged=True)
            self._flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.finish()
            self.writer = None

    def window_open(self) -> bool:
        return not self.show or self._plt.fignum_exists(self.fig.number)

    def draw(self, snapshot: IterationSnapshot, *, converged: bool = False) -> None:
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
        if self.restarts:
            self.restart_marks.set_data(*zip(*self.restarts, strict=True))
        ax = self.residual_axes
        values = np.concatenate([self.residuals[name] for name in self.residuals])
        values = values[np.isfinite(values) & (values > 0.0)]
        low = min(values.min() if values.size else 1.0, self.ftol or 1.0)
        high = values.max() if values.size else 1.0
        ax.set_ylim(low * 0.3, high * 3.0)
        ax.set_xlim(0, max(self.count, 10))
        self._set_title(snapshot, converged=converged)
        self._flush()
        self.drawn_at = self.count

    def _set_title(self, snapshot: IterationSnapshot, *, converged: bool) -> None:
        fsq = snapshot.fsqr + snapshot.fsqz + snapshot.fsql
        state = "converged" if converged else "iterating"
        self.fig.suptitle(
            f"{state}: iteration {snapshot.iteration}, ns = {snapshot.ns}, "
            f"delt = {snapshot.delt:.3g}, fsq = {fsq:.2e}"
        )

    def _flush(self) -> None:
        if self.window_open() and self.show:
            self._plt.pause(1.0e-3)
        if self.writer is not None:
            self.writer.grab_frame()
        elif not self.show:
            self.fig.canvas.draw()

    def _stage_transform(self):
        from matplotlib import transforms  # noqa: PLC0415

        ax = self.residual_axes
        return transforms.blended_transform_factory(ax.transData, ax.transAxes)

    def _widen_limits(self, r: np.ndarray, z: np.ndarray) -> None:
        """Widen the shared cross-section limits to this boundary with a margin, never
        shrinking them."""
        margin = 0.05 * max(r.max() - r.min(), z.max() - z.min())
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
