# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The per-iteration callback of ``run`` and the live view built on it."""

from pathlib import Path

import jax
import matplotlib as mpl
import numpy as np
import pytest
from PIL import Image

import vmecpp
from vmecpp import _watch
from vmecpp import geometry as vmec_geometry
from vmecpp.cpp import _vmecpp  # type: ignore

mpl.use("Agg")
jax.config.update("jax_enable_x64", True)

REPO_ROOT = Path(__file__).parent.parent
TEST_DATA_DIR = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"
SOLOVEV = REPO_ROOT / "examples" / "data" / "solovev.json"

# RestartReason::NO_RESTART of the C++ solver.
NO_RESTART = 1


def _collect(vmec_input):
    snapshots = []
    output = vmecpp.run(vmec_input, verbose=False, iteration_callback=snapshots.append)
    return snapshots, output


@pytest.mark.parametrize(
    "case",
    [SOLOVEV, TEST_DATA_DIR / "cth_like_fixed_bdy.json"],
    ids=["solovev", "cth_like_fixed_bdy"],
)
def test_callback_sees_every_iteration_of_the_residual_history(case):
    """One snapshot per force iteration: the residual history of the wout, in
    order and bit for bit, plus the converged iteration that closes each multigrid
    stage, which the solver does not record."""
    vmec_input = vmecpp.VmecInput.from_file(case)
    snapshots, output = _collect(vmec_input)
    wout = output.wout

    stage_ends = {}
    for index, snapshot in enumerate(snapshots):
        stage_ends[snapshot.multigrid_step] = index
    assert sorted(stage_ends) == list(range(len(vmec_input.ns_array)))
    for stage, index in stage_ends.items():
        assert snapshots[index].ns == vmec_input.ns_array[stage]
        assert snapshots[index].ftol == vmec_input.ftol_array[stage]

    recorded = [
        s
        for index, s in enumerate(snapshots)
        if s.restart_reason == NO_RESTART and index not in stage_ends.values()
    ]
    assert len(recorded) == wout.fsqt.size
    np.testing.assert_array_equal(
        [s.fsqr for s in recorded], np.asarray(wout.force_residual_r)
    )
    np.testing.assert_array_equal(
        [s.fsqz for s in recorded], np.asarray(wout.force_residual_z)
    )
    np.testing.assert_array_equal(
        [s.fsql for s in recorded], np.asarray(wout.force_residual_lambda)
    )

    last = snapshots[-1]
    assert last.fsqr == wout.fsqr
    assert last.fsqz == wout.fsqz
    assert last.fsql == wout.fsql
    assert max(last.fsqr, last.fsqz, last.fsql) <= last.ftol


@pytest.mark.parametrize(
    "case",
    [SOLOVEV, TEST_DATA_DIR / "cth_like_fixed_bdy_asym.json"],
    ids=["solovev", "cth_like_fixed_bdy_asym"],
)
def test_last_snapshot_carries_the_converged_geometry(case):
    vmec_input = vmecpp.VmecInput.from_file(case)
    snapshots, _ = _collect(vmec_input)
    cpp_output = _vmecpp.run(
        vmec_input._to_cpp_vmecindata(), verbose=_vmecpp.OutputMode.SILENT
    )
    live = vmec_geometry.from_cpp(snapshots[-1].geometry)
    final = vmec_geometry.make(cpp_output)
    for name in (
        "r_cc",
        "r_ss",
        "r_sc",
        "r_cs",
        "z_sc",
        "z_cs",
        "z_cc",
        "z_ss",
        "lambda_sc",
        "lambda_cs",
        "lambda_cc",
        "lambda_ss",
        "toroidal_flux",
        "poloidal_flux",
    ):
        np.testing.assert_allclose(
            np.asarray(getattr(live, name)),
            np.asarray(getattr(final, name)),
            rtol=0.0,
            atol=1.0e-13,
            err_msg=name,
        )


def test_returning_false_stops_the_run_with_the_state_reached():
    vmec_input = vmecpp.VmecInput.from_file(SOLOVEV)
    seen = []

    def stop_at_twenty(snapshot):
        seen.append(snapshot.iteration)
        return snapshot.iteration < 20

    output = vmecpp.run(vmec_input, verbose=False, iteration_callback=stop_at_twenty)
    assert seen == list(range(1, 21))
    assert output.wout.ns == vmec_input.ns_array[0]
    assert output.wout.ier_flag == 2
    assert "stopped by the iteration callback" in output.wout.reason
    assert output.wout.fsqt.size == 20


def test_exception_in_the_callback_propagates():
    vmec_input = vmecpp.VmecInput.from_file(SOLOVEV)

    def fail_at_five(snapshot):
        if snapshot.iteration == 5:
            msg = "stop here"
            raise KeyError(msg)

    with pytest.raises(KeyError, match="stop here"):
        vmecpp.run(vmec_input, verbose=False, iteration_callback=fail_at_five)


@pytest.mark.parametrize(
    "case",
    [
        TEST_DATA_DIR / "cth_like_fixed_bdy.json",
        TEST_DATA_DIR / "cth_like_fixed_bdy_asym.json",
    ],
    ids=["symmetric", "asymmetric"],
)
def test_surface_curves_match_the_geometry_evaluator(case):
    """The cross-sections drawn by watch agree with vmecpp.geometry on every surface."""
    vmec_input = vmecpp.VmecInput.from_file(case)
    cpp_output = _vmecpp.run(
        vmec_input._to_cpp_vmecindata(), verbose=_vmecpp.OutputMode.SILENT
    )
    geometry = _vmecpp.make_geometry(cpp_output)
    jax_geometry = vmec_geometry.from_cpp(geometry)
    ns = geometry.dimensions.ns
    rows = _watch.surface_indices(ns, 6)
    assert rows[-1] == ns - 1
    theta = np.linspace(0.0, 2.0 * np.pi, 7)
    for zeta in (0.0, 0.37, np.pi / vmec_input.nfp):
        r, z = _watch.surface_curves(geometry, rows, theta, zeta)
        for i, j in enumerate(rows):
            for k, t in enumerate(theta):
                jet = np.asarray(
                    vmec_geometry.evaluate(
                        jax_geometry, np.array([j / (ns - 1), t, zeta])
                    )
                )
                assert abs(r[i, k] - jet[0, 0]) < 1.0e-12
                assert abs(z[i, k] - jet[1, 0]) < 1.0e-12
        r_axis, z_axis = _watch.magnetic_axis(geometry, zeta)
        axis = np.asarray(
            vmec_geometry.evaluate(jax_geometry, np.array([0.0, 0.0, zeta]))
        )
        assert abs(r_axis - axis[0, 0]) < 1.0e-12
        assert abs(z_axis - axis[1, 0]) < 1.0e-12


def test_watch_records_the_solve(tmp_path):
    vmec_input = vmecpp.VmecInput.from_file(SOLOVEV)
    path = tmp_path / "solve.gif"
    output = vmecpp.watch(vmec_input, save=path, every=25, show=False)
    reference = vmecpp.run(vmec_input, verbose=False)

    assert output.wout.ier_flag == 0
    np.testing.assert_array_equal(output.wout.rmnc, reference.wout.rmnc)
    np.testing.assert_array_equal(output.wout.fsqt, reference.wout.fsqt)

    iterations = output.wout.fsqt.size + len(vmec_input.ns_array)
    frames = iterations // 25 + (1 if iterations % 25 else 0)
    with Image.open(path) as image:
        assert image.n_frames == frames


def test_watch_without_a_window_or_a_file_is_refused():
    vmec_input = vmecpp.VmecInput.from_file(SOLOVEV)
    with pytest.raises(ValueError, match="nothing to show"):
        vmecpp.watch(vmec_input, show=False)


def test_closing_the_window_stops_the_solve(monkeypatch):
    vmec_input = vmecpp.VmecInput.from_file(SOLOVEV)
    drawn = []

    def window_open(_view):
        return len(drawn) < 3

    original = _watch._SolveView.draw

    def draw(self, snapshot, **kwargs):
        drawn.append(snapshot.iteration)
        original(self, snapshot, **kwargs)

    monkeypatch.setattr(_watch._SolveView, "window_open", window_open)
    monkeypatch.setattr(_watch._SolveView, "draw", draw)
    monkeypatch.setattr(
        _watch._SolveView, "_flush", lambda self: self.fig.canvas.draw()
    )
    monkeypatch.setattr(_watch, "_backend_is_interactive", lambda: True)
    output = vmecpp.watch(vmec_input, every=5, block=False)
    assert output.wout.ier_flag == 2
    assert output.wout.fsqt.size == 15
