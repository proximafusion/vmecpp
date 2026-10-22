# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Regression tests for the CUDA execution path.

These require a CUDA-enabled build (-DVMECPP_USE_CUDA=ON) and an NVIDIA GPU and are
skipped in other builds. The W7-X cases carry the slow marker (pytest -m slow).

The references are the converged values of the CPU build (VMECPP_USE_CUDA off) of the
same commit, which the CUDA path reproduces iteration for iteration; the two agree on
them to 5e-10 relative or better. They are quantities of the solved interior: volume and
aspect ratio of a fixed-boundary run follow from its input boundary alone.
examples/cuda_cross_validation.py compares every output field of the two builds.
"""

import itertools
from pathlib import Path

import netCDF4
import numpy as np
import pytest

import vmecpp
from vmecpp.cpp import _vmecpp

REPO_ROOT = Path(__file__).parent.parent
TEST_DATA_DIR = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"
W7X_JSON = REPO_ROOT / "examples" / "data" / "w7x.json"

pytestmark = pytest.mark.skipif(
    not vmecpp.has_cuda(), reason="needs a build with VMECPP_USE_CUDA"
)

_REL = 1e-9

CMA = {
    "wb": 0.0011391226791305227,
    "raxis_cc0": 0.9994198576413145,
    "iota_axis": 0.570264288752222,
    "iota_edge": 0.6218448872334087,
    "rbtor": 0.47722561589354373,
}
# cma with its boundary scaled by 1.02
CMA_SCALED = {
    "wb": 0.001116786940324043,
    "raxis_cc0": 1.0194082547955607,
    "iota_axis": 0.5702642887440372,
    "iota_edge": 0.6218448872372695,
    "rbtor": 0.46786825087607187,
}
CTH_LIKE = {
    "wb": 0.0012834668943108437,
    "raxis_cc0": 0.7841884662485746,
    "iota_axis": 1.2427503576420111,
    "iota_edge": 0.8342641484308997,
    "rbtor": -0.4514075779540231,
    "rbtor0": -0.4613299766938168,
    "ctor": 43229.12865555726,
}
CTH_LIKE_ITERATIONS = 119
CTH_LIKE_FSQT = (0.06042998432095694, 0.038753752806344216, 0.06251715684555496)
# cth_like_fixed_bdy with the enclosed current set to zero from the first iteration
CTH_LIKE_ZERO_CURRENT = {
    "wb": 0.0012705664179153695,
    "raxis_cc0": 0.7855978015543246,
    "iota_axis": 0.2964248061428383,
    "iota_edge": 0.3553193105083082,
    "rbtor": -0.45662957807029814,
}
CTH_LIKE_ZERO_CURRENT_ITERATIONS = 150
# Free-boundary references: the boundary moves, so volume and aspect test the solve.
CTH_LIKE_FREE = {
    "wb": 0.0012836107618590692,
    "raxis_cc0": 0.7733364878403519,
    "iota_axis": 1.2689320157649617,
    "iota_edge": 0.8648074234240525,
    "rbtor": -0.4512378780387714,
    "volume": 0.3069922312210493,
    "aspect": 5.433239629580864,
}
CTH_LIKE_FREE_MULTIGRID = {
    "wb": 0.001283593620531752,
    "raxis_cc0": 0.7732583956350197,
    "iota_axis": 1.2633673053424452,
    "iota_edge": 0.8689511268445942,
    "rbtor": -0.4512153121231798,
    "volume": 0.3074173204816206,
    "aspect": 5.433276425079897,
}
W7X = {
    "wb": 1.858625986732257,
    "raxis_cc0": 5.644890855046078,
    "iota_axis": 0.8763391713648294,
    "iota_edge": 0.9513373357979356,
    "rbtor": -13.684233710613205,
}
W7X_ITERATIONS = 2788

# The quantities whose bit identity across runs and modes the tests require.
_INTERIOR_FIELDS = (
    "rmnc",
    "zmns",
    "lmns",
    "iotaf",
    "bmnc",
    "gmnc",
    "bsupumnc",
    "bsupvmnc",
    "wb",
    "rbtor",
)


def _quantities(wout) -> dict[str, float]:
    return {
        "wb": wout.wb,
        "raxis_cc0": wout.raxis_cc[0],
        "iota_axis": wout.iotaf[0],
        "iota_edge": wout.iotaf[-1],
        "rbtor": wout.rbtor,
        "rbtor0": wout.rbtor0,
        "ctor": wout.ctor,
        "b0": wout.b0,
        "volume": wout.volume,
        "aspect": wout.aspect,
    }


def _assert_reference(wout, reference: dict[str, float], rel: float = _REL) -> None:
    actual = _quantities(wout)
    for name, value in reference.items():
        assert actual[name] == pytest.approx(value, rel=rel), name


def _assert_bit_identical(actual, expected) -> None:
    for name in _INTERIOR_FIELDS:
        np.testing.assert_array_equal(
            np.asarray(getattr(actual, name)),
            np.asarray(getattr(expected, name)),
            err_msg=name,
        )


def _cma_input():
    return vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cma.json")


def _cth_like_input():
    return vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_fixed_bdy.json")


def _free_boundary_input(name: str = "cth_like_free_bdy"):
    vi = vmecpp.VmecInput.from_file(TEST_DATA_DIR / f"{name}.json")
    vi.mgrid_file = str(TEST_DATA_DIR / "mgrid_cth_like.nc")
    return vi


def _distinct_pair():
    vi = _cma_input()
    vi2 = vi.model_copy(deep=True)
    vi2.rbc = vi2.rbc * 1.02
    vi2.zbs = vi2.zbs * 1.02
    return [vi._to_cpp_vmecindata(), vi2._to_cpp_vmecindata()]


def test_single_cma_canonical():
    out = vmecpp.run(_cma_input(), max_threads=1, verbose=False)
    _assert_reference(out.wout, CMA)


def test_three_dimensional_trajectory_follows_the_cpu():
    # The CUDA path gates the m = 1 Z force and backs up the evaluated state as
    # the CPU path does, so a three-dimensional run takes the CPU's iterations;
    # the first residuals differ only by the summation order of the transforms.
    out = vmecpp.run(_cth_like_input(), max_threads=1, verbose=False)
    assert out.wout.itfsq == CTH_LIKE_ITERATIONS
    np.testing.assert_allclose(
        np.asarray(out.wout.fsqt)[1:4], CTH_LIKE_FSQT, rtol=1e-10
    )
    _assert_reference(out.wout, CTH_LIKE)


def test_toroidal_current_scalars_are_reported():
    # rbtor, rbtor0 and ctor come from the last force evaluation's bvcoH and
    # bucoH, which the end-of-run flush brings to the host.
    wout = vmecpp.run(_cth_like_input(), max_threads=1, verbose=False).wout
    for name in ("rbtor", "rbtor0", "ctor"):
        assert getattr(wout, name) == pytest.approx(CTH_LIKE[name], rel=_REL), name


def test_iteration_callback_receives_the_current_state():
    # The state the callback reads is the device state of that iteration, and a
    # callback that changes nothing leaves the run unchanged to the last bit.
    geometries = []

    def record(state):
        geometries.append(np.asarray(state.geometry.r_cc).copy())

    with_callback = vmecpp.run(
        _cth_like_input(), max_threads=1, verbose=False, iteration_callback=record
    )
    for earlier, later in itertools.pairwise(geometries[:6]):
        assert np.any(earlier != later)
    without = vmecpp.run(_cth_like_input(), max_threads=1, verbose=False)
    _assert_bit_identical(with_callback.wout, without.wout)


def test_iteration_callback_current_reaches_the_device():
    # A current the callback prescribes is staged to the device for the
    # following force evaluations, as on the CPU path.

    def zero_current(state):
        state.curr_h[:] = 0.0

    wout = vmecpp.run(
        _cth_like_input(), max_threads=1, verbose=False, iteration_callback=zero_current
    ).wout
    assert wout.itfsq == CTH_LIKE_ZERO_CURRENT_ITERATIONS
    assert abs(wout.ctor) < 1e-6 * abs(CTH_LIKE["ctor"])
    _assert_reference(wout, CTH_LIKE_ZERO_CURRENT)


def test_hot_restart_takes_its_own_state():
    # A hot restart that converges on its first evaluation reports the state it
    # was given, not the device state the previous run left behind.
    vi = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "solovev.json")
    base = vmecpp.run(vi, max_threads=1, verbose=False)
    rescaled = vmecpp.rescale(base, b_scale=1.5, r_scale=2.0, scale_pressure=True)
    np.testing.assert_allclose(
        np.asarray(rescaled.wout.rmnc), 2.0 * np.asarray(base.wout.rmnc), rtol=1e-12
    )


def test_geometry_cap_freezes_the_capped_modes():
    # mpol_geometry and ntor_geometry hold R and Z above the cap at zero on the
    # device state while lambda is resolved there.
    vi = _cma_input().model_copy(
        update={
            "mpol_geometry": 3,
            "ntor_geometry": 4,
            "ns_array": np.array([25]),
            "ftol_array": np.array([1e-12]),
            "niter_array": np.array([80]),
            "return_outputs_even_if_not_converged": True,
        }
    )
    wout = vmecpp.run(vi, max_threads=1, verbose=False).wout
    xm, xn = np.asarray(wout.xm), np.asarray(wout.xn)
    above = (xm >= 3) | (np.abs(xn) // vi.nfp > 4)
    assert np.max(np.abs(np.asarray(wout.rmnc)[above])) == 0.0
    assert np.max(np.abs(np.asarray(wout.zmns)[above])) == 0.0
    assert np.max(np.abs(np.asarray(wout.lmns)[above])) > 1e-10


def test_repeated_run_bit_identical():
    # The same input run twice in one process returns bitwise-identical
    # results; the persistent device state carries nothing between runs.
    first = vmecpp.run(_cma_input(), max_threads=1, verbose=False)
    second = vmecpp.run(_cma_input(), max_threads=1, verbose=False)
    _assert_bit_identical(second.wout, first.wout)


def test_broadcast_returns_single_output():
    # Broadcast mode solves one boundary in every slot and returns the
    # single converged result. This path also exercises the iteration-1
    # magnetic-axis recovery, whose re-staged device state must reach
    # every configuration slot.
    vi = _cma_input()
    outs = _vmecpp.run_batched_gpu(
        [vi._to_cpp_vmecindata()] * 4, max_threads=1, verbose=_vmecpp.OutputMode.SILENT
    )
    assert len(outs) == 1
    _assert_reference(vmecpp.VmecWOut._from_cpp_wout(outs[0].wout), CMA)


def test_two_call_distinct_identical():
    # Two consecutive distinct-mode calls in one process must return
    # bitwise-identical converged spectra: the second call restages its
    # inputs from clean device state.
    outs1, sp1 = _vmecpp.run_batched_gpu(
        _distinct_pair(),
        max_threads=1,
        verbose=_vmecpp.OutputMode.SILENT,
        return_spectra=True,
        distinct=True,
    )
    outs2, sp2 = _vmecpp.run_batched_gpu(
        _distinct_pair(),
        max_threads=1,
        verbose=_vmecpp.OutputMode.SILENT,
        return_spectra=True,
        distinct=True,
    )
    assert len(outs1) == 2
    first = [vmecpp.VmecWOut._from_cpp_wout(o.wout) for o in outs1]
    second = [vmecpp.VmecWOut._from_cpp_wout(o.wout) for o in outs2]
    _assert_reference(first[0], CMA)
    _assert_reference(first[1], CMA_SCALED)
    assert np.array_equal(sp1, sp2)
    for actual, expected in zip(second, first, strict=True):
        _assert_bit_identical(actual, expected)


def test_run_batch_distinct_api():
    # The high-level run_batch wrapper solves each input as its own
    # equilibrium and returns one VmecOutput per input.
    vi = _cma_input()
    vi2 = vi.model_copy(deep=True)
    vi2.rbc = vi2.rbc * 1.02
    vi2.zbs = vi2.zbs * 1.02
    outs = vmecpp.run_batch([vi, vi2], max_threads=1)
    assert len(outs) == 2
    _assert_reference(outs[0].wout, CMA)
    _assert_reference(outs[1].wout, CMA_SCALED)

    # A single-input broadcast call returns exactly one output.
    single = vmecpp.run_batch([vi], distinct=False, max_threads=1)
    assert len(single) == 1
    _assert_reference(single[0].wout, CMA)


def test_distinct_batch_raises_for_an_unconverged_configuration():
    # A configuration that misses the last stage's tolerance raises unless
    # return_outputs_even_if_not_converged is set.
    vi = _cma_input()
    vi2 = vi.model_copy(deep=True)
    vi2.rbc[1, vi.ntor] *= 1.05
    vi2.zbs[1, vi.ntor] *= 1.05
    singles = [vmecpp.run(v, max_threads=1, verbose=False) for v in (vi, vi2)]
    counts = [single.wout.niter for single in singles]
    fast = min(counts)
    assert fast < max(counts)
    niter_array = np.array(vi.niter_array)
    niter_array[-1] = fast - 1
    short = [v.model_copy(update={"niter_array": niter_array}) for v in (vi, vi2)]
    with pytest.raises(RuntimeError, match="did not converge"):
        vmecpp.run_batch(short, max_threads=1)
    best_effort = vmecpp.run_batch(
        [
            v.model_copy(update={"return_outputs_even_if_not_converged": True})
            for v in short
        ],
        max_threads=1,
    )
    assert [out.wout.niter for out in best_effort] == [fast, fast]
    slow = best_effort[counts.index(max(counts))].wout
    assert max(slow.fsqr, slow.fsqz, slow.fsql) > slow.ftolv


def test_distinct_batch_outputs_carry_their_own_traces():
    # Each output of a distinct batch carries its configuration's residual
    # trace and iteration count.
    vi = _cma_input()
    vi2 = vi.model_copy(deep=True)
    vi2.rbc[1, vi.ntor] *= 1.05
    vi2.zbs[1, vi.ntor] *= 1.05
    singles = [vmecpp.run(v, max_threads=1, verbose=False) for v in (vi, vi2)]
    outs = vmecpp.run_batch([vi, vi2], max_threads=1)
    for out, single in zip(outs, singles, strict=True):
        assert out.wout.itfsq == len(out.wout.restart_reason_timetrace) > 1
        assert out.wout.fsqt[0] == pytest.approx(single.wout.fsqt[0], rel=1e-10)
    assert outs[0].wout.fsqt[0] != pytest.approx(outs[1].wout.fsqt[0], rel=1e-3)
    assert outs[0].wout.niter != outs[1].wout.niter


def test_mixed_configuration_counts_one_process():
    # distinct N=2, broadcast N=4, distinct N=2, single: the
    # per-configuration buffers reallocate across the count changes and
    # every run returns its fresh-process result.
    outs = _vmecpp.run_batched_gpu(
        _distinct_pair(),
        max_threads=1,
        verbose=_vmecpp.OutputMode.SILENT,
        distinct=True,
    )
    _assert_reference(vmecpp.VmecWOut._from_cpp_wout(outs[1].wout), CMA_SCALED)

    vi = _cma_input()
    outs = _vmecpp.run_batched_gpu(
        [vi._to_cpp_vmecindata()] * 4, max_threads=1, verbose=_vmecpp.OutputMode.SILENT
    )
    assert len(outs) == 1
    _assert_reference(vmecpp.VmecWOut._from_cpp_wout(outs[0].wout), CMA)

    outs = _vmecpp.run_batched_gpu(
        _distinct_pair(),
        max_threads=1,
        verbose=_vmecpp.OutputMode.SILENT,
        distinct=True,
    )
    _assert_reference(vmecpp.VmecWOut._from_cpp_wout(outs[0].wout), CMA)
    _assert_reference(vmecpp.VmecWOut._from_cpp_wout(outs[1].wout), CMA_SCALED)

    out = vmecpp.run(_cma_input(), max_threads=1, verbose=False)
    _assert_reference(out.wout, CMA)


def test_scope_guards_and_memory_preflight():
    vi = _cma_input()

    vi_theta = vi.model_copy(deep=True)
    vi_theta.ntheta = 600  # nThetaReduced = ntheta/2 + 1 = 301 > 256 device cap
    with pytest.raises(Exception, match="nThetaReduced"):
        vmecpp.run(vi_theta, max_threads=1, verbose=False)

    vi_ns = vi.model_copy(deep=True)
    # Beyond CudaMaxRadialResolution() on any device (the PCR/block-Thomas
    # solver ceiling), so the radial-resolution scope guard rejects it.
    vi_ns.ns_array = np.array([100000])
    vi_ns.ftol_array = np.array([1e-6])
    vi_ns.niter_array = np.array([10])
    with pytest.raises(Exception, match="radial resolution"):
        vmecpp.run(vi_ns, max_threads=1, verbose=False)

    # The memory pre-flight rejects a batch whose device allocation exceeds the
    # free memory before allocating: 13000 slots at ns = 5 on a 200 x 100
    # angular grid need several hundred gigabytes.
    vi_big = vi.model_copy(
        update={
            "ntheta": 200,
            "nzeta": 100,
            "ns_array": np.array([5]),
            "ftol_array": np.array([1e-6]),
            "niter_array": np.array([10]),
        }
    )
    with pytest.raises(RuntimeError, match="free memory"):
        _vmecpp.run_batched_gpu(
            [vi_big._to_cpp_vmecindata()] * 13000,
            max_threads=1,
            verbose=_vmecpp.OutputMode.SILENT,
        )

    # The batched kernels index surface and configuration in one grid dimension
    # of at most 65535 blocks: 59 configurations at ns = 1101.
    vi_wide = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "solovev.json").model_copy(
        update={
            "ns_array": np.array([1101]),
            "ftol_array": np.array([1e-12]),
            "niter_array": np.array([10]),
        }
    )
    with pytest.raises(RuntimeError, match="holds at most 59 configurations"):
        vmecpp.run_batch([vi_wide] * 60, max_threads=1)

    # The same process recovers and runs normally after the rejections.
    out = vmecpp.run(vi, max_threads=1, verbose=False)
    _assert_reference(out.wout, CMA)


def test_free_boundary_multigrid_stage_transition():
    # The multigrid stage transition with the vacuum pressure already
    # active reallocates the device buffers mid-run; the constraint
    # origins enter the new stage zero-initialized because
    # rzConIntoVolume does not run there.
    out = vmecpp.run(
        _free_boundary_input("cth_like_free_bdy_multigrid"),
        max_threads=1,
        verbose=False,
    )
    _assert_reference(out.wout, CTH_LIKE_FREE_MULTIGRID)


def test_free_boundary_cth_like():
    out = vmecpp.run(_free_boundary_input(), max_threads=1, verbose=False)
    _assert_reference(out.wout, CTH_LIKE_FREE)


def test_free_boundary_wout_fields_match_reference():
    # Converged free-boundary wout against the committed reference file, at
    # the tolerance the CPU build is held to (tests/test_init.py); like the
    # reference, the run restarts from the advanced state.
    vi = _free_boundary_input()
    vi.backup_evaluated_state = False
    out = vmecpp.run(vi, max_threads=1, verbose=False)
    ref = netCDF4.Dataset(TEST_DATA_DIR / "wout_cth_like_free_bdy.nc")

    fields = [
        ("volume_p", "volume"),
        ("aspect", "aspect"),
        ("wb", "wb"),
        ("b0", "b0"),
        ("betatotal", "betatotal"),
        ("rbtor", "rbtor"),
        ("iotaf", "iotaf"),
        ("presf", "presf"),
        ("phipf", "phipf"),
        ("jcuru", "jcuru"),
        ("jcurv", "jcurv"),
        ("q_factor", "q_factor"),
        ("specw", "specw"),
    ]
    compared = 0
    for nc_name, attr in fields:
        if nc_name not in ref.variables or not hasattr(out.wout, attr):
            continue
        expected = np.asarray(ref[nc_name][...], dtype=float)
        actual = np.asarray(getattr(out.wout, attr), dtype=float)
        assert expected.shape == actual.shape, (nc_name, expected.shape, actual.shape)
        np.testing.assert_allclose(
            actual, expected, rtol=1e-6, atol=1e-7, err_msg=nc_name
        )
        compared += 1
    assert compared >= 8, f"only {compared} fields compared"


def test_free_boundary_broadcast_matches_single():
    # Free-boundary broadcast: per-configuration vacuum solves under a
    # batch-wide activation gate and ivacskip cadence. Configuration
    # zero's result agrees with the single run.
    vi = _free_boundary_input()
    single = vmecpp.run(vi, max_threads=1, verbose=False)
    outs = _vmecpp.run_batched_gpu(
        [vi._to_cpp_vmecindata()] * 2, max_threads=1, verbose=_vmecpp.OutputMode.SILENT
    )
    assert len(outs) == 1
    assert outs[0].wout.volume == pytest.approx(single.wout.volume, rel=1e-10)
    assert outs[0].wout.wb == pytest.approx(single.wout.wb, rel=1e-10)


def test_free_boundary_distinct_two_boundaries():
    # Distinct free-boundary batch: two boundaries against one coil set
    # converge to their own equilibria in one batched run. The batch
    # shares the vacuum activation iteration and the NESTOR cadence.
    a = _free_boundary_input()
    b = a.model_copy(deep=True)
    b.rbc = b.rbc * 1.005
    b.zbs = b.zbs * 1.005
    outs = _vmecpp.run_batched_gpu(
        [a._to_cpp_vmecindata(), b._to_cpp_vmecindata()],
        max_threads=1,
        verbose=_vmecpp.OutputMode.SILENT,
        distinct=True,
    )
    assert len(outs) == 2
    assert outs[0].wout.volume == pytest.approx(CTH_LIKE_FREE["volume"], rel=5e-4)
    assert outs[1].wout.volume > outs[0].wout.volume


def test_fixed_then_free_boundary_one_process():
    # A fixed-boundary run followed by a free-boundary run in the same
    # process: the vacuum block's host-side consumers (the edge-pressure
    # extrapolation reads presH) must see flushed device values, not
    # whatever the reallocated host arrays inherited from the prior run.
    vmecpp.run(_cma_input(), max_threads=1, verbose=False)
    out = vmecpp.run(_free_boundary_input(), max_threads=1, verbose=False)
    _assert_reference(out.wout, CTH_LIKE_FREE)


def test_hot_restart_reproduces_converged_state():
    # A run restarted from its own converged output terminates at once, at
    # the same equilibrium. Hot restart requires a single-stage ns_array.
    vi = _cma_input()
    vi.ns_array = vi.ns_array[-1:]
    vi.ftol_array = vi.ftol_array[-1:]
    vi.niter_array = vi.niter_array[-1:]
    cold = vmecpp.run(vi, max_threads=1, verbose=False)
    hot = vmecpp.run(vi, max_threads=1, verbose=False, restart_from=cold)
    assert hot.wout.itfsq == 0
    for name in _INTERIOR_FIELDS:
        expected = np.asarray(getattr(cold.wout, name))
        np.testing.assert_allclose(
            np.asarray(getattr(hot.wout, name)),
            expected,
            rtol=0.0,
            atol=1e-12 * np.max(np.abs(expected)),
            err_msg=name,
        )


@pytest.mark.slow
def test_w7x_bad_jacobian_recovery_converges():
    # W7-X exercises the mid-run restarts; with the evaluated state backed up
    # the CUDA path rewinds to the CPU's states and takes its iterations.
    out = vmecpp.run(vmecpp.VmecInput.from_file(W7X_JSON), max_threads=1, verbose=False)
    assert out.wout.itfsq == W7X_ITERATIONS
    _assert_reference(out.wout, W7X)


# ---------------------------------------------------------------------------
# Axisymmetric (ntor = 0) coverage. The 2D path runs nZeta = 1 through the
# 3D-symmetric device kernels and bypasses cuFFT; it agrees with the CPU
# build to 1e-14 across the single-grid, multigrid (solovev ns_array =
# [5, 11, 55]), and batched modes.
# ---------------------------------------------------------------------------
SOLOVEV = {
    "wb": 0.0645507521742048,
    "raxis_cc0": 3.989401072835298,
    "b0": 0.20333105580548258,
    "rbtor": 0.7230383356305398,
}
# solovev with its pressure scaled by 1.05
SOLOVEV_PRESSURE = {"raxis_cc0": 3.9894012650815647, "b0": 0.2033310263187132}
CIRCULAR_TOKAMAK = {
    "wb": 172.39494071067708,
    "raxis_cc0": 6.132188475454839,
    "b0": 5.241673569885228,
    "rbtor": 31.330680063374572,
}


def _solovev_input():
    return vmecpp.VmecInput.from_file(TEST_DATA_DIR / "solovev.json")


def _circular_tokamak_input():
    return vmecpp.VmecInput.from_file(TEST_DATA_DIR / "circular_tokamak.json")


def test_ntor0_solovev_canonical():
    out = vmecpp.run(_solovev_input(), max_threads=1, verbose=False)
    _assert_reference(out.wout, SOLOVEV)


def test_ntor0_circular_tokamak_canonical():
    # A second axisymmetric boundary (mpol = 8, single-grid ns = 17).
    out = vmecpp.run(_circular_tokamak_input(), max_threads=1, verbose=False)
    _assert_reference(out.wout, CIRCULAR_TOKAMAK)


def test_ntor0_repeated_run_bit_identical():
    first = vmecpp.run(_solovev_input(), max_threads=1, verbose=False)
    second = vmecpp.run(_solovev_input(), max_threads=1, verbose=False)
    _assert_bit_identical(second.wout, first.wout)


def test_ntor0_distinct_batch():
    # Batched distinct-mode axisymmetric run: each configuration converges to
    # its own single-run result.
    vi = _solovev_input()
    vi2 = vi.model_copy(deep=True)
    vi2.pres_scale = vi2.pres_scale * 1.05
    outs = _vmecpp.run_batched_gpu(
        [vi._to_cpp_vmecindata(), vi2._to_cpp_vmecindata()],
        max_threads=1,
        verbose=_vmecpp.OutputMode.SILENT,
        distinct=True,
    )
    assert len(outs) == 2
    _assert_reference(vmecpp.VmecWOut._from_cpp_wout(outs[0].wout), SOLOVEV)
    _assert_reference(vmecpp.VmecWOut._from_cpp_wout(outs[1].wout), SOLOVEV_PRESSURE)


def test_ntor0_broadcast_returns_single_output():
    # Broadcast mode with an axisymmetric seed returns the single converged
    # result, the same contract as the 3D broadcast path.
    vi = _solovev_input()
    outs = _vmecpp.run_batched_gpu(
        [vi._to_cpp_vmecindata()] * 3,
        max_threads=1,
        verbose=_vmecpp.OutputMode.SILENT,
    )
    assert len(outs) == 1
    _assert_reference(vmecpp.VmecWOut._from_cpp_wout(outs[0].wout), SOLOVEV)
