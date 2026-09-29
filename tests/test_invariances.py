# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Invariances of the solver whose answer a property of the representation fixes.

Each test solves one equilibrium along two paths that describe the same field and
compares the Fourier coefficients of the two runs directly.
"""

from pathlib import Path

import numpy as np
import pytest

import vmecpp

REPO_ROOT = Path(__file__).parent.parent
TEST_DATA_DIR = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"


def _run(vmec_input, restart_from=None):
    return vmecpp.run(
        vmec_input, max_threads=1, verbose=False, restart_from=restart_from
    )


def _converged(vmec_input, ftol=1e-13):
    """The input with every multigrid step run to ftol."""
    shape = np.asarray(vmec_input.ftol_array).shape
    return vmec_input.model_copy(
        update={
            "ftol_array": np.full(shape, ftol),
            "niter_array": np.full(shape, 20000, dtype=np.int64),
        }
    )


def _mode_index(wout):
    """(m, n) -> column of the coefficient arrays, n in absolute toroidal numbers."""
    xm = np.asarray(wout.xm).astype(int)
    xn = np.asarray(wout.xn).astype(int)
    return {(int(m), int(n)): k for k, (m, n) in enumerate(zip(xm, xn, strict=True))}


def _assert_coefficients_match(ref, test, ref_rows, test_rows, rtol):
    """Rmnc, zmns and lmns agree on matching modes, relative to the largest coefficient
    of the block."""
    for name in ("rmnc", "zmns", "lmns"):
        a = np.asarray(getattr(ref, name))[ref_rows]
        b = np.asarray(getattr(test, name))[test_rows]
        scale = np.max(np.abs(np.asarray(getattr(ref, name))))
        np.testing.assert_allclose(b, a, rtol=0, atol=rtol * scale, err_msg=name)


def _assert_coefficients_vanish(wout, rows, rtol):
    """The named modes carry nothing, relative to the largest coefficient."""
    for name in ("rmnc", "zmns", "lmns"):
        block = np.asarray(getattr(wout, name))
        scale = np.max(np.abs(block))
        assert np.max(np.abs(block[rows])) <= rtol * scale, name


def test_axisymmetric_input_through_the_three_dimensional_path():
    """An axisymmetric boundary solved with ntor = 4 reproduces the ntor = 0 run, with
    every n != 0 coefficient zero: every toroidal derivative of an axisymmetric field is
    zero exactly."""
    base = _converged(
        vmecpp.VmecInput.from_file(TEST_DATA_DIR / "circular_tokamak.json")
    )
    ntor = 4
    rbc = np.zeros((base.mpol, 2 * ntor + 1))
    zbs = np.zeros((base.mpol, 2 * ntor + 1))
    rbc[:, ntor] = np.asarray(base.rbc)[:, 0]
    zbs[:, ntor] = np.asarray(base.zbs)[:, 0]
    raxis_c = np.zeros(ntor + 1)
    raxis_c[0] = np.asarray(base.raxis_c)[0]
    three_d = base.model_copy(
        update={
            "ntor": ntor,
            "rbc": rbc,
            "zbs": zbs,
            "raxis_c": raxis_c,
            "zaxis_s": np.zeros(ntor + 1),
        }
    )
    ref = _run(base).wout
    test = _run(three_d).wout
    idx = _mode_index(test)
    n0 = [idx[(m, 0)] for m in range(base.mpol)]
    others = [k for k in range(len(idx)) if k not in n0]
    assert len(idx) == base.mpol * (2 * ntor + 1) - ntor
    _assert_coefficients_vanish(test, others, rtol=1e-12)
    _assert_coefficients_match(ref, test, slice(None), n0, rtol=1e-8)
    np.testing.assert_allclose(
        np.asarray(test.iotas), np.asarray(ref.iotas), rtol=0, atol=1e-12
    )


def test_one_field_period_and_the_whole_torus_agree():
    """A five-period stellarator solved over one period and over the whole torus, with
    the modes written in absolute toroidal numbers and the same toroidal sample points,
    gives the same coefficients, and every mode the torus admits and the period does not
    is zero."""
    base = _converged(
        vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_fixed_bdy.json")
    )
    nfp, ntor, mpol = base.nfp, base.ntor, base.mpol
    ntor_torus = nfp * ntor
    rbc = np.zeros((mpol, 2 * ntor_torus + 1))
    zbs = np.zeros((mpol, 2 * ntor_torus + 1))
    for n in range(-ntor, ntor + 1):
        rbc[:, ntor_torus + nfp * n] = np.asarray(base.rbc)[:, ntor + n]
        zbs[:, ntor_torus + nfp * n] = np.asarray(base.zbs)[:, ntor + n]
    raxis_c = np.zeros(ntor_torus + 1)
    zaxis_s = np.zeros(ntor_torus + 1)
    for n in range(ntor + 1):
        raxis_c[nfp * n] = np.asarray(base.raxis_c)[n]
        zaxis_s[nfp * n] = np.asarray(base.zaxis_s)[n]
    torus = base.model_copy(
        update={
            "nfp": 1,
            "ntor": ntor_torus,
            "nzeta": nfp * base.nzeta,
            "rbc": rbc,
            "zbs": zbs,
            "raxis_c": raxis_c,
            "zaxis_s": zaxis_s,
        }
    )
    ref = _run(base).wout
    test = _run(torus).wout
    idx = _mode_index(test)
    ref_idx = _mode_index(ref)
    shared = [idx[mn] for mn in ref_idx]
    others = [k for k in range(len(idx)) if k not in set(shared)]
    assert len(shared) == len(ref_idx)
    assert len(others) == len(idx) - len(shared)
    _assert_coefficients_match(ref, test, slice(None), shared, rtol=1e-8)
    _assert_coefficients_vanish(test, others, rtol=1e-10)
    np.testing.assert_allclose(
        np.asarray(test.iotas), np.asarray(ref.iotas), rtol=0, atol=1e-10
    )


def test_restart_with_the_gauge_mode_of_lambda_set():
    """A hot restart from a converged state whose (0,0) lambda coefficient is set
    converges at once to the same equilibrium: every angular derivative of lambda
    carries a factor m or n, so that coefficient drives nothing. The same restart
    with the (1,0) coefficient set instead iterates to convergence again. The
    restart reads lambda from lmns_full."""
    base = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "solovev.json")
    base = _converged(
        base.model_copy(
            update={
                "ns_array": np.asarray(base.ns_array)[-1:],
                "ftol_array": np.asarray(base.ftol_array)[-1:],
                "niter_array": np.asarray(base.niter_array)[-1:],
            }
        )
    )
    first = _run(base)
    idx = _mode_index(first.wout)

    def restarted(mode, amount):
        state = first.model_copy(deep=True)
        lmns_full = np.array(state.wout.lmns_full)
        lmns_full[idx[mode], :] += amount
        state.wout.lmns_full = lmns_full
        return _run(base, restart_from=state).wout

    gauge = restarted((0, 0), 0.05)
    _assert_coefficients_match(first.wout, gauge, slice(None), slice(None), rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(gauge.iotas), np.asarray(first.wout.iotas), rtol=0, atol=1e-12
    )
    assert gauge.niter <= 5
    physical = restarted((1, 0), 0.05)
    assert physical.niter > 50


if __name__ == "__main__":
    pytest.main([__file__])
