# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""A callback stop must return fields evaluated at the returned geometry."""

import numpy as np

import vmecpp
from vmecpp.cpp import _vmecpp  # type: ignore


def test_callback_stop_refreshes_final_fields_without_another_step():
    # A manufactured elliptic torus, deliberately stopped before convergence.
    rbc = np.zeros((4, 1))
    rbc[0, 0] = 3.0
    rbc[1, 0] = 1.0
    zbs = np.zeros((4, 1))
    zbs[1, 0] = 1.5
    vmec_input = vmecpp.VmecInput(
        mpol=4,
        ntor=0,
        ns_array=np.array([9]),
        niter_array=np.array([1]),
        ftol_array=np.array([1.0e-30]),
        ntheta=16,
        nzeta=1,
        phiedge=1.0,
        ai=np.array([0.4]),
        am=np.array([0.0]),
        raxis_c=np.array([3.0]),
        rbc=rbc,
        zbs=zbs,
        return_outputs_even_if_not_converged=True,
    )
    seen = []

    def stop(state):
        seen.append(state.iteration)
        return False

    stopped = _vmecpp.run(
        vmec_input._to_cpp_vmecindata(),
        max_threads=1,
        verbose=_vmecpp.OutputMode.SILENT,
        iteration_callback=stop,
    ).wout
    # Ordinary budget exhaustion already refreshes the final force evaluation.
    reference = _vmecpp.run(
        vmec_input._to_cpp_vmecindata(),
        max_threads=1,
        verbose=_vmecpp.OutputMode.SILENT,
    ).wout

    assert seen == [1]
    assert stopped.ier_flag == 2
    for name in ("rmnc", "zmns", "lmns"):
        np.testing.assert_array_equal(getattr(stopped, name), getattr(reference, name))
    for name in (
        "fsqr",
        "fsqz",
        "fsql",
        "gmnc",
        "bmnc",
        "bsupumnc",
        "bsupvmnc",
        "bsubumnc",
        "bsubvmnc",
        "force_residual_r",
        "force_residual_z",
        "force_residual_lambda",
    ):
        np.testing.assert_array_equal(
            getattr(stopped, name), getattr(reference, name), err_msg=name
        )
