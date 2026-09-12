# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
from pathlib import Path

import numpy as np
import pytest

import vmecpp

TEST_DATA = Path(__file__).resolve().parents[1] / "src/vmecpp/cpp/vmecpp/test_data"


@pytest.mark.parametrize("max_threads", [1, 4])
def test_initial_jacobian_recovery_matches_fortran(max_threads):
    vmec_input = vmecpp.VmecInput.from_file(
        TEST_DATA / "initial_jacobian_recovery.json"
    )
    before = vmec_input.model_dump_json()
    reference = vmecpp.VmecWOut.from_wout_file(
        TEST_DATA / "wout_initial_jacobian_recovery.nc"
    )
    actual = vmecpp.run(vmec_input, max_threads=max_threads, verbose=False).wout

    assert vmec_input.model_dump_json() == before
    assert actual.ier_flag == 0
    assert actual.ns == vmec_input.ns_array[-1]
    assert max(actual.fsqr, actual.fsqz, actual.fsql) <= vmec_input.ftol_array[-1]
    for name in ("rmnc", "zmns", "lmns", "iotaf"):
        np.testing.assert_allclose(
            getattr(actual, name), getattr(reference, name), rtol=1e-8, atol=1e-9
        )
    np.testing.assert_array_equal(actual.presf, reference.presf)
    np.testing.assert_allclose(actual.phi, reference.phi, rtol=1e-14, atol=0)
