# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
from pathlib import Path

import numpy as np

import vmecpp

REPO_ROOT = Path(__file__).parent.parent


def test_evaluated_backup_converges_faster_to_the_same_state():
    """Restart backups of the evaluated state take fewer iterations on W7-X.

    The two modes restart from different states, so the converged state differs at the
    level of the path dependence (2.4e-5 relative in R, Z and 1.6e-5 in iota at ftol
    1e-12), not at the level of ftol.
    """

    def run(backup_evaluated_state: bool) -> vmecpp.VmecWOut:
        vmec_input = vmecpp.VmecInput.from_file(REPO_ROOT / "examples/data/w7x.json")
        vmec_input.ftol_array = np.full_like(vmec_input.ftol_array, 1.0e-12)
        vmec_input.backup_evaluated_state = backup_evaluated_state
        return vmecpp.run(vmec_input, verbose=False).wout

    evaluated = run(True)
    advanced = run(False)

    assert evaluated.niter < advanced.niter
    scale = np.abs(advanced.rmnc).max()
    np.testing.assert_allclose(evaluated.rmnc, advanced.rmnc, rtol=0, atol=1e-4 * scale)
    np.testing.assert_allclose(evaluated.zmns, advanced.zmns, rtol=0, atol=1e-4 * scale)
    np.testing.assert_allclose(evaluated.iotaf, advanced.iotaf, rtol=0, atol=1e-4)
