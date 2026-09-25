# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The example examples/self_consistent_bootstrap_current.py: a Redl bootstrap current
made self-consistent through the callback of the Python-driven solve."""

import sys
from pathlib import Path

import numpy as np
from simsopt.mhd.profiles import ProfilePolynomial

import vmecpp

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "examples"))

import self_consistent_bootstrap_current as example  # type: ignore # noqa: E402

NE = 1.0e19 * np.array([1.0, 0.0, 0.0, 0.0, 0.0, -1.0])
TE = 4.0e3 * np.array([1.0, -1.0])
ZEFF = 1.0
HELICITY_N = -1


def _input(ns: int) -> vmecpp.VmecInput:
    return vmecpp.VmecInput.from_file(
        REPO_ROOT / "examples" / "data" / "input.nfp4_QH_warm_start"
    ).model_copy(
        update={
            "ns_array": np.asarray([ns]),
            "ftol_array": np.asarray([1.0e-11]),
            "niter_array": np.asarray([3000]),
            "pmass_type": "power_series",
            "am": example.kinetic_pressure(NE, TE, TE, ZEFF),
            "pres_scale": 1.0,
        }
    )


def test_the_equilibrium_carries_the_redl_current_by_simsopt_s_measure() -> None:
    """The closure reads the solver's real-space fields; SIMSOPT's
    VmecRedlBootstrapMismatch measure recomputes the Redl current from the wout spectrum
    of a separate solve with the resulting current profile.

    The two agree to a mismatch that falls with the radial resolution, while the zero-
    current equilibrium misses by the whole current.
    """
    ne = ProfilePolynomial(NE)
    te = ProfilePolynomial(TE)
    mismatches = []
    for ns in (16, 25):
        vmec_input = _input(ns)
        model, _ = example.solve_with_bootstrap_current(
            vmec_input, ne, te, te, ZEFF, HELICITY_N
        )
        buco = np.asarray(model.curr_h)
        output = vmecpp.run(
            example.with_current_profile(vmec_input, buco), verbose=False
        )
        np.testing.assert_allclose(
            np.asarray(output.wout.buco)[1:],
            buco,
            rtol=0.0,
            atol=1e-10 * abs(buco).max(),
        )
        mismatch = example.redl_mismatch(output, ne, te, te, ZEFF, HELICITY_N)
        mismatches.append(float(np.sqrt(np.sum(mismatch**2))))

        zero_current = vmecpp.run(vmec_input, verbose=False)
        zero_mismatch = example.redl_mismatch(
            zero_current, ne, te, te, ZEFF, HELICITY_N
        )
        assert np.sqrt(np.sum(zero_mismatch**2)) > 0.9

    assert mismatches[1] < mismatches[0] < 3.0e-2
