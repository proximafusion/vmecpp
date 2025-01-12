# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Tests for VMEC++'s Python API.

Here we just test that the Python bindings and the general API works as expected.
Physics correctness is checked at the level of the C++ core.
"""

from pathlib import Path

import pytest

import vmecpp

# We don't want to install tests and test data as part of the package,
# but scikit-build-core + hatchling does not support editable installs,
# so the tests live in the sources but the vmecpp module lives in site_packages.
# Therefore, in order to find the test data we use the relative path to this file.
# I'm very open to alternative solutions :)
REPO_ROOT = Path(__file__).parent.parent
TEST_DATA_DIR = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"


@pytest.mark.parametrize(
    ("max_threads", "input_file", "verbose"),
    [(None, "cma.json", True), (1, "input.cma", False)],
)
def test_run(max_threads, input_file, verbose):
    """Test that the Python API works with different combinations of parameters."""

    input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / input_file)
    out = vmecpp.run(input, max_threads=max_threads, verbose=verbose)

    assert out.wout is not None


# We trust the C++ tests to cover the hot restart functionality properly,
# here we just want to test that the Python API for it works.
def test_run_with_hot_restart():
    input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cma.json")

    # base run
    out = vmecpp.run(input, verbose=False)

    # now with hot restart
    # (only a single multigrid step is supported)
    input.ns_array = input.ns_array[-1:]
    input.ftol_array = input.ftol_array[-1:]
    input.niter_array = input.niter_array[-1:]
    hot_restarted_out = vmecpp.run(input, verbose=False, restart_from=out)

    assert hot_restarted_out.wout.niter == 1
