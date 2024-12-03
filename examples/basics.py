# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""How to run VMEC++ via the Python API."""

import vmecpp
from vmecpp._util import package_root

TEST_DATA_DIR = package_root() / "cpp" / "vmecpp" / "test_data"


def simple_run():
    # We need a VmecInput, a Python object that corresponds
    # to the classic "input.*" files.
    # We can construct it from such a classic VMEC++ input file:
    input_file = TEST_DATA_DIR / "input.solovev"
    input = vmecpp.VmecInput.from_file(input_file)

    # Now we can run VMEC++:
    output = vmecpp.run(input)
    # By default, VMEC++ runs with max_threads equal
    # to the number of logical cores on the machine.
    # The optional parameter max_threads=N controls
    # the level of parallelism in VMEC++. Note that
    # the actual level of parallelism is limited so
    # that each thread operates on at least two flux
    # surfaces, so VMEC++ might use less threads than
    # max_threads if there are few flux surfaces.

    # We can save the output wout as a classic wout
    # file if needed:
    output.wout.save("wout_solovev.nc")

    # Free-boundary runs work just the same, in which
    # case VmecInput will also include a path to an
    # "mgrid_*.nc" file produced by MAKEGRID.


if __name__ == "__main__":
    simple_run()
