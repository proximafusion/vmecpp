# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""How to modify a VmecInput object via the Python API."""

from pathlib import Path

import vmecpp


def modify_input_and_run():
    input_file = Path(__file__).parent / "data" / "input.solovev"
    input = vmecpp.VmecInput.from_file(input_file)

    # Since VmecInput is a Python object, we can easily
    # interact with it, e.g. by modifying the coefficients:
    input.rbc *= 1.1
    input.zbs *= 1.1

    # Note that the shapes of input arrays are validated and handled strictly,
    # so data isn't implicitly ignored or padded. In this example VmecInput
    # fails validation, because rbc has the wrong shape (2,1) when a configuration
    # with mpol=6 requires rbc.shape == (6,1):
    input.rbc = [[4.4], [1.13]]
    try:
        vmecpp.run(input)
    except ValueError as e:
        print(f"As expected: passing invalid shape arrays triggers the exception:\n{e}")

    # There are helper methods for explicitly resizing coefficient arrays to the correct shape
    input.rbc = vmecpp.VmecInput.resize_2d_coeff(input.rbc, input.mpol, input.ntor)
    vmecpp.run(input)


if __name__ == "__main__":
    modify_input_and_run()
