# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Instead of specifying the profiles using different kinds of parametrizations like
VMEC2000 does, vmecpp also supports setting the radial quantities exactly on flux
surfaces.

This example demonstrates how to use the ``set_profile`` function.
"""

from pathlib import Path

import vmecpp

input_file = Path(__file__).parent / "data" / "input.solovev"
vmec_input = vmecpp.VmecInput.from_file(input_file)


def pressure_profile_fn(s):
    """A function that computes the pressure profile at different values of s."""
    return 1 - s**4 + s**2


vmec_input = vmecpp.set_profile(vmec_input, "pressure", pressure_profile_fn)
output = vmecpp.run(vmec_input)

print("Evaluated pressure profile:\n", output.wout.presf)
