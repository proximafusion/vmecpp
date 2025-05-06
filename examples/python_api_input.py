# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""How to create a VmecInput object via the Python API."""

import numpy as np

import vmecpp

# We can construct a VmecInput object completely from Python.
# As a starting point, we can use the VMEC2000 defaults.
vmec_input = vmecpp.VmecInput.default()
print("default = \n", vmec_input.to_json(indent=2))

# Let's set a pressure profile that decays linearly from 0.125 to zero.
vmec_input.am = np.array([0.125, -0.125])
# ...a constant rotational transform of 1.0
vmec_input.ai = np.array([1.0])
# ...and increase the number of iterations so it has time to converge.
vmec_input.niter_array = np.array([200])

# Finally we construct boundary geometry that is approximately the Solovev
# equilibrium. There are helper methods for explicitly resizing coefficient
# arrays to the correct shape by padding them with zeros left and right.
vmec_input.rbc = vmecpp.VmecInput.resize_2d_coeff(
    np.array([[4.0], [1.0], [-0.068]]),
    vmec_input.mpol,
    vmec_input.ntor,
)
vmec_input.zbs = vmecpp.VmecInput.resize_2d_coeff(
    np.array([[0], [1.58], [0.01]]),
    vmec_input.mpol,
    vmec_input.ntor,
)

# Now it is time to run the equilibrium solver and inspect the results.
output = vmecpp.run(vmec_input)
# Once the residual force is small enough, the equilibrium is converged.
print(
    "Residual force: ",
    output.wout.fsqr,
    "is below the threshold of",
    vmec_input.ftol_array,
)
