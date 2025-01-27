# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Run VMEC++ via the Python API and take snapshots along the run."""

from pathlib import Path

import numpy as np

import vmecpp

cache_folder = Path("/home/jons/results/vmec_w7x/movie_cache")
Path.mkdir(cache_folder, parents=True, exist_ok=True)

# NOTE: This resolves to src/vmecpp/cpp/vmecpp/test_data in the repo.
TEST_DATA_DIR = (
    Path(__file__).parent.parent / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"
)

# input_file = TEST_DATA_DIR / "w7x.json"
input_file = TEST_DATA_DIR / "w7x_generic_initial_guess.json"
input = vmecpp.VmecInput.from_file(input_file)

# adjust as needed - we don't vendor the mgrid file, since it is too large
input.mgrid_file = "/home/jons/results/vmec_w7x/mgrid_w7x.nc"
# input.mgrid_file = "/home/jons/results/vmec_w7x/mgrid_w7x_nv72.nc"

input.return_outputs_even_if_not_converged = True

# # higher-res for nicer plots
# input.ntheta = 100
# input.nzeta = 72

maximum_iterations = 20000

# step = 100
step = 10

verbose = False
max_threads = 6

saved_steps = []

currently_allowed_num_iterations = 1
while currently_allowed_num_iterations < maximum_iterations:
    # only run up to given limit of number of iterations
    input.niter_array[0] = currently_allowed_num_iterations

    cpp_indata = input._to_cpp_vmecindatapywrapper()
    # start all over again, because flow control flags are not saved (yet) for restarting
    output = vmecpp._vmecpp.run(
        cpp_indata,
        max_threads=max_threads,
        verbose=verbose,
    )

    # print convergence progress
    print(
        "% 5d | % .3e | % .3e | % .3e"
        % (
            currently_allowed_num_iterations,
            output.wout.fsqr,
            output.wout.fsqz,
            output.wout.fsql,
        )
    )

    # save outputs for later plotting
    output.save(cache_folder / f"vmecpp_w7x_{currently_allowed_num_iterations:04d}.h5")
    saved_steps.append(currently_allowed_num_iterations)

    # early exis this loop when VMEC is converged
    if (
        output is not None
        and output.wout.fsqr < input.ftol_array[0]
        and output.wout.fsqz < input.ftol_array[0]
        and output.wout.fsql < input.ftol_array[0]
    ):
        print("converged after", output.wout.maximum_iterations, "iterations")
        break

    currently_allowed_num_iterations += step

np.savetxt(cache_folder / "saved_steps.dat", saved_steps, fmt="%d")
