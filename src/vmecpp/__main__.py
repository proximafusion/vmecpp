# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
import argparse
from pathlib import Path

import vmecpp


def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VMEC++ is a free-boundary ideal-MHD equilibrium solver for stellarators and tokamaks."
    )
    p.add_argument(
        "input_file",
        help="A VMEC input file either in the classic Fortran 'indata' format or in VMEC++'s JSON format.",
        type=Path,
    )
    p.add_argument(
        "-t",
        "--max-threads",
        help="maximum number of threads that VMEC++ should spawn. The actual number might still be lower that this in case there are too few flux surfaces to keep these many threads busy.",
        type=int,
    )
    return p.parse_args()


args = parse_arguments()

input = vmecpp.VmecInput.from_file(args.input_file)
output = vmecpp.run(input, args.max_threads)

configuration_name = vmecpp._util.get_vmec_configuration_name(args.input_file)
wout_file = Path(f"wout_{configuration_name}.nc")
output.wout.save(wout_file)

print(f"\nOutput written to {wout_file}.")  # noqa: T201
