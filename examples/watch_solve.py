# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Watch VMEC++ converge: the flux surfaces at two toroidal angles above the force
residuals of every iteration, drawn while the solve runs.

Run it with a display to get a live window, or pass ``--save solve.gif`` to record
the solve headlessly. The same per-iteration data is available to any script
through the ``iteration_callback`` argument of ``vmecpp.run``.
"""

import argparse
from pathlib import Path

import vmecpp

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "input",
    nargs="?",
    default=Path(__file__).parent / "data" / "solovev.json",
    type=Path,
    help="a VMEC++ JSON or INDATA file (default: examples/data/solovev.json)",
)
parser.add_argument(
    "--save", type=Path, help="record the solve to a .gif or video file"
)
parser.add_argument("--every", type=int, default=5, help="draw every N iterations")
parser.add_argument(
    "--planes", type=int, default=2, help="toroidal cross-sections to show"
)
args = parser.parse_args()

vmec_input = vmecpp.VmecInput.from_file(args.input)
output = vmecpp.watch(vmec_input, planes=args.planes, every=args.every, save=args.save)
print(
    f"{output.wout.reason}: fsqr = {output.wout.fsqr:.2e} after {output.wout.niter} iterations"
)
