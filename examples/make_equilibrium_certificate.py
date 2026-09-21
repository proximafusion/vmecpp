"""Certify a VMEC++ equilibrium with Stellarocq.

Runs VMEC++ on an input file, saves the wout, and hands it to the certificate
generator, the checker and the wout guard of a Stellarocq checkout
(https://github.com/CharlesCNorton/stellarocq); see
docs/proof_carrying_equilibria.md.

  python make_equilibrium_certificate.py --stellarocq ~/stellarocq
  python make_equilibrium_certificate.py --stellarocq ~/stellarocq --cells --nu 512

--mpol, --ntor, --ns and --ftol override the resolution and the force tolerance of
the input before the run, and --nodes, --nu and --nv are passed to
gen/make_cert.py of Stellarocq.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

import vmecpp


def run(*command):
    """Run a command and return its exit status."""
    return subprocess.run([str(c) for c in command], check=False).returncode


def certify():
    """Run VMEC++, write a certificate of the result and check it."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "input",
        nargs="?",
        default=str(Path(__file__).parent / "data" / "solovev.json"),
        help="a VMEC++ JSON or INDATA input file",
    )
    ap.add_argument(
        "--stellarocq",
        type=Path,
        default=None,
        help="a Stellarocq checkout; without one the wout is saved and not certified",
    )
    ap.add_argument(
        "--checker",
        type=Path,
        default=None,
        help="the checker binary; by default the one built inside the checkout",
    )
    ap.add_argument(
        "--cells",
        action="store_true",
        help="certify over cells of angles instead of at sampled angles",
    )
    ap.add_argument(
        "--ftol",
        type=float,
        default=None,
        help="converge to this force tolerance instead of the input's",
    )
    for option, meaning in (
        ("mpol", "run with this poloidal resolution instead of the input's"),
        ("ntor", "run with this toroidal resolution instead of the input's"),
        ("ns", "run on this many flux surfaces instead of the input's"),
        ("nodes", "how many full-grid nodes to certify"),
        ("nu", "poloidal angles, or poloidal cells, per node"),
        ("nv", "toroidal angles per node of a three-dimensional equilibrium"),
    ):
        ap.add_argument(f"--{option}", type=int, default=None, help=meaning)
    a = ap.parse_args()
    generator_args = [
        word
        for option in ("nodes", "nu", "nv")
        if getattr(a, option) is not None
        for word in (f"--{option}", getattr(a, option))
    ]
    name = Path(a.input).name
    case = name.removeprefix("input.") if name.startswith("input.") else Path(name).stem
    wout = Path(f"wout_{case}.nc")
    cert = Path(f"cert_{case}.txt")

    vmec_input = vmecpp.VmecInput.from_file(a.input)
    if a.mpol is not None:
        vmec_input.mpol = a.mpol
    if a.ntor is not None:
        vmec_input.ntor = a.ntor
    if a.ns is not None:
        vmec_input.ns_array = np.array([a.ns])
        vmec_input.ftol_array = vmec_input.ftol_array[-1:]
        vmec_input.niter_array = vmec_input.niter_array[-1:]
    if a.ftol is not None:
        vmec_input.ftol_array = np.full(len(vmec_input.ns_array), a.ftol)
    output = vmecpp.run(vmec_input)
    output.wout.save(wout)
    if a.stellarocq is None:
        print(f"saved {wout}; pass --stellarocq to certify it")
        return

    generator = a.stellarocq / "gen" / "make_cert.py"
    guard = a.stellarocq / "gen" / "verify_cert.py"
    checker = a.checker or a.stellarocq / "extract" / "_build" / "default" / "main.exe"
    if a.cells:
        # The generator leaves the bounds of a cell certificate blank and the
        # checker writes them; an ordinary run then establishes them.
        cells = Path(f"cells_{case}.txt")
        status = run(sys.executable, generator, wout, cells, "--cells", *generator_args)
        status = status or run(checker, "--tighten", cells, cert)
    else:
        status = run(sys.executable, generator, wout, cert, *generator_args)
    # the verdict, then the certificate against the wout it was made from
    status = status or run(checker, cert)
    status = status or run(sys.executable, guard, wout, cert)
    sys.exit(status)


if __name__ == "__main__":
    certify()
