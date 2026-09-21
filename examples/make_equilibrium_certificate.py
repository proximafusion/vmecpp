"""Certify a VMEC++ equilibrium with Stellarocq.

Runs VMEC++ on an input file, saves the wout, and hands it to the certificate
generator, the checker and the wout guard of a Stellarocq checkout
(https://github.com/CharlesCNorton/stellarocq); see
docs/proof_carrying_equilibria.md.

  python make_equilibrium_certificate.py --stellarocq ~/stellarocq
  python make_equilibrium_certificate.py --stellarocq ~/stellarocq --cells --nu 512

--nodes, --nu and --nv are passed to gen/make_cert.py of Stellarocq.
"""

import argparse
import subprocess
import sys
from pathlib import Path

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
        "--stellarocq", required=True, type=Path, help="a Stellarocq checkout"
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
    for option, meaning in (
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
    generator = a.stellarocq / "gen" / "make_cert.py"
    guard = a.stellarocq / "gen" / "verify_cert.py"
    checker = a.checker or a.stellarocq / "extract" / "_build" / "default" / "main.exe"

    name = Path(a.input).name
    case = name.removeprefix("input.") if name.startswith("input.") else Path(name).stem
    wout = Path(f"wout_{case}.nc")
    cert = Path(f"cert_{case}.txt")

    output = vmecpp.run(vmecpp.VmecInput.from_file(a.input))
    output.wout.save(wout)

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
