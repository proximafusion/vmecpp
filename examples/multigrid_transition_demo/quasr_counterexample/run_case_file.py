"""Like run_case.py but uses an on-disk mgrid .nc file (mgrid_file) instead of the in-
memory response table -- validates the makegrid mgrid against the in-memory result, and
is the exact same field PARVMEC reads.

Usage: run_case_file.py <cachedir> <mgrid_nc> <ns_csv> <delt> <niter> <ftol> [logfile]
"""

import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
from run_case import build_input  # reuse

import vmecpp


def main():
    cachedir = Path(sys.argv[1])
    mgrid_nc = Path(sys.argv[2])
    ns_array = np.array([int(x) for x in sys.argv[3].split(",")], dtype=np.int64)
    delt = float(sys.argv[4])
    niter = int(sys.argv[5])
    ftol = float(sys.argv[6])
    logfile = Path(sys.argv[7]) if len(sys.argv) > 7 else None

    meta = json.loads((cachedir / "meta.json").read_text())
    vi = build_input(meta, ns_array, delt, niter, ftol)
    vi.mgrid_file = str(mgrid_nc)
    if os.environ.get("EXTCUR_OVERRIDE"):
        vi.extcur = np.array([float(os.environ["EXTCUR_OVERRIDE"])])

    cap_fd, cap_path = tempfile.mkstemp(suffix=".log")
    saved = os.dup(1)
    os.dup2(cap_fd, 1)
    os.close(cap_fd)

    status = "converged"
    exc = ""
    niter_done = -1
    try:
        out = vmecpp.run(vi, verbose=1, max_threads=1)
        niter_done = int(out.wout.niter)
    except Exception as e:
        msg = str(e).lower()
        if any(k in msg for k in ("did not converge", "not converge", "maximum")):
            status = "maxiter"
        elif "jacobian" in msg:
            status = "badjac"
        else:
            status = "error"
        exc = f"{type(e).__name__}: {str(e)[:120]}"
    finally:
        os.dup2(saved, 1)
        os.close(saved)

    text = Path(cap_path).read_text()
    Path(cap_path).unlink()
    if exc:
        text += f"\nEXCEPTION: {exc}\n"
    last = None
    for m in re.finditer(
        r"^\s*(\d+)\s*\|\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*\|",
        text,
        re.MULTILINE,
    ):
        last = m
    fsqr = fsqz = fsql = float("nan")
    if last:
        if niter_done < 0:
            niter_done = int(last.group(1))
        fsqr, fsqz, fsql = (float(last.group(i)) for i in (2, 3, 4))
    if logfile:
        logfile.write_text(text)
    print(
        f"RESULT status={status} niter={niter_done} "
        f"fsqr={fsqr:.3e} fsqz={fsqz:.3e} fsql={fsql:.3e} ns_last={int(ns_array[-1])} exc={exc!r}"
    )


if __name__ == "__main__":
    main()
