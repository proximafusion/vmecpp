"""Run one free-boundary case from a cached QUASR config, report convergence.

Usage:
  run_case.py <cachedir> <ns_csv> <delt> <niter> <ftol> [logfile]

Captures the C++ solver stdout at the file-descriptor level (Python
redirect_stdout does not catch it). Prints a single RESULT line:
  RESULT status=<converged|maxiter|error> niter=<n> fsqr=.. fsqz=.. fsql=.. ns_last=..
"""

import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

import vmecpp


def build_input(meta, ns_array, delt, niter, ftol):
    vi = vmecpp.VmecInput.default()
    vi.lasym = False
    vi.nfp = meta["nfp"]
    vi.mpol = meta["mpol"]
    vi.ntor = meta["ntor"]
    vi.ntheta = 0
    vi.nzeta = meta["nzeta"]
    vi.ns_array = ns_array
    vi.ftol_array = np.full(len(ns_array), ftol, dtype=float)
    vi.niter_array = np.full(len(ns_array), niter, dtype=np.int64)
    vi.nstep = 1
    vi.delt = delt
    vi.phiedge = meta["phiedge"]
    vi.gamma = 0.0
    vi.bloat = 1.0
    vi.pmass_type = "power_series"
    vi.am = np.array([0.0])
    vi.pres_scale = 0.0
    vi.ncurr = 1
    vi.pcurr_type = "power_series"
    vi.ac = np.array([0.0])
    vi.curtor = 0.0
    vi.lfreeb = True
    vi.extcur = np.array([meta["extcur"]])
    vi.nvacskip = 6
    vi.raxis_c = np.zeros(meta["ntor"] + 1)
    vi.zaxis_s = np.zeros(meta["ntor"] + 1)
    vi.raxis_c[0] = meta["raxis_guess"]
    vi.rbc = np.array(meta["rbc"])
    vi.zbs = np.array(meta["zbs"])
    return vi


def main():
    cachedir = Path(sys.argv[1])
    ns_array = np.array([int(x) for x in sys.argv[2].split(",")], dtype=np.int64)
    delt = float(sys.argv[3])
    niter = int(sys.argv[4])
    ftol = float(sys.argv[5])
    logfile = Path(sys.argv[6]) if len(sys.argv) > 6 else None

    meta = json.loads((cachedir / "meta.json").read_text())
    response = vmecpp.MagneticFieldResponseTable.model_validate_json(
        (cachedir / "response.json").read_text()
    )
    vi = build_input(meta, ns_array, delt, niter, ftol)

    # capture C++ stdout at fd level (Python redirect_stdout does not catch it)
    cap_fd, cap_path = tempfile.mkstemp(suffix=".log")
    saved = os.dup(1)
    os.dup2(cap_fd, 1)
    os.close(cap_fd)

    status = "converged"
    exc = ""
    niter_done = -1
    try:
        out = vmecpp.run(vi, magnetic_field=response, verbose=1, max_threads=1)
        niter_done = int(out.wout.niter)
    except Exception as e:
        msg = str(e).lower()
        if any(k in msg for k in ("did not converge", "not converge", "maximum")):
            status = "maxiter"
        elif "jacobian" in msg:
            status = "badjac"
        else:
            status = "error"
        exc = f"{type(e).__name__}: {e}"
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
        f"fsqr={fsqr:.3e} fsqz={fsqz:.3e} fsql={fsql:.3e} ns_last={int(ns_array[-1])} "
        f"exc={exc!r}"
    )


if __name__ == "__main__":
    main()
