#!/usr/bin/env python
"""QUASR-only convergence comparison (default vs zeno vs map2disc).

Vacuum constant-pressure boundaries, low resolution (7 Fourier modes), single multigrid
step. Looser tolerance (1e-9) + lower iteration cap keeps the batch tractable; the
initial residual fsqt[0] and niter are still the comparison.
"""

from __future__ import annotations

import gzip
import json
import traceback
import urllib.request
from pathlib import Path

import numpy as np

import vmecpp

WORK = Path("/tmp/quasr_init_study")
QUASR = "https://quasr.flatironinstitute.org/"
METHODS = ["default", "zeno", "map2disc"]
NITER_CAP = 1200
FTOL = 1e-9
NS = 25
N_MODES = 7


def measure(vmec_input, ns, method):
    inp = vmec_input.model_copy(deep=True)
    inp.ns_array = np.array([ns], dtype=np.int64)
    inp.ftol_array = np.array([FTOL])
    inp.niter_array = np.array([NITER_CAP], dtype=np.int64)
    inp.return_outputs_even_if_not_converged = True
    restart = None
    try:
        if method == "zeno":
            # lmax kept low: zeno guess-construction (L-BFGS over the Fourier-
            # Zernike basis) is the dominant cost and scales steeply with lmax.
            restart = vmecpp.zeno_guess(vmec_input, ns=ns, lmax=3)
        elif method == "map2disc":
            restart = vmecpp.map2disc_guess(vmec_input, ns=ns)
    except Exception as exc:
        return {"method": method, "status": f"guess-error: {type(exc).__name__}: {exc}"}
    if restart is not None:
        restart.input = restart.input.model_copy(deep=True)
        restart.input.ns_array = np.array([ns], dtype=np.int64)
        restart.input.ftol_array = np.array([FTOL])
        restart.input.niter_array = np.array([NITER_CAP], dtype=np.int64)
    try:
        out = vmecpp.run(inp, restart_from=restart, verbose=False, max_threads=1)
    except Exception as exc:
        return {"method": method, "status": f"run-error: {type(exc).__name__}: {exc}"}
    w = out.wout
    f = np.asarray(w.fsqt)
    return {
        "method": method,
        "status": "ok",
        "fsqt0": float(f[0]) if f.size else float("nan"),
        "fsqt_final": float(f[-1]) if f.size else float("nan"),
        "niter": int(w.niter),
        "ftolv": float(w.ftolv),
        "converged": bool(f.size and f[-1] <= w.ftolv * 10.0),
        "hit_cap": int(w.niter) >= NITER_CAP - 1,
        "volume": float(w.volume),
    }


def quasr_db():
    p = WORK / "database.json.gz"
    with gzip.open(p) as f:
        return json.load(f)


def fetch_input(rid):
    i = f"{rid}".zfill(7)
    dst = WORK / f"input.{i}"
    if not dst.exists():
        url = f"{QUASR}nml/{i[:4]}/input.{i}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            dst.write_bytes(r.read())
    return dst


def make_vacuum_low_res(vi, n=N_MODES):
    inp = vi.model_copy(deep=True)
    inp.rbc = vmecpp.VmecInput.resize_2d_coeff(np.asarray(inp.rbc), n, n)
    inp.zbs = vmecpp.VmecInput.resize_2d_coeff(np.asarray(inp.zbs), n, n)

    def fix(a):
        a = np.asarray(a, dtype=float)
        out = np.zeros(n + 1)
        out[: min(len(a), n + 1)] = a[: n + 1]
        return out

    inp.raxis_c = fix(inp.raxis_c)
    inp.zaxis_s = fix(inp.zaxis_s)
    inp.mpol, inp.ntor = n, n
    inp.pmass_type = "power_series"
    inp.am = np.array([0.0])
    inp.pres_scale = 0.0
    inp.ncurr = 1
    inp.ac = np.array([0.0])
    inp.curtor = 0.0
    inp.gamma = 0.0
    return inp


def main():
    db = quasr_db()
    cols, data = db["columns"], db["data"]
    id_i, nfp_i = cols.index("ID"), cols.index("nfp")
    picks, want, got = [], {2: 4, 3: 4, 4: 4}, {2: 0, 3: 0, 4: 0}
    for row in data:
        nfp = int(row[nfp_i])
        if nfp in want and got[nfp] < want[nfp]:
            picks.append(int(row[id_i]))
            got[nfp] += 1
        if sum(got.values()) >= sum(want.values()):
            break
    print(f"QUASR picks (nfp {got}): {picks}", flush=True)

    rows = []
    for rid in picks:
        try:
            vi = make_vacuum_low_res(vmecpp.VmecInput.from_file(fetch_input(rid)))
            print(
                f"\n=== QUASR-{rid} (nfp{vi.nfp}, mpol={vi.mpol}, ns={NS}) ===",
                flush=True,
            )
            for m in METHODS:
                r = measure(vi, NS, m)
                r["config"] = f"QUASR-{rid}(nfp{vi.nfp})"
                rows.append(r)
                if r["status"] == "ok":
                    print(
                        f"  {m:9s} fsqt0={r['fsqt0']:.3e}  niter={r['niter']:5d}  "
                        f"cap={r['hit_cap']}  conv={r['converged']}  "
                        f"vol={r['volume']:.5f}",
                        flush=True,
                    )
                else:
                    print(f"  {m:9s} {r['status']}", flush=True)
                (WORK / "results_quasr.json").write_text(json.dumps(rows, indent=2))
        except Exception:
            print(f"QUASR-{rid} FAILED:\n{traceback.format_exc()}", flush=True)
    print(f"\nwrote results_quasr.json ({len(rows)} rows)")


if __name__ == "__main__":
    main()
