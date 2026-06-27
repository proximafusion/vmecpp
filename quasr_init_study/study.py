#!/usr/bin/env python
"""Investigate how the initial-guess method (default linear, zeno, map2disc) changes
VMEC++ convergence: initial force-balance residual and iterations to convergence.

Throwaway investigation script for PR #545 (not committed).
"""

from __future__ import annotations

import gzip
import json
import traceback
import urllib.request
from pathlib import Path

import numpy as np

import vmecpp

REPO = Path()
WORK = Path("/tmp/quasr_init_study")
WORK.mkdir(exist_ok=True)

METHODS = ["default", "zeno", "map2disc"]
NITER_CAP = 2000
FTOL = 1e-11


def measure(vmec_input: vmecpp.VmecInput, ns: int, method: str) -> dict:
    """Run one (config, method) at a single multigrid step and return metrics."""
    inp = vmec_input.model_copy(deep=True)
    inp.ns_array = np.array([ns], dtype=np.int64)
    inp.ftol_array = np.array([FTOL])
    inp.niter_array = np.array([NITER_CAP], dtype=np.int64)
    inp.return_outputs_even_if_not_converged = True

    restart = None
    try:
        if method == "zeno":
            restart = vmecpp.zeno_guess(vmec_input, ns=ns)
        elif method == "map2disc":
            restart = vmecpp.map2disc_guess(vmec_input, ns=ns)
    except Exception as exc:  # guess construction itself failed
        return {"method": method, "status": f"guess-error: {type(exc).__name__}: {exc}"}

    if restart is not None:
        # The PR keeps the original (multi-grid) input on the guess; hot restart
        # requires the restart state's ns_array to match the single-step run.
        restart.input = restart.input.model_copy(deep=True)
        restart.input.ns_array = np.array([ns], dtype=np.int64)
        restart.input.ftol_array = np.array([FTOL])
        restart.input.niter_array = np.array([NITER_CAP], dtype=np.int64)

    try:
        out = vmecpp.run(inp, restart_from=restart, verbose=False, max_threads=1)
    except Exception as exc:
        return {"method": method, "status": f"run-error: {type(exc).__name__}: {exc}"}

    w = out.wout
    fsqt = np.asarray(w.fsqt)
    # "converged" = reached within 10x of the requested force tolerance
    converged = bool(fsqt.size and fsqt[-1] <= w.ftolv * 10.0)
    return {
        "method": method,
        "status": "ok",
        "fsqt0": float(fsqt[0]) if fsqt.size else float("nan"),
        "fsqt_final": float(fsqt[-1]) if fsqt.size else float("nan"),
        "niter": int(w.niter),
        "ftolv": float(w.ftolv),
        "converged": converged,
        "volume": float(w.volume),
    }


def study_config(name: str, vmec_input: vmecpp.VmecInput, ns: int) -> list[dict]:
    rows = []
    print(
        f"\n=== {name}  (ns={ns}, mpol={vmec_input.mpol}, ntor={vmec_input.ntor}, "
        f"nfp={vmec_input.nfp}) ===",
        flush=True,
    )
    for m in METHODS:
        r = measure(vmec_input, ns, m)
        r["config"] = name
        rows.append(r)
        if r["status"] == "ok":
            print(
                f"  {m:9s} fsqt0={r['fsqt0']:.3e}  niter={r['niter']:5d}  "
                f"converged={r['converged']}  vol={r['volume']:.5f}",
                flush=True,
            )
        else:
            print(f"  {m:9s} {r['status']}", flush=True)
    return rows


# --------------------------------------------------------------------------
# QUASR helpers
# --------------------------------------------------------------------------
QUASR = "https://quasr.flatironinstitute.org/"


def quasr_db() -> dict:
    p = WORK / "database.json.gz"
    if not p.exists():
        print("downloading QUASR database index...", flush=True)
        req = urllib.request.Request(
            QUASR + "database.json.gz", headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            p.write_bytes(r.read())
    with gzip.open(p) as f:
        return json.load(f)


def quasr_fetch_input(rid: int) -> Path:
    i = f"{rid}".zfill(7)
    dst = WORK / f"input.{i}"
    if not dst.exists():
        url = f"{QUASR}nml/{i[:4]}/input.{i}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            dst.write_bytes(r.read())
    return dst


def make_vacuum_low_res(vmec_input: vmecpp.VmecInput, n_modes: int) -> vmecpp.VmecInput:
    """Force a vacuum constant-pressure profile and resolution of n_modes."""
    inp = vmec_input.model_copy(deep=True)
    mpol, ntor = n_modes, n_modes
    inp.rbc = vmecpp.VmecInput.resize_2d_coeff(np.asarray(inp.rbc), mpol, ntor)
    inp.zbs = vmecpp.VmecInput.resize_2d_coeff(np.asarray(inp.zbs), mpol, ntor)

    # axis: keep n=0..ntor, pad/truncate
    def fix_axis(a):
        a = np.asarray(a, dtype=float)
        out = np.zeros(ntor + 1)
        out[: min(len(a), ntor + 1)] = a[: ntor + 1]
        return out

    inp.raxis_c = fix_axis(inp.raxis_c)
    inp.zaxis_s = fix_axis(inp.zaxis_s)
    inp.mpol = mpol
    inp.ntor = ntor
    # vacuum: zero pressure, zero net current
    inp.pmass_type = "power_series"
    inp.am = np.array([0.0])
    inp.pres_scale = 0.0
    inp.ncurr = 1
    inp.ac = np.array([0.0])
    inp.curtor = 0.0
    inp.gamma = 0.0
    return inp


def dump(rows):
    (WORK / "results.json").write_text(json.dumps(rows, indent=2))


def main():
    all_rows: list[dict] = []

    # ---- Checked-in input files (fast ones first) ----
    # (name, path, ns_override).
    checked = [
        ("cth_like_fixed_bdy", REPO / "examples/data/cth_like_fixed_bdy.json", None),
        ("solovev", REPO / "examples/data/solovev.json", 51),
        (
            "circular_tokamak",
            REPO / "src/vmecpp/cpp/vmecpp/test_data/circular_tokamak.json",
            None,
        ),
        ("cma", REPO / "src/vmecpp/cpp/vmecpp/test_data/cma.json", None),
    ]
    for name, path, ns_over in checked:
        if not path.exists():
            print(f"skip {name}: missing {path}")
            continue
        vi = vmecpp.VmecInput.from_file(path)
        ns = ns_over or int(np.asarray(vi.ns_array)[-1])
        all_rows += study_config(name, vi, ns)
        dump(all_rows)

    # ---- QUASR configs ----
    try:
        db = quasr_db()
        cols = db["columns"]
        data = db["data"]
        id_i, nfp_i = cols.index("ID"), cols.index("nfp")
        # pick ~12 spanning nfp 2,3,4
        picks: list[int] = []
        want = {2: 4, 3: 4, 4: 4}
        got = {2: 0, 3: 0, 4: 0}
        for row in data:
            nfp = int(row[nfp_i])
            if nfp in want and got[nfp] < want[nfp]:
                picks.append(int(row[id_i]))
                got[nfp] += 1
            if sum(got.values()) >= sum(want.values()):
                break
        print(f"\nQUASR picks (nfp counts {got}): {picks}", flush=True)
        for rid in picks:
            try:
                path = quasr_fetch_input(rid)
                vi = vmecpp.VmecInput.from_file(path)
                vi = make_vacuum_low_res(vi, n_modes=7)
                all_rows += study_config(f"QUASR-{rid}(nfp{vi.nfp})", vi, ns=25)
                dump(all_rows)
            except Exception:
                print(f"QUASR-{rid} FAILED:\n{traceback.format_exc()}", flush=True)
    except Exception:
        print(f"QUASR phase failed:\n{traceback.format_exc()}", flush=True)

    # ---- w7x last: large 3D config, harder single-grid solve ----
    w7x = REPO / "examples/data/w7x.json"
    if w7x.exists():
        vi = vmecpp.VmecInput.from_file(w7x)
        all_rows += study_config("w7x", vi, ns=25)
        dump(all_rows)

    out = WORK / "results.json"
    print(f"\nwrote {out} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
