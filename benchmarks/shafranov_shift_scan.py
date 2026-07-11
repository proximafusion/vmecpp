# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Shafranov-shift pressure scan across the free-boundary solvers.

Verification experiment in the spirit of Hudson et al., Phys. Plasmas 32,
043906 (2025) and Conlin et al., arXiv:2412.05680: scan the pressure of a
free-boundary equilibrium and track the outward shift of the magnetic axis
(Shafranov shift). At high pressure the virtual-casing (plasma) contribution
to the vacuum field is large, making this a demanding test of the
free-boundary solver; comparing NESTOR, BIEST, and Vac2 on the same scan
isolates solver-discretization effects on the converged equilibrium.

Cases:
- solovev: axisymmetric (nfp=1) Solovev-like tokamak from the test fixtures.
  The magnetic field response is computed in-memory from coils.solovev at
  the requested nzeta so that the 3D-only solvers (BIEST, Vac2) can run the
  axisymmetric configuration as an nzeta > 1 case.
- cth_like: the 3D CTH-like stellarator fixture (nfp=5, mgrid file).

Usage:
    python benchmarks/shafranov_shift_scan.py --case solovev \
        --solvers nestor vac2 --scales 0.01 1 2 4 8
    python benchmarks/shafranov_shift_scan.py --case cth_like \
        --solvers nestor vac2 biest --scales 0.1 1 5 10 20

Outputs <out>/results_<case>.json and <out>/shafranov_shift_<case>.png.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import time
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import vmecpp

REPO_CPP = Path(__file__).resolve().parent.parent / "src" / "vmecpp" / "cpp"
TEST_DATA_LARGE = REPO_CPP / "vmecpp_large_cpp_tests" / "test_data"
TEST_DATA = REPO_CPP / "vmecpp" / "test_data"


def load_case(case: str, nzeta_3d: int):
    """Returns (VmecInput, MagneticFieldResponseTable | None)."""
    if case == "solovev":
        inp = vmecpp.VmecInput.from_file(TEST_DATA_LARGE / "solovev_free_bdy.json")
        # run the axisymmetric configuration as a 3D (nzeta > 1) case so that
        # BIEST and Vac2 can be used; the field is computed in-memory from
        # the coils file at matching toroidal resolution
        params = vmecpp.MakegridParameters.from_file(
            TEST_DATA_LARGE / "makegrid_parameters_solovev.json"
        )
        params.number_of_phi_grid_points = nzeta_3d
        # the fixture file requests raw fields (currents baked in); VMEC++
        # multiplies the response by extcur, so request the normalized
        # (per-unit-current) response instead
        params.normalize_by_currents = True
        # widen the grid box: at percent-level beta the Shafranov-shifted
        # boundary leaves the fixture's r <= 6 box and the interpolated
        # field gets clamped, silently corrupting the equilibrium
        params.r_grid_minimum = 1.5
        params.r_grid_maximum = 8.0
        params.z_grid_minimum = -3.0
        params.z_grid_maximum = 3.0
        params.number_of_r_grid_points = 261
        params.number_of_z_grid_points = 241
        field = vmecpp.MagneticFieldResponseTable.from_coils_file(
            TEST_DATA_LARGE / "coils.solovev", params
        )
        # VMEC++ does not support ntor = 0 with nzeta > 1: promote the
        # configuration to ntor = 1 with zero non-axisymmetric modes
        inp.nzeta = nzeta_3d
        mpol_rows = inp.rbc.shape[0]

        def pad(a):
            out = np.zeros((mpol_rows, 3))
            out[:, 1] = a[:, 0]
            return out

        inp.ntor = 1
        inp.rbc = pad(inp.rbc)
        inp.zbs = pad(inp.zbs)
        inp.raxis_c = np.array([inp.raxis_c[0], 0.0])
        inp.zaxis_s = np.array([0.0, 0.0])
        # keep the scan affordable; the fixture default is ftol 1e-14
        inp.ftol_array = np.array([1e-11] * len(inp.ns_array))
        return inp, field
    if case == "cth_like":
        inp = vmecpp.VmecInput.from_file(TEST_DATA / "cth_like_free_bdy.json")
        inp.mgrid_file = str(TEST_DATA / "mgrid_cth_like.nc")
        return inp, None
    msg = f"unknown case '{case}'"
    raise ValueError(msg)


def run_point(inp, field, max_threads: int) -> dict:
    """Worker executed in a subprocess: a LOG(FATAL) abort inside the C++
    core at extreme pressure must only lose this scan point, not the whole
    scan."""
    if field is not None:
        out = vmecpp.run(inp, field, max_threads=max_threads, verbose=False)
    else:
        out = vmecpp.run(inp, max_threads=max_threads, verbose=False)
    w = out.wout
    return {
        "converged": True,
        "niter": int(w.niter),
        "betatotal": float(w.betatotal),
        "betaxis": float(w.betaxis),
        "r_axis_phi0": float(np.sum(w.raxis_cc)),
        "volume_p": float(w.volume_p),
        "aspect": float(w.aspect),
        "b0": float(w.b0),
        "fsqr": float(w.fsqr),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["solovev", "cth_like"], required=True)
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=["nestor", "vac2"],
        choices=["nestor", "vac2", "biest"],
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=[0.01, 1.0, 2.0, 4.0, 8.0],
        help="multipliers on the fixture's pres_scale",
    )
    parser.add_argument("--nzeta-3d", type=int, default=36)
    parser.add_argument("--biest-digits", type=int, default=6)
    parser.add_argument("--max-threads", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("shafranov_shift_out"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    results = []

    base_inp, field = load_case(args.case, args.nzeta_3d)
    base_pres_scale = base_inp.pres_scale

    for solver in args.solvers:
        for scale in args.scales:
            inp = base_inp.model_copy(deep=True)
            inp.free_boundary_method = solver
            if solver == "biest":
                inp.biest_accuracy_digits = args.biest_digits
            inp.pres_scale = base_pres_scale * scale

            t0 = time.time()
            record = {
                "case": args.case,
                "solver": solver,
                "scale": scale,
                "pres_scale": inp.pres_scale,
            }
            ctx = multiprocessing.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=1, mp_context=ctx
            ) as pool:
                try:
                    record.update(
                        pool.submit(
                            run_point, inp, field, args.max_threads
                        ).result()
                    )
                    record["wall_s"] = time.time() - t0
                    print(
                        f"[{args.case}/{solver}] scale={scale:g}: "
                        f"beta={record['betatotal']:.4e} "
                        f"R_axis={record['r_axis_phi0']:.6f} "
                        f"({record['niter']} iters, {record['wall_s']:.1f}s)"
                    )
                except Exception as e:
                    record.update(
                        converged=False,
                        wall_s=time.time() - t0,
                        error=str(e)[:300],
                    )
                    print(f"[{args.case}/{solver}] scale={scale:g}: FAILED: {e}")
            results.append(record)

    results_file = args.out / f"results_{args.case}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {results_file}")

    # ------------------------------------------------------------------
    # plot: axis position and Shafranov shift vs beta, one line per solver
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for solver in args.solvers:
        recs = [
            r for r in results if r["solver"] == solver and r.get("converged", False)
        ]
        if not recs:
            continue
        recs.sort(key=lambda r: r["betatotal"])
        beta = np.array([r["betatotal"] for r in recs])
        r_axis = np.array([r["r_axis_phi0"] for r in recs])
        axs[0].plot(beta, r_axis, "o-", label=solver)
        axs[1].plot(beta, r_axis - r_axis[0], "o-", label=solver)
    axs[0].set_xlabel("<beta> total")
    axs[0].set_ylabel("R_axis(phi=0) [m]")
    axs[0].set_title(f"{args.case}: magnetic axis position")
    axs[1].set_xlabel("<beta> total")
    axs[1].set_ylabel("R_axis - R_axis(lowest beta) [m]")
    axs[1].set_title("Shafranov shift")
    for ax in axs:
        ax.legend()
        ax.grid(alpha=0.3)
    fig_file = args.out / f"shafranov_shift_{args.case}.png"
    fig.savefig(fig_file, dpi=150)
    print(f"wrote {fig_file}")


if __name__ == "__main__":
    main()
