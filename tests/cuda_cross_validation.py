# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Cross-validation of the CUDA iteration body against the CPU one.

Two builds of one commit, one configured with ``-DVMECPP_USE_CUDA=ON`` and one without,
run the same inputs in separate processes; every ``wout`` field of the CUDA run is then
compared with the CPU run on one thread, against the spread between the CPU runs on one
and on eight threads, which partition the radial reductions differently and so bound
what a different summation order alone produces. Each build is also hot-restarted from
the other's converged state: if both states solve the same discrete equations, the
restart stops within the iterations a restart from its own state takes. The force
residuals of the first iterations, before the two trajectories drift apart, compare the
whole chain of kernels at once.

    python tests/cuda_cross_validation.py run DIR cpu1 --threads 1     # CPU build
    python tests/cuda_cross_validation.py run DIR cpu8 --threads 8     # CPU build
    python tests/cuda_cross_validation.py run DIR gpu                  # CUDA build
    python tests/cuda_cross_validation.py restart DIR cpu1 cpu_from_cpu  # CPU build
    python tests/cuda_cross_validation.py restart DIR gpu cpu_from_gpu   # CPU build
    python tests/cuda_cross_validation.py restart DIR cpu1 gpu_from_cpu  # CUDA build
    python tests/cuda_cross_validation.py restart DIR gpu gpu_from_gpu   # CUDA build
    python tests/cuda_cross_validation.py compare DIR

``--cases`` restricts any command to a comma-separated subset of :data:`CASES`.
"""

from __future__ import annotations

import argparse
import json
import time
import types
from pathlib import Path

import numpy as np

import vmecpp

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"

# name: (input file, updates); the inputs cover axisymmetric and three-dimensional
# boundaries, prescribed current and prescribed iota, power-series and spline pressure,
# vacuum, lbsubs, nfp = 1 to 5, an odd nzeta, a missing axis guess, a bad initial
# Jacobian, the block-Thomas radial solver above 1024 surfaces, the largest poloidal
# grid the device kernels accept, and free boundary at two vacuum cadences
CASES: dict[str, tuple[Path, dict]] = {
    "cma": (TEST_DATA / "cma.json", {}),
    "solovev": (TEST_DATA / "solovev.json", {}),
    "solovev_no_axis": (TEST_DATA / "solovev_no_axis.json", {}),
    "circular_tokamak": (TEST_DATA / "circular_tokamak.json", {}),
    "cth_like_fixed_bdy": (TEST_DATA / "cth_like_fixed_bdy.json", {}),
    "cth_like_fixed_bdy_iota": (TEST_DATA / "cth_like_fixed_bdy_iota.json", {}),
    "cth_like_fixed_bdy_nzeta_37": (TEST_DATA / "cth_like_fixed_bdy_nzeta_37.json", {}),
    "cth_like_fixed_bdy_spline_pressure": (
        TEST_DATA / "cth_like_fixed_bdy_spline_pressure.json",
        {},
    ),
    "cth_like_fixed_bdy_lbsubs": (
        TEST_DATA / "cth_like_fixed_bdy.json",
        {"lbsubs": True},
    ),
    "cth_like_fixed_bdy_vacuum": (
        TEST_DATA / "cth_like_fixed_bdy.json",
        {"pres_scale": 0.0, "curtor": 0.0},
    ),
    "li383_low_res": (TEST_DATA / "li383_low_res.json", {}),
    "near_axis_iota_nfp4": (TEST_DATA / "near_axis_iota_nfp4.json", {}),
    "constellaration_bad_initial_jacobian": (
        TEST_DATA / "constellaration_bad_initial_jacobian.json",
        {},
    ),
    "solovev_ns1101": (
        TEST_DATA / "solovev.json",
        {
            "ns_array": np.array([1101]),
            "ftol_array": np.array([1.0e-12]),
            "niter_array": np.array([100000]),
        },
    ),
    "solovev_ntheta510": (TEST_DATA / "solovev.json", {"ntheta": 510}),
    "w7x": (REPO_ROOT / "examples" / "data" / "w7x.json", {}),
    "cth_like_free_bdy": (TEST_DATA / "cth_like_free_bdy.json", {}),
    "cth_like_free_bdy_multigrid": (TEST_DATA / "cth_like_free_bdy_multigrid.json", {}),
    "cth_like_free_bdy_nvacskip1": (
        TEST_DATA / "cth_like_free_bdy.json",
        {"nvacskip": 1},
    ),
    "solovev_free_bdy": (TEST_DATA / "solovev_free_bdy.json", {}),
}

# solver diagnostics compared as trajectories rather than as converged fields
_TRACES = ("fsqt", "force_residual_r", "force_residual_z", "force_residual_lambda")
_TRAJECTORY_FIELDS = {
    *_TRACES,
    "niter",
    "itfsq",
    "fsqr",
    "fsqz",
    "fsql",
    "delbsq",
    "restart_reason_timetrace",
    "wdot",
}
# relative difference below which two residual traces count as one trajectory
_TRACE_PARITY = 1.0e-10


def load_input(name: str) -> vmecpp.VmecInput:
    path, updates = CASES[name]
    indata = vmecpp.VmecInput.from_file(path)
    if indata.lfreeb:
        updates = {
            **updates,
            "mgrid_file": str(TEST_DATA / Path(indata.mgrid_file).name),
        }
    return indata.model_copy(update=updates)


def _selected(cases: str | None) -> list[str]:
    if not cases:
        return list(CASES)
    names = cases.split(",")
    unknown = sorted(set(names) - set(CASES))
    if unknown:
        msg = f"unknown cases: {', '.join(unknown)}"
        raise SystemExit(msg)
    return names


def _summary(wout: vmecpp.VmecWOut, seconds: float) -> dict:
    return {
        "seconds": seconds,
        "ier_flag": int(wout.ier_flag),
        "iterations": int(wout.itfsq),
        "fsqr": float(wout.fsqr),
        "fsqz": float(wout.fsqz),
        "fsql": float(wout.fsql),
    }


def _solve_with_fixed_m1_gauge(
    indata: vmecpp.VmecInput, threads: int | None
) -> vmecpp.VmecWOut:
    """The solve with the m = 1 Z force zeroed from the first iteration and the gauge
    set from the boundary at every multigrid step, through the C++ run that carries the
    switch."""
    cpp_indata = indata._to_cpp_vmecindata()
    cpp_indata.always_fix_m1_gauge = True
    output = vmecpp._vmecpp.run(
        cpp_indata, max_threads=threads, verbose=vmecpp._vmecpp.OutputMode.SILENT
    )
    return vmecpp.VmecWOut._from_cpp_wout(output.wout)


def _run_cases(
    out: Path,
    names: list[str],
    threads: int | None,
    make_run,
    fix_m1_gauge: bool = False,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    for name in names:
        start = time.perf_counter()
        try:
            indata, kwargs = make_run(name)
            if fix_m1_gauge:
                wout = _solve_with_fixed_m1_gauge(indata, threads)
            else:
                wout = vmecpp.run(
                    indata, max_threads=threads, verbose=False, **kwargs
                ).wout
        except Exception as error:
            status[name] = {"error": f"{type(error).__name__}: {error}"}
            print(f"{name}: {status[name]['error']}", flush=True)
        else:
            wout.save(out / f"{name}.nc")
            status[name] = _summary(wout, time.perf_counter() - start)
            print(f"{name}: {status[name]}", flush=True)
        status_path.write_text(json.dumps(status, indent=1))


def run(args: argparse.Namespace) -> None:
    """Solve every case with the installed build."""
    _run_cases(
        Path(args.dir) / args.tag,
        _selected(args.cases),
        args.threads,
        lambda name: (load_input(name), {}),
        fix_m1_gauge=args.fix_m1_gauge,
    )


def restart(args: argparse.Namespace) -> None:
    """Hot-restart every case at its final radial resolution from another run's
    state."""
    source = Path(args.dir) / args.source

    def make_run(name: str):
        original = load_input(name)
        state = types.SimpleNamespace(
            wout=vmecpp.VmecWOut.from_wout_file(source / f"{name}.nc"), input=original
        )
        final = original.model_copy(
            update={
                "ns_array": np.asarray(original.ns_array)[-1:],
                "ftol_array": np.asarray(original.ftol_array)[-1:],
                "niter_array": np.asarray(original.niter_array)[-1:],
            }
        )
        return final, {"restart_from": state}

    _run_cases(Path(args.dir) / args.tag, _selected(args.cases), args.threads, make_run)


def _relative(actual: np.ndarray, expected: np.ndarray) -> float:
    if actual.shape != expected.shape:
        return np.inf
    scale = float(np.max(np.abs(expected), initial=0.0))
    difference = float(np.max(np.abs(actual - expected), initial=0.0))
    if scale == 0.0:
        return difference
    return difference / scale


def _fields(path: Path) -> dict[str, np.ndarray]:
    wout = vmecpp.VmecWOut.from_wout_file(path)
    fields = {}
    for name, value in wout.model_dump().items():
        if value is None or isinstance(value, str | bool):
            continue
        array = np.asarray(value)
        if array.dtype.kind in "fiu":
            fields[name] = array.astype(np.float64)
    return fields


def _trace_parity(actual: np.ndarray, expected: np.ndarray) -> tuple[int, float]:
    """The number of leading iterations whose residuals agree to _TRACE_PARITY, and the
    relative difference at the first iteration."""
    length = min(actual.size, expected.size)
    if length == 0:
        return 0, np.nan
    a, e = actual[:length], expected[:length]
    relative = np.abs(a - e) / np.maximum(np.abs(e), np.finfo(float).tiny)
    apart = np.flatnonzero(relative > _TRACE_PARITY)
    return (int(apart[0]) if apart.size else length), float(relative[0])


def compare(args: argparse.Namespace) -> None:
    """Every field of the CUDA run against the CPU one, against the CPU thread
    spread."""
    root = Path(args.dir)
    tags = (args.reference, args.spread, args.test)
    lines = [
        f"| case | iterations {' / '.join(tags)} | largest field deviation, "
        f"{args.test} | largest field spread, {args.spread} | fields beyond 10x the "
        "spread | leading iterations in parity | restarts cpu<-cpu / cpu<-gpu / "
        "gpu<-cpu / gpu<-gpu |",
        "|---|---|---|---|---|---|---|",
    ]
    details = []
    for name in _selected(args.cases):
        paths = {tag: root / tag / f"{name}.nc" for tag in tags}
        if not all(path.exists() for path in paths.values()):
            missing = [tag for tag, path in paths.items() if not path.exists()]
            lines.append(f"| {name} | missing {', '.join(missing)} | | | | | |")
            continue
        cpu1, cpu8, gpu = (_fields(paths[tag]) for tag in tags)
        worst_gpu, worst_cpu, beyond = (0.0, ""), (0.0, ""), []
        for field, expected in cpu1.items():
            if field in _TRAJECTORY_FIELDS or field not in gpu or field not in cpu8:
                continue
            deviation = _relative(gpu[field], expected)
            spread = _relative(cpu8[field], expected)
            worst_gpu = max(worst_gpu, (deviation, field))
            worst_cpu = max(worst_cpu, (spread, field))
            if deviation > 10.0 * max(spread, 1.0e-14):
                beyond.append((deviation, spread, field))
        beyond.sort(reverse=True)
        leading, first = _trace_parity(gpu["fsqt"], cpu1["fsqt"])
        iterations = " / ".join(
            str(int(fields["itfsq"])) for fields in (cpu1, cpu8, gpu)
        )
        restarts = []
        for tag in ("cpu_from_cpu", "cpu_from_gpu", "gpu_from_cpu", "gpu_from_gpu"):
            status_path = root / tag / "status.json"
            entry = (
                json.loads(status_path.read_text()).get(name, {})
                if status_path.exists()
                else {}
            )
            restarts.append(str(entry.get("iterations", "-")))
        lines.append(
            f"| {name} | {iterations} | {worst_gpu[0]:.1e} ({worst_gpu[1]}) | "
            f"{worst_cpu[0]:.1e} ({worst_cpu[1]}) | {len(beyond)} | "
            f"{leading} (first {first:.0e}) | {' / '.join(restarts)} |"
        )
        if beyond:
            details.append(f"\n{name}: fields beyond 10x the CPU spread")
            details.extend(
                f"  {field}: gpu {deviation:.2e}, cpu spread {spread:.2e}"
                for deviation, spread, field in beyond[:15]
            )
    report = "\n".join(lines + details)
    (root / f"compare_{args.reference}_{args.test}.md").write_text(report + "\n")
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(required=True)
    for command, function in (("run", run), ("restart", restart)):
        sub = commands.add_parser(command, help=function.__doc__)
        sub.add_argument("dir")
        if command == "restart":
            sub.add_argument("source", help="the run whose converged states seed it")
        sub.add_argument("tag", help="the subdirectory that receives the outputs")
        sub.add_argument("--threads", type=int, default=None)
        sub.add_argument("--cases")
        if command == "run":
            sub.add_argument(
                "--fix-m1-gauge",
                action="store_true",
                help="zero the m = 1 Z force throughout, as the CUDA path does",
            )
        sub.set_defaults(function=function)
    sub = commands.add_parser("compare", help=compare.__doc__)
    sub.add_argument("dir")
    sub.add_argument("--cases")
    sub.add_argument("--reference", default="cpu1", help="the run compared against")
    sub.add_argument("--spread", default="cpu8", help="the run that sets the spread")
    sub.add_argument("--test", default="gpu", help="the run under test")
    sub.set_defaults(function=compare)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
