# SPDX-FileCopyrightText: 2026 Proxima Fusion GmbH <info@proximafusion.com>
# SPDX-License-Identifier: MIT
"""Compare vacuum interpolation at fixed coils, plasma resolution and tolerance.

Generate a fine response table from coils, or use a supplied MGRID file. Retain raw
outputs and run each measured solve in a separate process. Timings include VMEC output
computation but exclude table generation, reading and validation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import shutil
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

import vmecpp


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(args):
    folder = args.output / "inputs"
    folder.mkdir()
    inputs = vmecpp.VmecInput.from_file(args.input)
    if not inputs.lfreeb:
        message = "A free-boundary input is required"
        raise ValueError(message)
    inputs.ftol_array[-1] = args.ftol
    inputs.niter_array = np.maximum(inputs.niter_array, args.iterations)
    inputs.return_outputs_even_if_not_converged = False
    inputs.mgrid_file = ""
    inputs.save(folder / "input.json")
    provenance = {"input_sha256": digest(args.input), "ftol": args.ftol}
    started = time.perf_counter()
    if args.coils:
        if args.parameters is None:
            message = "--parameters is required with --coils"
            raise ValueError(message)
        parameters = vmecpp.MakegridParameters.from_file(args.parameters)
        parameters.number_of_r_grid_points = args.finest
        parameters.number_of_z_grid_points = args.finest
        field = vmecpp.MagneticFieldResponseTable.from_coils_file(
            args.coils, parameters
        )
        arrays = {key: getattr(field, key) for key in ("b_r", "b_p", "b_z")}
        provenance.update(
            coils_sha256=digest(args.coils),
            reference="finer table from identical source coils",
        )
    else:
        with Dataset(args.mgrid) as data:

            def scalar(name):
                return np.asarray(data.variables[name][...]).item()

            parameters = vmecpp.MakegridParameters(
                normalize_by_currents=b"S"
                in np.asarray(data.variables["mgrid_mode"][...]).tobytes(),
                assume_stellarator_symmetry=not inputs.lasym,
                number_of_field_periods=int(scalar("nfp")),
                r_grid_minimum=scalar("rmin"),
                r_grid_maximum=scalar("rmax"),
                z_grid_minimum=scalar("zmin"),
                z_grid_maximum=scalar("zmax"),
                number_of_r_grid_points=int(scalar("ir")),
                number_of_z_grid_points=int(scalar("jz")),
                number_of_phi_grid_points=int(scalar("kp")),
            )
            arrays = {
                key: np.stack(
                    [
                        np.asarray(data.variables[f"{prefix}_{i:03d}"][...]).ravel()
                        for i in range(1, int(scalar("nextcur")) + 1)
                    ]
                )
                for key, prefix in (("b_r", "br"), ("b_p", "bp"), ("b_z", "bz"))
            }
        provenance.update(
            mgrid_sha256=digest(args.mgrid),
            reference="original supplied table; no independent finer coil data",
        )
    provenance["table_preparation_seconds"] = time.perf_counter() - started
    np.savez_compressed(
        folder / "response.npz",
        b_r=arrays["b_r"],
        b_p=arrays["b_p"],
        b_z=arrays["b_z"],
    )
    save(folder / "field.json", parameters.model_dump(mode="json"))
    provenance.update(
        response_sha256=digest(folder / "response.npz"),
        parameters_sha256=digest(folder / "field.json"),
        prepared_input_sha256=digest(folder / "input.json"),
    )
    save(folder / "provenance.json", provenance)
    return parameters.number_of_r_grid_points


def worker(args):
    load_started = time.perf_counter()
    folder = args.output.parent / "inputs"
    inputs = vmecpp.VmecInput.from_file(folder / "input.json")
    inputs.mgrid_interpolation = vmecpp.MGridInterpolation(args.scheme)
    parameters = vmecpp.MakegridParameters(
        **json.loads((folder / "field.json").read_text())
    )
    nr = parameters.number_of_r_grid_points
    nz = parameters.number_of_z_grid_points
    stride = (nr - 1) // (args.grid - 1)
    if stride < 1 or (nr - 1) % (args.grid - 1) or (nz - 1) % stride:
        message = "The smaller grid must be an exact stride of the source grid"
        raise ValueError(message)
    with np.load(folder / "response.npz") as archive:
        arrays = {}
        for key in ("b_r", "b_p", "b_z"):
            values = archive[key]
            grid = values.reshape(-1, parameters.number_of_phi_grid_points, nz, nr)
            arrays[key] = np.ascontiguousarray(grid[:, :, ::stride, ::stride]).reshape(
                values.shape[0], -1
            )
    parameters.number_of_r_grid_points = args.grid
    parameters.number_of_z_grid_points = (nz - 1) // stride + 1
    field = vmecpp.MagneticFieldResponseTable(parameters=parameters, **arrays)
    inputs.save(args.output / "input.json")
    result = {
        "scheme": args.scheme,
        "grid": args.grid,
        "z_points": parameters.number_of_z_grid_points,
        "repeat": args.repeat,
        "input_sha256": digest(args.output / "input.json"),
        "converged": False,
        "started_at": time.time(),
        "load_seconds": time.perf_counter() - load_started,
    }
    started = time.perf_counter()
    try:
        output = vmecpp.run(inputs, field, max_threads=args.threads, verbose=False)
        result["seconds"] = time.perf_counter() - started
        w = output.wout
        residuals = [float(w.fsqr), float(w.fsqz), float(w.fsql)]
        keys = [
            "rmnc",
            "zmns",
            "lmns",
            "bmnc",
            "bsupumnc",
            "bsupvmnc",
            "iotaf",
            "presf",
            "phi",
        ]
        if inputs.lasym:
            keys += ["rmns", "zmnc", "lmnc", "bmns", "bsupumns", "bsupvmns"]
        finite = all(np.isfinite(getattr(w, key)).all() for key in keys)
        result.update(
            iterations=int(w.itfsq),
            residuals=residuals,
            converged=bool(
                w.ier_flag == 0
                and w.ns == inputs.ns_array[-1]
                and w.mpol == inputs.mpol
                and w.ntor == inputs.ntor
                and finite
                and np.isfinite(residuals).all()
                and max(residuals) <= inputs.ftol_array[-1]
            ),
        )
        with gzip.open(args.output / "output.json.gz", "wt") as stream:
            stream.write(output.model_dump_json())
    except RuntimeError as error:
        result.update(seconds=time.perf_counter() - started, error=str(error))
    result["finished_at"] = time.time()
    save(args.output / "result.json", result)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    finest = prepare(args)
    script = args.output / "runner.py"
    shutil.copy2(__file__, script)
    native = Path(vmecpp._vmecpp.__file__)
    save(
        args.output / "environment.json",
        {
            "script_sha256": digest(script),
            "threads": args.threads,
            "python": sys.version,
            "package": version("vmecpp"),
            "binaries": {
                p.name: digest(p)
                for p in native.parent.iterdir()
                if p.suffix in (".so", ".dylib")
            },
        },
    )
    pairs = [(grid, repeat) for grid in args.grids for repeat in range(args.repeats)]
    rng = random.Random(20260907)
    rng.shuffle(pairs)
    schedule = [(finest, -1, "cubic")]
    for grid, repeat in pairs:
        schemes = ["linear", "cubic"]
        rng.shuffle(schemes)
        schedule.extend((grid, repeat, scheme) for scheme in schemes)
    save(args.output / "schedule.json", schedule)
    rows = []
    reference = None
    for grid, repeat, scheme in schedule:
        target = args.output / f"{scheme}-{grid}-{repeat}"
        target.mkdir()
        command = [
            sys.executable,
            str(script),
            "--worker",
            "--output",
            str(target),
            "--grid",
            str(grid),
            "--scheme",
            scheme,
            "--repeat",
            str(repeat),
            "--threads",
            str(args.threads),
        ]
        with (target / "process.log").open("w") as log:
            try:
                process = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                    check=False,
                )
                result = (
                    json.loads((target / "result.json").read_text())
                    if (target / "result.json").exists()
                    else {"converged": False, "exit_code": process.returncode}
                )
            except subprocess.TimeoutExpired:
                result = {"converged": False, "timeout": args.timeout}
        result.update(scheme=scheme, grid=grid, repeat=repeat)
        if result["converged"]:
            with gzip.open(target / "output.json.gz", "rt") as stream:
                w = json.load(stream)["wout"]
            if reference is None:
                reference = w
            for key in ("rmnc", "zmns", "lmns", "bmnc", "iotaf"):
                result[key + "_maximum_difference"] = float(
                    np.max(np.abs(np.asarray(w[key]) - np.asarray(reference[key])))
                )
        elif repeat == -1:
            save(args.output / "results.json", [result])
            message = "The reference solve failed; see its process log"
            raise RuntimeError(message)
        rows.append(result)
        save(args.output / "results.json", rows)
        print(json.dumps(result), flush=True)  # noqa: T201


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--coils", type=Path)
    source.add_argument("--mgrid", type=Path)
    parser.add_argument("--parameters", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--finest", type=int, default=401)
    parser.add_argument("--grids", type=int, nargs="+", default=[51, 101])
    parser.add_argument("--ftol", type=float, default=1e-14)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--grid", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--scheme", choices=["linear", "cubic"], help=argparse.SUPPRESS)
    parser.add_argument("--repeat", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.worker:
        worker(args)
    else:
        if args.input is None or not (args.coils or args.mgrid):
            parser.error("Supply --input and either --coils or --mgrid")
        if min(args.grids) < 2 or min(args.repeats, args.threads) < 1:
            parser.error(
                "Grid sizes must be at least two; repeats and threads positive"
            )
        run(args)


if __name__ == "__main__":
    main()
