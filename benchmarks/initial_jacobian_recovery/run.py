# SPDX-License-Identifier: MIT
"""Compare VMEC++ installations on frozen QUASR initialization failures.

Prepare inputs once, then pass one or more (label, Python executable) pairs to run. Each
equilibrium is evaluated in a fresh process with a wall-time limit.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from simsopt._core import load
from simsopt.field import BiotSavart
from simsopt.geo import SurfaceRZFourier

import vmecpp

REPO = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("cases.csv")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def cases(args):
    with args.manifest.open() as stream:
        rows = list(csv.DictReader(stream))
    if args.subset != "all":
        rows = [r for r in rows if r["subset"] == args.subset]
    if args.ids:
        rows = [r for r in rows if int(r["case_id"]) in args.ids]
    return rows


def prepare(args):
    helper_path = REPO / "tests/test_free_boundary_quasr.py"
    spec = importlib.util.spec_from_file_location("quasr_inputs", helper_path)
    assert spec is not None
    assert spec.loader is not None
    helper: Any = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    serials = args.work_dir / "serials"
    serials.mkdir(parents=True, exist_ok=True)
    for row in cases(args):
        cid = int(row["case_id"])
        serial = serials / f"serial{cid:07d}.json"
        if not serial.exists():
            cached = args.serial_cache / serial.name if args.serial_cache else None
            if cached and cached.exists():
                shutil.copy2(cached, serial)
            else:
                prefix = f"{cid:07d}"[:4]
                url = f"https://quasr.flatironinstitute.org/simsopt_serials/{prefix}/{serial.name}"
                with urllib.request.urlopen(url, timeout=60) as response:
                    serial.write_bytes(response.read())
        if sha256(serial) != row["serial_sha256"]:
            message = f"QUASR source hash changed for {cid}"
            raise ValueError(message)
        folder = args.work_dir / "inputs" / str(cid)
        if folder.exists():
            raise FileExistsError(folder)
        folder.mkdir(parents=True)
        helper.LOCAL_DATA_DIR = serials
        started = time.perf_counter()
        config = helper._load_config(cid)
        inp = helper._make_input(config, helper.VACUUM_PROFILE, helper.NS_ARRAY)
        inp.ftol_array = np.full(len(inp.ns_array), args.ftol)
        inp.niter_array = np.full(len(inp.ns_array), args.niter, dtype=np.int64)
        inp.return_outputs_even_if_not_converged = False
        inp.save(folder / "input.json")
        field = config.response
        np.savez_compressed(
            folder / "response.npz", b_r=field.b_r, b_p=field.b_p, b_z=field.b_z
        )
        save_json(folder / "field.json", field.parameters.model_dump(mode="json"))
        save_json(
            folder / "provenance.json",
            {
                **row,
                "input_sha256": sha256(folder / "input.json"),
                "response_sha256": sha256(folder / "response.npz"),
                "helper_sha256": sha256(helper_path),
                "prepare_seconds": time.perf_counter() - started,
            },
        )
        logging.info("Prepared %s", cid)


def normal_field(wout_path, serial_path, nfp):
    _, coils = load(str(serial_path))
    surface = SurfaceRZFourier.from_wout(
        str(wout_path),
        quadpoints_phi=(np.arange(48) + 0.37) / (48 * nfp),
        quadpoints_theta=(np.arange(96) + 0.23) / 96,
    )
    field = BiotSavart(coils)
    field.set_points(surface.gamma().reshape(-1, 3))
    b = field.B()
    normal = surface.normal().reshape(-1, 3)
    area = np.linalg.norm(normal, axis=1)
    ratio = np.sum(b * normal, axis=1) / (area * np.linalg.norm(b, axis=1))
    return float(np.sqrt(np.sum(area * ratio**2) / np.sum(area)))


def worker(args):
    folder = args.work_dir / "inputs" / str(args.case_id)
    inp = vmecpp.VmecInput.from_file(folder / "input.json")
    with np.load(folder / "response.npz") as values:
        field = vmecpp.MagneticFieldResponseTable(
            parameters=json.loads((folder / "field.json").read_text()), **dict(values)
        )
    inp.save(args.output / "input.json")
    native = Path(vmecpp._vmecpp.__file__)
    record = {
        "case_id": args.case_id,
        "label": args.label,
        "repeat": args.repeat,
        "threads": args.threads,
        "input_sha256": sha256(folder / "input.json"),
        "response_sha256": sha256(folder / "response.npz"),
        "package_path": str(Path(vmecpp.__file__).resolve()),
        "binary_sha256": {
            p.name: sha256(p)
            for p in native.parent.iterdir()
            if p.suffix in (".so", ".dylib")
        },
        "converged": False,
    }
    started = time.perf_counter()
    phase = "solve"
    try:
        out = vmecpp.run(
            inp, magnetic_field=field, max_threads=args.threads, verbose=False
        )
        record["solve_seconds"] = time.perf_counter() - started
        w = out.wout
        residuals = [float(w.fsqr), float(w.fsqz), float(w.fsql)]
        record.update(
            ier_flag=int(w.ier_flag),
            ns=int(w.ns),
            residuals=residuals,
            ftol=float(inp.ftol_array[-1]),
            iterations=int(w.itfsq),
            converged=bool(
                w.ier_flag == 0
                and w.ns == inp.ns_array[-1]
                and np.all(np.isfinite(residuals))
                and max(residuals) <= inp.ftol_array[-1]
            ),
        )
        phase = "output"
        w.save(args.output / "wout.nc")
        with gzip.open(args.output / "output.json.gz", "wt") as stream:
            stream.write(out.model_dump_json())
        phase = "validation"
        started_check = time.perf_counter()
        record["coil_normal_field_rms"] = normal_field(
            args.output / "wout.nc",
            args.work_dir / "serials" / f"serial{args.case_id:07d}.json",
            w.nfp,
        )
        record["validation_seconds"] = time.perf_counter() - started_check
    except Exception as error:
        record.setdefault("solve_seconds", time.perf_counter() - started)
        record[f"{phase}_error"] = f"{type(error).__name__}: {error}"
    save_json(args.output / "result.json", record)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    runners = args.runner or [("current", sys.executable)]
    groups = [
        (int(row["case_id"]), rep) for row in cases(args) for rep in range(args.repeats)
    ]
    rng = random.Random(20260906)
    rng.shuffle(groups)
    schedule = []
    for cid, rep in groups:
        paired = [(cid, rep, label, python) for label, python in runners]
        rng.shuffle(paired)
        schedule.extend(paired)
    save_json(args.output / "schedule.json", schedule)
    save_json(
        args.output / "run.json",
        {
            "script_sha256": sha256(__file__),
            "manifest_sha256": sha256(args.manifest),
            "threads": args.threads,
            "timeout_seconds": args.timeout,
            "runners": runners,
            "repeats": args.repeats,
        },
    )
    shutil.copy2(__file__, args.output / "runner.py")
    for cid, rep, label, python in schedule:
        target = args.output / f"{cid}-{label}-{rep}"
        target.mkdir()
        command = [
            python,
            str(Path(__file__).resolve()),
            "worker",
            "--work-dir",
            str(args.work_dir),
            "--output",
            str(target),
            "--case-id",
            str(cid),
            "--label",
            label,
            "--repeat",
            str(rep),
            "--threads",
            str(args.threads),
        ]
        env = os.environ.copy()
        env.update(
            OMP_NUM_THREADS=str(args.threads),
            OPENBLAS_NUM_THREADS="1",
            VECLIB_MAXIMUM_THREADS="1",
        )
        started = time.perf_counter()
        with (target / "process.log").open("w") as stream:
            try:
                result = subprocess.run(
                    command,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                    check=False,
                )
                status = {"returncode": result.returncode}
            except subprocess.TimeoutExpired:
                status = {"timeout_seconds": args.timeout}
        record = (
            json.loads((target / "result.json").read_text())
            if (target / "result.json").exists()
            else {"case_id": cid, "label": label, "repeat": rep, "converged": False}
        )
        record.update(status, process_seconds=time.perf_counter() - started)
        save_json(target / "process.json", status)
        with (args.output / "results.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        logging.info("%s %s: converged=%s", cid, label, record["converged"])


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep, execute, child = (
        sub.add_parser(name) for name in ("prepare", "run", "worker")
    )
    for item in (prep, execute, child):
        item.add_argument("--work-dir", type=Path, required=True)
    for item in (prep, execute):
        item.add_argument("--manifest", type=Path, default=MANIFEST)
        item.add_argument(
            "--subset", choices=["all", "discovery", "held-out"], default="all"
        )
        item.add_argument("--ids", type=int, nargs="*")
    prep.add_argument("--serial-cache", type=Path)
    prep.add_argument("--ftol", type=float, default=1e-9)
    prep.add_argument("--niter", type=int, default=4000)
    execute.add_argument(
        "--runner", action="append", nargs=2, metavar=("LABEL", "PYTHON")
    )
    execute.add_argument("--repeats", type=int, default=1)
    execute.add_argument("--timeout", type=float, default=120)
    for item in (execute, child):
        item.add_argument("--output", type=Path, required=True)
        item.add_argument("--threads", type=int, default=4)
    child.add_argument("--case-id", type=int, required=True)
    child.add_argument("--label", required=True)
    child.add_argument("--repeat", type=int, required=True)
    args = parser.parse_args()
    args.work_dir = args.work_dir.resolve()
    if hasattr(args, "output"):
        args.output = args.output.resolve()
    {"prepare": prepare, "run": run, "worker": worker}[args.command](args)


if __name__ == "__main__":
    main()
