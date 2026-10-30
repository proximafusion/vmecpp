# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH
# SPDX-License-Identifier: MIT
"""Measure lbsubs currents against vacuum and exact 3D equilibria.

Run with --stage before before rebuilding, then --stage after. Results and figures are
written beside this script by default.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import tri

import vmecpp

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from examples import exact_equilibria as ee  # noqa: E402

MU_0 = 4e-7 * np.pi
MEMBERS = ee.MEMBERS | {
    "sheared-highbeta": dataclasses.replace(
        ee.SHEARED, name="sheared-highbeta", k_b=0.4
    )
}


def make_input(case, ns):
    if case.startswith("sheared"):
        member = MEMBERS[case]
        indata = member.vmec_input(ns, 12, 10)
    else:
        indata = vmecpp.VmecInput.from_file(ROOT / "examples/data/input.w7x")
        indata = indata.model_copy(deep=True)
        if case.startswith("vacuum"):
            indata.am = np.zeros(1)
            indata.ncurr = 1
            indata.curtor = 0.0
            indata.ac = np.zeros(1)
            if case == "vacuum-asym":
                ntor = int(np.asarray(indata.ntor).item())
                rbs = np.zeros_like(indata.rbc)
                zbc = np.zeros_like(indata.zbs)
                rbs[1, ntor + 1] = 0.003
                zbc[1, ntor - 1] = 0.002
                indata = indata.model_copy(
                    update={
                        "lasym": True,
                        "rbs": rbs,
                        "zbc": zbc,
                        "raxis_s": np.zeros(ntor + 1),
                        "zaxis_c": np.zeros(ntor + 1),
                    }
                )
            if "-m" in case:
                modes = int(case.split("-m")[1])
                indata.resize(modes, modes)
                indata.nzeta = 3 * modes
        else:
            indata.am = np.array([6.5e4, -6.5e4, 0.0, 0.0, -6.5e4, 6.5e4])
            if case == "w7x-highbeta":
                indata.am *= 4.0
    grids = [n for n in (9, 17) if n < ns] + [ns]
    if case == "sheared-A":
        grids = [ns]
    indata.ns_array = np.asarray(grids)
    indata.ftol_array = np.full(len(grids), 1e-16)
    if "tight" in case:
        indata.ftol_array[:] = 1e-20
    indata.niter_array = np.full(len(grids), 20000)
    indata.lbsubs = True
    return indata


def section(wout, zeta=0.0):
    """Physical toroidal current on full surfaces, from exported currents."""
    theta = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    radial = np.arange(1, wout.ns - 1)

    def series(cos_name, sin_name, xm, xn, indices):
        arg = xm[:, None] * theta - xn[:, None] * zeta
        cosine = getattr(wout, cos_name)
        value = np.zeros((len(indices), len(theta)))
        if cosine is not None:
            value += cosine[:, indices].T @ np.cos(arg)
        sine = getattr(wout, sin_name)
        if sine is not None:
            value += sine[:, indices].T @ np.sin(arg)
        return value

    xm, xn = wout.xm, wout.xn
    r = series("rmnc", "rmns", xm, xn, radial)
    z = series("zmnc", "zmns", xm, xn, radial)
    xm, xn = wout.xm_nyq, wout.xn_nyq
    sqrtg = 0.5 * (
        series("gmnc", "gmns", xm, xn, radial)
        + series("gmnc", "gmns", xm, xn, radial + 1)
    )
    current = r * series("currvmnc", "currvmns", xm, xn, radial) / sqrtg
    return r, z, current


def exact_current(member, r, z, phi, h=1e-5):
    """Curl of the analytical Cartesian field with a five-point derivative."""
    xyz = np.array([r * np.cos(phi), r * np.sin(phi), z])
    derivatives = []
    for axis in np.eye(3):
        step = h * axis[:, None, None]
        samples = [
            np.array(member.field(*(xyz + k * step))[:3]) for k in (-2, -1, 1, 2)
        ]
        derivatives.append(
            (samples[0] - 8 * samples[1] + 8 * samples[2] - samples[3]) / (12 * h)
        )
    jx = derivatives[1][2] - derivatives[2][1]
    jy = derivatives[2][0] - derivatives[0][2]
    return (-np.sin(phi) * jx + np.cos(phi) * jy) / MU_0


def normalize_metrics(metrics):
    case = metrics["case"]
    metrics["reference_kind"] = (
        "analytical_curl"
        if case.startswith("sheared")
        else "zero_current"
        if case.startswith("vacuum")
        else "none"
    )
    if metrics["reference_kind"] == "none":
        for key in list(metrics):
            if "error_A_m2" in key:
                metrics[key.replace("error_A_m2", "current_A_m2")] = metrics.pop(key)
        metrics.pop("reference_max_A_m2", None)
    return metrics


def run_case(case, ns, stage, destination):
    indata = make_input(case, ns)
    start = time.perf_counter()
    output = vmecpp.run(indata, max_threads=4, verbose=False)
    elapsed = time.perf_counter() - start
    wout = output.wout
    sections = [section(wout, phi) for phi in np.arange(4) * np.pi / (2 * wout.nfp)]
    r, z, current = (np.stack([s[k] for s in sections]) for k in range(3))
    exact = np.zeros_like(current)
    if case.startswith("sheared"):
        for k, phi in enumerate(np.arange(4) * np.pi / (2 * wout.nfp)):
            exact[k] = exact_current(MEMBERS[case], r[k], z[k], phi)
    error = current - exact
    if case == "w7x":
        error = current
    s = np.arange(1, ns - 1) / (ns - 1)
    metrics = {
        "case": case,
        "ns": ns,
        "stage": stage,
        "lbsubs_requested": bool(indata.lbsubs),
        "seconds": elapsed,
        "fsq": max(float(wout.fsqr), float(wout.fsqz), float(wout.fsql)),
        "beta": float(wout.betatotal),
        "max_error_A_m2": float(np.max(np.abs(error))),
        "rms_error_A_m2": float(np.sqrt(np.mean(error**2))),
        "first_surface_max_error_A_m2": float(np.max(np.abs(error[:, 0]))),
        "bulk_max_error_A_m2": float(np.max(np.abs(error[:, s >= 0.1]))),
        "reference_max_A_m2": float(np.max(np.abs(exact))),
        "DMerc_first": float(wout.DMerc[1]),
    }
    ntor = int(np.asarray(indata.ntor).item())
    nzeta = max(indata.nzeta, 2 * ntor + 4) if ntor else max(indata.nzeta, 1)
    ntheta = output.jxbout.izeta.shape[1] // nzeta
    ntheta_even = ntheta if indata.lasym else 2 * (ntheta - 1)
    phi = 2 * np.pi * np.arange(nzeta) / (nzeta * wout.nfp)
    theta = 2 * np.pi * np.arange(ntheta) / ntheta_even
    angles = (
        wout.xm_nyq[:, None, None] * theta - wout.xn_nyq[:, None, None] * phi[:, None]
    )
    from_wout = wout.currvmnc[:, 1:-1].T @ np.cos(angles).reshape(len(wout.xm_nyq), -1)
    if wout.currvmns is not None:
        from_wout += wout.currvmns[:, 1:-1].T @ np.sin(angles).reshape(
            len(wout.xm_nyq), -1
        )
    metrics["wout_jxbout_max_difference"] = float(
        np.max(np.abs(from_wout - output.jxbout.izeta[1:-1]))
    )
    metrics = normalize_metrics(metrics)
    stem = destination / f"{case}_{ns}_{stage}"
    wout.save(stem.with_suffix(".nc"))
    np.savez_compressed(
        stem.with_suffix(".npz"),
        r=r,
        z=z,
        current=current,
        exact=exact,
        s=s,
        currvmnc=wout.currvmnc,
        currvmns=wout.currvmns,
        xm=wout.xm_nyq,
        xn=wout.xn_nyq,
        DMerc=wout.DMerc,
    )
    stem.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")
    logging.info(json.dumps(metrics))


def plot_results(destination):
    def mesh(r, z):
        nrad, ntheta = r.shape
        angle = np.arange(ntheta)
        a = (np.arange(nrad - 1)[:, None] * ntheta + angle).ravel()
        b = (np.arange(nrad - 1)[:, None] * ntheta + (angle + 1) % ntheta).ravel()
        triangles = np.r_[np.c_[a, b, a + ntheta], np.c_[b, b + ntheta, a + ntheta]]
        return tri.Triangulation(r.ravel(), z.ravel(), triangles)

    for path in destination.glob("*_*.json"):
        metrics = normalize_metrics(json.loads(path.read_text()))
        path.write_text(json.dumps(metrics, indent=2) + "\n")
    cases = sorted(
        {
            p.name.split("_")[0]
            for p in destination.glob("*_after.json")
            if p.with_name(p.name.replace("_after", "_before")).exists()
        }
    )
    reference_cases = [case for case in cases if not case.startswith("w7x")]
    fig_convergence, axes = plt.subplots(
        1,
        len(reference_cases),
        figsize=(4 * len(reference_cases), 3.8),
        squeeze=False,
        constrained_layout=True,
    )
    for case in cases:
        paired = [
            json.loads(p.read_text())
            for p in sorted(destination.glob(f"{case}_*_after.json"))
            if p.with_name(p.name.replace("_after", "_before")).exists()
        ]
        paired.sort(key=lambda m: m["ns"])
        ns = paired[-1]["ns"]
        before = np.load(destination / f"{case}_{ns}_before.npz", allow_pickle=True)
        after = np.load(destination / f"{case}_{ns}_after.npz", allow_pickle=True)
        if case == "sheared-asym":
            fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
            all_errors = [d["current"] - d["exact"] for d in (before, after)]
            plane = int(np.argmax(np.max(np.abs(all_errors[0]), axis=(1, 2))))
            errors = [e[plane] for e in all_errors]
            field = "current density" if case.startswith("w7x") else "current error"
            common_limit = max(np.max(np.abs(e)) for e in errors) / 1000
            for ax, data, error, title, limit in zip(
                list(axs.flat)[:3],
                (before, after, after),
                (errors[0], errors[1], errors[1]),
                (
                    f"Before: {field}",
                    f"After: {field} (same scale)",
                    f"After: {field} (expanded scale)",
                ),
                (
                    common_limit,
                    common_limit,
                    max(np.max(np.abs(errors[1])) / 1000, 1e-12),
                ),
                strict=True,
            ):
                artist = ax.tricontourf(
                    mesh(data["r"][plane], data["z"][plane]),
                    error.ravel() / 1000,
                    levels=np.linspace(-limit, limit, 41),
                    cmap="RdBu_r",
                    extend="both",
                )
                fig.colorbar(artist, ax=ax, label="J_phi [kA/m^2]")
                ax.set(title=title, xlabel="R [m]", ylabel="Z [m]")
                ax.set_aspect("equal")
            ax = axs[1, 1]
            for data, label in ((before, "before"), (after, "after")):
                radial_max = np.max(
                    np.abs(data["current"] - data["exact"]), axis=(0, 2)
                )
                ax.semilogy(
                    np.sqrt(data["s"]), np.maximum(radial_max, 1e-12), label=label
                )
            ax.set(
                xlabel="sqrt(s)",
                ylabel=f"max |{field}| [A/m^2]",
                title="Maximum over four toroidal planes",
            )
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.suptitle(f"{case}, ns = {ns}, N_fp phi / pi = {plane / 2:g}")
            fig.savefig(destination.parent / f"{case}_current.png", dpi=150)
            plt.close(fig)

        if case in reference_cases:
            ax = axes[0, reference_cases.index(case)]
            for stage, style in (("before", "o--"), ("after", "o-")):
                metrics = [
                    json.loads(
                        (destination / f"{case}_{p['ns']}_{stage}.json").read_text()
                    )
                    for p in paired
                ]
                ax.loglog(
                    [p["ns"] - 1 for p in metrics],
                    [p["max_error_A_m2"] for p in metrics],
                    style,
                    label=stage,
                )
            ax.set(
                title=case, xlabel="ns - 1", ylabel="max |J_phi - reference| [A/m^2]"
            )
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
    fig_convergence.savefig(destination.parent / "current_convergence.png", dpi=150)
    plt.close(fig_convergence)

    audit_path = destination.parent / "baseline_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for ax, case in zip(axes, ("sheared", "vacuum"), strict=True):
            for run in audit["runs"]:
                if run["case"] != case:
                    continue
                for name, marker in (("wout", "o"), ("jxbout", "x")):
                    ax.semilogy(
                        np.sqrt(run["s"]),
                        run[f"{name}_radial_max_error_A_m2"],
                        marker + ("-" if run["lbsubs"] else "--"),
                        label=f"{name}, lbsubs={run['lbsubs']}",
                    )
            ax.set(
                title=f"Baseline {case}: same native angular nodes",
                xlabel="sqrt(s)",
                ylabel="max |J_phi - reference| [A/m^2]",
            )
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig.savefig(destination.parent / "baseline_current_paths.png", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    trace_cases = ("vacuum", "vacuum-asym", "sheared", "sheared-asym")
    for ax, case in zip(axes.flat, trace_cases, strict=True):
        paths = sorted(
            destination.glob(f"{case}_*_after.npz"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        finest = np.load(paths[-1])
        residual = finest["current"] - finest["exact"]
        plane_index, angle_index = np.unravel_index(
            np.argmax(np.max(np.abs(residual), axis=1)),
            (residual.shape[0], residual.shape[2]),
        )
        for path in paths:
            data = np.load(path)
            error = data["current"] - data["exact"]
            ax.plot(
                np.sqrt(data["s"]),
                error[plane_index, :, angle_index],
                ".-",
                label=f"ns={path.stem.split('_')[1]}",
            )
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set(
            title=(
                f"{case}: N_fp phi / pi = {plane_index / 2:g}, "
                f"theta / pi = {2 * angle_index / residual.shape[2]:.3f}"
            ),
            xlabel="sqrt(s)",
            ylabel="J_phi - reference [A/m^2]",
        )
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.savefig(destination.parent / "signed_residuals.png", dpi=150)
    plt.close(fig)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("before", "after"))
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--cases", nargs="+", default=["vacuum", "w7x", "sheared", "sheared-asym"]
    )
    parser.add_argument("--ns", nargs="+", type=int, default=[17, 33, 65])
    parser.add_argument(
        "--out", type=Path, default=ROOT / "docs/reports/lbsubs/data_final"
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.plot:
        plot_results(args.out)
        return
    if args.stage is None:
        parser.error("--stage is required when running equilibria")
    for case in args.cases:
        for ns in args.ns:
            run_case(case, ns, args.stage, args.out)


if __name__ == "__main__":
    main()
