"""Plots 2/3: native interim Taylor remainders and signed-mode FD comparisons."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tests"))

from adjoint_verification.harness import INPUT_NAMES  # noqa: E402 # pyright: ignore
from adjoint_verification.reference import (  # noqa: E402 # pyright: ignore
    COMPONENT_STEPS,
    DEFAULT_STEPS,
    OBJECTIVES,
    ReferenceStudy,
    reference_input,
)
from plot_style import STYLE, mode_color, save_plot  # noqa: E402


def plot_taylor(study, results, output, case):
    import matplotlib.pyplot as plt  # noqa: PLC0415

    for result in results:
        if result["status"] == "inactive":
            continue
        name = result["objective"]
        direction = result["direction_name"]
        with plt.rc_context(STYLE):
            figure, axes = plt.subplots()
            h, error = result["steps"], result["remainder"]
            positive = error > 0
            axes.loglog(h[positive], error[positive], "o-", label="absolute remainder")
            if positive.any() and result["slope_applicable"]:
                anchor = int(np.flatnonzero(positive)[0])
                axes.loglog(h, error[anchor] * (h / h[anchor]) ** 2, "--", label="h^2")
            axes.axhline(10 * result["noise_floor"], color="gray", label="10 x noise")
            axes.set(xlabel="boundary step h", ylabel="|J(b+hd)-J(b)-h g.d|")
            title = f"{case}: {name}, {direction}; {result['status']}"
            if result["slope_applicable"]:
                title += (
                    f", slope={result['fit_slope']:.2f}, points={result['fit_stop']}"
                )
            else:
                title += "; prescribed iota, no slope test"
            axes.set_title(title)
            axes.legend()
            save_plot(
                figure,
                output,
                f"plot_02_taylor_{case}_{name}_{direction}",
                **result,
                boundary=study.boundary,
                base_state=np.asarray(study.model.get_state()),
                noise_probe_direction=study.noise_probe_direction,
                gradient_ad=study.gradients[name],
                solve_records=np.asarray(study.records),
                solve_record_columns=np.asarray(
                    ["status", "ftol", "fsqr", "fsqz", "fsql"]
                ),
                solve_failures=np.asarray(study.failures),
                provenance=study.provenance(),
            )
            plt.close(figure)


def plot_components(study, results, output, case):
    import matplotlib.pyplot as plt  # noqa: PLC0415

    for result in results:
        # Plot a fixed middle step, never an error-minimizing per-mode choice.
        middle = len(result["steps"]) // 2
        ad = result["gradient_ad"]
        fd = result["gradient_fd"][middle]
        active = result["active"]
        with plt.rc_context(STYLE):
            figure, axes = plt.subplots(1, 3, figsize=(11, 3.5))
            poloidal = np.broadcast_to(result["m"][None, :, None], ad.shape)
            for m in result["m"]:
                selected = active & (poloidal == m)
                axes[0].scatter(
                    ad[selected],
                    fd[selected],
                    s=8,
                    color=mode_color(int(m)),
                    label=f"m={m}",
                )
            extent = max(
                np.max(np.abs(ad[active])), np.max(np.abs(fd[active])), 1.0e-15
            )
            axes[0].plot([-extent, extent], [-extent, extent], "k--")
            axes[0].set(xlabel="exact AD derivative", ylabel="central FD derivative")
            axes[0].legend(fontsize=7)
            for kind, axis in enumerate(axes[1:]):
                error = result["relative_error"][middle, kind].copy()
                error[~active[kind] | result["near_zero"][middle, kind]] = np.nan
                image = axis.imshow(
                    error,
                    origin="lower",
                    aspect="auto",
                    extent=(
                        result["n"][0] - 0.5,
                        result["n"][-1] + 0.5,
                        -0.5,
                        len(result["m"]) - 0.5,
                    ),
                )
                axis.set(xlabel="signed n", ylabel="m", title=("rbc", "zbs")[kind])
                figure.colorbar(image, ax=axis, label="relative error")
            name = result["objective"]
            figure.suptitle(
                f"{case}: {name}; {result['status']}; h={result['steps'][middle]:.1e}"
            )
            figure.tight_layout()
            save_plot(
                figure,
                output,
                f"plot_03_components_{case}_{name}",
                **result,
                plotted_step_index=np.asarray(middle),
                boundary=study.boundary,
                base_state=np.asarray(study.model.get_state()),
                noise_probe_direction=study.noise_probe_direction,
                solve_records=np.asarray(study.records),
                solve_record_columns=np.asarray(
                    ["status", "ftol", "fsqr", "fsqz", "fsql"]
                ),
                solve_failures=np.asarray(study.failures),
                provenance=study.provenance(),
            )
            plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", nargs="+", choices=("small", *INPUT_NAMES), default=["small"]
    )
    parser.add_argument(
        "--objectives", nargs="+", choices=OBJECTIVES, default=["boundary"]
    )
    parser.add_argument(
        "--studies", nargs="+", choices=("taylor", "components"), default=["taylor"]
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        choices=("random", "m1", "highm", "highn"),
        default=["random"],
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=float,
        help="decreasing step series; defaults: 11 Taylor steps, 3 component steps",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hot-restart", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "resume per-case component checkpoints in --output-dir; "
            "Taylor studies are not rerun"
        ),
    )
    args = parser.parse_args()
    if args.resume:
        if "components" not in args.studies:
            parser.error("--resume requires --studies components")
        for case in args.cases:
            checkpoint = args.output_dir / f"components_{case}.checkpoint.npz"
            if not checkpoint.is_file():
                parser.error(f"--resume requires component checkpoint {checkpoint}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failed = False
    for case in args.cases:
        try:
            study = ReferenceStudy(
                reference_input(case),
                args.objectives,
                seed=args.seed,
                hot_restart=args.hot_restart,
            )
            summaries = []
            for name in args.studies:
                if args.resume and name == "taylor":
                    summaries.append(
                        {
                            "study": "taylor",
                            "status": "skipped",
                            "reason": "--resume resumes components only",
                        }
                    )
                    continue
                results = (
                    study.taylor(
                        DEFAULT_STEPS if args.steps is None else args.steps,
                        args.directions,
                    )
                    if name == "taylor"
                    else study.components(
                        COMPONENT_STEPS if args.steps is None else args.steps,
                        checkpoint=args.output_dir
                        / f"components_{case}.checkpoint.npz",
                        resume=args.resume,
                    )
                )
                if name == "taylor":
                    plot_taylor(study, results, args.output_dir, case)
                else:
                    plot_components(study, results, args.output_dir, case)
                for result in results:
                    summary = {
                        key: result[key]
                        for key in ("objective", "direction_name", "status", "passed")
                        if key in result
                    }
                    summary["study"] = name
                    summaries.append(summary)
                    failed |= result["status"] not in ("pass", "inactive", "constant")
            report = {
                "case": case,
                "provenance": json.loads(study.provenance()),
                "results": summaries,
            }
        except (OSError, RuntimeError, ValueError) as error:
            report = {
                "case": case,
                "status": "blocked",
                "error": str(error),
                "input": json.loads(reference_input(case).to_json()),
                "steps": {
                    name: (
                        args.steps
                        if args.steps is not None
                        else (
                            DEFAULT_STEPS if name == "taylor" else COMPONENT_STEPS
                        ).tolist()
                    )
                    for name in args.studies
                },
                "objectives": args.objectives,
                "studies": args.studies,
                "seed": args.seed,
                "reference": "native VmecModel.solve, interim ftol=1e-16",
            }
            failed = True
        (args.output_dir / f"reference_{case}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        sys.stdout.write(json.dumps(report) + "\n")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
