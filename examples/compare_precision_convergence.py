"""Compare force-balance convergence between double and long-double builds.

Workflow:
  1. On the `main` (double) branch:    python examples/compare_precision_convergence.py --save-as double
  2. On this `long double` branch:     python examples/compare_precision_convergence.py --save-as long_double
  3. Either of the above runs, with no --save-as, will plot whichever traces
     have been saved so far (so step 3 is implicit if you run with --save-as
     on the second branch).

Each run drops a JSON next to this script with the four force-residual traces
(R, Z, lambda, total). When more than one trace file is present, an overlay
plot is shown.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

import vmecpp

INPUT_FILE = Path("examples/data/w7x.json")
TRACE_DIR = Path(__file__).parent / "convergence_traces"


def run_and_extract() -> dict:
    indata = vmecpp.VmecInput.from_file(INPUT_FILE)
    output = vmecpp.run(indata)
    wout = output.wout
    return {
        "force_residual_r": np.asarray(wout.force_residual_r).flatten().tolist(),
        "force_residual_z": np.asarray(wout.force_residual_z).flatten().tolist(),
        "force_residual_lambda": (
            np.asarray(wout.force_residual_lambda).flatten().tolist()
        ),
        "fsqt": np.asarray(wout.fsqt).flatten().tolist(),
        "ftolv": float(wout.ftolv),
    }


def save_trace(label: str, trace: dict) -> Path:
    TRACE_DIR.mkdir(exist_ok=True)
    path = TRACE_DIR / f"{label}.json"
    path.write_text(json.dumps(trace))
    return path


def plot_overlay() -> None:
    if not TRACE_DIR.exists():
        print("No traces saved yet.")
        return
    traces = sorted(TRACE_DIR.glob("*.json"))
    if not traces:
        print("No trace files in", TRACE_DIR)
        return

    components = [
        ("force_residual_r", "FSQR"),
        ("force_residual_z", "FSQZ"),
        ("force_residual_lambda", "FSQL"),
        ("fsqt", "FSQ_total"),
    ]
    fig = go.Figure()
    for path in traces:
        label = path.stem
        data = json.loads(path.read_text())
        for key, short in components:
            fig.add_trace(
                go.Scatter(
                    y=data[key],
                    mode="lines",
                    name=f"{short} [{label}]",
                    legendgroup=label,
                )
            )
    # Use the smallest tolerance found across the saved runs as the reference
    tols = [json.loads(p.read_text())["ftolv"] for p in traces]
    fig.add_hline(
        y=min(tols),
        line={"color": "red", "dash": "dash"},
        annotation_text="ftolv",
        annotation_position="bottom right",
    )
    fig.update_yaxes(type="log", title="Force residual")
    fig.update_xaxes(title="Iteration")
    fig.update_layout(title="Force-residual convergence: double vs long double")
    fig.show()
    # out_html = TRACE_DIR / "convergence_overlay.html"
    # fig.write_html(str(out_html))
    # print(f"Overlay written to {out_html}")
    print("Final residuals per run:")
    for path in traces:
        d = json.loads(path.read_text())
        print(
            f"  {path.stem:>15s}  "
            f"FSQR={d['force_residual_r'][-1]:.3e}  "
            f"FSQZ={d['force_residual_z'][-1]:.3e}  "
            f"FSQL={d['force_residual_lambda'][-1]:.3e}  "
            f"FSQt={d['fsqt'][-1]:.3e}  "
            f"iters={len(d['fsqt'])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-as",
        help="Label to save this run's trace under (e.g. 'double', 'long_double').",
    )
    args = parser.parse_args()

    if args.save_as:
        trace = run_and_extract()
        path = save_trace(args.save_as, trace)
        print(f"Saved {len(trace['fsqt'])} iterations to {path}")
    plot_overlay()


if __name__ == "__main__":
    main()
