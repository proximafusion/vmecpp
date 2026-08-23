# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Convert pytest-benchmark or Google Benchmark JSON to seconds.

github-action-benchmark's built-in "pytest" and "googlecpp" tool parsers plot whatever
value each framework happens to report natively -- pytest-benchmark's "iter/sec" and
Google Benchmark's raw per-iteration time in its own time_unit (commonly nanoseconds).
Charting those side by side, or even a single suite whose time_unit isn't fixed,
requires the reader (or the chart's JS) to know and convert between units.

This script normalizes both to plain wall-clock seconds and emits the action's generic
"customSmallerIsBetter" schema, so the chart always plots seconds and the benchmarks.js
frontend needs no unit-detection logic.
"""

import argparse
import json


def convert_pytest(data):
    results = []
    for bench in data["benchmarks"]:
        stats = bench["stats"]
        results.append(
            {
                "name": bench["fullname"],
                "unit": "seconds",
                "value": stats["mean"],
                "range": f"stddev: {stats['stddev']}",
                "extra": f"rounds: {stats['rounds']}",
            }
        )
    return results


_GOOGLECPP_TIME_UNIT_TO_SECONDS = {
    "s": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "ns": 1e-9,
}


def convert_googlecpp(data):
    results = []
    for bench in data["benchmarks"]:
        if bench.get("error_occurred"):
            continue
        factor = _GOOGLECPP_TIME_UNIT_TO_SECONDS[bench["time_unit"]]
        results.append(
            {
                "name": bench["name"],
                "unit": "seconds",
                "value": bench["real_time"] * factor,
                "extra": (
                    f"iterations: {bench['iterations']}\n"
                    f"cpu: {bench['cpu_time'] * factor} seconds\n"
                    f"threads: {bench.get('threads', 1)}"
                ),
            }
        )
    return results


_CONVERTERS = {
    "pytest": convert_pytest,
    "googlecpp": convert_googlecpp,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("format", choices=sorted(_CONVERTERS))
    parser.add_argument("input", help="Raw benchmark JSON produced by the tool.")
    parser.add_argument("output", help="Where to write the customSmallerIsBetter JSON.")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    results = _CONVERTERS[args.format](data)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
