#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
#
# Build and run the C++ Google Benchmark microbenchmarks, merging their JSON
# outputs into a single file consumable by benchmark-action/github-action-benchmark
# with tool: "googlecpp".
#
# Output: writes cpp_benchmark_results.json to the repository root (the current
# working directory when this script is invoked from CI).
#
# Usage:
#   ./benchmarks/run_cpp_benchmarks.sh [output_file]
#
# The single merged JSON has the Google Benchmark schema the action expects:
# a top-level object with a "context" object and a "benchmarks" array.

set -euo pipefail

# Resolve paths relative to this script so it works regardless of CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CPP_DIR="${REPO_ROOT}/src/vmecpp/cpp"

# Absolute path for the merged output so a later `cd` doesn't confuse it.
OUT_FILE="${1:-${REPO_ROOT}/cpp_benchmark_results.json}"
case "${OUT_FILE}" in
  /*) ;;                                   # already absolute
  *) OUT_FILE="$(pwd)/${OUT_FILE}" ;;      # make relative paths absolute
esac

# The benchmark targets to build and run.  Each is a Google Benchmark binary
# (cc_binary linking @google_benchmark//:benchmark_main).
TARGETS=(
  "//vmecpp/vmec/ideal_mhd_model:fft_toroidal_bench"
  "//vmecpp/vmec/ideal_mhd_model:dealias_constraint_force_bench"
  "//vmecpp/free_boundary/laplace_solver:laplace_solver_bench"
  "//vmecpp/vmec/output_quantities:output_quantities_bench"
)

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

cd "${CPP_DIR}"

# --config=perf: optimized build with frame pointers and symbols retained,
# matching the existing .bazelrc convention for performance work.
echo "Building C++ benchmark targets..."
bazel build --config=perf -- "${TARGETS[@]}"

# Run each target.  Use `bazel run` so runfiles (e.g. test_data for the
# output-quantities benchmark) are available in the working directory.
echo "Running C++ benchmarks..."
for target in "${TARGETS[@]}"; do
  name="${target##*:}"
  out="${TMP_DIR}/${name}.json"
  # --benchmark_min_time keeps CI runtime bounded while still averaging enough
  # iterations for a stable measurement.
  bazel run --config=perf -- "${target}" \
    --benchmark_min_time=0.2s \
    --benchmark_format=json \
    --benchmark_out="${out}"
done

# Merge all per-target JSON files into one, concatenating their "benchmarks"
# arrays and taking "context" from the first run.
echo "Merging benchmark results into ${OUT_FILE}..."
python3 - "${TMP_DIR}" "${OUT_FILE}" <<'PY'
import glob
import json
import os
import sys

tmp_dir, out_file = sys.argv[1], sys.argv[2]
merged = {"context": None, "benchmarks": []}
skipped = 0
for path in sorted(glob.glob(os.path.join(tmp_dir, "*.json"))):
    with open(path) as f:
        data = json.load(f)
    if merged["context"] is None:
        merged["context"] = data.get("context")
    for bench in data.get("benchmarks", []):
        # Drop benchmarks that reported an error (e.g. an FFT-path benchmark
        # skipped because no FFTX codelet exists for that resolution): they
        # carry no valid timing for the regression tracker to consume.
        if bench.get("error_occurred"):
            skipped += 1
            continue
        merged["benchmarks"].append(bench)

with open(out_file, "w") as f:
    json.dump(merged, f, indent=2)

print(f"Merged {len(merged['benchmarks'])} benchmark entries "
      f"({skipped} skipped/errored).")
PY

echo "Done."
