# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
import platform
import statistics
import time
from datetime import datetime
from pathlib import Path

import vmecpp


def _format_results(
    avg_cold_start,
    std_dev_cold_start,
    cold_start_iterations,
    avg_hot_restart,
    std_dev_hot_restart,
    hot_restart_iterations,
    vmec_input_filename,
    script_name,
    original_cold_start_time=None,
    original_hot_restart_time=None,
):
    """Formats the benchmark results into a human-readable string, similar to the
    BENCHMARK_RESULTS.txt file."""
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    system_info = platform.system()

    output = f"""Benchmark Results Summary

Date: {current_date}
System: {system_info}
Script: {script_name}
Input: {vmec_input_filename}

--- Original Baseline (Before HandoverStorage Refactor) ---
Cold Start Average Time: {original_cold_start_time if original_cold_start_time is not None else 'N/A'} s (Iterations: N/A)
Hot Restart Average Time: {original_hot_restart_time if original_hot_restart_time is not None else 'N/A'} s (Iterations: N/A)
Details: Cold start averaged over 3 runs. Hot restart averaged over 5 runs.

--- Latest Results (After HandoverStorage Refactor and subsequent fixes) ---
Cold Start Average Time: {avg_cold_start:.4f} s +/- {std_dev_cold_start:.4f} s (Iterations: {cold_start_iterations})
Hot Restart Average Time: {avg_hot_restart:.4f} s +/- {std_dev_hot_restart:.4f} s (Iterations: {hot_restart_iterations})
Details: Cold start averaged over 3 runs. Hot restart averaged over 5 runs.

--- Performance Comparison (Latest vs. Original Baseline) ---
Cold Start: """
    if original_cold_start_time is not None:
        if avg_cold_start < original_cold_start_time:
            improvement = (
                (original_cold_start_time - avg_cold_start) / original_cold_start_time
            ) * 100
            output += f"Improved by ~{improvement:.2f}% ({original_cold_start_time:.4f}s -> {avg_cold_start:.4f}s)\n"
        else:
            degradation = (
                (avg_cold_start - original_cold_start_time) / original_cold_start_time
            ) * 100
            output += f"Degraded by ~{degradation:.2f}% ({original_cold_start_time:.4f}s -> {avg_cold_start:.4f}s)\n"
    else:
        output += "N/A\n"

    output += "Hot Restart: "
    if original_hot_restart_time is not None:
        if avg_hot_restart < original_hot_restart_time:
            improvement = (
                (original_hot_restart_time - avg_hot_restart)
                / original_hot_restart_time
            ) * 100
            output += f"Improved by ~{improvement:.2f}% ({original_hot_restart_time:.4f}s -> {avg_hot_restart:.4f}s)\n"
        else:
            degradation = (
                (avg_hot_restart - original_hot_restart_time)
                / original_hot_restart_time
            ) * 100
            output += f"Degraded by ~{degradation:.2f}% ({original_hot_restart_time:.4f}s -> {avg_hot_restart:.4f}s)\n"
    else:
        output += "N/A\n"

    output += """
Note: All hot restart runs involved a 1% perturbation of magnetic field components (b_p, b_r, b_z).
The benchmark measures the full vmecpp.run call, including both the core solver and the NESTOR interface for free-boundary cases.
"""
    return output


def benchmark():
    # Load the VMEC++ JSON indata file.
    TEST_DATA_DIR = Path("src") / "vmecpp" / "cpp" / "vmecpp" / "test_data"
    vmec_input_filename = TEST_DATA_DIR / "cth_like_free_bdy.json"
    coils_fname = TEST_DATA_DIR / "coils.cth_like"
    makegrid_params_fname = TEST_DATA_DIR / "makegrid_parameters_cth_like.json"

    print(f"Loading input from {vmec_input_filename}...")
    vmec_input = vmecpp.VmecInput.from_file(vmec_input_filename)
    vmec_input.mgrid_file = ""

    mgrid_params = vmecpp.MakegridParameters.from_file(makegrid_params_fname)
    magnetic_response_table = vmecpp.MagneticFieldResponseTable.from_coils_file(
        coils_fname, mgrid_params
    )

    print("Starting benchmark runs...")

    num_cold_runs = 3
    cold_start_times: list[float] = []
    cold_start_iterations = 0  # To store iterations from the last cold run
    vmec_output: vmecpp.VmecOutput | None = None

    print("\n--- Cold Start Runs ---")
    for i in range(num_cold_runs):
        start_time = time.perf_counter()
        vmec_output = vmecpp.run(vmec_input, magnetic_field=magnetic_response_table)
        end_time = time.perf_counter()
        duration = end_time - start_time
        cold_start_times.append(duration)
        cold_start_iterations = (
            vmec_output.wout.itfsq
        )  # Store iterations from the last run
        print(
            f"Cold start run {i+1}: {duration:.4f} seconds (Iterations: {cold_start_iterations})"
        )

    avg_cold_start = statistics.mean(cold_start_times)
    std_dev_cold_start = statistics.stdev(cold_start_times)

    print(
        f"\nAverage Cold Start Time: {avg_cold_start:.4f} s +/- {std_dev_cold_start:.4f} s (Iterations: {cold_start_iterations})"
    )

    # Hot restart benchmark
    # We will perturb the field and run a hot restart multiple times to get an average
    num_hot_runs = 5
    hot_restart_times = []
    hot_restart_iterations = 0  # To store iterations from the last hot run

    # Perturb field once
    magnetic_response_table.b_p *= 1.01
    magnetic_response_table.b_r *= 1.01
    magnetic_response_table.b_z *= 1.01
    print("\n--- Hot Restart Runs ---")
    assert vmec_output is not None, "Cold start must run at least once"
    for i in range(num_hot_runs):
        start_time = time.perf_counter()
        perturbed_output = vmecpp.run(
            vmec_input, magnetic_field=magnetic_response_table, restart_from=vmec_output
        )
        end_time = time.perf_counter()
        duration = end_time - start_time
        hot_restart_times.append(duration)
        hot_restart_iterations = (
            perturbed_output.wout.itfsq
        )  # Store iterations from the last run
        print(
            f"Hot restart run {i+1}: {duration:.4f} seconds (Iterations: {hot_restart_iterations})"
        )

    avg_hot_restart = statistics.mean(hot_restart_times)
    std_dev_hot_restart = statistics.stdev(hot_restart_times)

    avg_hot_restart = statistics.mean(hot_restart_times)
    std_dev_hot_restart = statistics.stdev(hot_restart_times)

    # Original baseline values from BENCHMARK_RESULTS.txt
    original_cold_start_time = 0.5946
    original_hot_restart_time = 0.2438

    # Format results
    results_str = _format_results(
        avg_cold_start=avg_cold_start,
        std_dev_cold_start=std_dev_cold_start,
        cold_start_iterations=cold_start_iterations,
        avg_hot_restart=avg_hot_restart,
        std_dev_hot_restart=std_dev_hot_restart,
        hot_restart_iterations=hot_restart_iterations,
        vmec_input_filename=vmec_input_filename,
        script_name="examples/benchmark_vmec.py",
        original_cold_start_time=original_cold_start_time,
        original_hot_restart_time=original_hot_restart_time,
    )

    # Write results to BENCHMARK_RESULTS.txt
    with open("BENCHMARK_RESULTS.txt", "w") as f:
        f.write(results_str)

    print("\nBenchmark Results written to BENCHMARK_RESULTS.txt")


if __name__ == "__main__":
    benchmark()
