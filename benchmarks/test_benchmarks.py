# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Benchmarks for VMEC++ using pytest-benchmark.

Run locally with:
    pytest benchmarks/test_benchmarks.py -v
    pytest benchmarks/test_benchmarks.py --benchmark-json=benchmark_results.json
"""

import subprocess
import sys
from pathlib import Path

import pytest

import vmecpp

REPO_ROOT = Path(__file__).parent.parent
TEST_DATA_DIR = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"
EXAMPLES_DATA_DIR = REPO_ROOT / "examples" / "data"


# ---------------------------------------------------------------------------
# Module-scoped fixtures: load inputs and pre-compute expensive objects once
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def w7x_input():
    return vmecpp.VmecInput.from_file(EXAMPLES_DATA_DIR / "w7x.json")


@pytest.fixture(scope="module")
def cma_input():
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cma.json")
    vmec_input.ftol_array[0] = 1e-8
    vmec_input.ftol_array[0] = 1e-10
    return vmec_input


@pytest.fixture(scope="module")
def free_boundary_input():
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_free_bdy.json")
    vmec_input.nzeta = 136
    return vmec_input


@pytest.fixture(scope="module")
def makegrid_params():
    params = vmecpp.MakegridParameters.from_file(
        TEST_DATA_DIR / "makegrid_parameters_cth_like.json"
    )
    # Lower the makegrid resolution (same as test_free_boundary.py)
    params.number_of_r_grid_points = 61
    params.number_of_phi_grid_points = 136
    params.number_of_z_grid_points = 60
    return params


@pytest.fixture(scope="module")
def response_table(makegrid_params):
    """Pre-computed magnetic field response table for free-boundary benchmarks."""
    return vmecpp.MagneticFieldResponseTable.from_coils_file(
        TEST_DATA_DIR / "coils.cth_like", makegrid_params
    )


# ---------------------------------------------------------------------------
# CLI benchmarks
# ---------------------------------------------------------------------------


def test_bench_cli_startup(benchmark):
    """Benchmark CLI startup time via `vmecpp -h`."""
    result = benchmark(
        subprocess.run,
        [sys.executable, "-m", "vmecpp", "-h"],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0


def test_bench_cli_invalid_input(benchmark):
    """Benchmark CLI error path via `vmecpp invalid_input`."""

    def run_invalid():
        return subprocess.run(
            [sys.executable, "-m", "vmecpp", "invalid_input"],
            capture_output=True,
            check=False,
        )

    result = benchmark(run_invalid)
    assert result.returncode != 0


# ---------------------------------------------------------------------------
# Fixed-boundary solver benchmarks
# ---------------------------------------------------------------------------


def test_bench_fixed_boundary_w7x(benchmark, w7x_input):
    """Benchmark W7-X equilibrium (5-period stellarator, mpol=12, ntor=12, ns=99)."""
    result = benchmark.pedantic(
        vmecpp.run,
        args=(w7x_input,),
        kwargs={"max_threads": 4},
        rounds=3,
        warmup_rounds=0,
    )
    assert result.wout.volume == pytest.approx(27.78, rel=1e-3)


def test_bench_fixed_boundary_cma(benchmark, cma_input):
    """Benchmark CMA equilibrium (stellarator, ntor=6, mpol=5)."""
    result = benchmark.pedantic(
        vmecpp.run,
        args=(cma_input,),
        kwargs={"max_threads": 1},
        rounds=3,
        warmup_rounds=1,
    )
    assert result.wout.volume == pytest.approx(0.5014, rel=1e-3)


# ---------------------------------------------------------------------------
# Free-boundary benchmarks
# ---------------------------------------------------------------------------


def test_bench_response_table_from_coils(benchmark, makegrid_params):
    """Benchmark MagneticFieldResponseTable.from_coils_file() creation."""
    benchmark.pedantic(
        vmecpp.MagneticFieldResponseTable.from_coils_file,
        args=(TEST_DATA_DIR / "coils.cth_like", makegrid_params),
        rounds=3,
        warmup_rounds=1,
    )


def test_bench_free_boundary(benchmark, free_boundary_input, response_table):
    """Benchmark free-boundary solve with pre-computed response table."""
    result = benchmark.pedantic(
        vmecpp.run,
        args=(free_boundary_input, response_table),
        kwargs={"max_threads": 1},
        rounds=3,
        warmup_rounds=1,
    )
    assert result.wout.volume == pytest.approx(0.3075, rel=1e-3)
