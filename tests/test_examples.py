# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Smoke tests for repository example scripts."""

import os
import subprocess
import sys
from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]

REPO_ROOT = Path(__file__).parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
MAX_OUTPUT_CHARS = 3000

SKIPPED_EXAMPLES = {
    "compare_vmecpp_to_parvmec.py": (
        "requires a PARVMEC reference file (examples/data/wout_w7x.nc) "
        "that is not tracked in this repository"
    ),
    "mpi_finite_difference.py": "requires an MPI runtime and mpi4py configured against it",
    "simsopt_qh_fixed_resolution.py": "requires an MPI runtime and mpi4py configured against it",
    "visualize_magnetic_field.py": "requires optional dependency pyvista",
}
MPI_RUNTIME_OPTIONAL_EXAMPLES = {"simsopt_integration.py"}


def _has_working_mpi_runtime() -> bool:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank())",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _prepare_example_workdir(workdir: Path) -> None:
    """Create an isolated cwd that still resolves repository-relative paths.

    Some example scripts use relative paths like ``examples/...`` or ``src/...``,
    so we provide symlinks in a temporary working directory to avoid polluting the
    repository root with generated files.
    """
    (workdir / "examples").symlink_to(EXAMPLES_DIR, target_is_directory=True)
    (workdir / "src").symlink_to(REPO_ROOT / "src", target_is_directory=True)


@pytest.mark.examples
@pytest.mark.parametrize(
    "example_path",
    sorted(EXAMPLES_DIR.glob("*.py")),
    ids=lambda path: path.name,
)
def test_example_script_runs(example_path: Path, tmp_path: Path) -> None:
    """Run each example script as a subprocess in a clean working directory."""
    skip_reason = SKIPPED_EXAMPLES.get(example_path.name)
    if skip_reason is not None:
        pytest.skip(skip_reason)

    if (
        example_path.name in MPI_RUNTIME_OPTIONAL_EXAMPLES
        and not _has_working_mpi_runtime()
    ):
        pytest.skip("requires either no mpi4py install or a working MPI runtime")

    _prepare_example_workdir(tmp_path)
    environment = os.environ.copy() | {"MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1"}
    result = subprocess.run(
        [sys.executable, str(example_path)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, (
        f"Example script failed: {example_path.name}\n"
        f"stdout (tail):\n{result.stdout[-MAX_OUTPUT_CHARS:]}\n"
        f"stderr (tail):\n{result.stderr[-MAX_OUTPUT_CHARS:]}"
    )
