# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Guarantee that the public input files in the main repo
(``vmecpp/test_data``) and the instrumented copies used by the large
integration tests (``vmecpp_large_cpp_tests/test_data``) describe identical
equilibria.

The large-tests inputs carry an extra block of ``dump_*`` flags (and
``iter2_to_dump``) that make Fortran VMEC emit the per-step reference data the
large C++ tests compare against. Those flags are intentionally kept out of the
more public-facing inputs in the main repo. ``indata2json`` ignores them, so
the JSON representations must agree on every physics field. The ``mgrid_file``
entry is an environment-specific path rather than a physics parameter and is
ignored.

The check only runs when the ``vmecpp_large_cpp_tests`` checkout is available
(it is checked out next to the main sources in the integration CI; a checkout
elsewhere can be pointed at via ``VMECPP_LARGE_CPP_TESTS_TEST_DATA``).
"""

import json
import os
from pathlib import Path

import pytest

from vmecpp import _util

# tests/ live in the sources while the vmecpp module is installed separately,
# so locate the data via the path to this file (see tests/test_util.py).
REPO_ROOT = Path(__file__).parent.parent
MAIN_TEST_DATA = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"

_LARGE_ENV = os.environ.get("VMECPP_LARGE_CPP_TESTS_TEST_DATA")
LARGE_TEST_DATA = (
    Path(_LARGE_ENV)
    if _LARGE_ENV
    else REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp_large_cpp_tests" / "test_data"
)

# Keys that legitimately differ and carry no physics meaning.
IGNORED_KEYS = {"mgrid_file"}


def _overlapping_cases() -> list[str]:
    if not LARGE_TEST_DATA.is_dir():
        return []
    prefix = "input."
    main = {p.name[len(prefix) :] for p in MAIN_TEST_DATA.glob("input.*")}
    large = {p.name[len(prefix) :] for p in LARGE_TEST_DATA.glob("input.*")}
    return sorted(main & large)


def _indata_as_json(input_path: Path, work_dir: Path) -> dict:
    """Convert a Fortran INDATA file to a JSON dict via indata2json."""
    work_dir.mkdir(parents=True, exist_ok=True)
    with _util.change_working_directory_to(work_dir):
        json_path = _util.indata_to_json(input_path)
        with json_path.open() as json_file:
            data = json.load(json_file)
    for key in IGNORED_KEYS:
        data.pop(key, None)
    return data


_CASES = _overlapping_cases()
_LARGE_AVAILABLE = LARGE_TEST_DATA.is_dir()
_SKIP_REASON = (
    f"vmecpp_large_cpp_tests checkout not found at {LARGE_TEST_DATA}; set "
    "VMECPP_LARGE_CPP_TESTS_TEST_DATA to its test_data directory to enable."
)


@pytest.mark.skipif(not _LARGE_AVAILABLE, reason=_SKIP_REASON)
def test_overlapping_cases_present():
    # Guard against silently testing nothing if the layout changes.
    assert _CASES, (
        "Expected at least one input case present in both "
        f"{MAIN_TEST_DATA} and {LARGE_TEST_DATA}."
    )


@pytest.mark.skipif(not _LARGE_AVAILABLE, reason=_SKIP_REASON)
@pytest.mark.parametrize("case", _CASES)
def test_input_files_are_consistent(case: str, tmp_path: Path):
    main_json = _indata_as_json(MAIN_TEST_DATA / f"input.{case}", tmp_path / "main")
    large_json = _indata_as_json(LARGE_TEST_DATA / f"input.{case}", tmp_path / "large")
    assert main_json == large_json, (
        f"input.{case} differs between vmecpp/test_data and "
        f"vmecpp_large_cpp_tests/test_data after indata2json conversion "
        f"(ignoring {sorted(IGNORED_KEYS)}). The instrumented large-tests input "
        "and the public input must describe the same equilibrium."
    )

    # The main C++ tests load the committed <case>.json, so it must describe the
    # same equilibrium as the Fortran input that the large-tests reference data
    # is generated from. This closes the gap if input.<case> and <case>.json
    # ever diverge within the main repo.
    canonical_json_path = MAIN_TEST_DATA / f"{case}.json"
    if canonical_json_path.is_file():
        with canonical_json_path.open() as canonical_file:
            canonical_json = json.load(canonical_file)
        for key in IGNORED_KEYS:
            canonical_json.pop(key, None)
        assert main_json == canonical_json, (
            f"{case}.json (loaded by the main C++ tests) differs from "
            f"input.{case} after indata2json conversion (ignoring "
            f"{sorted(IGNORED_KEYS)})."
        )
