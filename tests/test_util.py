# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
import json
import os
import tempfile
from pathlib import Path

import pytest

from vmecpp import _util

# We don't want to install tests and test data as part of the package,
# but scikit-build-core + hatchling does not support editable installs,
# so the tests live in the sources but the vmecpp module lives in site_packages.
# Therefore, in order to find the test data we use the relative path to this file.
# I'm very open to alternative solutions :)
REPO_ROOT = Path(__file__).parent.parent
TEST_DATA_DIR = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"


def test_indata_to_json_success():
    with tempfile.TemporaryDirectory() as tmpdir, _util.change_working_directory_to(
        Path(tmpdir)
    ):
        test_file = TEST_DATA_DIR / "input.cma"
        json_input_file = _util.indata_to_json(test_file)
        expected_json_input_file = Path(tmpdir, "cma.json").resolve()
        assert json_input_file.exists()
        assert json_input_file == expected_json_input_file


def test_indata_to_json_output_override():
    with tempfile.TemporaryDirectory() as tmpdir, _util.change_working_directory_to(
        Path(tmpdir)
    ):
        test_file = TEST_DATA_DIR / "input.cma"
        json_input_file = _util.indata_to_json(
            test_file, output_override=Path("override_path")
        )
        expected_json_input_file = Path("override_path").resolve()
        assert json_input_file.exists()
        assert json_input_file == expected_json_input_file
        assert json_input_file.parent == Path.cwd()


def test_indata_to_json_not_found_file():
    test_file = Path("input.i_do_not_exist")
    with pytest.raises(FileNotFoundError):
        _util.indata_to_json(test_file)


def test_package_root():
    # check we can find a file that should be there from the repo root
    assert Path(_util.package_root(), "_util.py").is_file()


def test_is_vmec2000_input():
    vmec2000_input_file = TEST_DATA_DIR / "input.cma"
    vmecpp_input_file = TEST_DATA_DIR / "cma.json"

    assert _util.is_vmec2000_input(vmec2000_input_file)
    assert not _util.is_vmec2000_input(vmecpp_input_file)


def test_ensure_vmec2000_input_noop():
    vmec2000_input_file = TEST_DATA_DIR / "input.cma"

    with _util.ensure_vmec2000_input(vmec2000_input_file) as indata_file:
        assert indata_file == vmec2000_input_file


def test_ensure_vmecpp_input_noop():
    vmecpp_input_file = TEST_DATA_DIR / "cma.json"

    with _util.ensure_vmecpp_input(vmecpp_input_file) as new_input_file:
        assert new_input_file == vmecpp_input_file


def test_ensure_vmecpp_input():
    vmec2000_input_file = TEST_DATA_DIR / "input.cma"

    with _util.ensure_vmecpp_input(vmec2000_input_file) as vmecpp_input_file:
        assert vmecpp_input_file == TEST_DATA_DIR / f"cma.{os.getpid()}.json"
        with open(vmecpp_input_file) as f:
            vmecpp_input_dict = json.load(f)
            # check the output is remotely sensible: we don't want to test indata_to_json's
            # correctness here, just that nothing went terribly wrong
            assert vmecpp_input_dict["mpol"] == 5
            assert vmecpp_input_dict["ntor"] == 6
