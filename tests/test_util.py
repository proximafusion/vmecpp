# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
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
        expected_json_input_file = Path("cma.json")
        assert json_input_file.exists()
        assert json_input_file == expected_json_input_file


def test_indata_to_json_not_found_file():
    test_file = Path("input.i_do_not_exist")
    with pytest.raises(FileNotFoundError):
        _util.indata_to_json(test_file)
