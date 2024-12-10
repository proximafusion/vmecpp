# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
import tempfile
from pathlib import Path

import pytest

from vmecpp import _util
from vmecpp.cpp.third_party.indata2json import indata_to_json


def test_indata_to_json_success():
    with tempfile.TemporaryDirectory() as tmpdir, _util.change_working_directory_to(
        Path(tmpdir)
    ):
        test_file = Path(
            _util.package_root(), "cpp", "vmecpp", "test_data", "input.cma"
        )
        json_input_file = indata_to_json.indata_to_json(test_file)
        expected_json_input_file = Path("cma.json")
        assert json_input_file.exists()
        assert json_input_file == expected_json_input_file


def test_indata_to_json_not_found_file():
    test_file = Path("input.i_do_not_exist")
    with pytest.raises(FileNotFoundError):
        indata_to_json.indata_to_json(test_file)
