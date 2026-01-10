# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
import tempfile
from pathlib import Path

import numpy as np
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
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        _util.change_working_directory_to(Path(tmpdir)),
    ):
        test_file = TEST_DATA_DIR / "input.cma"
        json_input_file = _util.indata_to_json(test_file)
        expected_json_input_file = Path(tmpdir, "cma.json").resolve()
        assert json_input_file.exists()
        assert json_input_file == expected_json_input_file


def test_indata_to_json_output_override():
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        _util.change_working_directory_to(Path(tmpdir)),
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


def test_distributino_root():
    assert _util.distribution_root().is_dir()
    assert (
        _util.distribution_root()
        / "cpp"
        / "third_party"
        / "indata2json"
        / "indata2json"
    ).is_file()


def test_sparse_to_dense_coefficients_out_of_bounds():
    sparse_data = [{"m": 2, "n": 0, "value": 1.0}]
    with pytest.raises(ValueError):  # noqa: PT011
        _util.sparse_to_dense_coefficients(sparse_data, mpol=2, ntor=1)


def test_dense_to_sparse_truncation():
    result_sparse = _util.dense_to_sparse_coefficients(
        np.array([[0, 0, 0, 0, 0], [0, 0, 0, 1, 0]])
    )
    expeted_sparse = [{"m": 1, "n": 1, "value": 1.0}]
    assert expeted_sparse == result_sparse


def test_to_and_from_sparse():
    sparse_data = [
        {"m": 0, "n": -1, "value": 10.0},
        {"m": 1, "n": -1, "value": -1.5},
        {"m": 1, "n": 1, "value": 1.5},
        {"m": 2, "n": 0, "value": 5.0},
        {"m": 3, "n": 2, "value": 7.0},
    ]
    dense_result = _util.sparse_to_dense_coefficients_implicit(sparse_data)
    sparse_result = _util.dense_to_sparse_coefficients(dense_result)

    assert sparse_data == sparse_result
