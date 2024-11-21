import pathlib

import pytest
from third_party.indata2json import indata_to_json
from util import workspace

STARFINDER_INPUTS_DIR = pathlib.Path(workspace.workspace_root(), "starfinder/inputs")


def test_indata_to_json_success():
    filename = "input.cma"
    test_file = pathlib.Path(STARFINDER_INPUTS_DIR, filename)
    json_input_file = indata_to_json.indata_to_json(test_file)
    expected_json_input_file = pathlib.Path("cma.json")
    assert json_input_file.exists()
    assert json_input_file == expected_json_input_file


def test_indata_to_json_not_found_file():
    test_file = pathlib.Path("input.i_do_not_exist")
    with pytest.raises(FileNotFoundError):
        indata_to_json.indata_to_json(test_file)
