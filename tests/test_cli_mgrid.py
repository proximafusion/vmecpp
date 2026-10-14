# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
from pathlib import Path

import netCDF4
import pytest

from vmecpp import __main__ as cli

REPO_ROOT = Path(__file__).parent.parent
COILS_FILE = REPO_ROOT / "tests/data/quasr/coils.quasr0019493"
PARAMETERS_FILE = (
    REPO_ROOT / "src/vmecpp/cpp/makegrid/example_inputs/makegrid_parameters.json"
)


def test_mgrid_writes_netcdf_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv", ["vmecpp", str(PARAMETERS_FILE), "--mgrid", str(COILS_FILE)]
    )

    cli.main()

    with netCDF4.Dataset(tmp_path / "mgrid_quasr0019493.nc") as mgrid:
        assert mgrid["br_001"].shape == (18, 13, 11)


def test_mgrid_argument_selects_mgrid_command():
    args = cli.parse_arguments(["params.json", "--mgrid", "coils.x"])
    assert args.command == "mgrid"
    assert args.mgrid == Path("coils.x")


if __name__ == "__main__":
    pytest.main([__file__])
