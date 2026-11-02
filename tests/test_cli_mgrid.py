# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
from pathlib import Path
from types import SimpleNamespace

import netCDF4
import pytest

from vmecpp import __main__ as cli

REPO_ROOT = Path(__file__).parent.parent
COILS_FILE = REPO_ROOT / "src/vmecpp/cpp/vmecpp/test_data/coils.solovev"
PARAMETERS_FILE = (
    REPO_ROOT / "src/vmecpp/cpp/makegrid/example_inputs/makegrid_parameters.json"
)


def test_mgrid_writes_netcdf_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv", ["vmecpp", "makegrid", str(PARAMETERS_FILE), str(COILS_FILE)]
    )

    cli.main()

    with netCDF4.Dataset(tmp_path / "mgrid_solovev.nc") as mgrid:
        assert mgrid["br_001"].shape == (18, 13, 11)


@pytest.mark.parametrize("subcommand", ["makegrid", "mgrid"])
def test_makegrid_subcommand_arguments(subcommand):
    args = cli.parse_arguments([subcommand, "params.json", "coils.x"])
    assert args.command == "makegrid"
    assert args.makegrid_parameters == Path("params.json")
    assert args.coils_file == Path("coils.x")


def test_help_lists_subcommands(capsys):
    with pytest.raises(SystemExit):
        cli.parse_arguments(["--help"])
    out = capsys.readouterr().out
    assert "convert" in out
    assert "makegrid" in out
    assert "mgrid" in out


def test_quiet_run_does_not_print_output_message(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["vmecpp", "-q", "input.test"])
    monkeypatch.setattr(cli.vmecpp.VmecInput, "from_file", lambda _: object())
    monkeypatch.setattr(
        cli.vmecpp,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            wout=SimpleNamespace(save=lambda _: None)
        ),
    )
    monkeypatch.setattr(
        cli.vmecpp._util, "get_vmec_configuration_name", lambda _: "test"
    )

    cli.main()

    assert capsys.readouterr().out == ""


if __name__ == "__main__":
    pytest.main([__file__])
