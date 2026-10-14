# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
import argparse
import importlib.metadata
import sys
from pathlib import Path

import vmecpp
from vmecpp._free_boundary import write_mgrid_file


def _parse_convert_arguments(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="vmecpp convert",
        description="Convert a Fortran indata file to VMEC++'s JSON format.",
    )
    p.add_argument(
        "input_file",
        help="A VMEC input file either in the classic Fortran 'indata' format or in VMEC++'s JSON format.",
        type=Path,
    )
    args = p.parse_args(argv)
    args.command = "convert"
    return args


def _parse_makegrid_arguments(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="vmecpp makegrid",
        description="Run MAKEGRID on a coils file and write the mgrid NetCDF file "
        "mgrid_<name>.nc for the coils file coils.<name>.",
    )
    p.add_argument(
        "makegrid_parameters",
        help="A MakegridParameters JSON file defining the grid of the mgrid file.",
        type=Path,
    )
    p.add_argument(
        "coils_file",
        help="A coils file in MAKEGRID format, e.g. coils.w7x.",
        type=Path,
    )
    args = p.parse_args(argv)
    args.command = "makegrid"
    return args


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[1:]

    if argv and argv[0] == "convert":
        return _parse_convert_arguments(argv[1:])
    if argv and argv[0] in ("makegrid", "mgrid"):
        return _parse_makegrid_arguments(argv[1:])

    p = argparse.ArgumentParser(
        description="VMEC++ is a free-boundary ideal-MHD equilibrium solver for stellarators and tokamaks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""commands (run 'vmecpp <command> --help' for details):
  convert                      convert an input file to VMEC++'s JSON format
  makegrid (alias: mgrid)      compute an mgrid file from a coils file with MAKEGRID""",
    )
    p.add_argument(
        "input_file",
        help="A VMEC input file either in the classic Fortran 'indata' format or in VMEC++'s JSON format.",
        type=Path,
    )
    p.add_argument(
        "-t",
        "--max-threads",
        help="Maximum number of threads that VMEC++ should spawn. The actual number might still be lower than this in case there are too few flux surfaces to keep these many threads busy.",
        type=int,
    )
    p.add_argument(
        "-q",
        "--quiet",
        help="If present, silences the printing of VMEC++ logs to standard output.",
        action="store_true",
    )
    p.add_argument(
        "-v",
        "--version",
        help="Print VMEC++ version information and exit.",
        action="version",
        version=f"vmecpp v{importlib.metadata.version('vmecpp')}",
    )
    p.add_argument(
        "--legacy",
        help="Show the legacy table output instead of animated progress bars.",
        action="store_true",
    )
    args = p.parse_args(argv)
    args.command = "run"
    return args


_RESET, _RED, _GREEN, _YELLOW = "\033[0m", "\033[31m", "\033[32m", "\033[33m"
_MAKEGRID_STAGES = {"B": "B field", "A": "A potential"}


def _progress_bar(done: int, total: int, width: int = 50) -> str:
    """Bar in the style of the VMEC++ run progress display."""
    filled = width * done // total
    color = _GREEN if done == total else _YELLOW if 2 * done > total else _RED
    fill = f"{color}{'=' * filled}{_RESET}" if filled else ""
    return f"[{fill}{'-' * (width - filled)}]"


class _MakegridProgress:
    """One progress bar per MAKEGRID stage, redrawn in place on a TTY."""

    def __init__(self) -> None:
        self._tty = sys.stdout.isatty()
        self._state: dict[str, tuple[int, int]] = dict.fromkeys(
            _MAKEGRID_STAGES, (0, 1)
        )
        self._drawn = False

    def _line(self, stage: str) -> str:
        done, total = self._state[stage]
        return f"  {_MAKEGRID_STAGES[stage]:<11}: {_progress_bar(done, total)} {done}/{total}"

    def __call__(self, stage: str, done: int, total: int) -> None:
        self._state[stage] = (done, total)
        if self._tty:
            up = f"\033[{len(self._state)}A" if self._drawn else ""
            print(up + "\n".join(self._line(s) for s in self._state))  # noqa: T201
            self._drawn = True
        elif done == total:
            print(self._line(stage))  # noqa: T201


def _describe_makegrid_grid(p: vmecpp.MakegridParameters) -> str:
    return (
        f"Grid: {p.number_of_r_grid_points} (R) x {p.number_of_z_grid_points} (Z) "
        f"x {p.number_of_phi_grid_points} (phi) per field period\n"
        f"R in [{p.r_grid_minimum:g}, {p.r_grid_maximum:g}] m, "
        f"Z in [{p.z_grid_minimum:g}, {p.z_grid_maximum:g}] m\n"
        f"{p.number_of_field_periods} field periods, "
        f"stellarator symmetry: {'yes' if p.assume_stellarator_symmetry else 'no'}"
    )


def main() -> None:
    args = parse_arguments()

    if args.command == "convert":
        vmec_input = vmecpp.VmecInput.from_file(args.input_file)
        json_name = args.input_file.name.replace("input.", "")
        json_file = Path(f"{args.input_file.parent}/{json_name}.json")
        vmec_input.save(json_file, indent=4)
        print(f"Converted {args.input_file} to {json_file}")  # noqa: T201
        return

    if args.command == "makegrid":
        makegrid_parameters = vmecpp.MakegridParameters.from_file(
            args.makegrid_parameters
        )
        coils_name = args.coils_file.name.removeprefix("coils.")
        mgrid_file = Path(f"mgrid_{coils_name}.nc")
        print(_describe_makegrid_grid(makegrid_parameters))  # noqa: T201
        write_mgrid_file(
            args.coils_file,
            makegrid_parameters,
            mgrid_file,
            progress=_MakegridProgress(),
        )
        print(f"Output written to {mgrid_file}")  # noqa: T201
        return

    if args.quiet:
        verbose = 0
    elif args.legacy:
        verbose = 1
    else:
        verbose = 2
        print(  # noqa: T201
            "Tip: Use the --legacy flag for classic table output."
        )
        vmecpp._progress_tip_shown = True

    vmec_input = vmecpp.VmecInput.from_file(args.input_file)
    output = vmecpp.run(vmec_input, max_threads=args.max_threads, verbose=verbose)

    configuration_name = vmecpp._util.get_vmec_configuration_name(args.input_file)
    wout_file = Path(f"wout_{configuration_name}.nc")
    output.wout.save(wout_file)

    print(f"\nOutput written to {wout_file}")  # noqa: T201


if __name__ == "__main__":
    main()
