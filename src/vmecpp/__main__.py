# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
import argparse
import importlib.metadata
import sys
from pathlib import Path

import vmecpp

_COMMANDS = ("run", "convert")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VMEC++ is a free-boundary ideal-MHD equilibrium solver for stellarators and tokamaks."
    )
    p.add_argument(
        "-v",
        "--version",
        help="Print VMEC++ version information and exit.",
        action="version",
        version=f"vmecpp v{importlib.metadata.version('vmecpp')}",
    )
    subparsers = p.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run", help="Run VMEC++ on an input file. This is the default command."
    )
    run_parser.add_argument(
        "input_file",
        help="A VMEC input file either in the classic Fortran 'indata' format or in VMEC++'s JSON format.",
        type=Path,
    )
    run_parser.add_argument(
        "-t",
        "--max-threads",
        help="Maximum number of threads that VMEC++ should spawn. The actual number might still be lower that this in case there are too few flux surfaces to keep these many threads busy.",
        type=int,
    )
    run_parser.add_argument(
        "-q",
        "--quiet",
        help="If present, silences the printing of VMEC++ logs to standard output.",
        action="store_true",
    )
    run_parser.add_argument(
        "--legacy",
        help="Show the legacy table output instead of animated progress bars.",
        action="store_true",
    )

    convert_parser = subparsers.add_parser(
        "convert", help="Convert a Fortran indata file to VMEC++'s JSON format."
    )
    convert_parser.add_argument(
        "input_file",
        help="A VMEC input file either in the classic Fortran 'indata' format or in VMEC++'s JSON format.",
        type=Path,
    )

    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] not in (*_COMMANDS, "-h", "--help", "-v", "--version"):
        # No subcommand given: default to "run" for backwards compatibility.
        argv = ["run", *argv]

    return p.parse_args(argv)


def main() -> None:
    args = parse_arguments()

    if args.command == "convert":
        vmec_input = vmecpp.VmecInput.from_file(args.input_file)
        json_name = args.input_file.name.replace("input.", "")
        json_file = Path(f"{args.input_file.parent}/{json_name}.json")
        vmec_input.save(json_file, indent=4)
        print(f"Converted {args.input_file} to {json_file}")  # noqa: T201
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
