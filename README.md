# VMEC++

<!--
[![PyPI - Version](https://img.shields.io/pypi/v/vmecpp.svg)](https://pypi.org/project/vmecpp)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/vmecpp.svg)](https://pypi.org/project/vmecpp)
-->

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![MIT license](https://img.shields.io/badge/license-MIT-blue)](https://github.com/jons-pf/vmecpp?tab=MIT-1-ov-file#readme)
[![CI](https://github.com/jons-pf/vmecpp/actions/workflows/tests.yaml/badge.svg)](https://github.com/jons-pf/vmecpp/actions/workflows/tests.yaml)
[![Python version](https://img.shields.io/badge/python-3.10-blue)]()

VMEC++ is a Python-friendly, from-scratch reimplementation in C++ of the Variational Moments Equilibrium Code (VMEC),
a free-boundary ideal-MHD equilibrium solver for stellarators and tokamaks.

The original version was written by Steven P. Hirshman and colleagues in the 1980s and 1990s.
The latest version of the original code is called `PARVMEC` and is available [here](https://github.com/ORNL-Fusion/PARVMEC).

VMEC++ is typically just as fast or faster than its Fortran predecessor, uses a lighter-weight
multi-thread (OpenMP) parallelization scheme to Fortran VMEC's MPI parallelization and implements
some extra features such as hot-restart. As a result it can run on a laptop, but it is a suitable component
for large-scale stellarator optimization pipelines.

On the other hand, some features of the original Fortran VMEC are not available in VMEC++.
See [below](#known-differences-with-respect-to-parvmecvmec2000) for more details.

⚠️ VMEC++ is a production-ready physics simulator but still in beta as a standalone Python package ⚠️

-----

## Table of Contents

- [Usage](#usage)
  - [As a Python package](#as-a-python-package)
  - [With SIMSOPT](#with-simsopt)
  - [As a command line tool](#as-a-command-line-tool)
- [Installation](#installation)
- [Known differences with respect to PARVMEC/VMEC2000](#known-differences-with-respect-to-parvmecvmec2000)
- [License](#license)

## Usage

This is a quick overview of the three main ways in which you can use VMEC++.
See [examples/](examples/) for some actual example scripts.

### As a Python package

VMEC++ offers a simple Python API:

```python
import vmecpp

# construct a VmecInput object, e.g. from a classic Fortran input file
input = vmecpp.VmecInput.from_file("input.w7x")

# run VMEC++
output = vmecpp.run(input)

# inspect the results or save them as a classic wout file
output.wout.save("wout_solovev.nc")
```

Note that other output files are planned to be accessible via members of the `output` object called `threed1`, `jxbout` and `mercier` soon.

### With SIMSOPT

[SIMSOPT](https://simsopt.readthedocs.io) is a popular stellarator optimization framework.
VMEC++ implements a SIMSOPT-friendly wrapper that makes it easy to use it with SIMSOPT.

```python
import vmecpp.simsopt_compat

vmec = vmecpp.simsopt_compat.Vmec("input.w7x")
print(f"Computed plasma volume: {vmec.volume()}")
```

### As a command line tool

You can use VMEC++ directly as a CLI tool.
In a terminal in which Python has access to the VMEC++ package:

```console
# run on a given input file -> produce corresponding wout_w7x.nc
python -m vmecpp "input.w7x"

# check all options
python -m vmecpp --help
```

## Installation

The only supported platform at the moment is Ubuntu 22.04, but installation on any sufficiently recent Linux machine should work fine following a very similar process.

### Pre-requisites

Some libraries are required even when installing VMEC++ as a Python package.

On Ubuntu 22.04, they are available as these system packages:

```console
sudo apt-get install build-essential libnetcdf-dev liblapacke-dev libopenmpi-dev
```

You also need [bazelisk](https://github.com/bazelbuild/bazelisk), see link for up to date installation instructions.
For convenience, here is one possible way to install bazelisk on a Linux amd64 machine (please make sure this makes sense for your setup before running the commands):

```console
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.24.1/bazelisk-linux-amd64
sudo mv bazelisk-linux-amd64 /usr/local/bin/bazel
sudo chmod u+x /usr/local/bin/bazel
```

### Python package

Assuming git has access to the SSH key you use to log into GitHub (only until we make the repo public and publish official wheels to PyPI):

```console
pip install git+ssh://git@github.com/jons-pf/vmecpp.git
```

The procedure will take a few minutes as it will build VMEC++ and some dependencies from source.

A common issue on Ubuntu 22.04 is a build failure due to no `python` executable being available in PATH, since on Ubuntu 22.04 the executable is called `python3`.
When installing in a virtual environment (which is always a good idea anyways) `python` will be present.
Otherwise the Ubuntu package `python-is-python3` provides the `python` alias.

### C++ build from source

```console
git clone https://github.com/jons-pf/vmecpp
cd vmecpp/src/vmecpp/cpp
bazel build --config=opt //...
```

All artifacts are now under `./bazel-bin/vmecpp`.

The main C++ source code tree starts in [`src/vmecpp/cpp/vmecpp`](src/vmecpp/cpp/vmecpp).

## Known differences with respect to PARVMEC/VMEC2000

- VMEC++ has a zero-crash policy and reports issues via standard Python exceptions
- VMEC++ allows hot-restarting a run from a previous converged state (see [#hot-restart])
- VMEC++'s parallelization strategy is the same as Fortran VMEC, but it leverages OpenMP for a multi-thread implementation rather than Fortran VMEC's MPI parallelization
- VMEC++ implements the iteration algorithm of Fortran VMEC 8.52, which has sometimes different convergence behavior from (PAR)VMEC 9.0: some configurations might converge with VMEC++ and not with (PAR)VMEC 9.0, and vice versa
- VMEC++ supports inputs in the classic INDATA format as well as JSON files; it is also simple to construct input objects programmatically in Python

### Known limitations with respect to the Fortran implementations
- non-stellarator-symmetric terms (`lasym == true`) are not supported yet
- free-boundary works only for `ntor > 0` - axisymmetric (`ntor = 0`) free-boundary runs don't work yet
- `lgiveup`/`fgiveup` logic for early termination of a multi-grid sequence is not implemented yet
- `lbsubs` logic in computing outputs is not implemented yet
- `lforbal` logic for non-variational forces near the magnetic axis is not implemented yet
- `lrfp` is not implemented yet - only stellarators/Tokamaks for now
- several profile parameterizations are not fully implemented yet:
   * `gauss_trunc`
   * `two_power_gs`
   * `akima_spline`
   * `akima_spline_i`
   * `akima_spline_ip`
   * `cubic_spline`
   * `cubic_spline_i`
   * `cubic_spline_ip`
   * `pedestal`
   * `rational`
   * `line_segment`
   * `line_segment_i`
   * `line_segment_ip`
   * `nice_quadratic`
   * `sum_cossq_s`
   * `sum_cossq_sqrts`
   * `sum_cossq_s_free`
- some (rarely used) free-boundary-related output quantities are not implemented yet:
   * `curlabel` - declared but not populated yet
   * `potvac` - declared but not populated yet
   * `xmpot` - not declared yet
   * `xnpot` - not declared yet
- 2D preconditioning using block-tridiagonal solver (`BCYCLIC`) is not implemented
- VMEC++ only computes the output quantities if the run converged
- The Fortran version falls back to fixed-boundary computation if the `mgrid` file cannot be found; VMEC++ (gracefully) errors out instead.
- The Fortran version accepts both the full path or filename of the input file as well as the "extension", i.e., the part after `input.`; VMEC++ only supports a valid filename or full path to an existing input file.

## Roadmap

Some of the things we are planning for VMEC++'s future:
- VMEC++ usable as a C++ bazel module
- readthedocs docs
- free-boundary hot-restart in Python
- open-sourcing the full VMEC++ test suite (including the Verification&Validation part that compares `wout` contents)
- open-sourcing the source code to reproduce VMEC++'s performance benchmarks

Some items we do not plan to work on, but where community ownership is welcome:
- adding cmake as an alternative C++ build system, making VMEC++ a well-behaved CMake dependency
- packaging VMEC++ for other platforms or package managers (e.g. conda, homebrew, ...)
- macOS and native Windows support

## License

`vmecpp` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
