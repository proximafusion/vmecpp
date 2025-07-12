<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/978b76bc-cd9b-4af8-b1f3-18efde7c079f">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/ec4e391a-9044-44ae-93f0-9dd8bed70001">
  <img alt="A dark Proxima logo in light color mode and a light one in dark color mode." src="https://github.com/user-attachments/assets/ec4e391a-9044-44ae-93f0-9dd8bed70001" width=400px>
</picture>

# VMEC++

<!--
[![PyPI - Version](https://img.shields.io/pypi/v/vmecpp.svg)](https://pypi.org/project/vmecpp)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/vmecpp.svg)](https://pypi.org/project/vmecpp)
-->

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![MIT license](https://img.shields.io/badge/license-MIT-blue)](LICENSE.txt)
![Python version](https://img.shields.io/badge/python-3.10-blue)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14800158.svg)](https://doi.org/10.5281/zenodo.14800158)

[![CI](https://github.com/proximafusion/vmecpp/actions/workflows/tests.yaml/badge.svg)](https://github.com/proximafusion/vmecpp/actions/workflows/tests.yaml)
[![C++ core tests](https://github.com/proximafusion/vmecpp/actions/workflows/test_bazel.yaml/badge.svg)](https://github.com/proximafusion/vmecpp/actions/workflows/test_bazel.yaml)
[![Publish wheels to PyPI](https://github.com/proximafusion/vmecpp/actions/workflows/pypi_publish.yml/badge.svg)](https://github.com/proximafusion/vmecpp/actions/workflows/pypi_publish.yml)

VMEC++ is a Python-friendly, from-scratch reimplementation in C++ of the Variational Moments Equilibrium Code (VMEC),
a free-boundary ideal-MHD equilibrium solver for stellarators and tokamaks.

The original version was written by Steven P. Hirshman and colleagues in the 1980s and 1990s.
The latest version of the original code is called `PARVMEC` and is available [here](https://github.com/ORNL-Fusion/PARVMEC).

Compared to its Fortran predecessors, VMEC++:
- has a zero-crash policy and reports issues via standard Python exceptions
- allows hot-restarting a run from a previous converged state (see [Hot restart](#hot-restart))
- supports inputs in the classic INDATA format as well as simpler-to-parse JSON files; it is also simple to construct input objects programmatically in Python
- typically runs just as fast or faster
- comes with [substantial documentation of its internal numerics](https://github.com/proximafusion/vmecpp/blob/main/docs/the_numerics_of_vmecpp.pdf)

VMEC++ can run on a laptop, but it is a suitable component for large-scale stellarator optimization pipelines.

On the other hand, some features of the original Fortran VMEC are not available in VMEC++.
See [below](#differences-with-respect-to-parvmecvmec2000) for more details.

<!-- NOTE: if you change the intro above, remember to also adapt docs/index.md! -->

-----

## Table of Contents

- [Usage](#usage)
  - [As a Python package](#as-a-python-package)
  - [With SIMSOPT](#with-simsopt)
  - [As a command line tool](#as-a-command-line-tool)
  - [As a Docker image](#as-a-docker-image)
- [Installation](#installation)
  - [Ubuntu](#ubuntudebian)
  - [Arch](#arch-linux)
  - [Fedora](#fedora)
  - [MacOS](#macos)
  - [As part of a conda environment](#as-part-of-a-conda-environment)
  - [C++ build from source](#c-build-from-source)
- [Hot restart](#hot-restart)
- [Differences with respect to PARVMEC/VMEC2000](#differences-with-respect-to-parvmecvmec2000)
- [Roadmap](#roadmap)
- [Related repositories](#related-repositories)
- [License](#license)

<!-- SPHINX-START -->

## Usage

This is a quick overview of the three main ways in which you can use VMEC++.
See [examples/](https://github.com/proximafusion/vmecpp/blob/main/examples/) for some actual example scripts.
Suitable input files are found in [`examples/data`](https://github.com/proximafusion/vmecpp/blob/main/examples/data).
If unsure where to start, we suggest to give the [`w7x`](https://github.com/proximafusion/vmecpp/blob/main/examples/data/w7x.json) case a try, which is a five-field-period stellarator case for the [Wendelstein 7-X](https://www.ipp.mpg.de/w7x) stellarator.

For example [`examples/force_residual_convergence.py`](https://github.com/proximafusion/vmecpp/blob/main/examples/force_residual_convergence.py) runs fixed-boundary VMEC++ on the W7-X case and plots the convergence of the force residuals.
![W7-X force residual convergence](docs/w7x_force_convergence.png)

### As a Python package

VMEC++ offers a simple Python API:

```python
import vmecpp

# Construct a VmecInput object, e.g. from a classic Fortran input file
vmec_input = vmecpp.VmecInput.from_file("input.w7x")  # or VMEC++'s w7x.json format
# This is a normal Python object: it can be constructed and modified programmatically
vmec_input.rbc[0, 0] *= 1.1

# Run VMEC++
vmec_output = vmecpp.run(vmec_input)

# Inspect the results programmatically or save them as a classic wout file
print(vmec_output.mercier.iota)
vmec_output.wout.save("wout_w7x.nc")
```

All other output files are accessible via members of the `output` object called `threed1_volumetrics`, `jxbout` and `mercier`.

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
# vmecpp is a python module and can be either run with `python -m` or directly as a script
vmecpp examples/data/input.w7x

# check all options
vmecpp --help
```

### As a Docker image

A pre-built Docker image is available at https://github.com/proximafusion/vmecpp/pkgs/container/vmecpp.
Note that at present it is only updated occasionally.

See [docker/README.md](https://github.com/proximafusion/vmecpp/blob/main/docker/README.md) for
more information and instructions on how to build a new image.

## Installation

The easiest method for installing `vmecpp` is using pip:
```shell
pip install vmecpp
```

For usage as part of MPI-parallelized SIMSOPT applications, you might want to also install MPI on your machine and `pip install mpi4py`.

Alternatively you can build the latest `vmecpp` directly from source according to the appropriate instructions below.

### Ubuntu/Debian

Ubuntu 22.04 and 24.04, as well as Debian 12 are officially supported.

1. Install required system packages:
```shell
sudo apt-get install -y build-essential cmake libnetcdf-dev liblapack-dev libomp-dev libhdf5-dev python3-dev
```

2. Install VMEC++ as a Python package (possibly after creating a dedicated virtual environment):

```shell
pip install git+https://github.com/proximafusion/vmecpp
```

The procedure will take a few minutes as it will build VMEC++ and some dependencies from source.

A common issue on Ubuntu is a build failure due to no `python` executable being available in PATH, since on Ubuntu the executable is called `python3`.
When installing in a virtual environment (which is always a good idea anyways) `python` will be present.
Otherwise the Ubuntu package `python-is-python3` provides the `python` alias.

### Arch Linux

1. Install required system packages:

```shell
pacman -Sy --noconfirm python-pip gcc gcc-fortran openmp hdf5 netcdf lapack
```

2. Install VMEC++ as a Python package (possibly after creating a virtual environment):

```shell
python -m pip install git+https://github.com/proximafusion/vmecpp
```

### Fedora

[Fedora 41](https://docs.fedoraproject.org/en-US/fedora/f41/release-notes/) is officially supported.

1. Install required system packages:

```shell
dnf install -y python3.10-devel cmake g++ gfortran libomp-devel hdf5-devel netcdf-devel lapack-devel
```

2. Install VMEC++ as a Python package (possibly after creating a virtual environment):

```shell
# If you are installing with MPI support, remember to source the mpi compiler first
. /etc/profile.d/modules.sh
python3.10 -m pip install git+https://github.com/proximafusion/vmecpp
```


### MacOS

1. Install dependencies via [Homebrew](https://brew.sh/):

```shell
brew install python@3.10 gcc cmake ninja libomp netcdf-cxx git
```

2. Install VMEC++ as a Python package (possibly after creating a virtual environment):

```shell
# tell cmake where to find gfortran and gcc as they have non-standard names
export FC=$(which gfortran-14)
# OpenMP headers live under a different path newer OS-X versions, so CMake can't find them
export OpenMP_ROOT=$(brew --prefix)/opt/libomp
export HDF5_ROOT=$(brew --prefix hdf5)
python3.10 -m pip install git+https://github.com/proximafusion/vmecpp
```

### As part of a conda environment

VMEC++ is currently not packaged for conda, but all its dependencies are and VMEC++
can be installed inside a conda environment. An example `environment.yml` file is
provided [here](https://github.com/proximafusion/vmecpp/blob/main/environment.yml) that
can be used, after cloning the `vmecpp` repository, as:

```shell
git clone https://github.com/proximafusion/vmecpp.git
cd vmecpp
# this creates a "vmecpp" conda environment
conda env create --file environment.yml
# use the environment as usual
conda activate vmecpp
```

### C++ build from source

After having installed the build dependencies as shown above, you can compile
the C++ core of VMEC++ via CMake or Bazel. E.g. with CMake:

```shell
git clone https://github.com/proximafusion/vmecpp.git
cd vmecpp
cmake -B build  # create and configure build directory
cmake --build build --parallel  # build VMEC++
# you can now use the vmec_standalone C++ executable to run VMEC on a VMEC++ input JSON file, e.g.
./build/vmec_standalone ./examples/data/solovev.json
```

The main C++ source code tree is located at [`src/vmecpp/cpp/vmecpp`](https://github.com/proximafusion/vmecpp/blob/main/src/vmecpp/cpp/vmecpp).

## Hot restart

By passing the output of a VMEC++ run as initial state for a subsequent one,
VMEC++ is initialized using the previously converged equilibrium.
This can dramatically decrease the number of iterations to convergence when running
VMEC++ on a configuration that is very similar to the converged equilibrium.

### Example

```python
import vmecpp

input = vmecpp.VmecInput.from_file("w7x.json")

# Base run
output = vmecpp.run(input)

# Now let's perturb the plasma boundary a little bit...
input.rbc[0, 0] *= 0.8
input.rbc[1, 0] *= 1.2
# ...and fix up the multigrid steps: hot-restarted runs only allow a single step
input.ns_array = input.ns_array[-1:]
input.ftol_array = input.ftol_array[-1:]
input.niter_array = input.niter_array[-1:]

# We can now run with hot restart:
# passing the previously obtained output ensures that
# the run starts already close to the equilibrium, so it will take
# very few iterations to converge this time!
hot_restarted_output = vmecpp.run(input, restart_from=output)
```

## Full tests and validation against the reference Fortran VMEC v8.52

When developing the C++ core, it's advisable to locally run the full C++ tests for debugging or to validate changes before submitting them.
The full tests are not stored in the sources of this repo, but in a separate repo: https://github.com/proximafusion/vmecpp_large_cpp_tests .
See the instructions there for how to run those tests locally. The CI of this repo includes those tests too.

The single-thread runtimes as well as the contents of the "wout" file produced by VMEC++ can be compared with those of Fortran VMEC v8.52.
The full validation test can be found at https://github.com/proximafusion/vmecpp-validation, including a set of sensible input configurations,
parameter scan values and tolerances that make the comparison pass. See that repo for more information.

## Differences with respect to PARVMEC/VMEC2000

VMEC++:
- reports issues via standard Python exceptions and has a zero crash policy
- allows hot-restarting a run from a previous converged state (see [Hot restart](#hot-restart))
- supports inputs in the classic INDATA format as well as simpler-to-parse JSON files; it is also simple to construct input objects programmatically in Python
- employs the same parallelization strategy as Fortran VMEC, but VMEC++ leverages OpenMP for a multi-thread implementation rather than Fortran VMEC's MPI parallelization: as a consequence it cannot parallelize over multiple nodes
- implements the iteration algorithm of Fortran VMEC 8.52, which sometimes has different convergence behavior from (PAR)VMEC 9.0: some configurations might converge with VMEC++ and not with (PAR)VMEC 9.0, and vice versa

### Limitations with respect to the Fortran implementations
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
- 2D preconditioning using block-tridiagonal solver ([`BCYCLIC`](https://www.sciencedirect.com/science/article/abs/pii/S0021999110002536)) is not implemented;
  neither are the associated input fields `precon_type` and `prec2d_threshold`
- VMEC++ only computes the output quantities if the run converged
- The Fortran version falls back to fixed-boundary computation if the `mgrid` file cannot be found; VMEC++ (gracefully) errors out instead.
- The Fortran version accepts both the full path or filename of the input file as well as the "extension", i.e., the part after `input.`; VMEC++ only supports a valid filename or full path to an existing input file.

## Roadmap

Some of the things we are planning for VMEC++'s future:
- [x] free-boundary hot-restart in Python
- [X] open-sourcing the full VMEC++ test suite (including the Verification&Validation part that compares `wout` contents)
- [x] open-sourcing the source code to reproduce VMEC++'s performance benchmarks
- [ ] VMEC++ usable as a C++ bazel module

Some items we do not plan to work on, but where community ownership is welcome:
- [ ] packaging VMEC++ for platforms or package managers other than pip (e.g. conda, homebrew, ...)
- [ ] native Windows support
- [ ] ARM support
- [ ] 2D preconditioner using [`bcyclic_plus_plus`](https://code.ornl.gov/m4c/bcyclic_plus_plus)

## Related repositories

* [`proximafusion/vmecpp-validation`](https://github.com/proximafusion/vmecpp-validation) - Validation tests for VMEC++
* [`proximafusion/vmecpp_large_cpp_tests`](https://github.com/proximafusion/vmecpp_large_cpp_tests) - Large C++ tests for VMEC++
* [`proximafusion/the_numerics_of_vmecpp`](https://github.com/proximafusion/the_numerics_of_vmecpp) - Documentation of the numerical details of VMEC++
* [`proximafusion/vmecpp-benchmarks`](https://github.com/proximafusion/vmecpp-benchmarks) - Performance benchmarks comparing VMEC++ and VMEC2000

## License

`vmecpp` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
