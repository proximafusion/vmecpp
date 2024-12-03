# VMEC++

[![PyPI - Version](https://img.shields.io/pypi/v/vmecpp.svg)](https://pypi.org/project/vmecpp)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/vmecpp.svg)](https://pypi.org/project/vmecpp)

-----

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [As a Python package](#as-a-python-package)
  - [With SIMSOPT](#with-simsopt)
  - [As a command line tool](#as-a-command-line-tool)
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

### With SIMSOPT

[SIMSOPT](https://simsopt.readthedocs.io) is a popular stellarator optimization framework.
VMEC++ implements a SIMSOPT-friendly wrapper that makes it possible to use it with SIMSOPT.

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
sudo apt-get install libnetcdf-dev liblapacke-dev libopenmpi-dev
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
pip install git+ssh://git@github.com/proximafusion/vmecpp-dev.git
```

The procedure will take a few minutes as it will build VMEC++ and some dependencies from source.

A common issue on Ubuntu 22.04 is a build failure due to no `python` executable being available in PATH, since on Ubuntu 22.04 the executable is called `python3`.
When installing in a virtual environment (which is always a good idea anyways) `python` will be present.
Otherwise the Ubuntu package `python-is-python3` provides the `python` alias.

### C++ build from source

```console
git clone https://github.com/proximafusion/vmecpp-dev
cd vmecpp-dev/src/vmecpp/cpp
bazel build --config=opt //...
```

All artifacts are now under `./bazel-bin/vmecpp`.

## License

`vmecpp` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
