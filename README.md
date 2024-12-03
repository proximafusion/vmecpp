# VMEC++

[![PyPI - Version](https://img.shields.io/pypi/v/vmecpp.svg)](https://pypi.org/project/vmecpp)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/vmecpp.svg)](https://pypi.org/project/vmecpp)

-----

## Table of Contents

- [Installation](#installation)
- [License](#license)

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
