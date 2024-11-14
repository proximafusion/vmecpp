# VMEC++

[![PyPI - Version](https://img.shields.io/pypi/v/vmecpp.svg)](https://pypi.org/project/vmecpp)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/vmecpp.svg)](https://pypi.org/project/vmecpp)

-----

## Table of Contents

- [Installation](#installation)
- [License](#license)

## Installation

### Python package

```console
sudo apt-get install libnetcdf-dev liblapacke-dev libopenmpi-dev
git clone https://github.com/proximafusion/vmecpp
pip install ./vmecpp
```

### C++ build from source

On Ubuntu 22.04, having installed [bazelisk](https://github.com/bazelbuild/bazelisk):

```console
sudo apt-get install libnetcdf-dev liblapacke-dev
cd src/cpp
bazel build --config=opt //...
```

Now you have a standalone vmecpp executable available:

```
./bazel-bin/vmecpp/vmec/vmec_standalone/vmec_standalone
```

## License

`vmecpp` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
