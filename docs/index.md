# VMEC++

```{toctree}
:maxdepth: 2
:hidden:

Reference <api/vmecpp>
```

![MIT license](https://img.shields.io/badge/license-MIT-blue)
![Python version](https://img.shields.io/badge/python-3.10-blue)

VMEC++ is a Python-friendly, from-scratch reimplementation in C++ of the Variational Moments Equilibrium Code (VMEC),
a free-boundary ideal-MHD equilibrium solver for stellarators and tokamaks.

The original version was written by Steven P. Hirshman and colleagues in the 1980s and 1990s.
The latest version of the original code is called `PARVMEC` and is available [here](https://github.com/ORNL-Fusion/PARVMEC).

VMEC++ is typically just as fast or faster than its Fortran predecessor, uses a lighter-weight
multi-thread (OpenMP) parallelization scheme to Fortran VMEC's MPI parallelization and implements
some extra features such as hot-restart. As a result it can run on a laptop, but it is a suitable component
for large-scale stellarator optimization pipelines.

On the other hand, some features of the original Fortran VMEC are not available in VMEC++.
See [below](#known-differences-with-respect-to-parvmec-vmec2000) for more details.

```{include} ../README.md
:start-after: <!-- SPHINX-START -->
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
