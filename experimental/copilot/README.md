# ASCOT5 experimental integration

This directory contains an experimental integration of
[ASCOT5](https://github.com/ascot4fusion/ascot5) (a high-performance orbit-
following code for fusion plasma physics) with the vmecpp workflow.

ASCOT5 is added as a git submodule in `third_party/ascot5`.

## Setup

1. Initialise the submodule (if not done already):

   ```bash
   git submodule update --init third_party/ascot5
   ```

2. Build `libascot.so` and install the `a5py` Python package:

   ```bash
   bash experimental/copilot/setup_ascot5.sh
   ```

   **Prerequisites (Debian/Ubuntu):**
   ```bash
   sudo apt-get install libhdf5-dev
   ```

   **Prerequisites (macOS):**
   ```bash
   brew install hdf5
   ```

## Running the demo

```bash
python experimental/copilot/ascot5_demo.py
```

The demo script:
1. Creates an ASCOT5 HDF5 input file (`ascot.h5`) in this directory.
2. Populates it with an analytical ITER-like equilibrium magnetic field,
   a flat plasma, a rectangular wall, and 100 fusion alpha guiding-centre
   markers.
3. Prints the input data tree.
4. Loads the magnetic field into memory via `libascot.so` and evaluates
   `psi` and `rho` at a point on the midplane.

## ASCOT5 documentation

- Introduction tutorial: <https://ascot4fusion.github.io/ascot5/>
- Input generation: <https://ascot4fusion.github.io/ascot5/inputgen.html>
- Python API reference: <https://ascot4fusion.github.io/ascot5/papi/>
