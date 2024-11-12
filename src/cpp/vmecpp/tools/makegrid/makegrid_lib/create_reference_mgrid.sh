#!/bin/bash

# This needs the official Fortran MAKEGRID executable `mgrid` on $PATH.
# The actual code is hosted here: https://github.com/ORNL-Fusion/MAKEGRID
# but keep in mind that you need to go throught the `Stellarator-Tools` repo to get the build system:
# https://github.com/ORNL-Fusion/Stellarator-Tools

# delete previous outputs
rm -fi extcur.test_* mgrid_test_*

OMP_NUM_THREADS=1 mgrid coils.test_non_symmetric
OMP_NUM_THREADS=1 mgrid coils.test_symmetric_even
OMP_NUM_THREADS=1 mgrid coils.test_symmetric_odd
