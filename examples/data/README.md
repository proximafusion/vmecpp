# Example input and output files for VMEC++

A few cases are available for testing VMCE++ and experimenting with it:

1. `cth_like_fixed_bdy.json` - Stellarator case, similar to the Compact Toroidal Hybrid ([CTH](https://www.auburn.edu/cosam/departments/physics/research/plasma_physics/compact_toroidal_hybrid/index.htm)) device
   1. `input.cth_like_fixed_bdy` - Fortran namelist input file to be used with Fortran VMEC
   1. `cth_like_fixed_bdy.json` - JSON input file for VMEC++, derived from `input.cth_like_fixed_bdy` using [`indata2json`](https://github.com/jonathanschilling/indata2json)
   1. `wout_cth_like_fixed_bdy.nc` - NetCDF output file, produced using [`PARVMEC`](https://github.com/ORNL-Fusion/PARVMEC) from `input.cth_like_fixed_bdy`, for testing the loading of a `wout` file using VMEC++'s tooling

1. `input.nfp4_QH_warm_start` - quasi-helically example for use with SIMSOPT

1. `solovev` - axisymmetric Tokamak case, similar to the Solov'ev equilibrium used in the [1983 Hirshman & Whitson article](https://doi.org/10.1063/1.864116)
    1. `input.solovev` - Fortran namelist input file for use with Fortran VMEC
    1. `solovev.json` - JSON input file for VMEC++, derived from `input.solovev` using [`indata2json`](https://github.com/jonathanschilling/indata2json)

1. `w7x` - Wendelstein 7-X ([W7-X](https://www.ipp.mpg.de/w7x)) Stellarator
    1. `input.w7x` - Fortran namelist input file for use with Fortran VMEC
    1. `w7x.json` - JSON input file for VMEC++, derived from `input.w7x` using [`indata2json`](https://github.com/jonathanschilling/indata2json)
