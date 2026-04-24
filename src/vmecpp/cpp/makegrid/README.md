# makegrid

This is a C++ implementation of the Fortran `MAKEGRID` code
used to generate the response function tables for coils
needed for a free-boundary VMEC run.

Compile with `bazel build --config=opt //makegrid/...`.

Run with `./bazel-bin/makegrid/makegrid/makegrid <makegrid_parameters>.json coils.<runId>`.
The first argument `<makegrid_parameters>.json` contains the desired grid extent and resolution.
An example file for that is available in `makegrid/example_inputs/makegrid_parameters.json`.
The filename has to have the `.json` filename extension.

The second argument `coils.<runId>` contains the geometry of coil filaments in MAKEGRID format.
An example file for that is available in `../vmecpp/test_data/coils.cth_like`.
The filename has to start with `coils.`.

The tool produces a file `mgrid_<runId>.nc` in the current working directory,
where `<runId>` was extracted from the name of the coils file.

The computation is parallelized with OpenMP
and can make use of as many threads as independent coil circuits
are present in the coils file.

## Further information

Some documentation on the Fortran `MAKEGRID` implementation can be found here:
https://github.com/jonathanschilling/vmec-internals/blob/master/mgrid_file.md
