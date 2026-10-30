#!/bin/bash

# Activate the inofficial Bash Strict Mode.
# Reference: http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'

# Make sure this script is run from inside //vmecpp/test_data.
if [[ $PWD != *vmecpp_large_cpp_tests/test_data ]]
then
  echo "You must run this script from within //vmecpp_large_cpp_tests/test_data."
  exit 255
fi

# Use indata2json to convert the Fortran input files to JSON.
# You can get it from here: https://github.com/jonathanschilling/indata2json
# Re-generate JSON input files from input files for Fortran educational_VMEC.
# The Fortran files have the mgrid file relative to this folder, i.e., without subfolder.
# The JSON files need to have vmecpp/test_data as prefix to work in the Bazel tests,
# so add that prefix via the new --mgrid_folder option of indata2json.
# Only do this for the free-boundary cases,
# as we would otherwise write "mgrid_file":"vmecpp/test_data/NONE".
indata2json input.solovev
indata2json input.solovev_analytical
indata2json --mgrid_folder vmecpp_large_cpp_tests/test_data input.solovev_free_bdy
indata2json input.solovev_no_axis
indata2json input.cth_like_fixed_bdy
indata2json input.cth_like_fixed_bdy_nzeta_37
indata2json input.cma
indata2json --mgrid_folder vmecpp_large_cpp_tests/test_data input.cth_like_free_bdy

# Re-do mgrid file for the cth_like_free_bdy case.
makegrid makegrid_parameters_cth_like.json coils.cth_like
makegrid makegrid_parameters_solovev.json coils.solovev

# educational_VMEC is used to generate the reference data.
# You can get it from here: https://github.com/jonathanschilling/educational_VMEC
EDUCATIONAL_VMEC_EXECUTABLE=/home/jons/code/educational_VMEC/build/bin/xvmec

# now re-do VMEC runs
rm -rf solovev                     && ${EDUCATIONAL_VMEC_EXECUTABLE} input.solovev
rm -rf solovev_analytical          && ${EDUCATIONAL_VMEC_EXECUTABLE} input.solovev_analytical
rm -rf solovev_free_bdy            && ${EDUCATIONAL_VMEC_EXECUTABLE} input.solovev_free_bdy
rm -rf solovev_no_axis             && ${EDUCATIONAL_VMEC_EXECUTABLE} input.solovev_no_axis
rm -rf cth_like_fixed_bdy          && ${EDUCATIONAL_VMEC_EXECUTABLE} input.cth_like_fixed_bdy
rm -rf cth_like_fixed_bdy_nzeta_37 && ${EDUCATIONAL_VMEC_EXECUTABLE} input.cth_like_fixed_bdy_nzeta_37
rm -rf cma                         && ${EDUCATIONAL_VMEC_EXECUTABLE} input.cma
rm -rf cth_like_free_bdy           && ${EDUCATIONAL_VMEC_EXECUTABLE} input.cth_like_free_bdy

echo "formatting JSON files..."
find . -name "*.json" -print0 | while IFS= read -r -d '' f; do
  python -m json.tool --indent 4 "$f" > "$f.tmp" && mv "$f.tmp" "$f"
done
echo "done"
