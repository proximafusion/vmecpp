# Replace the macro-named zlib include in HDF5's deflate filter with a literal one.
set(_file "${HDF5_SRC}/src/H5Zdeflate.c")
file(READ "${_file}" _content)
string(REPLACE "#include H5_ZLIB_HEADER" "#include \"zlib.h\"" _content "${_content}")
file(WRITE "${_file}" "${_content}")
