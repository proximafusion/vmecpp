# Copy the installed static zlib archive to a fixed name.
foreach(_name zs zlibstatic z zlib)
  if(EXISTS "${LIB_DIR}/${_name}.lib")
    file(COPY_FILE "${LIB_DIR}/${_name}.lib" "${LIB_DIR}/zlib_static.lib")
    return()
  endif()
endforeach()
message(FATAL_ERROR "No static zlib archive found in ${LIB_DIR}")
