cmake_minimum_required(VERSION 3.4...3.18)

set(CMAKE_CXX_STANDARD 20)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-fPIC -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fno-math-errno")

project(vmecpp C CXX)

find_package(HDF5 REQUIRED COMPONENTS C CXX)
include_directories(${HDF5_INCLUDE_DIRS} ${HDF5_CXX_INCLUDE_DIRS})

find_package (netCDF REQUIRED)
include_directories(${NETCDF_INCLUDES})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(nlohmann_json REQUIRED)

find_package(LAPACK REQUIRED C)

find_package(OpenMP)

include_directories(${PROJECT_SOURCE_DIR}/abscab-cpp)
set (abscab_sources "abscab-cpp/abscab/abscab.cc" "abscab-cpp/abscab/abscab.hh")

include_directories(${PROJECT_SOURCE_DIR}/abseil-cpp)
add_subdirectory(${PROJECT_SOURCE_DIR}/abseil-cpp)

# Allow to retain include paths as used for Bazel build.
# This needs to be defined before add_subdirectory(src) is called,
# which starts including files that want to pull in header files
# specified relative to `${PROJECT_SOURCE_DIR}/src/vmecpp/cpp`.
include_directories(${PROJECT_SOURCE_DIR}/src/vmecpp/cpp)

# Assemble the VMEC++ source tree.
# Start out with ABSCAB sources - no need for a separate library for ABSCAB.
set (vmecpp_sources ${abscab_sources})
add_subdirectory(src)

# Define a static library for the actual computation core of VMEC++.
add_library (vmecpp_core STATIC ${vmecpp_sources})
target_link_libraries(vmecpp_core PRIVATE ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES})
target_link_libraries(vmecpp_core PRIVATE netCDF::netcdf)
target_link_libraries(vmecpp_core PRIVATE nlohmann_json::nlohmann_json)
if(APPLE)
  # We assume we are building against dependencies installed via homebrew.
  # These must be public for the _vmecpp target to pick them up transitively.
  target_include_directories(vmecpp_core PUBLIC /opt/homebrew/opt/lapack/include)
  target_link_directories(vmecpp_core PUBLIC /opt/homebrew/opt/lapack/lib)
  target_link_libraries(vmecpp_core PUBLIC LAPACK::LAPACK -llapacke)
else()
  target_link_libraries(vmecpp_core PRIVATE LAPACK::LAPACK -llapacke)
endif()
target_link_libraries(vmecpp_core PRIVATE absl::algorithm absl::base absl::synchronization absl::strings absl::str_format absl::log absl::string_view absl::check absl::status absl::statusor)
if(OpenMP_CXX_FOUND)
  target_link_libraries(vmecpp_core PRIVATE OpenMP::OpenMP_CXX)
endif()

# Now also add the vmec_standalone executable.
add_executable (vmec_standalone ${PROJECT_SOURCE_DIR}/src/vmecpp/cpp/vmecpp/vmec/vmec_standalone/vmec_standalone.cc)
target_link_libraries (vmec_standalone vmecpp_core)
target_link_libraries(vmecpp_core PRIVATE absl::strings)

# Now add the pybind11 module for VMEC++.
add_subdirectory(pybind11)
set (vmecpp_pybind11_sources
  ${PROJECT_SOURCE_DIR}/src/vmecpp/cpp/vmecpp/vmec/pybind11/pybind_vmec.cc
  ${PROJECT_SOURCE_DIR}/src/vmecpp/cpp/vmecpp/vmec/pybind11/vmec_indata_pywrapper.cc
  ${PROJECT_SOURCE_DIR}/src/vmecpp/cpp/vmecpp/vmec/pybind11/vmec_indata_pywrapper.h
)
pybind11_add_module(_vmecpp ${vmecpp_pybind11_sources})
target_link_libraries(_vmecpp PRIVATE vmecpp_core)

install(TARGETS _vmecpp LIBRARY DESTINATION vmecpp/cpp/.)
