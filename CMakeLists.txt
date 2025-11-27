cmake_minimum_required(VERSION 3.15)
project(vmecpp C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-fPIC -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fno-math-errno")

# use ccache if available
find_program(CCACHE_COMMAND NAMES ccache ccache-swig)
if(EXISTS ${CCACHE_COMMAND})
  message(STATUS "Found ccache: ${CCACHE_COMMAND}")
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_COMMAND})
else()
  message(STATUS "Could NOT find ccache")
endif()

# First check if required libraries are installed locally
find_package(OpenMP REQUIRED)

# Optional CUDA support for GPU acceleration
option(VMECPP_ENABLE_CUDA "Enable CUDA backend for GPU acceleration" OFF)
if(VMECPP_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # Use the same C++ standard for CUDA as for C++
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    add_compile_definitions(VMECPP_HAS_CUDA)
    message(STATUS "CUDA support enabled")
    message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA version: ${CMAKE_CUDA_COMPILER_VERSION}")
  else()
    message(WARNING "CUDA requested but no CUDA compiler found. Disabling CUDA support.")
    set(VMECPP_ENABLE_CUDA OFF)
  endif()
endif()

find_package(HDF5 REQUIRED COMPONENTS C CXX)
include_directories(${HDF5_INCLUDE_DIRS} ${HDF5_CXX_INCLUDE_DIRS})

find_package(netCDF)
if(NOT netCDF_FOUND)
  # Finds the netCDF installation using CMake's PkgConfig
  set(netCDF_PC_FILE netcdf)
  find_package(PkgConfig REQUIRED)
  set(PKG_CONFIG_USE_CMAKE_PREFIX_PATH TRUE)
  pkg_check_modules(netCDF REQUIRED ${netCDF_PC_FILE}>=4.3.0 IMPORTED_TARGET)
  pkg_get_variable(netCDF_PREFIX ${netCDF_PC_FILE} prefix)
  message(STATUS "netCDF prefix: ${netCDF_PREFIX}")
  message(STATUS "netCDF include dirs: ${netCDF_INCLUDE_DIRS}")
  message(STATUS "netCDF libraries: ${netCDF_LIBRARIES}")
endif()

include_directories(${netCDF_INCLUDE_DIRS})

# Fetch all the remote dependencies
include(FetchContent)
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  # Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
  cmake_policy(SET CMP0135 NEW)
endif()
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG tags/3.4.0
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(eigen)
include_directories(${eigen_SOURCE_DIR})

FetchContent_Declare(nlohmann_json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(nlohmann_json)

find_package(LAPACK REQUIRED)


FetchContent_Declare(
  abseil-cpp
  GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
  GIT_TAG 4447c7562e3bc702ade25105912dce503f0c4010
  GIT_SHALLOW TRUE
)
FetchContent_Declare(
  indata2json
  GIT_REPOSITORY https://github.com/jonathanschilling/indata2json.git
  GIT_TAG f59e3ddd66486b63536f141a786d39c23d654c77
  GIT_SHALLOW TRUE
)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG "v2.13.6"
  GIT_SHALLOW TRUE
)
FetchContent_Declare(
  abscab-cpp
  GIT_REPOSITORY https://github.com/jonathanschilling/abscab-cpp.git
  GIT_TAG 5cfa473b90aab06d7f70d986da0c46c46c1ebe9c
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(abscab-cpp)
include_directories(${abscab-cpp_SOURCE_DIR})
set(abscab_sources "${abscab-cpp_SOURCE_DIR}/abscab/abscab.cc" "${abscab-cpp_SOURCE_DIR}/abscab/abscab.hh")

# Fix deprecation warning, abseil will change this default soon.
set(ABSL_PROPAGATE_CXX_STD "ON")
FetchContent_MakeAvailable(abseil-cpp indata2json)
include_directories(${abseil-cpp_SOURCE_DIR})

# Allow to retain include paths as used for Bazel build.
# This needs to be defined before add_subdirectory(src) is called,
# which starts including files that want to pull in header files
# specified relative to `${PROJECT_SOURCE_DIR}/src/vmecpp/cpp`.
include_directories(${PROJECT_SOURCE_DIR}/src/vmecpp/cpp)

# Assemble the VMEC++ source tree.
# Start out with ABSCAB sources - no need for a separate library for ABSCAB.
set(vmecpp_sources ${abscab_sources})
add_subdirectory(src)

# Define a static library for the actual computation core of VMEC++.
add_library(vmecpp_core STATIC ${vmecpp_sources})
target_link_libraries(vmecpp_core PRIVATE ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES})
target_link_libraries(vmecpp_core PRIVATE ${netCDF_LIBRARIES})
target_link_libraries(vmecpp_core PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(vmecpp_core PRIVATE LAPACK::LAPACK)
target_link_libraries(vmecpp_core PRIVATE absl::algorithm absl::base absl::synchronization absl::strings absl::str_format absl::log absl::string_view absl::check absl::status absl::statusor)

if(OpenMP_CXX_FOUND)
  target_link_libraries(vmecpp_core PRIVATE OpenMP::OpenMP_CXX)
endif()

# CUDA backend library (built separately due to CUDA compilation requirements)
if(VMECPP_ENABLE_CUDA)
  set(vmecpp_cuda_sources
    ${PROJECT_SOURCE_DIR}/src/vmecpp/cpp/vmecpp/common/compute_backend/cuda/compute_backend_cuda.cu
  )
  add_library(vmecpp_cuda STATIC ${vmecpp_cuda_sources})
  target_include_directories(vmecpp_cuda PRIVATE ${PROJECT_SOURCE_DIR}/src/vmecpp/cpp)
  target_link_libraries(vmecpp_cuda PRIVATE CUDA::cudart)
  set_target_properties(vmecpp_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
  )
  target_link_libraries(vmecpp_core PRIVATE vmecpp_cuda CUDA::cudart)
endif()


# Now also add the vmec_standalone executable.
add_executable(vmec_standalone ${PROJECT_SOURCE_DIR}/src/vmecpp/cpp/vmecpp/vmec/vmec_standalone/vmec_standalone.cc)
target_link_libraries(vmec_standalone vmecpp_core)
target_link_libraries(vmecpp_core PRIVATE absl::strings)

# Now add the pybind11 module for VMEC++.
FetchContent_MakeAvailable(pybind11)
set(vmecpp_pybind11_sources
  ${PROJECT_SOURCE_DIR}/src/vmecpp/cpp/vmecpp/vmec/pybind11/pybind_vmec.cc
)
pybind11_add_module(_vmecpp ${vmecpp_pybind11_sources})
target_link_libraries(_vmecpp PRIVATE vmecpp_core)

install(TARGETS _vmecpp LIBRARY DESTINATION vmecpp/cpp/.)
install(TARGETS indata2json DESTINATION vmecpp/cpp/third_party/indata2json/)
