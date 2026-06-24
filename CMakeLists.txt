cmake_minimum_required(VERSION 3.15)
project(vmecpp C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-fPIC -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -fno-math-errno")

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
  GIT_TAG tags/5.0.1
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
  # 20260107.1 LTS: older abseil fails to compile under Clang >= 21 (the
  # Enzyme build) on absl::Nonnull SFINAE in absl/strings/ascii.cc.
  GIT_TAG 255c84dadd029fd8ad25c5efb5933e47beaa00c7
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
  GIT_TAG "v3.0.0"
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

# FFTX (SPIRAL-generated batched IPRDFT/PRDFT) for the toroidal FFT hot path.
# The codelets are vendored under src/vmecpp/cpp/third_party/fftx_codelets/;
# see that directory's README.md for what they cover and how to regenerate.
# Pass -DVMECPP_USE_FFTX=OFF to fall back to the partial-DFT path.
option(VMECPP_USE_FFTX "Use FFTX/SPIRAL kernels for toroidal transforms" ON)
if(VMECPP_USE_FFTX)
  set(_fftx_dir "${PROJECT_SOURCE_DIR}/src/vmecpp/cpp/third_party/fftx_codelets")
  file(GLOB _fftx_iprdft_srcs CONFIGURE_DEPENDS
       "${_fftx_dir}/lib_fftx_iprdftbat_cpu_srcs/*.cpp")
  file(GLOB _fftx_prdft_srcs  CONFIGURE_DEPENDS
       "${_fftx_dir}/lib_fftx_prdftbat_cpu_srcs/*.cpp")
  add_library(fftx_codelets STATIC ${_fftx_iprdft_srcs} ${_fftx_prdft_srcs})
  target_include_directories(fftx_codelets PUBLIC
    "${_fftx_dir}"                                                 # for <include/omega64.h>
    "${_fftx_dir}/include"                                         # for fftx_minimal.hpp
    "${_fftx_dir}/lib_fftx_iprdftbat_cpu_srcs"
    "${_fftx_dir}/lib_fftx_prdftbat_cpu_srcs")
  # Match vmecpp's release flags; codelets are >30k lines and benefit from -O3.
  target_compile_options(fftx_codelets PRIVATE -fPIC)
  target_link_libraries(vmecpp_core PRIVATE fftx_codelets)
  target_compile_definitions(vmecpp_core PUBLIC VMECPP_USE_FFTX)
  list(LENGTH _fftx_iprdft_srcs _n_iprdft)
  list(LENGTH _fftx_prdft_srcs  _n_prdft)
  message(STATUS "FFTX vendored codelets: ${_n_iprdft} iprdft + ${_n_prdft} prdft sources")
endif()
target_link_libraries(vmecpp_core PRIVATE absl::algorithm absl::base absl::synchronization absl::strings absl::str_format absl::log absl::string_view absl::check absl::status absl::statusor)

if(OpenMP_CXX_FOUND)
  target_link_libraries(vmecpp_core PRIVATE OpenMP::OpenMP_CXX)
endif()

# We multithread at the outer loop level, not for the individual Eigen operations
# https://libeigen.gitlab.io/eigen/docs-3.3/TopicMultiThreading.html
target_compile_definitions(vmecpp_core PRIVATE EIGEN_DONT_PARALLELIZE)


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
# The VmecModel iteration bindings run the forward model inside a single-thread
# OpenMP parallel region, so the module itself must be compiled with OpenMP
# (matching the context Vmec::SolveEquilibrium provides for the omp single/barrier
# directives inside IdealMhdModel::update).
if(OpenMP_CXX_FOUND)
  target_link_libraries(_vmecpp PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS _vmecpp LIBRARY DESTINATION vmecpp/cpp/.)
install(TARGETS indata2json DESTINATION vmecpp/cpp/third_party/indata2json/)

# Optional Enzyme automatic-differentiation targets.
# Enzyme (https://enzyme.mit.edu) differentiates LLVM IR via a Clang plugin, so
# these targets need a Clang frontend and the matching ClangEnzyme plugin. OFF
# by default; the production build and all existing targets are unaffected.
# Enable with:
#   -DVMECPP_ENABLE_ENZYME=ON -DVMECPP_ENZYME_PLUGIN=/path/to/ClangEnzyme-NN.so
option(VMECPP_ENABLE_ENZYME "Build Enzyme autodiff targets" OFF)
if(VMECPP_ENABLE_ENZYME)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|IntelLLVM")
    message(FATAL_ERROR
      "VMECPP_ENABLE_ENZYME requires a Clang-based compiler (got "
      "${CMAKE_CXX_COMPILER_ID}); Enzyme attaches as a Clang plugin. "
      "Intel oneAPI icx/icpx (IntelLLVM) is Clang-based and supported.")
  endif()
  set(VMECPP_ENZYME_PLUGIN "" CACHE FILEPATH "Path to ClangEnzyme-NN.so")
  # The Enzyme pass is an LLVM-version-specific plugin. icx/icpx (IntelLLVM)
  # cannot load a plugin built against a different upstream LLVM. Enzyme's output
  # is an ordinary object file, though, so for such compilers the single Enzyme
  # TU (exact_force_jvp.cc) can be precompiled with a matching ClangEnzyme and
  # linked in via VMECPP_ENZYME_OBJECT, while the rest is built with icx.
  set(VMECPP_ENZYME_OBJECT "" CACHE FILEPATH
    "Prebuilt exact_force_jvp object (compiled elsewhere with a matching ClangEnzyme)")
  if(VMECPP_ENZYME_OBJECT)
    if(NOT EXISTS "${VMECPP_ENZYME_OBJECT}")
      message(FATAL_ERROR "VMECPP_ENZYME_OBJECT not found: ${VMECPP_ENZYME_OBJECT}")
    endif()
    message(STATUS "Enzyme: linking prebuilt object ${VMECPP_ENZYME_OBJECT}")
    set_source_files_properties("${VMECPP_ENZYME_OBJECT}" PROPERTIES
      EXTERNAL_OBJECT TRUE GENERATED TRUE)
    target_sources(vmecpp_core PRIVATE "${VMECPP_ENZYME_OBJECT}")
    target_compile_definitions(vmecpp_core PUBLIC VMECPP_ENABLE_ENZYME)
  else()
  if(NOT VMECPP_ENZYME_PLUGIN OR NOT EXISTS "${VMECPP_ENZYME_PLUGIN}")
    message(FATAL_ERROR
      "VMECPP_ENABLE_ENZYME=ON requires "
      "-DVMECPP_ENZYME_PLUGIN=/path/to/ClangEnzyme-NN.so "
      "(or -DVMECPP_ENZYME_OBJECT=/path/to/exact_force_jvp.o for icx)")
  endif()
  message(STATUS "Enzyme plugin: ${VMECPP_ENZYME_PLUGIN}")
  enable_testing()
  # Exact autodiff (forward and reverse) of a real VMEC nonlinear kernel: the
  # half-grid Jacobian. Self-contained over flat buffers; checks Jv and J^T u
  # against finite differences and against each other (adjoint identity).
  add_executable(jacobian_kernel_autodiff_test
    ${PROJECT_SOURCE_DIR}/src/vmecpp/cpp/vmecpp/common/enzyme/jacobian_kernel_autodiff_test.cc)
  target_compile_options(jacobian_kernel_autodiff_test PRIVATE
    -O2 -fplugin=${VMECPP_ENZYME_PLUGIN})
  add_test(NAME jacobian_kernel_autodiff COMMAND jacobian_kernel_autodiff_test)
  # Exact Hessian of the composed local force map (all six force-chain kernels):
  # forward/reverse Jacobian vs finite differences, and JVP cost vs FD-HVP.
  add_executable(local_force_hessian_test
    ${PROJECT_SOURCE_DIR}/src/vmecpp/cpp/vmecpp/common/enzyme/local_force_hessian_test.cc
    ${PROJECT_SOURCE_DIR}/src/vmecpp/cpp/vmecpp/vmec/ideal_mhd_model/exact_force_jvp.cc)
  target_compile_options(local_force_hessian_test PRIVATE
    -O2 -fplugin=${VMECPP_ENZYME_PLUGIN})
  add_test(NAME local_force_hessian COMMAND local_force_hessian_test)

  # Wire the exact force Hessian-vector product into the core library and the
  # Python extension: the single Enzyme forward pass lives in exact_force_jvp.cc
  # (compiled with the plugin), the rest of the core stays normally compiled.
  # VMECPP_ENABLE_ENZYME guards the callers so the default (plugin-free) build is
  # unaffected.
  target_sources(vmecpp_core PRIVATE
    ${PROJECT_SOURCE_DIR}/src/vmecpp/cpp/vmecpp/vmec/ideal_mhd_model/exact_force_jvp.cc
    ${PROJECT_SOURCE_DIR}/src/vmecpp/cpp/vmecpp/vmec/ideal_mhd_model/exact_force_vjp.cc)
  set_source_files_properties(
    ${PROJECT_SOURCE_DIR}/src/vmecpp/cpp/vmecpp/vmec/ideal_mhd_model/exact_force_jvp.cc
    ${PROJECT_SOURCE_DIR}/src/vmecpp/cpp/vmecpp/vmec/ideal_mhd_model/exact_force_vjp.cc
    PROPERTIES COMPILE_OPTIONS "-O2;-fplugin=${VMECPP_ENZYME_PLUGIN}")
  target_compile_definitions(vmecpp_core PUBLIC VMECPP_ENABLE_ENZYME)
  endif()
endif()
