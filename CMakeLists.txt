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

# LTO: cross-TU inlining and dead-code elimination, ~1/3 smaller binaries.
include(CheckIPOSupported)
check_ipo_supported(RESULT VMECPP_IPO_SUPPORTED OUTPUT VMECPP_IPO_ERROR)
if(VMECPP_IPO_SUPPORTED)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
else()
  message(STATUS "LTO unavailable, building without it: ${VMECPP_IPO_ERROR}")
endif()

# -march raises Eigen's EIGEN_MAX_ALIGN_BYTES from 16 to 32, which changes
# Eigen::aligned_allocator and is ABI-affecting. Pin it so cores built with
# different -march settings stay interchangeable.
add_compile_definitions(EIGEN_MAX_ALIGN_BYTES=32 EIGEN_MAX_STATIC_ALIGN_BYTES=32)

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

# HDF5 and netCDF-C are built from source, as static libraries, rather than
# relying on whatever happens to be installed on the system. This is to
# ensure reproducibility and minimize binary size by turning off netcdf
# features we don't use. Versions must stay in sync with the Bazel build's
# src/vmecpp/cpp/third_party/non_module_deps.bzl.
include(ExternalProject)
set(_native_deps_prefix "${CMAKE_BINARY_DIR}/native-deps")
set(_native_deps_lib "${_native_deps_prefix}/lib")
set(_native_deps_include "${_native_deps_prefix}/include")
# INTERFACE_INCLUDE_DIRECTORIES below must exist already at generate time.
file(MAKE_DIRECTORY "${_native_deps_include}")

# HDF5 is built with zlib support (required by netCDF-4), but as a static
# lib it doesn't bundle zlib's own object code, so it must be linked in
# explicitly wherever hdf5_c is used.
find_package(ZLIB REQUIRED)

set(_native_deps_cmake_args
  -DCMAKE_INSTALL_PREFIX=${_native_deps_prefix}
  -DCMAKE_INSTALL_LIBDIR=lib
  -DCMAKE_PREFIX_PATH=${_native_deps_prefix}
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
  -DBUILD_SHARED_LIBS=OFF
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  -DBUILD_TESTING=OFF)

if(APPLE)
  # CC=brew's gcc-14 (set by some CI jobs) chokes on Apple's SDK math.h for
  # plain C. -D always wins over the CC/CXX env vars, unlike omitting it.
  list(APPEND _native_deps_cmake_args
    -DCMAKE_C_COMPILER=/usr/bin/cc
    -DCMAKE_CXX_COMPILER=/usr/bin/c++)
endif()

ExternalProject_Add(hdf5_native
  URL "https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5-1_14_3.tar.gz"
  URL_HASH SHA256=df5ee33c74d5efb59738075ef96f4201588e1f1eeb233f047ac7fd1072dee1f6
  CMAKE_ARGS
    ${_native_deps_cmake_args}
    -DHDF5_BUILD_CPP_LIB=ON
    -DHDF5_ENABLE_Z_LIB_SUPPORT=ON
    -DHDF5_ENABLE_SZIP_SUPPORT=OFF
    -DHDF5_ENABLE_SZIP_ENCODING=OFF
    -DHDF5_BUILD_EXAMPLES=OFF
    -DHDF5_BUILD_TOOLS=OFF
  BUILD_BYPRODUCTS
    "${_native_deps_lib}/libhdf5.a"
    "${_native_deps_lib}/libhdf5_cpp.a"
    "${_native_deps_lib}/libhdf5_hl.a")

ExternalProject_Add(netcdf_native
  DEPENDS hdf5_native
  URL "https://github.com/Unidata/netcdf-c/archive/refs/tags/v4.9.3.tar.gz"
  URL_HASH SHA256=990f46d49525d6ab5dc4249f8684c6deeaf54de6fec63a187e9fb382cc0ffdff
  CMAKE_ARGS
    ${_native_deps_cmake_args}
    -DNETCDF_ENABLE_DAP=OFF
    -DNETCDF_ENABLE_DAP2=OFF
    -DNETCDF_ENABLE_DAP4=OFF
    -DNETCDF_ENABLE_NCZARR=OFF
    -DNETCDF_ENABLE_NCZARR_ZIP=OFF
    -DNETCDF_ENABLE_TESTS=OFF
    -DNETCDF_BUILD_UTILITIES=OFF
  BUILD_BYPRODUCTS
    "${_native_deps_lib}/libnetcdf.a")

# Imported targets pointing at the static libraries the ExternalProjects
# above install at build time
add_library(hdf5_c STATIC IMPORTED GLOBAL)
set_target_properties(hdf5_c PROPERTIES
  IMPORTED_LOCATION "${_native_deps_lib}/libhdf5.a"
  INTERFACE_INCLUDE_DIRECTORIES "${_native_deps_include}"
  INTERFACE_LINK_LIBRARIES ZLIB::ZLIB)
add_dependencies(hdf5_c hdf5_native)

add_library(hdf5_hl STATIC IMPORTED GLOBAL)
set_target_properties(hdf5_hl PROPERTIES
  IMPORTED_LOCATION "${_native_deps_lib}/libhdf5_hl.a"
  INTERFACE_LINK_LIBRARIES hdf5_c)
add_dependencies(hdf5_hl hdf5_native)

add_library(hdf5_cpp STATIC IMPORTED GLOBAL)
set_target_properties(hdf5_cpp PROPERTIES
  IMPORTED_LOCATION "${_native_deps_lib}/libhdf5_cpp.a"
  INTERFACE_LINK_LIBRARIES hdf5_c)
add_dependencies(hdf5_cpp hdf5_native)

# netCDF's static lib references HDF5 HL symbols internally, so link it in
# too even though vmecpp's own code never calls it directly.
add_library(netcdf STATIC IMPORTED GLOBAL)
set_target_properties(netcdf PROPERTIES
  IMPORTED_LOCATION "${_native_deps_lib}/libnetcdf.a"
  INTERFACE_INCLUDE_DIRECTORIES "${_native_deps_include}"
  INTERFACE_LINK_LIBRARIES "hdf5_hl;hdf5_c")
add_dependencies(netcdf netcdf_native)

add_library(vmecpp_hdf5_netcdf INTERFACE)
target_link_libraries(vmecpp_hdf5_netcdf INTERFACE netcdf hdf5_cpp hdf5_hl hdf5_c)

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
# The only dense linear solve in this project (LaplaceSolver::DecomposeMatrix/
# SolveForPotential) uses Eigen::PartialPivLU, which is pure C++ with no
# external LAPACK/BLAS dependency as long as EIGEN_USE_LAPACKE/EIGEN_USE_BLAS/
# EIGEN_USE_MKL are never defined. Do not define those macros without
# reintroducing an explicit LAPACK/BLAS dependency here.

FetchContent_Declare(nlohmann_json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(nlohmann_json)

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

# The computation core, built once per target ISA. Variants share one SONAME and
# are told apart by directory, so the loader can substitute them (see the
# glibc-hwcaps install rules below).
function(vmecpp_add_core target)
  cmake_parse_arguments(CORE "" "HWCAPS_SUBDIR" "COMPILE_OPTIONS" ${ARGN})

  add_library(${target} SHARED ${vmecpp_sources})
  set_target_properties(${target} PROPERTIES OUTPUT_NAME vmecpp_core)
  if(CORE_HWCAPS_SUBDIR)
    set_target_properties(${target} PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/glibc-hwcaps/${CORE_HWCAPS_SUBDIR}")
  endif()
  target_compile_options(${target} PRIVATE ${CORE_COMPILE_OPTIONS})

  target_link_libraries(${target} PUBLIC vmecpp_hdf5_netcdf)
  target_link_libraries(${target} PUBLIC nlohmann_json::nlohmann_json)
  target_link_libraries(${target} PUBLIC absl::algorithm absl::base
    absl::synchronization absl::strings absl::str_format absl::log
    absl::string_view absl::check absl::status absl::statusor)

  if(VMECPP_USE_FFTX)
    # Built per variant: the codelets must match the ISA of the core linking them.
    add_library(${target}_fftx STATIC ${_fftx_iprdft_srcs} ${_fftx_prdft_srcs})
    target_include_directories(${target}_fftx PUBLIC
      "${_fftx_dir}"                                                 # for <include/omega64.h>
      "${_fftx_dir}/include"                                         # for fftx_minimal.hpp
      "${_fftx_dir}/lib_fftx_iprdftbat_cpu_srcs"
      "${_fftx_dir}/lib_fftx_prdftbat_cpu_srcs")
    target_compile_options(${target}_fftx PRIVATE -fPIC ${CORE_COMPILE_OPTIONS})
    target_link_libraries(${target} PRIVATE ${target}_fftx)
    target_compile_definitions(${target} PUBLIC VMECPP_USE_FFTX)
  endif()

  if(OpenMP_CXX_FOUND)
    target_link_libraries(${target} PUBLIC OpenMP::OpenMP_CXX)
  endif()

  # We multithread at the outer loop level, not for the individual Eigen operations
  # https://libeigen.gitlab.io/eigen/docs-3.3/TopicMultiThreading.html
  target_compile_definitions(${target} PRIVATE EIGEN_DONT_PARALLELIZE)
endfunction()

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
  list(LENGTH _fftx_iprdft_srcs _n_iprdft)
  list(LENGTH _fftx_prdft_srcs  _n_prdft)
  message(STATUS "FFTX vendored codelets: ${_n_iprdft} iprdft + ${_n_prdft} prdft sources")
endif()

vmecpp_add_core(vmecpp_core)

set(VMECPP_HWCAPS_DISPATCH_DEFAULT OFF)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
   CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  set(VMECPP_HWCAPS_DISPATCH_DEFAULT ON)
endif()
option(VMECPP_HWCAPS_DISPATCH
       "Also build x86-64-v3 core variant selected at load time via glibc-hwcaps"
       ${VMECPP_HWCAPS_DISPATCH_DEFAULT})
if(VMECPP_HWCAPS_DISPATCH)
  if(NOT VMECPP_HWCAPS_DISPATCH_DEFAULT)
    message(FATAL_ERROR
      "VMECPP_HWCAPS_DISPATCH requires Linux on an x86-64 processor")
  endif()
  vmecpp_add_core(vmecpp_core_v3 HWCAPS_SUBDIR x86-64-v3
                  COMPILE_OPTIONS -march=x86-64-v3)
endif()


# Now also add the vmec_standalone executable.
add_executable(vmec_standalone ${PROJECT_SOURCE_DIR}/src/vmecpp/cpp/vmecpp/vmec/vmec_standalone/vmec_standalone.cc)
target_link_libraries(vmec_standalone vmecpp_core)
if(APPLE)
  set(_vmecpp_loader_path "@loader_path")
else()
  set(_vmecpp_loader_path "$ORIGIN")
endif()
set_target_properties(vmec_standalone PROPERTIES
                      BUILD_RPATH "${_vmecpp_loader_path}")

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

set_target_properties(_vmecpp PROPERTIES
                      INSTALL_RPATH "${_vmecpp_loader_path}")
install(TARGETS _vmecpp LIBRARY DESTINATION vmecpp/cpp/.)
# glibc >= 2.33 prefers glibc-hwcaps/ subdirectories; older loaders ignore
# them and pick up the baseline next to the extension module.
install(TARGETS vmecpp_core LIBRARY DESTINATION vmecpp/cpp/.)
install(FILES
  ${PROJECT_SOURCE_DIR}/src/vmecpp/cpp/vmecpp/vmec/geometry/geometry_c_api.h
  DESTINATION vmecpp/include)
if(VMECPP_HWCAPS_DISPATCH)
  install(TARGETS vmecpp_core_v3
          LIBRARY DESTINATION vmecpp/cpp/glibc-hwcaps/x86-64-v3/.)
endif()
install(TARGETS indata2json DESTINATION vmecpp/cpp/third_party/indata2json/)

# Optional Enzyme automatic-differentiation target: exact autodiff (forward and
# reverse) of a real VMEC nonlinear kernel, the half-grid Jacobian. Enzyme
# (https://enzyme.mit.edu) differentiates LLVM IR via a Clang plugin, so it needs
# a Clang frontend and the matching ClangEnzyme plugin. OFF by default; the
# production build and all existing targets are unaffected. The AD-toolchain
# smoke test lives in bazel (//vmecpp/common/enzyme:enzyme_smoke_test, built with
# --config=enzyme). Enable this target with:
#   -DVMECPP_ENABLE_ENZYME=ON -DVMECPP_ENZYME_PLUGIN=/path/to/ClangEnzyme-NN.so
option(VMECPP_ENABLE_ENZYME "Build Enzyme autodiff targets" OFF)
if(VMECPP_ENABLE_ENZYME)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR
      "VMECPP_ENABLE_ENZYME requires a Clang compiler (got "
      "${CMAKE_CXX_COMPILER_ID}); Enzyme attaches as a Clang plugin.")
  endif()
  set(VMECPP_ENZYME_PLUGIN "" CACHE FILEPATH "Path to ClangEnzyme-NN.so")
  if(NOT VMECPP_ENZYME_PLUGIN OR NOT EXISTS "${VMECPP_ENZYME_PLUGIN}")
    message(FATAL_ERROR
      "VMECPP_ENABLE_ENZYME=ON requires "
      "-DVMECPP_ENZYME_PLUGIN=/path/to/ClangEnzyme-NN.so")
  endif()
  message(STATUS "Enzyme plugin: ${VMECPP_ENZYME_PLUGIN}")
  enable_testing()
  # Self-contained over flat buffers; checks Jv and J^T u against finite
  # differences and against each other (adjoint identity). Enzyme runs as an
  # optimization-time pass, so it needs -O2 and the plugin attached. -O2 also
  # guards against a Debug (-O0) configuration where the AD pass would not run.
  add_executable(jacobian_kernel_autodiff_test
    ${PROJECT_SOURCE_DIR}/src/vmecpp/cpp/vmecpp/common/enzyme/jacobian_kernel_autodiff_test.cc)
  target_compile_options(jacobian_kernel_autodiff_test PRIVATE
    -O2 -fplugin=${VMECPP_ENZYME_PLUGIN})
  add_test(NAME jacobian_kernel_autodiff COMMAND jacobian_kernel_autodiff_test)
endif()
