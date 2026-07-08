cmake_minimum_required(VERSION 3.15)
project(vmecpp C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  # cl.exe does not accept the GCC/Clang flags below. /bigobj is required:
  # the CUDA translation units exceed the default object section limit.
  # _USE_MATH_DEFINES exposes M_PI; NOMINMAX keeps windows.h from defining
  # min/max macros that clobber std::min/std::max.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /bigobj")
  set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
  add_compile_definitions(_USE_MATH_DEFINES NOMINMAX)
  # The vendored abscab dependency uses malloc/free without including
  # <cstdlib>, relying on transitive inclusion that MSVC does not provide.
  # Force-include it for C++ sources (CUDA host compilation is unaffected).
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/FIcstdlib>)
else()
  set(CMAKE_CXX_FLAGS "-fPIC -Wall -Wextra")
  set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -fno-math-errno")
endif()

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

if(MSVC)
  # vcpkg's clapack (the only LAPACK provider on Windows that needs no Fortran
  # toolchain) ships its LAPACK::LAPACK target through a FindLAPACK wrapper
  # registered at a nonstandard path the toolchain does not auto-apply. Take
  # its `lapack` config target directly and alias LAPACK::LAPACK to it, which
  # is exactly what that wrapper does.
  find_package(clapack CONFIG REQUIRED)
  if(NOT TARGET LAPACK::LAPACK)
    add_library(LAPACK::LAPACK INTERFACE IMPORTED)
    target_link_libraries(LAPACK::LAPACK INTERFACE lapack)
  endif()
else()
  find_package(LAPACK REQUIRED)
endif()


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
if(MSVC)
  # indata2json is a Fortran tool that converts classic VMEC namelist inputs
  # to JSON; it is invoked as a subprocess, not linked into any target. Its
  # Fortran build needs a compiler compatible with the MSVC/Ninja generator,
  # which is not present on this toolchain, so it is skipped on Windows. The
  # JSON-input path (VmecInput.from_file) does not require it.
  FetchContent_MakeAvailable(abseil-cpp)
else()
  FetchContent_MakeAvailable(abseil-cpp indata2json)
endif()
include_directories(${abseil-cpp_SOURCE_DIR})

# Allow to retain include paths as used for Bazel build.
# This needs to be defined before add_subdirectory(src) is called,
# which starts including files that want to pull in header files
# specified relative to `${PROJECT_SOURCE_DIR}/src/vmecpp/cpp`.
include_directories(${PROJECT_SOURCE_DIR}/src/vmecpp/cpp)

# CUDA-backed toroidal FFT (optional). Must be enabled BEFORE add_subdirectory
# so the .cu source gets picked up by the ideal_mhd_model subdir's CMakeLists.
option(VMECPP_USE_CUDA "Use CUDA-backed toroidal FFT kernels (requires CUDA toolkit + cuFFT)" OFF)
if(VMECPP_USE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  # The .cu objects link into vmecpp_core and then into the shared Python
  # module. On a GCC/Clang host nvcc needs -fPIC; on the MSVC host -fPIC is
  # not a valid flag, and the host compilation instead needs /bigobj (the
  # CUDA units are large) and /EHsc passed through -Xcompiler.
  if(MSVC)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/bigobj -Xcompiler=/EHsc")
    # CMake injects a bare /utf-8 into the nvcc command line for MSVC, which
    # nvcc misparses as a second input file; no CMake variable suppresses it.
    # Route nvcc through a launcher that rewraps /utf-8 as -Xcompiler=/utf-8.
    # Compiler launchers are honored by the Ninja and Makefile generators (use
    # one of those on Windows, not the Visual Studio generator). find_program
    # rather than find_package keeps this off pybind11's Python discovery.
    find_program(VMECPP_LAUNCHER_PYTHON NAMES python3 python py REQUIRED)
    set(CMAKE_CUDA_COMPILER_LAUNCHER
        "${VMECPP_LAUNCHER_PYTHON}"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/nvcc_utf8_launcher.py")
  elseif(CMAKE_CUDA_COMPILER_ID MATCHES "Clang")
    # Clang CUDA takes host flags directly; -Xcompiler is nvcc-only syntax. This
    # keeps the clang CUDA path (used by VMECPP_ENABLE_CUDA_ENZYME) building and
    # leaves the default nvcc flags untouched.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -fPIC")
  else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC")
  endif()
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # sm_70 covers V100, sm_80 covers A100/H100, sm_89 covers Ada.
    set(CMAKE_CUDA_ARCHITECTURES "70;80;89")
  endif()
  message(STATUS "VMECPP_USE_CUDA: enabled, CUDA arches=${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Optional CUDA-Enzyme target: the first proof that a VMEC++ CUDA device kernel
# (the half-grid Jacobian) can be differentiated by Enzyme on the GPU. OFF by
# default; the default CPU, CPU-Enzyme, and CUDA builds are unaffected. Requires
# the CUDA path, a Clang CUDA compiler (Enzyme differentiates Clang's device IR,
# not nvcc's), and the same ClangEnzyme plugin as the CPU Enzyme path. The test
# target itself lives in the ideal_mhd_model subdirectory. Enable with:
#   -DVMECPP_USE_CUDA=ON -DVMECPP_ENABLE_CUDA_ENZYME=ON \
#   -DVMECPP_ENZYME_PLUGIN=/path/to/ClangEnzyme-NN.so \
#   -DCMAKE_CUDA_COMPILER=clang++-NN -DCMAKE_CXX_COMPILER=clang++-NN
option(VMECPP_ENABLE_CUDA_ENZYME "Build the CUDA-Enzyme Jacobian JVP target" OFF)
if(VMECPP_ENABLE_CUDA_ENZYME)
  if(NOT VMECPP_USE_CUDA)
    message(FATAL_ERROR
      "VMECPP_ENABLE_CUDA_ENZYME=ON requires VMECPP_USE_CUDA=ON.")
  endif()
  if(NOT CMAKE_CUDA_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR
      "VMECPP_ENABLE_CUDA_ENZYME requires a Clang CUDA compiler (got "
      "'${CMAKE_CUDA_COMPILER_ID}'); Enzyme differentiates Clang's device IR. "
      "Configure with -DCMAKE_CUDA_COMPILER=clang++-NN.")
  endif()
  set(VMECPP_ENZYME_PLUGIN "" CACHE FILEPATH "Path to ClangEnzyme-NN.so")
  if(NOT VMECPP_ENZYME_PLUGIN OR NOT EXISTS "${VMECPP_ENZYME_PLUGIN}")
    message(FATAL_ERROR
      "VMECPP_ENABLE_CUDA_ENZYME=ON requires "
      "-DVMECPP_ENZYME_PLUGIN=/path/to/ClangEnzyme-NN.so")
  endif()
  message(STATUS "CUDA-Enzyme plugin: ${VMECPP_ENZYME_PLUGIN}")
  enable_testing()
endif()

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
# The vendored FFTX codelets are machine-generated code that does not compile
# under MSVC (over-long string literals, non-standard cast syntax), so the
# option defaults off on Windows. The CUDA path uses cuFFT and the CPU
# fallback is the partial-DFT transform, so neither build needs FFTX.
if(MSVC)
  option(VMECPP_USE_FFTX "Use FFTX/SPIRAL kernels for toroidal transforms" OFF)
else()
  option(VMECPP_USE_FFTX "Use FFTX/SPIRAL kernels for toroidal transforms" ON)
endif()
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

# CUDA-backed toroidal FFT (links cuFFT + cuBLAS + cudart, defines VMECPP_USE_CUDA).
if(VMECPP_USE_CUDA)
  target_link_libraries(vmecpp_core PRIVATE CUDA::cudart CUDA::cufft CUDA::cublas)
  target_compile_definitions(vmecpp_core PUBLIC VMECPP_USE_CUDA)
  message(STATUS "VMECPP_USE_CUDA: linked CUDA::cudart + CUDA::cufft + CUDA::cublas")
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
if(NOT MSVC)
  install(TARGETS indata2json DESTINATION vmecpp/cpp/third_party/indata2json/)
endif()

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
