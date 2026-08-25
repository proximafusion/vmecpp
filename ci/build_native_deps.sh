#!/usr/bin/env bash
# Builds HDF5 and netCDF-C from source as static, position-independent
# libraries with OPeNDAP/DAP support disabled, and installs them into a
# single prefix.
#
# Why this exists: the manylinux "netcdf-devel"/"hdf5-devel" distro packages
# used previously by the Linux wheel build (see pyproject.toml,
# [tool.cibuildwheel.linux]) are built WITH DAP/OPeNDAP support enabled. That
# drags in libcurl and its whole TLS/Kerberos/LDAP dependency closure
# (libssl, libcrypto, libkrb5, libldap, libsasl2, libgssapi, libnghttp2,
# libidn2, libpsl, libssh, libunistring) into the wheel via auditwheel, even
# though vmecpp only ever does local-file netCDF/HDF5 I/O and never uses
# OPeNDAP/remote access. That one dependency chain alone accounts for
# roughly a quarter of the published wheel size.
#
# This script builds both libraries the same way the repo's Bazel build
# already does for local development/testing (see
# src/vmecpp/cpp/third_party/hdf5/BUILD.bazel,
# src/vmecpp/cpp/third_party/netcdf4/BUILD.bazel and
# src/vmecpp/cpp/third_party/non_module_deps.bzl for the pinned versions),
# but for the CMake-based wheel-build path. It is intended to be invoked
# from `before-build` in `[tool.cibuildwheel.linux]` inside the manylinux
# container, before CMake configures the vmecpp extension.
#
# Usage:
#   ./ci/build_native_deps.sh [install_prefix]
#
# The install prefix can also be provided via the VMECPP_DEPS_PREFIX
# environment variable; the positional argument takes precedence. Defaults
# to /tmp/vmecpp-deps if neither is given.
#
# After running, point CMake at the result with:
#   -DCMAKE_PREFIX_PATH=<install_prefix>

set -euo pipefail

# Versions/URLs must stay in sync with
# src/vmecpp/cpp/third_party/non_module_deps.bzl.
readonly HDF5_TAG="hdf5-1_14_3"
readonly HDF5_URL="https://github.com/HDFGroup/hdf5/archive/refs/tags/${HDF5_TAG}.tar.gz"
readonly HDF5_SRC_DIRNAME="hdf5-${HDF5_TAG}"

readonly NETCDF_TAG="v4.9.3"
readonly NETCDF_URL="https://github.com/Unidata/netcdf-c/archive/refs/tags/${NETCDF_TAG}.tar.gz"
readonly NETCDF_SRC_DIRNAME="netcdf-c-4.9.3"

PREFIX="${1:-${VMECPP_DEPS_PREFIX:-/tmp/vmecpp-deps}}"
mkdir -p "${PREFIX}"
PREFIX="$(cd "${PREFIX}" && pwd)"
echo "Installing HDF5 ${HDF5_TAG} and netCDF-C ${NETCDF_TAG} into: ${PREFIX}"

NPROC="$(nproc)"

WORKDIR="$(mktemp -d -t vmecpp-native-deps.XXXXXX)"
trap 'rm -rf "${WORKDIR}"' EXIT

echo "== Downloading and building HDF5 ${HDF5_TAG} =="
curl -fsSL "${HDF5_URL}" -o "${WORKDIR}/hdf5.tar.gz"
tar -xzf "${WORKDIR}/hdf5.tar.gz" -C "${WORKDIR}"

cmake -S "${WORKDIR}/${HDF5_SRC_DIRNAME}" -B "${WORKDIR}/hdf5-build" \
  -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
  -DCMAKE_PREFIX_PATH="${PREFIX}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DHDF5_BUILD_CPP_LIB=ON \
  -DHDF5_ENABLE_Z_LIB_SUPPORT=ON \
  -DHDF5_ENABLE_SZIP_SUPPORT=OFF \
  -DHDF5_ENABLE_SZIP_ENCODING=OFF \
  -DHDF5_BUILD_EXAMPLES=OFF \
  -DHDF5_BUILD_TOOLS=OFF \
  -DBUILD_TESTING=OFF

cmake --build "${WORKDIR}/hdf5-build" --parallel "${NPROC}"
cmake --install "${WORKDIR}/hdf5-build"

# NETCDF_ENABLE_TESTS/NETCDF_BUILD_UTILITIES: we only need libnetcdf.a for
# linking into vmecpp, not netCDF's own test/utility binaries (ncdump,
# ncgen, nctest, ...). Building them also fails here: CMake's FindHDF5
# module places the C library before the HL library on the final static
# link line, which is backwards (HL depends on symbols defined in the C
# library) and produces undefined-reference errors for those executables
# with a fully static HDF5. Since we don't build or ship them, this sidesteps
# the ordering bug entirely rather than patching it.
echo "== Downloading and building netCDF-C ${NETCDF_TAG} =="
curl -fsSL "${NETCDF_URL}" -o "${WORKDIR}/netcdf-c.tar.gz"
tar -xzf "${WORKDIR}/netcdf-c.tar.gz" -C "${WORKDIR}"

cmake -S "${WORKDIR}/${NETCDF_SRC_DIRNAME}" -B "${WORKDIR}/netcdf-c-build" \
  -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
  -DCMAKE_PREFIX_PATH="${PREFIX}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DBUILD_TESTING=OFF \
  -DNETCDF_ENABLE_DAP=OFF \
  -DNETCDF_ENABLE_DAP2=OFF \
  -DNETCDF_ENABLE_DAP4=OFF \
  -DNETCDF_ENABLE_NCZARR=OFF \
  -DNETCDF_ENABLE_NCZARR_ZIP=OFF \
  -DNETCDF_ENABLE_TESTS=OFF \
  -DNETCDF_BUILD_UTILITIES=OFF

cmake --build "${WORKDIR}/netcdf-c-build" --parallel "${NPROC}"
cmake --install "${WORKDIR}/netcdf-c-build"

echo "== Done. HDF5 ${HDF5_TAG} and netCDF-C ${NETCDF_TAG} installed into ${PREFIX} =="
