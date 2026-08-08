# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Module extension for VMEC++ dependencies not yet available as Bazel modules.

These were previously declared in WORKSPACE.bazel. WORKSPACE repositories do not
propagate to modules that depend on vmecpp, so a module extension is used
instead: it exposes the repositories through bzlmod, which lets a downstream
`bazel_dep(name = "vmecpp")` resolve them automatically.
"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:local.bzl", "new_local_repository")

# The HDF5 and NetCDF archives ship only their sources here; the actual builds
# (via rules_foreign_cc) live in //third_party/hdf5 and //third_party/netcdf4.
_ALL_CONTENT = """\
filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)
"""

def _non_module_deps_impl(_module_ctx):
    # BIEST, the Boundary Integral Equation Solver for Toroidal systems
    # (Apache-2.0), used by the BIEST free-boundary solver. See
    # //third_party/biest/README.md for the local warm-start patch.
    http_archive(
        name = "biest",
        urls = ["https://github.com/dmalhotra/BIEST/archive/97329a2e76799aed158266dedf28e3cfa7323df4.tar.gz"],
        strip_prefix = "BIEST-97329a2e76799aed158266dedf28e3cfa7323df4",
        sha256 = "4cd84deb60ba26b9a97bf4ee231faed9c2debca60dbc99b1c4b11cc0ed034987",
        build_file = "//third_party/biest:biest.BUILD",
        patches = ["//third_party/biest:warm_start.patch"],
        patch_args = ["-p1"],
    )

    # SCTL (Apache-2.0), BIEST's only dependency (header-only).
    http_archive(
        name = "sctl",
        urls = ["https://github.com/dmalhotra/SCTL/archive/af59e54adaf0e3bd25643df2d32967bd05f370fa.tar.gz"],
        strip_prefix = "SCTL-af59e54adaf0e3bd25643df2d32967bd05f370fa",
        sha256 = "5c6337010f3bfd4aeb0dbd81d0af479d6aa9f8abcfbddb771f6ba123f79f13bd",
        build_file = "//third_party/sctl:sctl.BUILD",
        patches = ["//third_party/sctl:cxx20.patch"],
        patch_args = ["-p1"],
    )

    # Intel MKL's FFTW3 interface headers (system installation; Ubuntu:
    # libmkl-dev). The FFT provider for SCTL and the Vac2 solver -- FFTW
    # itself is GPL and cannot be linked into the MIT-licensed VMEC++.
    new_local_repository(
        name = "mkl_fftw",
        path = "/usr/include/mkl",
        build_file = "//third_party/mkl:mkl_fftw.BUILD",
    )

    # Accurate Biot-Savart routines with Correct Asymptotic Behaviour (C++):
    # https://github.com/jonathanschilling/abscab-cpp
    http_archive(
        name = "abscab_cpp",
        urls = ["https://github.com/jonathanschilling/abscab-cpp/archive/refs/tags/v1.0.3.tar.gz"],
        strip_prefix = "abscab-cpp-1.0.3",
        sha256 = "d7d4d8060117ac047ca7a3c5824f79cc7d8f42c538e542946650b188c7d2e145",
    )

    # HDF5. Switch to a Bazel module when one is available:
    # https://github.com/bazelbuild/bazel-central-registry/issues/1327
    http_archive(
        name = "hdf5",
        build_file_content = _ALL_CONTENT,
        sha256 = "df5ee33c74d5efb59738075ef96f4201588e1f1eeb233f047ac7fd1072dee1f6",
        urls = ["https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5-1_14_3.tar.gz"],
        strip_prefix = "hdf5-hdf5-1_14_3",
    )

    http_archive(
        name = "netcdf4",
        build_file_content = _ALL_CONTENT,
        integrity = "sha256-mQ9G1JUl1qtdxCSfhoTG3ur1Teb+xjoYfp+zgswP/f8=",
        urls = ["https://github.com/Unidata/netcdf-c/archive/refs/tags/v4.9.3.tar.gz"],
        strip_prefix = "netcdf-c-4.9.3",
    )

non_module_deps = module_extension(implementation = _non_module_deps_impl)
