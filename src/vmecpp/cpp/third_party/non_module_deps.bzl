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
