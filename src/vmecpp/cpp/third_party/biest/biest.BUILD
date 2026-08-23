# Build overlay for the BIEST archive fetched in
# //third_party:non_module_deps.bzl (header-only).
cc_library(
    name = "biest",
    hdrs = glob(
        ["include/**"],
        # the archive embeds a stale copy of SCTL; use @sctl instead
        exclude = [
            "include/sctl/**",
            "include/sctl.hpp",
        ],
    ),
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = ["@sctl"],
)

filegroup(
    name = "geom",
    srcs = glob(["geom/**"]),
    visibility = ["//visibility:public"],
)
