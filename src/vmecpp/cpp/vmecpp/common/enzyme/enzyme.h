// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_COMMON_ENZYME_ENZYME_H_
#define VMECPP_COMMON_ENZYME_ENZYME_H_

// Thin declarations for the Enzyme automatic-differentiation intrinsics.
//
// Enzyme (https://enzyme.mit.edu) differentiates at the LLVM-IR level via a
// Clang plugin (ClangEnzyme-NN.so). These intrinsics are resolved by that
// plugin; without it the symbols do not link, so any translation unit that
// calls them must be compiled with -fplugin=<ClangEnzyme>. The CMake option
// VMECPP_ENABLE_ENZYME wires that flag and is OFF by default.
//
// Differentiation activity is selected per argument with the marker globals
// below: pass `enzyme_dup, primal_ptr, shadow_ptr` for an active buffer (the
// gradient accumulates into shadow_ptr) and `enzyme_const, value` for an input
// held fixed. Enzyme matches these markers by symbol name.
//
// Constraint that shapes how differentiable kernels here are written: Enzyme's
// allocation analysis does not track Eigen's aligned allocator, so a heap
// temporary from an Eigen expression (e.g. a dynamic-size `A * x`) crossing the
// differentiated call aborts with "freeing without malloc". Differentiable
// kernels therefore operate on caller-owned buffers via Eigen::Map and avoid
// allocating expression temporaries on the differentiated path.

// The marker globals and the __enzyme_* intrinsic names are part of the Enzyme
// ABI: the plugin matches them by exact symbol name, so they cannot be const or
// renamed to satisfy the in-tree naming/identifier lint rules.

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
extern int enzyme_dup;
extern int enzyme_const;
extern int enzyme_dupnoneed;
extern int enzyme_out;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

// Reverse mode: returns nothing useful here; gradients land in the shadow
// buffers passed alongside each `enzyme_dup` argument.
template <typename... Args>
void __enzyme_autodiff(  // NOLINT(bugprone-reserved-identifier,readability-identifier-naming)
    void*, Args...);

// Forward mode: propagates the seed in the shadow argument to the directional
// derivative of the result.
template <typename Return, typename... Args>
Return
__enzyme_fwddiff(  // NOLINT(bugprone-reserved-identifier,readability-identifier-naming)
    void*, Args...);

#endif  // VMECPP_COMMON_ENZYME_ENZYME_H_
