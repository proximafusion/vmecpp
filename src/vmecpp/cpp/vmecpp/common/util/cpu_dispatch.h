// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_COMMON_UTIL_CPU_DISPATCH_H_
#define VMECPP_COMMON_UTIL_CPU_DISPATCH_H_

// VMECPP_TARGET_CLONES: apply to a free-function definition to emit an AVX2
// variant and an SSE2 fallback ("default") in the same binary. The compiler
// inserts a one-time ifunc resolver that picks the right clone at load time.
// Expands to nothing on non-x86 or non-GCC/Clang compilers.
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
#define VMECPP_TARGET_CLONES \
  __attribute__((target_clones("avx2,bmi,bmi2,popcnt", "default")))
#else
#define VMECPP_TARGET_CLONES
#endif

#endif  // VMECPP_COMMON_UTIL_CPU_DISPATCH_H_
