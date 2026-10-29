// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
//
// Portability shims so the same source builds under MSVC (native Windows) as
// well as GCC/Clang.
#ifndef VMECPP_COMMON_UTIL_OS_COMPAT_H_
#define VMECPP_COMMON_UTIL_OS_COMPAT_H_

#ifdef _WIN32
#define VMECPP_UNREACHABLE() __assume(0)
#else
#define VMECPP_UNREACHABLE() __builtin_unreachable()
#endif

#endif  // VMECPP_COMMON_UTIL_OS_COMPAT_H_
