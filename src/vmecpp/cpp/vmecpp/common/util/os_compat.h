// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
//
// Small portability shims so the same source builds under MSVC (native
// Windows) as well as GCC/Clang. On POSIX every name below resolves to the
// platform function unchanged, so the Linux build is unaffected; only the
// MSVC branch substitutes the Windows equivalent.
#ifndef VMECPP_COMMON_UTIL_OS_COMPAT_H_
#define VMECPP_COMMON_UTIL_OS_COMPAT_H_

#include <cstdlib>
#include <string>

#ifdef _WIN32
#include <process.h>  // _getpid

// POSIX environment and process shims. setenv ignores the overwrite flag
// (the call sites here always overwrite); unsetenv removes the variable;
// getpid maps to the underscore-prefixed MSVC name.
static inline int setenv(const char* name, const char* value, int /*ovr*/) {
  return _putenv_s(name, value);
}
static inline int unsetenv(const char* name) { return _putenv_s(name, ""); }
static inline int getpid() { return _getpid(); }

#define VMECPP_UNREACHABLE() __assume(0)
#else
#include <unistd.h>  // getpid, setenv, unsetenv via <cstdlib>

#define VMECPP_UNREACHABLE() __builtin_unreachable()
#endif

namespace vmecpp {

// Directory for transient files (the distinct-mode batch-output staging and
// the diagnostic dumps), with a trailing separator. POSIX uses /tmp/; Windows
// uses %TEMP%, then %TMP%, then the current directory.
inline std::string OsTmpDir() {
#ifdef _WIN32
  if (const char* t = std::getenv("TEMP")) return std::string(t) + "\\";
  if (const char* t = std::getenv("TMP")) return std::string(t) + "\\";
  return ".\\";
#else
  return "/tmp/";
#endif
}

}  // namespace vmecpp

#endif  // VMECPP_COMMON_UTIL_OS_COMPAT_H_
