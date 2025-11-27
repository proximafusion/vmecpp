// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_COMMON_COMPUTE_BACKEND_COMPUTE_BACKEND_FACTORY_H_
#define VMECPP_COMMON_COMPUTE_BACKEND_COMPUTE_BACKEND_FACTORY_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "vmecpp/common/compute_backend/compute_backend.h"

namespace vmecpp {

// Factory for creating compute backend instances.
//
// This factory provides a centralized way to create and manage compute
// backends. It supports runtime backend selection via environment variables
// or explicit configuration, and handles fallback logic when requested
// backends are unavailable.
//
// Usage:
//   auto backend = ComputeBackendFactory::Create(config);
//   if (backend.ok()) {
//     backend.value()->FourierToReal(...);
//   }
//
// Environment variable:
//   VMECPP_BACKEND: Set to "cpu" or "cuda" to override default backend.
class ComputeBackendFactory {
 public:
  // Creates a compute backend instance based on the provided configuration.
  //
  // If the requested backend type is not available (e.g., CUDA requested
  // but no GPU present), returns an error status.
  //
  // Parameters:
  //   config: Backend configuration specifying type and options.
  //
  // Returns:
  //   A unique_ptr to the created backend, or an error status.
  static absl::StatusOr<std::unique_ptr<ComputeBackend>> Create(
      const BackendConfig& config);

  // Creates a compute backend instance, with automatic fallback.
  //
  // If the requested backend is unavailable, falls back to CPU backend.
  // This is useful for deployments where CUDA may or may not be available.
  //
  // Parameters:
  //   config: Backend configuration specifying type and options.
  //
  // Returns:
  //   A unique_ptr to the created backend (never fails, always returns CPU
  //   as fallback).
  static std::unique_ptr<ComputeBackend> CreateWithFallback(
      const BackendConfig& config);

  // Creates a compute backend based on environment configuration.
  //
  // Checks the VMECPP_BACKEND environment variable:
  //   - "cpu": Use CPU backend
  //   - "cuda": Use CUDA backend (with fallback to CPU if unavailable)
  //   - unset/empty: Use CPU backend
  //
  // Returns:
  //   A unique_ptr to the created backend.
  static std::unique_ptr<ComputeBackend> CreateFromEnvironment();

  // Returns a list of available backend types on this system.
  //
  // CPU is always available. CUDA is available only if compiled with
  // CUDA support and a compatible GPU is detected.
  static std::vector<BackendType> GetAvailableBackends();

  // Returns true if the specified backend type is available.
  static bool IsBackendAvailable(BackendType type);

  // Converts a BackendType to its string representation.
  static std::string BackendTypeToString(BackendType type);

  // Parses a string to BackendType.
  //
  // Accepts: "cpu", "CPU", "cuda", "CUDA" (case-insensitive).
  // Returns kCpu for unrecognized strings.
  static BackendType StringToBackendType(const std::string& str);
};

}  // namespace vmecpp

#endif  // VMECPP_COMMON_COMPUTE_BACKEND_COMPUTE_BACKEND_FACTORY_H_
