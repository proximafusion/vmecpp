// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include "vmecpp/common/compute_backend/compute_backend_factory.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>

#include "absl/status/status.h"
#include "vmecpp/common/compute_backend/compute_backend_cpu.h"

#ifdef VMECPP_HAS_CUDA
#include "vmecpp/common/compute_backend/cuda/compute_backend_cuda.h"
#endif

namespace vmecpp {

absl::StatusOr<std::unique_ptr<ComputeBackend>> ComputeBackendFactory::Create(
    const BackendConfig& config) {
  switch (config.type) {
    case BackendType::kCpu:
      return std::make_unique<ComputeBackendCpu>();

    case BackendType::kCuda:
#ifdef VMECPP_HAS_CUDA
      {
        auto cuda_backend = std::make_unique<ComputeBackendCuda>(
            config.cuda_device_id, config.cuda_num_streams);
        if (!cuda_backend->IsAvailable()) {
          return absl::UnavailableError(
              "CUDA backend requested but no compatible GPU found");
        }
        return cuda_backend;
      }
#else
      return absl::UnavailableError(
          "CUDA backend requested but VMEC++ was not compiled with CUDA "
          "support. Rebuild with -DVMECPP_ENABLE_CUDA=ON");
#endif

    default:
      return absl::InvalidArgumentError("Unknown backend type");
  }
}

std::unique_ptr<ComputeBackend> ComputeBackendFactory::CreateWithFallback(
    const BackendConfig& config) {
  auto result = Create(config);
  if (result.ok()) {
    return std::move(result.value());
  }

  // Fall back to CPU backend.
  return std::make_unique<ComputeBackendCpu>();
}

std::unique_ptr<ComputeBackend> ComputeBackendFactory::CreateFromEnvironment() {
  BackendConfig config;

  const char* env_backend = std::getenv("VMECPP_BACKEND");
  if (env_backend != nullptr) {
    config.type = StringToBackendType(env_backend);
  }

  const char* env_device = std::getenv("VMECPP_CUDA_DEVICE");
  if (env_device != nullptr) {
    config.cuda_device_id = std::atoi(env_device);
  }

  return CreateWithFallback(config);
}

std::vector<BackendType> ComputeBackendFactory::GetAvailableBackends() {
  std::vector<BackendType> backends;
  backends.push_back(BackendType::kCpu);  // CPU is always available.

#ifdef VMECPP_HAS_CUDA
  if (IsBackendAvailable(BackendType::kCuda)) {
    backends.push_back(BackendType::kCuda);
  }
#endif

  return backends;
}

bool ComputeBackendFactory::IsBackendAvailable(BackendType type) {
  switch (type) {
    case BackendType::kCpu:
      return true;

    case BackendType::kCuda:
#ifdef VMECPP_HAS_CUDA
      {
        // Check if CUDA is functional by attempting to create a backend.
        ComputeBackendCuda test_backend(0, 1);
        return test_backend.IsAvailable();
      }
#else
      return false;
#endif

    default:
      return false;
  }
}

std::string ComputeBackendFactory::BackendTypeToString(BackendType type) {
  switch (type) {
    case BackendType::kCpu:
      return "cpu";
    case BackendType::kCuda:
      return "cuda";
    default:
      return "unknown";
  }
}

BackendType ComputeBackendFactory::StringToBackendType(const std::string& str) {
  std::string lower_str = str;
  std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (lower_str == "cuda" || lower_str == "gpu") {
    return BackendType::kCuda;
  }

  // Default to CPU for any unrecognized string.
  return BackendType::kCpu;
}

}  // namespace vmecpp
