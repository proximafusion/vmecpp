// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/geometry/geometry_c_api.h"

#include <algorithm>
#include <exception>
#include <memory>
#include <optional>
#include <string>

#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/geometry/geometry.h"
#include "vmecpp/vmec/geometry/vmec_geometry.h"
#include "vmecpp/vmec/vmec/vmec.h"

struct vmecpp_geometry_handle {
  vmecpp::Geometry geometry;
};

namespace {

std::string& ErrorMessage() {
  thread_local std::string message;
  return message;
}

void Copy(const vmecpp::GeometryJet& source,
          double (&m_target)[VMECPP_GEOMETRY_JET_SIZE]) {
  std::copy(source.begin(), source.end(), m_target);
}

int Fail(const std::string& message) {
  ErrorMessage() = message;
  return 1;
}

}  // namespace

extern "C" int vmecpp_geometry_create(const char* input_path,
                                      vmecpp_geometry_handle** output) {
  if (input_path == nullptr || output == nullptr) {
    return Fail("input_path and output must not be null");
  }
  *output = nullptr;
  try {
    const vmecpp::VmecINDATA indata = vmecpp::VmecINDATA::FromFile(input_path);
    auto result = vmecpp::run(indata, std::nullopt, std::nullopt,
                              vmecpp::OutputMode::kSilent);
    if (!result.ok()) return Fail(std::string(result.status().message()));
    auto handle = std::make_unique<vmecpp_geometry_handle>();
    handle->geometry =
        vmecpp::MakeGeometry(indata, result->vmec_internal_results);
    *output = handle.release();
    ErrorMessage().clear();
    return 0;
  } catch (const std::exception& error) {
    return Fail(error.what());
  } catch (...) {
    return Fail("unknown VMEC++ error");
  }
}

extern "C" void vmecpp_geometry_destroy(vmecpp_geometry_handle* handle) {
  delete handle;  // NOLINT(cppcoreguidelines-owning-memory)
}

extern "C" int vmecpp_geometry_get_metadata(
    const vmecpp_geometry_handle* handle, vmecpp_geometry_metadata* output) {
  if (handle == nullptr || output == nullptr) {
    return Fail("handle and output must not be null");
  }
  const vmecpp::Geometry& geometry = handle->geometry;
  const vmecpp::GeometryDimensions& dimensions = geometry.dimensions;
  output->nfp = dimensions.nfp;
  const int boundary_r00 =
      (dimensions.ns - 1) * dimensions.mpol * (dimensions.ntor + 1);
  output->major_radius = geometry.coefficients.r_cc[boundary_r00];
  ErrorMessage().clear();
  return 0;
}

extern "C" int vmecpp_geometry_evaluate(const vmecpp_geometry_handle* handle,
                                        double s, double theta, double zeta,
                                        vmecpp_geometry_point* output) {
  if (handle == nullptr || output == nullptr) {
    return Fail("handle and output must not be null");
  }
  try {
    const vmecpp::GeometryPoint point =
        vmecpp::EvaluateGeometry(handle->geometry, s, theta, zeta);
    Copy(point.r, output->r);
    Copy(point.z, output->z);
    Copy(point.lambda, output->lambda);
    Copy(point.toroidal_flux, output->toroidal_flux);
    Copy(point.poloidal_flux, output->poloidal_flux);
    ErrorMessage().clear();
    return 0;
  } catch (const std::exception& error) {
    return Fail(error.what());
  } catch (...) {
    return Fail("unknown VMEC++ error");
  }
}

extern "C" const char* vmecpp_geometry_error(void) {
  return ErrorMessage().c_str();
}
