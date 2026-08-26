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

thread_local std::string error_message;

void Copy(const vmecpp::GeometryJet& source, double (&m_target)[4]) {
  std::copy(source.begin(), source.end(), m_target);
}

int Fail(const std::string& message) {
  error_message = message;
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
    error_message.clear();
    return 0;
  } catch (const std::exception& error) {
    return Fail(error.what());
  } catch (...) {
    return Fail("unknown VMEC++ error");
  }
}

extern "C" void vmecpp_geometry_destroy(vmecpp_geometry_handle* handle) {
  delete handle;
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
    error_message.clear();
    return 0;
  } catch (const std::exception& error) {
    return Fail(error.what());
  } catch (...) {
    return Fail("unknown VMEC++ error");
  }
}

extern "C" const char* vmecpp_geometry_error(void) {
  return error_message.c_str();
}
