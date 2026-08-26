// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/geometry/vmec_geometry.h"

#include <cmath>
#include <vector>

namespace vmecpp {
namespace {

std::vector<double> Scale(const RowMatrixXd& source, int mpol, int ntor) {
  std::vector<double> result(source.size());
  for (int j = 0; j < source.rows(); ++j) {
    for (int m = 0; m < mpol; ++m) {
      for (int n = 0; n <= ntor; ++n) {
        const int index = (j * mpol + m) * (ntor + 1) + n;
        const double mscale = m == 0 ? 1.0 : std::sqrt(2.0);
        const double nscale = n == 0 ? 1.0 : std::sqrt(2.0);
        result[index] = source(index) * mscale * nscale;
      }
    }
  }
  return result;
}

void ScaleLambda(std::vector<double>& m_coefficients,
                 const VmecInternalResults& internal, int modes_per_surface) {
  if (m_coefficients.empty()) return;
  for (int j = 0; j < internal.num_full; ++j) {
    const double factor = internal.lamscale / internal.phipF[j];
    for (int mode = 0; mode < modes_per_surface; ++mode) {
      m_coefficients[j * modes_per_surface + mode] *= factor;
    }
  }
}

}  // namespace

Geometry MakeGeometry(const VmecINDATA& indata,
                      const VmecInternalResults& internal) {
  Geometry result{
      .dimensions = {.ns = internal.num_full,
                     .mpol = indata.mpol,
                     .ntor = indata.ntor,
                     .nfp = indata.nfp},
      .toroidal_flux = std::vector<double>(
          internal.phiF.data(), internal.phiF.data() + internal.num_full),
      .poloidal_flux = std::vector<double>(internal.num_full, 0.0),
      .coefficients = {},
  };
  const double delta_s = 1.0 / (internal.num_full - 1);
  for (int j = 1; j < internal.num_full; ++j) {
    result.poloidal_flux[j] = result.poloidal_flux[j - 1] +
                              internal.sign_of_jacobian * 2.0 * M_PI * delta_s *
                                  internal.phipH[j - 1] * internal.iotaH[j - 1];
  }

  GeometryCoefficients& coefficients = result.coefficients;
  coefficients.r_cc = Scale(internal.rmncc, indata.mpol, indata.ntor);
  coefficients.z_sc = Scale(internal.zmnsc, indata.mpol, indata.ntor);
  coefficients.lambda_sc = Scale(internal.lmnsc, indata.mpol, indata.ntor);
  if (indata.ntor > 0) {
    coefficients.r_ss = Scale(internal.rmnss, indata.mpol, indata.ntor);
    coefficients.z_cs = Scale(internal.zmncs, indata.mpol, indata.ntor);
    coefficients.lambda_cs = Scale(internal.lmncs, indata.mpol, indata.ntor);
  }
  if (indata.lasym) {
    coefficients.r_sc = Scale(internal.rmnsc, indata.mpol, indata.ntor);
    coefficients.z_cc = Scale(internal.zmncc, indata.mpol, indata.ntor);
    coefficients.lambda_cc = Scale(internal.lmncc, indata.mpol, indata.ntor);
    if (indata.ntor > 0) {
      coefficients.r_cs = Scale(internal.rmncs, indata.mpol, indata.ntor);
      coefficients.z_ss = Scale(internal.zmnss, indata.mpol, indata.ntor);
      coefficients.lambda_ss = Scale(internal.lmnss, indata.mpol, indata.ntor);
    }
  }

  const int modes_per_surface = indata.mpol * (indata.ntor + 1);
  ScaleLambda(coefficients.lambda_sc, internal, modes_per_surface);
  ScaleLambda(coefficients.lambda_cs, internal, modes_per_surface);
  ScaleLambda(coefficients.lambda_cc, internal, modes_per_surface);
  ScaleLambda(coefficients.lambda_ss, internal, modes_per_surface);

  // Undo the solver's m=1 constraint to recover physical product-basis
  // coefficients.
  if (indata.mpol > 1 && indata.ntor > 0) {
    for (int j = 0; j < internal.num_full; ++j) {
      for (int n = 0; n <= indata.ntor; ++n) {
        const int index = (j * indata.mpol + 1) * (indata.ntor + 1) + n;
        const double old_r_ss = coefficients.r_ss[index];
        coefficients.r_ss[index] = old_r_ss + coefficients.z_cs[index];
        coefficients.z_cs[index] = old_r_ss - coefficients.z_cs[index];
      }
    }
  }
  if (indata.mpol > 1 && indata.lasym) {
    for (int j = 0; j < internal.num_full; ++j) {
      for (int n = 0; n <= indata.ntor; ++n) {
        const int index = (j * indata.mpol + 1) * (indata.ntor + 1) + n;
        const double old_r_sc = coefficients.r_sc[index];
        coefficients.r_sc[index] = old_r_sc + coefficients.z_cc[index];
        coefficients.z_cc[index] = old_r_sc - coefficients.z_cc[index];
      }
    }
  }
  return result;
}

}  // namespace vmecpp
