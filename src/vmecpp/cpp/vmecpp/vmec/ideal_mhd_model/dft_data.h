// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_DFT_DATA_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_DFT_DATA_H_

#include <span>

namespace vmecpp {

// A bundle of views over (const) data required by the "ForcesToFourier"
// calculations
// TODO(eguiraud): use this struct as a data member in IdealMHDModel (with
// vectors instead of spans).
struct RealSpaceForces {
  std::span<const double> armn_e;
  std::span<const double> armn_o;
  std::span<const double> azmn_e;
  std::span<const double> azmn_o;
  std::span<const double> blmn_e;
  std::span<const double> blmn_o;
  std::span<const double> brmn_e;
  std::span<const double> brmn_o;
  std::span<const double> bzmn_e;
  std::span<const double> bzmn_o;
  std::span<const double> clmn_e;
  std::span<const double> clmn_o;
  std::span<const double> crmn_e;
  std::span<const double> crmn_o;
  std::span<const double> czmn_e;
  std::span<const double> czmn_o;
  std::span<const double> frcon_e;
  std::span<const double> frcon_o;
  std::span<const double> fzcon_e;
  std::span<const double> fzcon_o;
};

// A bundle of views over (non-const) data required by the "FourierToReal"
// calculations
struct RealSpaceGeometry {
  std::span<double> r1_e;
  std::span<double> r1_o;
  std::span<double> ru_e;
  std::span<double> ru_o;
  std::span<double> rv_e;
  std::span<double> rv_o;
  std::span<double> z1_e;
  std::span<double> z1_o;
  std::span<double> zu_e;
  std::span<double> zu_o;
  std::span<double> zv_e;
  std::span<double> zv_o;
  std::span<double> lu_e;
  std::span<double> lu_o;
  std::span<double> lv_e;
  std::span<double> lv_o;
  std::span<double> rCon;
  std::span<double> zCon;
};

}  // namespace vmecpp

#endif  // VMECPP_VMEC_IDEAL_MHD_MODEL_DFT_DATA_H_
