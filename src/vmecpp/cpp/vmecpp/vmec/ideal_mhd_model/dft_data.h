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
  std::span<const real_t> armn_e;
  std::span<const real_t> armn_o;
  std::span<const real_t> azmn_e;
  std::span<const real_t> azmn_o;
  std::span<const real_t> blmn_e;
  std::span<const real_t> blmn_o;
  std::span<const real_t> brmn_e;
  std::span<const real_t> brmn_o;
  std::span<const real_t> bzmn_e;
  std::span<const real_t> bzmn_o;
  std::span<const real_t> clmn_e;
  std::span<const real_t> clmn_o;
  std::span<const real_t> crmn_e;
  std::span<const real_t> crmn_o;
  std::span<const real_t> czmn_e;
  std::span<const real_t> czmn_o;
  std::span<const real_t> frcon_e;
  std::span<const real_t> frcon_o;
  std::span<const real_t> fzcon_e;
  std::span<const real_t> fzcon_o;
};

// A bundle of views over (non-const) data required by the "FourierToReal"
// calculations
struct RealSpaceGeometry {
  std::span<real_t> r1_e;
  std::span<real_t> r1_o;
  std::span<real_t> ru_e;
  std::span<real_t> ru_o;
  std::span<real_t> rv_e;
  std::span<real_t> rv_o;
  std::span<real_t> z1_e;
  std::span<real_t> z1_o;
  std::span<real_t> zu_e;
  std::span<real_t> zu_o;
  std::span<real_t> zv_e;
  std::span<real_t> zv_o;
  std::span<real_t> lu_e;
  std::span<real_t> lu_o;
  std::span<real_t> lv_e;
  std::span<real_t> lv_o;
  std::span<real_t> rCon;
  std::span<real_t> zCon;
};

}  // namespace vmecpp

#endif  // VMECPP_VMEC_IDEAL_MHD_MODEL_DFT_DATA_H_
