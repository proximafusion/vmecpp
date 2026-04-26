// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/ideal_mhd_model/force_symmetry.h"

#include "absl/log/check.h"

namespace vmecpp {

namespace {

int ReflectedToroidalIndex(const Sizes& sizes, int k) {
  return (sizes.nZeta - k) % sizes.nZeta;
}

int ReflectedPoloidalIndex(const Sizes& sizes, int l) {
  if (l == 0) {
    return 0;
  }
  return sizes.nThetaEven - l;
}

}  // namespace

void DecomposeForceComponent(const Sizes& sizes, std::span<const double> full,
                             ReflectionParity parity, std::span<double> sym,
                             std::span<double> asym) {
  CHECK(sizes.lasym);
  const std::size_t nZnT = static_cast<std::size_t>(sizes.nZnT);
  CHECK_EQ(full.size() % nZnT, std::size_t{0});

  const std::size_t nZnTReduced =
      static_cast<std::size_t>(sizes.nZeta * sizes.nThetaReduced);
  const std::size_t num_surfaces = full.size() / nZnT;
  CHECK_EQ(sym.size(), num_surfaces * nZnTReduced);
  CHECK_EQ(asym.size(), num_surfaces * nZnTReduced);

  for (std::size_t j = 0; j < num_surfaces; ++j) {
    for (int k = 0; k < sizes.nZeta; ++k) {
      const int k_ref = ReflectedToroidalIndex(sizes, k);
      for (int l = 0; l < sizes.nThetaReduced; ++l) {
        const int l_ref = ReflectedPoloidalIndex(sizes, l);

        const std::size_t idx_full =
            ((j * static_cast<std::size_t>(sizes.nZeta) + k) *
             static_cast<std::size_t>(sizes.nThetaEff)) +
            l;
        const std::size_t idx_ref =
            ((j * static_cast<std::size_t>(sizes.nZeta) + k_ref) *
             static_cast<std::size_t>(sizes.nThetaEff)) +
            l_ref;
        const std::size_t idx_reduced =
            ((j * static_cast<std::size_t>(sizes.nZeta) + k) *
             static_cast<std::size_t>(sizes.nThetaReduced)) +
            l;

        const double value = full[idx_full];
        const double reflected = full[idx_ref];

        if (parity == ReflectionParity::kStandard) {
          sym[idx_reduced] = 0.5 * (value + reflected);
          asym[idx_reduced] = 0.5 * (value - reflected);
        } else {
          sym[idx_reduced] = 0.5 * (value - reflected);
          asym[idx_reduced] = 0.5 * (value + reflected);
        }
      }
    }
  }
}

}  // namespace vmecpp
