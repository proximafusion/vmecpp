// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_FREE_BOUNDARY_MGRID_PROVIDER_MGRID_PROVIDER_H_
#define VMECPP_FREE_BOUNDARY_MGRID_PROVIDER_MGRID_PROVIDER_H_

#include <Eigen/Dense>
#include <filesystem>

#include "absl/status/status.h"
#include "vmecpp/common/makegrid_lib/makegrid_lib.h"
#include "vmecpp/common/sizes/sizes.h"

namespace vmecpp {

class MGridProvider {
 public:
  MGridProvider();

  absl::Status LoadFile(const std::filesystem::path& filename,
                        const Eigen::VectorXd& coil_currents);

  // May return an error status, when the response table resolution doesn't
  // match coil_currents.size()
  absl::Status LoadFields(
      const makegrid::MagneticFieldResponseTable& magnetic_response_table,
      const Eigen::VectorXd& coil_currents);

  void SetFixedMagneticField(const Eigen::VectorXd& fixed_br,
                             const Eigen::VectorXd& fixed_bp,
                             const Eigen::VectorXd& fixed_bz);

  // Interpolate [ztMin, ztMax) of nZnT points; error if outside the grid.
  [[nodiscard]] absl::Status interpolate(int ztMin, int ztMax, int nZeta,
                                         int nZnT, const Eigen::VectorXd& r,
                                         const Eigen::VectorXd& z,
                                         Eigen::VectorXd& m_interpBr,
                                         Eigen::VectorXd& m_interpBp,
                                         Eigen::VectorXd& m_interpBz) const;

  // mgrid internals below

  Eigen::VectorXd bR;
  Eigen::VectorXd bP;
  Eigen::VectorXd bZ;

  int nfp;

  int numR;
  double minR;
  double maxR;
  double deltaR;

  int numZ;
  double minZ;
  double maxZ;
  double deltaZ;

  int numPhi;

  int nextcur;

  std::string mgrid_mode;

  bool IsLoaded() const { return has_mgrid_loaded_; }

 private:
  // Size the accumulation arrays to the current grid and clear them.
  void ResetAccumulatedField();

  // Add one circuit's contribution, weighted by its current.
  // `contribution_at(linear_index)` returns that circuit's {b_r, b_p, b_z} at
  // one grid point. LoadFile and LoadFields differ only in where the
  // contribution comes from, so the weighting lives here once.
  template <typename ContributionAt>
  void AccumulateCircuit(double coil_current, ContributionAt contribution_at) {
    const int num_grid_points = numPhi * numZ * numR;
    for (int linear_index = 0; linear_index < num_grid_points; ++linear_index) {
      const auto [b_r, b_p, b_z] = contribution_at(linear_index);
      bR[linear_index] += b_r * coil_current;
      bP[linear_index] += b_p * coil_current;
      bZ[linear_index] += b_z * coil_current;
    }  // linear_index
  }

  bool has_mgrid_loaded_;
  bool has_fixed_field_;

  Eigen::VectorXd fixed_br_;
  Eigen::VectorXd fixed_bp_;
  Eigen::VectorXd fixed_bz_;
};

}  // namespace vmecpp

#endif  // VMECPP_FREE_BOUNDARY_MGRID_PROVIDER_MGRID_PROVIDER_H_
