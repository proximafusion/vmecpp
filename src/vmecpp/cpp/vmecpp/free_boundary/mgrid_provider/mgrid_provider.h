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

  // Interpolate the vacuum magnetic field onto the tangential grid points
  // [ztMin, ztMax) of the plasma boundary. nZnT is the number of tangential
  // grid points on the whole boundary, of which [ztMin, ztMax) is this
  // thread's slice.
  //
  // Returns a kFailedPrecondition status if the boundary left the domain of
  // the vacuum field grid. The interpolated field is still written in that
  // case, clamped to the grid edge, so that the caller sees finite values and
  // can carry the status to a safe place before acting on it; the clamped
  // field is not physically meaningful and must not be used to advance the
  // equilibrium.
  //
  // The status describes this thread's tangential slice only. The caller runs
  // this inside the nested vacuum OpenMP team, so the result has to be reduced
  // across that team before it is acted upon. The extents named in the error
  // message, on the other hand, are taken over the whole boundary [0, nZnT),
  // so that every thread that reports produces the same message no matter
  // which one wins the reduction.
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
  bool has_mgrid_loaded_;
  bool has_fixed_field_;

  Eigen::VectorXd fixed_br_;
  Eigen::VectorXd fixed_bp_;
  Eigen::VectorXd fixed_bz_;
};

}  // namespace vmecpp

#endif  // VMECPP_FREE_BOUNDARY_MGRID_PROVIDER_MGRID_PROVIDER_H_
