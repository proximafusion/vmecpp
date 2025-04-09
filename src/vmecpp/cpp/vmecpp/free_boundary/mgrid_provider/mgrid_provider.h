// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_FREE_BOUNDARY_MGRID_PROVIDER_MGRID_PROVIDER_H_
#define VMECPP_FREE_BOUNDARY_MGRID_PROVIDER_MGRID_PROVIDER_H_

#include <string>
#include <vector>

#include "vmecpp/common/makegrid_lib/makegrid_lib.h"
#include "vmecpp/common/sizes/sizes.h"

namespace vmecpp {

class MGridProvider {
 public:
  MGridProvider();

  // TODO(jons): return an absl::Status, rename to LoadFile, fix casing
  int loadFromMGrid(const std::string& filename,
                    const std::vector<double>& coilCurrents);

  absl::Status LoadFields(
      const makegrid::MakegridParameters& mgrid_params,
      const makegrid::MagneticFieldResponseTable& magnetic_response_table,
      const std::vector<double>& coilCurrents);

  void SetFixedMagneticField(const std::vector<double>& fixed_br,
                             const std::vector<double>& fixed_bp,
                             const std::vector<double>& fixed_bz);

  void interpolate(int ztMin, int ztMax, int nZeta,
                   const std::vector<double>& r, const std::vector<double>& z,
                   std::vector<double>& m_interpBr,
                   std::vector<double>& m_interpBp,
                   std::vector<double>& m_interpBz) const;

  // mgrid internals below

  std::vector<double> bR;
  std::vector<double> bP;
  std::vector<double> bZ;

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

 private:
  bool hasMgridLoaded;
  bool has_fixed_field_;

  std::vector<double> fixed_br_;
  std::vector<double> fixed_bp_;
  std::vector<double> fixed_bz_;
};

}  // namespace vmecpp

#endif  // VMECPP_FREE_BOUNDARY_MGRID_PROVIDER_MGRID_PROVIDER_H_
