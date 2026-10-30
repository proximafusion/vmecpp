// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_FREE_BOUNDARY_SURFACE_GEOMETRY_MOCKUP_SURFACE_GEOMETRY_MOCKUP_H_
#define VMECPP_FREE_BOUNDARY_SURFACE_GEOMETRY_MOCKUP_SURFACE_GEOMETRY_MOCKUP_H_
#include <string>
#include <vector>

#include "vmecpp/common/fourier_basis_fast_toroidal/fourier_basis_fast_toroidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/free_boundary/surface_geometry/surface_geometry.h"
#include "vmecpp/free_boundary/tangential_partitioning/tangential_partitioning.h"

namespace vmecpp {

class SurfaceGeometryMockup {
 public:
  const int signOfJacobian = -1;

  // Setup SurfaceGeometry and all needed ingredients (for testing)
  // based on a static W7-X LCFS geometry provided with the code in CSV form.
  static SurfaceGeometryMockup InitializeFromFile(
      const std::string& filename =
          "vmecpp/free_boundary/surface_geometry_mockup/"
          "lcfs.SurfaceRZFourier.csv",
      bool lasym = false, int nphi = 36, int ntheta = 0, int nfp = 5);

  SurfaceGeometryMockup(bool lasym, int nfp, int mpol, int ntor, int ntheta,
                        int nphi, std::vector<double>& m_rmnc,
                        std::vector<double>& m_rmns,
                        std::vector<double>& m_zmns,
                        std::vector<double>& m_zmnc, int num_threads = 1,
                        int thread_id = 0);

  bool lasym;
  std::vector<double> rmnc;
  std::vector<double> rmns;
  std::vector<double> zmns;
  std::vector<double> zmnc;

  Sizes s;
  FourierBasisFastToroidal fb;
  TangentialPartitioning tp;
  SurfaceGeometry sg;
};

}  // namespace vmecpp

#endif  // VMECPP_FREE_BOUNDARY_SURFACE_GEOMETRY_MOCKUP_SURFACE_GEOMETRY_MOCKUP_H_
