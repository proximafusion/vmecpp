// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/free_boundary/surface_geometry_mockup/surface_geometry_mockup.h"

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "util/file_io/file_io.h"
#include "vmecpp/common/composed_types_lib/composed_types_lib.h"

using composed_types::SurfaceRZFourier;
using composed_types::SurfaceRZFourierFromCsv;

namespace vmecpp {

SurfaceGeometryMockup SurfaceGeometryMockup::InitializeFromFile(
    const std::string& filename, bool lasym, int nphi, int ntheta, int nfp) {
  absl::StatusOr<std::string> maybe_lcfs = file_io::ReadFile(filename);
  if (!maybe_lcfs.ok()) {
    LOG(FATAL) << maybe_lcfs.status().message();
  }

  absl::StatusOr<SurfaceRZFourier> maybe_surface =
      SurfaceRZFourierFromCsv(*maybe_lcfs);
  if (!maybe_surface.ok()) {
    LOG(FATAL) << maybe_surface.status().message();
  }

  absl::StatusOr<std::vector<int>> xm =
      composed_types::PoloidalModeNumbers(*maybe_surface);
  if (!xm.ok()) {
    LOG(FATAL) << xm.status().message();
  }

  // These do not contain the nfp factor that is present in VMEC's xn.
  absl::StatusOr<std::vector<int>> xn =
      composed_types::ToroidalModeNumbers(*maybe_surface);
  if (!xn.ok()) {
    LOG(FATAL) << xn.status().message();
  }

  absl::StatusOr<std::vector<real_t>> rmnc_double =
      composed_types::CoefficientsRCos(*maybe_surface);
  if (!rmnc_double.ok()) {
    LOG(FATAL) << rmnc_double.status().message();
  }
  std::vector<real_t> rmnc(rmnc_double->begin(), rmnc_double->end());

  absl::StatusOr<std::vector<real_t>> rmns_double =
      composed_types::CoefficientsRSin(*maybe_surface);
  if (!rmns_double.ok()) {
    LOG(FATAL) << rmns_double.status().message();
  }
  std::vector<real_t> rmns(rmns_double->begin(), rmns_double->end());

  absl::StatusOr<std::vector<real_t>> zmnc_double =
      composed_types::CoefficientsZCos(*maybe_surface);
  if (!zmnc_double.ok()) {
    LOG(FATAL) << zmnc_double.status().message();
  }
  std::vector<real_t> zmnc(zmnc_double->begin(), zmnc_double->end());

  absl::StatusOr<std::vector<real_t>> zmns_double =
      composed_types::CoefficientsZSin(*maybe_surface);
  if (!zmns_double.ok()) {
    LOG(FATAL) << zmns_double.status().message();
  }
  std::vector<real_t> zmns(zmns_double->begin(), zmns_double->end());

  const int mpol = std::ranges::max(*xm) + 1;

  // Note that this xn does not contain the nfp factor, as VMEC's xn does.
  const int ntor = std::ranges::max(*xn);

  return SurfaceGeometryMockup(lasym, nfp, mpol, ntor, ntheta, nphi, rmnc, rmns,
                               zmns, zmnc);
}  // InitializeFromFile

SurfaceGeometryMockup::SurfaceGeometryMockup(bool lasym, int nfp, int mpol,
                                             int ntor, int ntheta, int nphi,
                                             std::vector<real_t>& m_rmnc,
                                             std::vector<real_t>& m_rmns,
                                             std::vector<real_t>& m_zmns,
                                             std::vector<real_t>& m_zmnc,
                                             int num_threads, int thread_id)
    : lasym(lasym),
      rmnc(m_rmnc),
      rmns(m_rmns),
      zmns(m_zmns),
      zmnc(m_zmnc),
      s(lasym, nfp, mpol, ntor, ntheta, nphi),
      fb(&s),
      tp(s.nZnT, num_threads, thread_id),
      sg(&s, &fb, &tp) {
  // convert into 2D Fourier coefficient arrays
  std::vector<real_t> rCC(s.mnsize);
  std::vector<real_t> rSS(s.mnsize);
  std::vector<real_t> rSC;
  std::vector<real_t> rCS;
  fb.cos_to_cc_ss(m_rmnc, rCC, rSS, s.ntor, s.mpol);
  if (s.lasym) {
    rSC.resize(s.mnsize, 0.0);
    rCS.resize(s.mnsize, 0.0);
    fb.sin_to_sc_cs(m_rmns, rSC, rCS, s.ntor, s.mpol);
  }

  std::vector<real_t> zSC(s.mnsize);
  std::vector<real_t> zCS(s.mnsize);
  std::vector<real_t> zCC;
  std::vector<real_t> zSS;
  fb.sin_to_sc_cs(m_zmns, zSC, zCS, s.ntor, s.mpol);
  if (s.lasym) {
    zCC.resize(s.mnsize, 0.0);
    zSS.resize(s.mnsize, 0.0);
    fb.cos_to_cc_ss(m_zmnc, zCC, zSS, s.ntor, s.mpol);
  }

  // perform inv-DFT and related stuff in SurfaceGeometry
  bool fullUpdate = true;
  sg.update(rCC, rSS, rSC, rCS, zSC, zCS, zCC, zSS, signOfJacobian, fullUpdate);
}  // SurfaceGeometryMockup

}  // namespace vmecpp
