// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
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

  absl::StatusOr<std::vector<double>> rmnc =
      composed_types::CoefficientsRCos(*maybe_surface);
  if (!rmnc.ok()) {
    LOG(FATAL) << rmnc.status().message();
  }

  absl::StatusOr<std::vector<double>> rmns =
      composed_types::CoefficientsRSin(*maybe_surface);
  if (!rmns.ok()) {
    LOG(FATAL) << rmns.status().message();
  }

  absl::StatusOr<std::vector<double>> zmnc =
      composed_types::CoefficientsZCos(*maybe_surface);
  if (!zmnc.ok()) {
    LOG(FATAL) << zmnc.status().message();
  }

  absl::StatusOr<std::vector<double>> zmns =
      composed_types::CoefficientsZSin(*maybe_surface);
  if (!zmns.ok()) {
    LOG(FATAL) << zmns.status().message();
  }

  const int mpol = std::ranges::max(*xm) + 1;

  // Note that this xn does not contain the nfp factor, as VMEC's xn does.
  const int ntor = std::ranges::max(*xn);

  return SurfaceGeometryMockup(lasym, nfp, mpol, ntor, ntheta, nphi, *rmnc,
                               *rmns, *zmns, *zmnc);
}  // InitializeFromFile

SurfaceGeometryMockup::SurfaceGeometryMockup(bool lasym, int nfp, int mpol,
                                             int ntor, int ntheta, int nphi,
                                             std::vector<double>& m_rmnc,
                                             std::vector<double>& m_rmns,
                                             std::vector<double>& m_zmns,
                                             std::vector<double>& m_zmnc,
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
  std::vector<double> r_cc(s.mnsize);
  std::vector<double> r_ss(s.mnsize);
  std::vector<double> r_sc;
  std::vector<double> r_cs;
  fb.cos_to_cc_ss(m_rmnc, r_cc, r_ss, s.ntor, s.mpol);
  if (s.lasym) {
    r_sc.resize(s.mnsize, 0.0);
    r_cs.resize(s.mnsize, 0.0);
    fb.sin_to_sc_cs(m_rmns, r_sc, r_cs, s.ntor, s.mpol);
  }

  std::vector<double> z_sc(s.mnsize);
  std::vector<double> z_cs(s.mnsize);
  std::vector<double> z_cc;
  std::vector<double> z_ss;
  fb.sin_to_sc_cs(m_zmns, z_sc, z_cs, s.ntor, s.mpol);
  if (s.lasym) {
    z_cc.resize(s.mnsize, 0.0);
    z_ss.resize(s.mnsize, 0.0);
    fb.cos_to_cc_ss(m_zmnc, z_cc, z_ss, s.ntor, s.mpol);
  }

  // perform inv-DFT and related stuff in SurfaceGeometry
  bool full_update = true;
  sg.update(r_cc, r_ss, r_sc, r_cs, z_sc, z_cs, z_cc, z_ss, signOfJacobian, full_update);
}  // SurfaceGeometryMockup

}  // namespace vmecpp
