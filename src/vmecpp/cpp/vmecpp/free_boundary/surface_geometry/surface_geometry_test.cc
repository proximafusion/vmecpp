// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/free_boundary/surface_geometry/surface_geometry.h"

#include <array>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "util/file_io/file_io.h"
#include "util/testing/numerical_comparison_lib.h"
#include "vmecpp/common/fourier_basis_fast_toroidal/fourier_basis_fast_toroidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/free_boundary/surface_geometry_mockup/surface_geometry_mockup.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace vmecpp {

using nlohmann::json;

using file_io::ReadFile;
using testing::IsCloseRelAbs;

using ::testing::TestWithParam;
using ::testing::Values;

TEST(TestSurfaceGeometry, CheckConstants) {
  constexpr double kTolerance = 1.0e-12;

  const SurfaceGeometryMockup& surface_geometry_mockup =
      SurfaceGeometryMockup::InitializeFromFile();

  const Sizes& s = surface_geometry_mockup.s;
  const SurfaceGeometry& sg = surface_geometry_mockup.sg;

  const double omega_phi = 2.0 * M_PI / (s.nfp * s.nZeta);
  for (int k = 0; k < s.nZeta; ++k) {
    const double phi = omega_phi * k;

    EXPECT_TRUE(IsCloseRelAbs(std::cos(phi), sg.cos_phi[k], kTolerance))
        << "at k=" << k;
    EXPECT_TRUE(IsCloseRelAbs(std::sin(phi), sg.sin_phi[k], kTolerance))
        << "at k=" << k;
  }
}

TEST(TestSurfaceGeometry, CheckInvDFT) {
  constexpr double kTolerance = 1.0e-12;

  // tolerance for finite-difference derivative approximations
  constexpr double kFdTol = 1.0e-6;

  SurfaceGeometryMockup surface_geometry_mockup =
      SurfaceGeometryMockup::InitializeFromFile();

  const Sizes& s = surface_geometry_mockup.s;
  const SurfaceGeometry& sg = surface_geometry_mockup.sg;
  const FourierBasisFastToroidal& fb = surface_geometry_mockup.fb;

  // reference inv-DFT
  std::vector<double> ref_r(s.nThetaEven * s.nZeta);  // full surface
  std::vector<double> ref_z(s.nThetaEven * s.nZeta);  // full surface

  std::vector<double> ref_r_tp(s.nZnT);     // theta + eps
  std::vector<double> ref_r_tm(s.nZnT);     // theta - eps
  std::vector<double> ref_r_zp(s.nZnT);     //  phi + eps
  std::vector<double> ref_r_zm(s.nZnT);     //  phi - eps
  std::vector<double> ref_r_tp_zp(s.nZnT);  // theta + eps, phi + eps
  std::vector<double> ref_r_tp_zm(s.nZnT);  // theta + eps, phi - eps
  std::vector<double> ref_r_tm_zp(s.nZnT);  // theta - eps, phi + eps
  std::vector<double> ref_r_tm_zm(s.nZnT);  // theta - eps, phi - eps
  std::vector<double> ref_ru(s.nZnT);
  std::vector<double> ref_rv(s.nZnT);
  std::vector<double> ref_ruu(s.nZnT);
  std::vector<double> ref_ruv(s.nZnT);
  std::vector<double> ref_rvv(s.nZnT);

  std::vector<double> ref_z_tp(s.nZnT);     // theta + eps
  std::vector<double> ref_z_tm(s.nZnT);     // theta - eps
  std::vector<double> ref_z_zp(s.nZnT);     //  phi + eps
  std::vector<double> ref_z_zm(s.nZnT);     //  phi - eps
  std::vector<double> ref_z_tp_zp(s.nZnT);  // theta + eps, phi + eps
  std::vector<double> ref_z_tp_zm(s.nZnT);  // theta + eps, phi - eps
  std::vector<double> ref_z_tm_zp(s.nZnT);  // theta - eps, phi + eps
  std::vector<double> ref_z_tm_zm(s.nZnT);  // theta - eps, phi - eps
  std::vector<double> ref_zu(s.nZnT);
  std::vector<double> ref_zv(s.nZnT);
  std::vector<double> ref_zuu(s.nZnT);
  std::vector<double> ref_zuv(s.nZnT);
  std::vector<double> ref_zvv(s.nZnT);

  // relative finite-difference step
  constexpr double kEps = 1.0e-5;

  std::vector<double> theta(s.nThetaEven * s.nZeta);
  std::vector<double> phi(s.nThetaEven * s.nZeta);

  const double omega_theta = 2.0 * M_PI / s.nThetaEven;
  const double omega_phi = 2.0 * M_PI / (s.nfp * s.nZeta);

  for (int l = 0; l < s.nThetaEven; ++l) {
    for (int k = 0; k < s.nZeta; ++k) {
      const int kl = l * s.nZeta + k;

      theta[kl] = omega_theta * l;
      phi[kl] = omega_phi * k;

      ref_r[kl] = 0.0;
      ref_z[kl] = 0.0;
      for (int mn = 0; mn < s.mnmax; ++mn) {
        const double kernel = fb.xm[mn] * theta[kl] - fb.xn[mn] * phi[kl];
        ref_r[kl] += surface_geometry_mockup.rmnc[mn] * cos(kernel);
        ref_z[kl] += surface_geometry_mockup.zmns[mn] * sin(kernel);
        if (s.lasym) {
          ref_r[kl] += surface_geometry_mockup.rmns[mn] * sin(kernel);
          ref_z[kl] += surface_geometry_mockup.zmnc[mn] * cos(kernel);
        }  // lasym
      }    // mn
    }      // k
  }        // l

  for (int l = 0; l < s.nThetaReduced; ++l) {
    for (int k = 0; k < s.nZeta; ++k) {
      const int kl = l * s.nZeta + k;

      theta[kl] = omega_theta * l;
      phi[kl] = omega_phi * k;

      ref_r_tp[kl] = 0.0;
      ref_r_tm[kl] = 0.0;
      ref_r_zp[kl] = 0.0;
      ref_r_zm[kl] = 0.0;
      ref_r_tp_zp[kl] = 0.0;
      ref_r_tp_zm[kl] = 0.0;
      ref_r_tm_zp[kl] = 0.0;
      ref_r_tm_zm[kl] = 0.0;
      ref_ru[kl] = 0.0;
      ref_rv[kl] = 0.0;
      ref_ruu[kl] = 0.0;
      ref_ruv[kl] = 0.0;
      ref_rvv[kl] = 0.0;

      ref_z_tp[kl] = 0.0;
      ref_z_tm[kl] = 0.0;
      ref_z_zp[kl] = 0.0;
      ref_z_zm[kl] = 0.0;
      ref_z_tp_zp[kl] = 0.0;
      ref_z_tp_zm[kl] = 0.0;
      ref_z_tm_zp[kl] = 0.0;
      ref_z_tm_zm[kl] = 0.0;
      ref_zu[kl] = 0.0;
      ref_zv[kl] = 0.0;
      ref_zuu[kl] = 0.0;
      ref_zuv[kl] = 0.0;
      ref_zvv[kl] = 0.0;

      for (int mn = 0; mn < s.mnmax; ++mn) {
        const double kernel = fb.xm[mn] * theta[kl] - fb.xn[mn] * phi[kl];
        const double kernel_tp =
            fb.xm[mn] * (theta[kl] + kEps) - fb.xn[mn] * phi[kl];
        const double kernel_tm =
            fb.xm[mn] * (theta[kl] - kEps) - fb.xn[mn] * phi[kl];
        const double kernel_zp =
            fb.xm[mn] * theta[kl] - fb.xn[mn] * (phi[kl] + kEps);
        const double kernel_zm =
            fb.xm[mn] * theta[kl] - fb.xn[mn] * (phi[kl] - kEps);

        const double kernel_tp_zp =
            fb.xm[mn] * (theta[kl] + kEps) - fb.xn[mn] * (phi[kl] + kEps);
        const double kernel_tp_zm =
            fb.xm[mn] * (theta[kl] + kEps) - fb.xn[mn] * (phi[kl] - kEps);
        const double kernel_tm_zp =
            fb.xm[mn] * (theta[kl] - kEps) - fb.xn[mn] * (phi[kl] + kEps);
        const double kernel_tm_zm =
            fb.xm[mn] * (theta[kl] - kEps) - fb.xn[mn] * (phi[kl] - kEps);

        ref_r_tp[kl] += surface_geometry_mockup.rmnc[mn] * cos(kernel_tp);
        ref_r_tm[kl] += surface_geometry_mockup.rmnc[mn] * cos(kernel_tm);
        ref_r_zp[kl] += surface_geometry_mockup.rmnc[mn] * cos(kernel_zp);
        ref_r_zm[kl] += surface_geometry_mockup.rmnc[mn] * cos(kernel_zm);
        ref_r_tp_zp[kl] += surface_geometry_mockup.rmnc[mn] * cos(kernel_tp_zp);
        ref_r_tp_zm[kl] += surface_geometry_mockup.rmnc[mn] * cos(kernel_tp_zm);
        ref_r_tm_zp[kl] += surface_geometry_mockup.rmnc[mn] * cos(kernel_tm_zp);
        ref_r_tm_zm[kl] += surface_geometry_mockup.rmnc[mn] * cos(kernel_tm_zm);
        ref_ru[kl] +=
            surface_geometry_mockup.rmnc[mn] * sin(kernel) * (-fb.xm[mn]);
        ref_rv[kl] += surface_geometry_mockup.rmnc[mn] * sin(kernel) * fb.xn[mn];
        ref_ruu[kl] += surface_geometry_mockup.rmnc[mn] * cos(kernel) *
                      (-fb.xm[mn] * fb.xm[mn]);
        ref_ruv[kl] += surface_geometry_mockup.rmnc[mn] * cos(kernel) *
                      fb.xn[mn] * fb.xm[mn];
        ref_rvv[kl] += surface_geometry_mockup.rmnc[mn] * cos(kernel) *
                      (-fb.xn[mn] * fb.xn[mn]);

        ref_z_tp[kl] += surface_geometry_mockup.zmns[mn] * sin(kernel_tp);
        ref_z_tm[kl] += surface_geometry_mockup.zmns[mn] * sin(kernel_tm);
        ref_z_zp[kl] += surface_geometry_mockup.zmns[mn] * sin(kernel_zp);
        ref_z_zm[kl] += surface_geometry_mockup.zmns[mn] * sin(kernel_zm);
        ref_z_tp_zp[kl] += surface_geometry_mockup.zmns[mn] * sin(kernel_tp_zp);
        ref_z_tp_zm[kl] += surface_geometry_mockup.zmns[mn] * sin(kernel_tp_zm);
        ref_z_tm_zp[kl] += surface_geometry_mockup.zmns[mn] * sin(kernel_tm_zp);
        ref_z_tm_zm[kl] += surface_geometry_mockup.zmns[mn] * sin(kernel_tm_zm);
        ref_zu[kl] += surface_geometry_mockup.zmns[mn] * cos(kernel) * fb.xm[mn];
        ref_zv[kl] +=
            surface_geometry_mockup.zmns[mn] * cos(kernel) * (-fb.xn[mn]);
        ref_zuu[kl] += surface_geometry_mockup.zmns[mn] * sin(kernel) *
                      (-fb.xm[mn] * fb.xm[mn]);
        ref_zuv[kl] += surface_geometry_mockup.zmns[mn] * sin(kernel) *
                      fb.xn[mn] * fb.xm[mn];
        ref_zvv[kl] += surface_geometry_mockup.zmns[mn] * sin(kernel) *
                      (-fb.xn[mn] * fb.xn[mn]);

        if (s.lasym) {
          ref_r_tp[kl] += surface_geometry_mockup.rmns[mn] * sin(kernel_tp);
          ref_r_tm[kl] += surface_geometry_mockup.rmns[mn] * sin(kernel_tm);
          ref_r_zp[kl] += surface_geometry_mockup.rmns[mn] * sin(kernel_zp);
          ref_r_zm[kl] += surface_geometry_mockup.rmns[mn] * sin(kernel_zm);
          ref_r_tp_zp[kl] +=
              surface_geometry_mockup.rmns[mn] * sin(kernel_tp_zp);
          ref_r_tp_zm[kl] +=
              surface_geometry_mockup.rmns[mn] * sin(kernel_tp_zm);
          ref_r_tm_zp[kl] +=
              surface_geometry_mockup.rmns[mn] * sin(kernel_tm_zp);
          ref_r_tm_zm[kl] +=
              surface_geometry_mockup.rmns[mn] * sin(kernel_tm_zm);
          ref_ru[kl] +=
              surface_geometry_mockup.rmns[mn] * cos(kernel) * (-fb.xm[mn]);
          ref_rv[kl] +=
              surface_geometry_mockup.rmns[mn] * cos(kernel) * fb.xn[mn];
          ref_ruu[kl] += surface_geometry_mockup.rmns[mn] * sin(kernel) *
                        (-fb.xm[mn] * fb.xm[mn]);
          ref_ruv[kl] += surface_geometry_mockup.rmns[mn] * sin(kernel) *
                        fb.xn[mn] * fb.xm[mn];
          ref_rvv[kl] += surface_geometry_mockup.rmns[mn] * sin(kernel) *
                        (-fb.xn[mn] * fb.xn[mn]);

          ref_z_tp[kl] += surface_geometry_mockup.zmnc[mn] * cos(kernel_tp);
          ref_z_tm[kl] += surface_geometry_mockup.zmnc[mn] * cos(kernel_tm);
          ref_z_zp[kl] += surface_geometry_mockup.zmnc[mn] * cos(kernel_zp);
          ref_z_zm[kl] += surface_geometry_mockup.zmnc[mn] * cos(kernel_zm);
          ref_z_tp_zp[kl] +=
              surface_geometry_mockup.zmnc[mn] * cos(kernel_tp_zp);
          ref_z_tp_zm[kl] +=
              surface_geometry_mockup.zmnc[mn] * cos(kernel_tp_zm);
          ref_z_tm_zp[kl] +=
              surface_geometry_mockup.zmnc[mn] * cos(kernel_tm_zp);
          ref_z_tm_zm[kl] +=
              surface_geometry_mockup.zmnc[mn] * cos(kernel_tm_zm);
          ref_zu[kl] +=
              surface_geometry_mockup.zmnc[mn] * sin(kernel) * fb.xm[mn];
          ref_zv[kl] +=
              surface_geometry_mockup.zmnc[mn] * sin(kernel) * (-fb.xn[mn]);
          ref_zuu[kl] += surface_geometry_mockup.zmnc[mn] * cos(kernel) *
                        (-fb.xm[mn] * fb.xm[mn]);
          ref_zuv[kl] += surface_geometry_mockup.zmnc[mn] * cos(kernel) *
                        fb.xn[mn] * fb.xm[mn];
          ref_zvv[kl] += surface_geometry_mockup.zmnc[mn] * cos(kernel) *
                        (-fb.xn[mn] * fb.xn[mn]);
        }  // lasym
      }    // mn
    }      // m
  }        // l

  // compare results

  // a) direct evaluation of geometry should match exactly (on full surface)
  for (int l = 0; l < s.nThetaEven; ++l) {
    for (int k = 0; k < s.nZeta; ++k) {
      const int kl = l * s.nZeta + k;

      // R
      EXPECT_TRUE(IsCloseRelAbs(ref_r[kl], sg.r1b[kl], kTolerance))
          << "at k=" << k << " l=" << l;

      // Z
      EXPECT_TRUE(IsCloseRelAbs(ref_z[kl], sg.z1b[kl], kTolerance))
          << "at k=" << k << " l=" << l;
    }  // k
  }    // l

  // b) finite-differences derivatives vs. analytical derivatives from inv-DFT:
  // should be roughly the same
  for (int l = 0; l < s.nThetaReduced; ++l) {
    for (int k = 0; k < s.nZeta; ++k) {
      const int kl = l * s.nZeta + k;

      // dR/dTheta
      double ru_approx = (ref_r_tp[kl] - ref_r_tm[kl]) / (2.0 * kEps);
      EXPECT_TRUE(IsCloseRelAbs(ru_approx, sg.rub[kl], kFdTol))
          << "at k=" << k << " l=" << l;

      // dR/dZeta
      double rv_approx = (ref_r_zp[kl] - ref_r_zm[kl]) / (2.0 * kEps);
      EXPECT_TRUE(IsCloseRelAbs(rv_approx, sg.rvb[kl], kFdTol))
          << "at k=" << k << " l=" << l;

      // ----------------

      // dZ/dTheta
      double zu_approx = (ref_z_tp[kl] - ref_z_tm[kl]) / (2.0 * kEps);
      EXPECT_TRUE(IsCloseRelAbs(zu_approx, sg.zub[kl], kFdTol))
          << "at k=" << k << " l=" << l;

      // dZ/dZeta
      double zv_approx = (ref_z_zp[kl] - ref_z_zm[kl]) / (2.0 * kEps);
      EXPECT_TRUE(IsCloseRelAbs(zv_approx, sg.zvb[kl], kFdTol))
          << "at k=" << k << " l=" << l;

      // --------------

      // d^2R/dTheta^2
      double ruu_approx =
          (ref_r_tp[kl] - 2.0 * ref_r[kl] + ref_r_tm[kl]) / (kEps * kEps);
      EXPECT_TRUE(IsCloseRelAbs(ruu_approx, sg.ruu[kl], sqrt(kFdTol)))
          << "at k=" << k << " l=" << l;

      // d^2R/(dTheta dZeta)
      double ruv_approx =
          (ref_r_tp_zp[kl] + ref_r_tm_zm[kl] - ref_r_tp_zm[kl] - ref_r_tm_zp[kl]) /
          (4.0 * kEps * kEps);
      EXPECT_TRUE(IsCloseRelAbs(ruv_approx, sg.ruv[kl], sqrt(kFdTol)))
          << "at k=" << k << " l=" << l;

      // d^2R/dZeta^2
      double rvv_approx =
          (ref_r_zp[kl] - 2.0 * ref_r[kl] + ref_r_zm[kl]) / (kEps * kEps);
      EXPECT_TRUE(IsCloseRelAbs(rvv_approx, sg.rvv[kl], sqrt(kFdTol)))
          << "at k=" << k << " l=" << l;

      // ----------------

      // d^2Z/dTheta^2
      double zuu_approx =
          (ref_z_tp[kl] - 2.0 * ref_z[kl] + ref_z_tm[kl]) / (kEps * kEps);
      EXPECT_TRUE(IsCloseRelAbs(zuu_approx, sg.zuu[kl], sqrt(kFdTol)))
          << "at k=" << k << " l=" << l;

      // d^2Z/(dTheta dZeta)
      double zuv_approx =
          (ref_z_tp_zp[kl] + ref_z_tm_zm[kl] - ref_z_tp_zm[kl] - ref_z_tm_zp[kl]) /
          (4.0 * kEps * kEps);
      EXPECT_TRUE(IsCloseRelAbs(zuv_approx, sg.zuv[kl], sqrt(kFdTol)))
          << "at k=" << k << " l=" << l;

      // d^2Z/dZeta^2
      double zvv_approx =
          (ref_z_zp[kl] - 2.0 * ref_z[kl] + ref_z_zm[kl]) / (kEps * kEps);
      EXPECT_TRUE(IsCloseRelAbs(zvv_approx, sg.zvv[kl], sqrt(kFdTol)))
          << "at k=" << k << " l=" << l;
    }  // k
  }    // l

  // c) inv-DFT vs. reference inv-DFT: should be exactly the same
  for (int l = 0; l < s.nThetaReduced; ++l) {
    for (int k = 0; k < s.nZeta; ++k) {
      const int kl = l * s.nZeta + k;

      // dR/dTheta
      EXPECT_TRUE(IsCloseRelAbs(ref_ru[kl], sg.rub[kl], kTolerance))
          << "at k=" << k << " l=" << l;

      // dR/dZeta
      EXPECT_TRUE(IsCloseRelAbs(ref_rv[kl], sg.rvb[kl], kTolerance))
          << "at k=" << k << " l=" << l;

      // -----------------

      // dZ/dTheta
      EXPECT_TRUE(IsCloseRelAbs(ref_zu[kl], sg.zub[kl], kTolerance))
          << "at k=" << k << " l=" << l;

      // dZ/dZeta
      EXPECT_TRUE(IsCloseRelAbs(ref_zv[kl], sg.zvb[kl], kTolerance))
          << "at k=" << k << " l=" << l;

      // -----------------

      // d^2R/dTheta^2
      EXPECT_TRUE(IsCloseRelAbs(ref_ruu[kl], sg.ruu[kl], kTolerance))
          << "at k=" << k << " l=" << l;

      // d^2R/(dTheta dZeta)
      EXPECT_TRUE(IsCloseRelAbs(ref_ruv[kl], sg.ruv[kl], kTolerance))
          << "at k=" << k << " l=" << l;

      // d^2R/dZeta^2
      EXPECT_TRUE(IsCloseRelAbs(ref_rvv[kl], sg.rvv[kl], kTolerance))
          << "at k=" << k << " l=" << l;

      // -----------------

      // d^2Z/dTheta^2
      EXPECT_TRUE(IsCloseRelAbs(ref_zuu[kl], sg.zuu[kl], kTolerance))
          << "at k=" << k << " l=" << l;

      // d^2Z/(dTheta dZeta)
      EXPECT_TRUE(IsCloseRelAbs(ref_zuv[kl], sg.zuv[kl], kTolerance))
          << "at k=" << k << " l=" << l;

      // d^2Z/dZeta^2
      EXPECT_TRUE(IsCloseRelAbs(ref_zvv[kl], sg.zvv[kl], kTolerance))
          << "at k=" << k << " l=" << l;
    }  // k
  }    // l
}

TEST(TestSurfaceGeometry, CheckDerivedQuantities) {
  constexpr double kTolerance = 1.0e-12;

  SurfaceGeometryMockup surface_geometry_mockup =
      SurfaceGeometryMockup::InitializeFromFile();

  const Sizes& s = surface_geometry_mockup.s;
  const FourierBasisFastToroidal& fb = surface_geometry_mockup.fb;
  const SurfaceGeometry& sg = surface_geometry_mockup.sg;

  // form vector-values quantities as vectors and perform actual dot-products
  // for verification

  // reference inv-DFT
  std::vector<double> ref_r(s.nThetaEven * s.nZeta);
  std::vector<double> ref_z(s.nThetaEven * s.nZeta);

  std::vector<double> ref_ru(s.nZnT);
  std::vector<double> ref_rv(s.nZnT);
  std::vector<double> ref_ruu(s.nZnT);
  std::vector<double> ref_ruv(s.nZnT);
  std::vector<double> ref_rvv(s.nZnT);

  std::vector<double> ref_zu(s.nZnT);
  std::vector<double> ref_zv(s.nZnT);
  std::vector<double> ref_zuu(s.nZnT);
  std::vector<double> ref_zuv(s.nZnT);
  std::vector<double> ref_zvv(s.nZnT);

  std::vector<double> theta(s.nThetaEven * s.nZeta);
  std::vector<double> phi(s.nThetaEven * s.nZeta);

  const double omega_theta = 2.0 * M_PI / s.nThetaEven;
  const double omega_phi = 2.0 * M_PI / (s.nfp * s.nZeta);

  for (int l = 0; l < s.nThetaEven; ++l) {
    for (int k = 0; k < s.nZeta; ++k) {
      const int kl = l * s.nZeta + k;

      theta[kl] = omega_theta * l;
      phi[kl] = omega_phi * k;

      ref_r[kl] = 0.0;
      ref_z[kl] = 0.0;
      for (int mn = 0; mn < s.mnmax; ++mn) {
        const double kernel = fb.xm[mn] * theta[kl] - fb.xn[mn] * phi[kl];
        ref_r[kl] += surface_geometry_mockup.rmnc[mn] * cos(kernel);
        ref_z[kl] += surface_geometry_mockup.zmns[mn] * sin(kernel);
        if (s.lasym) {
          ref_r[kl] += surface_geometry_mockup.rmns[mn] * sin(kernel);
          ref_z[kl] += surface_geometry_mockup.zmnc[mn] * cos(kernel);
        }  // lasym
      }    // mn
    }      // k
  }        // l

  for (int l = 0; l < s.nThetaReduced; ++l) {
    for (int k = 0; k < s.nZeta; ++k) {
      const int kl = l * s.nZeta + k;

      ref_ru[kl] = 0.0;
      ref_rv[kl] = 0.0;
      ref_ruu[kl] = 0.0;
      ref_ruv[kl] = 0.0;
      ref_rvv[kl] = 0.0;

      ref_zu[kl] = 0.0;
      ref_zv[kl] = 0.0;
      ref_zuu[kl] = 0.0;
      ref_zuv[kl] = 0.0;
      ref_zvv[kl] = 0.0;

      for (int mn = 0; mn < s.mnmax; ++mn) {
        const double kernel = fb.xm[mn] * theta[kl] - fb.xn[mn] * phi[kl];

        ref_ru[kl] +=
            surface_geometry_mockup.rmnc[mn] * sin(kernel) * (-fb.xm[mn]);
        ref_rv[kl] += surface_geometry_mockup.rmnc[mn] * sin(kernel) * fb.xn[mn];
        ref_ruu[kl] += surface_geometry_mockup.rmnc[mn] * cos(kernel) *
                      (-fb.xm[mn] * fb.xm[mn]);
        ref_ruv[kl] += surface_geometry_mockup.rmnc[mn] * cos(kernel) *
                      fb.xn[mn] * fb.xm[mn];
        ref_rvv[kl] += surface_geometry_mockup.rmnc[mn] * cos(kernel) *
                      (-fb.xn[mn] * fb.xn[mn]);

        ref_zu[kl] += surface_geometry_mockup.zmns[mn] * cos(kernel) * fb.xm[mn];
        ref_zv[kl] +=
            surface_geometry_mockup.zmns[mn] * cos(kernel) * (-fb.xn[mn]);
        ref_zuu[kl] += surface_geometry_mockup.zmns[mn] * sin(kernel) *
                      (-fb.xm[mn] * fb.xm[mn]);
        ref_zuv[kl] += surface_geometry_mockup.zmns[mn] * sin(kernel) *
                      fb.xn[mn] * fb.xm[mn];
        ref_zvv[kl] += surface_geometry_mockup.zmns[mn] * sin(kernel) *
                      (-fb.xn[mn] * fb.xn[mn]);

        if (s.lasym) {
          ref_ru[kl] +=
              surface_geometry_mockup.rmns[mn] * cos(kernel) * (-fb.xm[mn]);
          ref_rv[kl] +=
              surface_geometry_mockup.rmns[mn] * cos(kernel) * fb.xn[mn];
          ref_ruu[kl] += surface_geometry_mockup.rmns[mn] * sin(kernel) *
                        (-fb.xm[mn] * fb.xm[mn]);
          ref_ruv[kl] += surface_geometry_mockup.rmns[mn] * sin(kernel) *
                        fb.xn[mn] * fb.xm[mn];
          ref_rvv[kl] += surface_geometry_mockup.rmns[mn] * sin(kernel) *
                        (-fb.xn[mn] * fb.xn[mn]);

          ref_zu[kl] +=
              surface_geometry_mockup.zmnc[mn] * sin(kernel) * fb.xm[mn];
          ref_zv[kl] +=
              surface_geometry_mockup.zmnc[mn] * sin(kernel) * (-fb.xn[mn]);
          ref_zuu[kl] += surface_geometry_mockup.zmnc[mn] * cos(kernel) *
                        (-fb.xm[mn] * fb.xm[mn]);
          ref_zuv[kl] += surface_geometry_mockup.zmnc[mn] * cos(kernel) *
                        fb.xn[mn] * fb.xm[mn];
          ref_zvv[kl] += surface_geometry_mockup.zmnc[mn] * cos(kernel) *
                        (-fb.xn[mn] * fb.xn[mn]);
        }  // lasym
      }    // mn
    }      // k
  }        // l

  // guu == g_{theta,theta}
  for (int l = 0; l < s.nThetaReduced; ++l) {
    for (int k = 0; k < s.nZeta; ++k) {
      const int kl = l * s.nZeta + k;

      const std::array<double, 3> d_xd_theta = {
          ref_ru[kl] * sg.cos_phi[k], ref_ru[kl] * sg.sin_phi[k], ref_zu[kl]};

      const double ref_guu = dot3x3(d_xd_theta, d_xd_theta);

      EXPECT_TRUE(IsCloseRelAbs(ref_guu, sg.guu[kl], kTolerance))
          << "at k=" << k << " l=" << l;
    }  // k
  }    // l

  // 2 guv == 2 g_{theta,zeta}
  for (int l = 0; l < s.nThetaReduced; ++l) {
    for (int k = 0; k < s.nZeta; ++k) {
      const int kl = l * s.nZeta + k;

      const std::array<double, 3> d_xd_theta = {
          ref_ru[kl] * sg.cos_phi[k], ref_ru[kl] * sg.sin_phi[k], ref_zu[kl]};

      const std::array<double, 3> d_xd_phi = {
          ref_rv[kl] * sg.cos_phi[k] - ref_r[kl] * sg.sin_phi[k],
          ref_rv[kl] * sg.sin_phi[k] + ref_r[kl] * sg.cos_phi[k], ref_zv[kl]};

      const double ref_2guv = 2.0 * dot3x3(d_xd_theta, d_xd_phi) / s.nfp;

      EXPECT_TRUE(IsCloseRelAbs(ref_2guv, sg.guv[kl], kTolerance))
          << "at k=" << k << " l=" << l;
    }  // k
  }    // l

  // gvv == g_{zeta,zeta}
  for (int l = 0; l < s.nThetaReduced; ++l) {
    for (int k = 0; k < s.nZeta; ++k) {
      const int kl = l * s.nZeta + k;

      const std::array<double, 3> d_xd_phi = {
          ref_rv[kl] * sg.cos_phi[k] - ref_r[kl] * sg.sin_phi[k],
          ref_rv[kl] * sg.sin_phi[k] + ref_r[kl] * sg.cos_phi[k], ref_zv[kl]};

      const double ref_gvv = dot3x3(d_xd_phi, d_xd_phi) / (s.nfp * s.nfp);

      EXPECT_TRUE(IsCloseRelAbs(ref_gvv, sg.gvv[kl], kTolerance))
          << "at k=" << k << " l=" << l;
    }  // k
  }    // l

  // components of normal vector N (snr, snv, snz) and drv
  for (int l = 0; l < s.nThetaReduced; ++l) {
    for (int k = 0; k < s.nZeta; ++k) {
      const int kl = l * s.nZeta + k;

      // need to compute cross product in cylindrical coordinates
      // to get cylindrical components of normal vector
      const std::array<double, 3> d_xd_theta_cyl = {ref_ru[kl], 0.0, ref_zu[kl]};

      const std::array<double, 3> d_xd_phi_cyl = {ref_rv[kl], ref_r[kl], ref_zv[kl]};

      std::array<double, 3> n_cyl = {0.0, 0.0, 0.0};

      // -(X_u cross X_v) == (X_v cross X_u)
      cross3x3(/*m_result=*/n_cyl, d_xd_phi_cyl, d_xd_theta_cyl);

      // include sign of Jacobian
      n_cyl[0] *= surface_geometry_mockup.signOfJacobian;
      n_cyl[1] *= surface_geometry_mockup.signOfJacobian;
      n_cyl[2] *= surface_geometry_mockup.signOfJacobian;

      const double ref_drv = -(ref_r[kl] * n_cyl[0] + ref_z[kl] * n_cyl[2]);

      EXPECT_TRUE(IsCloseRelAbs(n_cyl[0], sg.snr[kl], kTolerance))
          << "at k=" << k << " l=" << l;
      EXPECT_TRUE(IsCloseRelAbs(n_cyl[1], sg.snv[kl], kTolerance))
          << "at k=" << k << " l=" << l;
      EXPECT_TRUE(IsCloseRelAbs(n_cyl[2], sg.snz[kl], kTolerance))
          << "at k=" << k << " l=" << l;

      EXPECT_TRUE(IsCloseRelAbs(ref_drv, sg.drv[kl], kTolerance))
          << "at k=" << k << " l=" << l;
    }  // k
  }    // l

  // A, B, C (==auu, auv, avv)
  for (int l = 0; l < s.nThetaReduced; ++l) {
    for (int k = 0; k < s.nZeta; ++k) {
      const int kl = l * s.nZeta + k;

      const std::array<double, 3> d2_xd_theta2 = {ref_ruu[kl], 0.0, ref_zuu[kl]};

      const std::array<double, 3> d2_xd_theta_d_zeta = {ref_ruv[kl], ref_ru[kl],
                                                    ref_zuv[kl]};

      const std::array<double, 3> d2_xd_zeta2 = {ref_rvv[kl] - ref_r[kl],
                                               2 * ref_rv[kl], ref_zvv[kl]};

      // already tested above
      const std::array<double, 3> n_cyl = {sg.snr[kl], sg.snv[kl], sg.snz[kl]};

      const double ref_auu = dot3x3(d2_xd_theta2, n_cyl) / 2;
      const double ref_auv = dot3x3(d2_xd_theta_d_zeta, n_cyl) / s.nfp;
      const double ref_avv = dot3x3(d2_xd_zeta2, n_cyl) / (2 * s.nfp * s.nfp);

      EXPECT_TRUE(IsCloseRelAbs(ref_auu, sg.auu[kl], kTolerance))
          << "at k=" << k << " l=" << l;
      EXPECT_TRUE(IsCloseRelAbs(ref_auv, sg.auv[kl], kTolerance))
          << "at k=" << k << " l=" << l;
      EXPECT_TRUE(IsCloseRelAbs(ref_avv, sg.avv[kl], kTolerance))
          << "at k=" << k << " l=" << l;
    }  // k
  }    // l

  // rzb2, rcosuv, rsinuv: full surface
  for (int l = 0; l < s.nThetaEven; ++l) {
    for (int k = 0; k < s.nZeta; ++k) {
      const int kl = l * s.nZeta + k;

      const double ref_rzb2 = ref_r[kl] * ref_r[kl] + ref_z[kl] * ref_z[kl];

      const double ref_x = ref_r[kl] * sg.cos_phi[k];
      const double ref_y = ref_r[kl] * sg.sin_phi[k];

      EXPECT_TRUE(IsCloseRelAbs(ref_rzb2, sg.rzb2[kl], kTolerance))
          << "at k=" << k << " l=" << l;

      EXPECT_TRUE(IsCloseRelAbs(ref_x, sg.rcosuv[kl], kTolerance))
          << "at k=" << k << " l=" << l;
      EXPECT_TRUE(IsCloseRelAbs(ref_y, sg.rsinuv[kl], kTolerance))
          << "at k=" << k << " l=" << l;
    }  // k
  }    // l
}

}  // namespace vmecpp
