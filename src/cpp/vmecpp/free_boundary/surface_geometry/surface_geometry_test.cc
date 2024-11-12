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

// dot product of two three-dimensional vectors
double dot3x3(const std::span<const double, 3> a,
              const std::span<const double, 3> b) {
  return (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
}

// cross product of two three-dimensional vectors
void cross3x3(std::span<double, 3> m_result, const std::span<const double, 3> a,
              const std::span<const double, 3> b) {
  m_result[0] = a[1] * b[2] - a[2] * b[1];
  m_result[1] = a[2] * b[0] - a[0] * b[2];
  m_result[2] = a[0] * b[1] - a[1] * b[0];
}

// used to specify case-specific tolerances
// and which iterations to test
struct DataSource {
  std::string identifier;
  double tolerance = 0.0;
  std::vector<int> iter2_to_test = {1, 2};
};

class SurfaceGeometryTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(SurfaceGeometryTest, CheckSurfaceGeometry) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  absl::StatusOr<VmecINDATA> vmec_indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  for (int number_of_iterations : data_source_.iter2_to_test) {
    Vmec vmec(*vmec_indata);
    const Sizes& s = vmec.s_;
    const FlowControl& fc = vmec.fc_;

    bool reached_checkpoint =
        vmec.run(VmecCheckpoint::VAC1_SURFACE, number_of_iterations).value();
    ASSERT_TRUE(reached_checkpoint);

    filename = absl::StrFormat(
        "vmecpp/test_data/%s/vac1n_surface/"
        "vac1n_surface_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, number_of_iterations,
        vmec.get_num_eqsolve_retries(), data_source_.identifier);

    std::ifstream ifs_vac1n_surface(filename);
    ASSERT_TRUE(ifs_vac1n_surface.is_open());
    json vac1n_surface = json::parse(ifs_vac1n_surface);

    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const Nestor& n = static_cast<const Nestor&>(*vmec.fb_[thread_id]);
      const TangentialPartitioning& tp = *vmec.tp_[thread_id];
      const SurfaceGeometry& sg = n.GetSurfaceGeometry();

      // full-surface quantities
      for (int kl = 0; kl < s.nZnT; ++kl) {
        const int l = kl / s.nZeta;
        const int k = kl % s.nZeta;

        EXPECT_TRUE(
            IsCloseRelAbs(vac1n_surface["r1b"][k][l], sg.r1b[kl], tolerance));
        EXPECT_TRUE(
            IsCloseRelAbs(vac1n_surface["z1b"][k][l], sg.z1b[kl], tolerance));
      }  // kl

      // local tangential partition
      for (int kl = tp.ztMin; kl < tp.ztMax; ++kl) {
        const int l = kl / s.nZeta;
        const int k = kl % s.nZeta;

        const int klRel = kl - tp.ztMin;

        EXPECT_TRUE(IsCloseRelAbs(vac1n_surface["rub"][k][l], sg.rub[klRel],
                                  tolerance));
        EXPECT_TRUE(IsCloseRelAbs(vac1n_surface["rvb"][k][l], sg.rvb[klRel],
                                  tolerance));
        EXPECT_TRUE(IsCloseRelAbs(vac1n_surface["zub"][k][l], sg.zub[klRel],
                                  tolerance));
        EXPECT_TRUE(IsCloseRelAbs(vac1n_surface["zvb"][k][l], sg.zvb[klRel],
                                  tolerance));

        EXPECT_TRUE(IsCloseRelAbs(vac1n_surface["snr"][k][l], sg.snr[klRel],
                                  tolerance));
        EXPECT_TRUE(IsCloseRelAbs(vac1n_surface["snv"][k][l], sg.snv[klRel],
                                  tolerance));
        EXPECT_TRUE(IsCloseRelAbs(vac1n_surface["snz"][k][l], sg.snz[klRel],
                                  tolerance));

        EXPECT_TRUE(IsCloseRelAbs(vac1n_surface["guu_b"][k][l], sg.guu[klRel],
                                  tolerance));
        EXPECT_TRUE(IsCloseRelAbs(vac1n_surface["guv_b"][k][l], sg.guv[klRel],
                                  tolerance));
        EXPECT_TRUE(IsCloseRelAbs(vac1n_surface["gvv_b"][k][l], sg.gvv[klRel],
                                  tolerance));
      }  // kl

      if (vmec.m_[0]->get_ivacskip() == 0) {
        for (int kl = tp.ztMin; kl < tp.ztMax; ++kl) {
          const int l = kl / s.nZeta;
          const int k = kl % s.nZeta;

          const int klRel = kl - tp.ztMin;

          EXPECT_TRUE(IsCloseRelAbs(vac1n_surface["ruu"][k][l], sg.ruu[klRel],
                                    tolerance));
          EXPECT_TRUE(IsCloseRelAbs(vac1n_surface["ruv"][k][l], sg.ruv[klRel],
                                    tolerance));
          EXPECT_TRUE(IsCloseRelAbs(vac1n_surface["rvv"][k][l], sg.rvv[klRel],
                                    tolerance));

          EXPECT_TRUE(IsCloseRelAbs(vac1n_surface["zuu"][k][l], sg.zuu[klRel],
                                    tolerance));
          EXPECT_TRUE(IsCloseRelAbs(vac1n_surface["zuv"][k][l], sg.zuv[klRel],
                                    tolerance));
          EXPECT_TRUE(IsCloseRelAbs(vac1n_surface["zvv"][k][l], sg.zvv[klRel],
                                    tolerance));

          EXPECT_TRUE(IsCloseRelAbs(vac1n_surface["auu"][k][l], sg.auu[klRel],
                                    tolerance));
          EXPECT_TRUE(IsCloseRelAbs(vac1n_surface["auv"][k][l], sg.auv[klRel],
                                    tolerance));
          EXPECT_TRUE(IsCloseRelAbs(vac1n_surface["avv"][k][l], sg.avv[klRel],
                                    tolerance));

          EXPECT_TRUE(IsCloseRelAbs(vac1n_surface["drv"][k][l], sg.drv[klRel],
                                    tolerance));
        }  // kl

        // full-surface quantities
        for (int kl = 0; kl < s.nZnT; ++kl) {
          const int l = kl / s.nZeta;
          const int k = kl % s.nZeta;

          EXPECT_TRUE(IsCloseRelAbs(vac1n_surface["rzb2"][k][l], sg.rzb2[kl],
                                    tolerance));

          EXPECT_TRUE(IsCloseRelAbs(vac1n_surface["rcosuv"][k][l],
                                    sg.rcosuv[kl], tolerance));
          EXPECT_TRUE(IsCloseRelAbs(vac1n_surface["rsinuv"][k][l],
                                    sg.rsinuv[kl], tolerance));
        }  // kl

        // non-zymmetry-reduced R and Z are also only available if a full update
        // is being done
        for (int kl = s.nZnT; kl < s.nThetaEven * s.nZeta; ++kl) {
          const int l = kl / s.nZeta;
          const int k = kl % s.nZeta;

          EXPECT_TRUE(
              IsCloseRelAbs(vac1n_surface["r1b"][k][l], sg.r1b[kl], tolerance));
          EXPECT_TRUE(
              IsCloseRelAbs(vac1n_surface["z1b"][k][l], sg.z1b[kl], tolerance));
        }  // kl
      }    // fullUpdate
    }      // thread_id
  }
}  // CheckSurfaceGeometry

INSTANTIATE_TEST_SUITE_P(TestSurfaceGeometry, SurfaceGeometryTest,
                         Values(DataSource{.identifier = "cth_like_free_bdy",
                                           .tolerance = 1.0e-12,
                                           .iter2_to_test = {53, 54}}));

TEST(TestSurfaceGeometry, CheckConstants) {
  const double tolerance = 1.0e-12;

  const SurfaceGeometryMockup& surface_geometry_mockup =
      SurfaceGeometryMockup::InitializeFromFile();

  const Sizes& s = surface_geometry_mockup.s;
  const SurfaceGeometry& sg = surface_geometry_mockup.sg;

  const double omega_phi = 2.0 * M_PI / (s.nfp * s.nZeta);
  for (int k = 0; k < s.nZeta; ++k) {
    const double phi = omega_phi * k;

    EXPECT_TRUE(IsCloseRelAbs(std::cos(phi), sg.cos_phi[k], tolerance))
        << "at k=" << k;
    EXPECT_TRUE(IsCloseRelAbs(std::sin(phi), sg.sin_phi[k], tolerance))
        << "at k=" << k;
  }
}

TEST(TestSurfaceGeometry, CheckInvDFT) {
  const double tolerance = 1.0e-12;

  // tolerance for finite-difference derivative approximations
  const double fdTol = 1.0e-6;

  SurfaceGeometryMockup surface_geometry_mockup =
      SurfaceGeometryMockup::InitializeFromFile();

  const Sizes& s = surface_geometry_mockup.s;
  const SurfaceGeometry& sg = surface_geometry_mockup.sg;
  const FourierBasisFastToroidal& fb = surface_geometry_mockup.fb;

  // reference inv-DFT
  std::vector<double> refR(s.nThetaEven * s.nZeta);  // full surface
  std::vector<double> refZ(s.nThetaEven * s.nZeta);  // full surface

  std::vector<double> refR_tp(s.nZnT);     // theta + eps
  std::vector<double> refR_tm(s.nZnT);     // theta - eps
  std::vector<double> refR_zp(s.nZnT);     //  phi + eps
  std::vector<double> refR_zm(s.nZnT);     //  phi - eps
  std::vector<double> refR_tp_zp(s.nZnT);  // theta + eps, phi + eps
  std::vector<double> refR_tp_zm(s.nZnT);  // theta + eps, phi - eps
  std::vector<double> refR_tm_zp(s.nZnT);  // theta - eps, phi + eps
  std::vector<double> refR_tm_zm(s.nZnT);  // theta - eps, phi - eps
  std::vector<double> refRu(s.nZnT);
  std::vector<double> refRv(s.nZnT);
  std::vector<double> refRuu(s.nZnT);
  std::vector<double> refRuv(s.nZnT);
  std::vector<double> refRvv(s.nZnT);

  std::vector<double> refZ_tp(s.nZnT);     // theta + eps
  std::vector<double> refZ_tm(s.nZnT);     // theta - eps
  std::vector<double> refZ_zp(s.nZnT);     //  phi + eps
  std::vector<double> refZ_zm(s.nZnT);     //  phi - eps
  std::vector<double> refZ_tp_zp(s.nZnT);  // theta + eps, phi + eps
  std::vector<double> refZ_tp_zm(s.nZnT);  // theta + eps, phi - eps
  std::vector<double> refZ_tm_zp(s.nZnT);  // theta - eps, phi + eps
  std::vector<double> refZ_tm_zm(s.nZnT);  // theta - eps, phi - eps
  std::vector<double> refZu(s.nZnT);
  std::vector<double> refZv(s.nZnT);
  std::vector<double> refZuu(s.nZnT);
  std::vector<double> refZuv(s.nZnT);
  std::vector<double> refZvv(s.nZnT);

  // relative finite-difference step
  const double eps = 1.0e-5;

  std::vector<double> theta(s.nThetaEven * s.nZeta);
  std::vector<double> phi(s.nThetaEven * s.nZeta);

  const double omega_theta = 2.0 * M_PI / s.nThetaEven;
  const double omega_phi = 2.0 * M_PI / (s.nfp * s.nZeta);

  for (int l = 0; l < s.nThetaEven; ++l) {
    for (int k = 0; k < s.nZeta; ++k) {
      const int kl = l * s.nZeta + k;

      theta[kl] = omega_theta * l;
      phi[kl] = omega_phi * k;

      refR[kl] = 0.0;
      refZ[kl] = 0.0;
      for (int mn = 0; mn < s.mnmax; ++mn) {
        const double kernel = fb.xm[mn] * theta[kl] - fb.xn[mn] * phi[kl];
        refR[kl] += surface_geometry_mockup.rmnc[mn] * cos(kernel);
        refZ[kl] += surface_geometry_mockup.zmns[mn] * sin(kernel);
        if (s.lasym) {
          refR[kl] += surface_geometry_mockup.rmns[mn] * sin(kernel);
          refZ[kl] += surface_geometry_mockup.zmnc[mn] * cos(kernel);
        }  // lasym
      }    // mn
    }      // k
  }        // l

  for (int l = 0; l < s.nThetaReduced; ++l) {
    for (int k = 0; k < s.nZeta; ++k) {
      const int kl = l * s.nZeta + k;

      theta[kl] = omega_theta * l;
      phi[kl] = omega_phi * k;

      refR_tp[kl] = 0.0;
      refR_tm[kl] = 0.0;
      refR_zp[kl] = 0.0;
      refR_zm[kl] = 0.0;
      refR_tp_zp[kl] = 0.0;
      refR_tp_zm[kl] = 0.0;
      refR_tm_zp[kl] = 0.0;
      refR_tm_zm[kl] = 0.0;
      refRu[kl] = 0.0;
      refRv[kl] = 0.0;
      refRuu[kl] = 0.0;
      refRuv[kl] = 0.0;
      refRvv[kl] = 0.0;

      refZ_tp[kl] = 0.0;
      refZ_tm[kl] = 0.0;
      refZ_zp[kl] = 0.0;
      refZ_zm[kl] = 0.0;
      refZ_tp_zp[kl] = 0.0;
      refZ_tp_zm[kl] = 0.0;
      refZ_tm_zp[kl] = 0.0;
      refZ_tm_zm[kl] = 0.0;
      refZu[kl] = 0.0;
      refZv[kl] = 0.0;
      refZuu[kl] = 0.0;
      refZuv[kl] = 0.0;
      refZvv[kl] = 0.0;

      for (int mn = 0; mn < s.mnmax; ++mn) {
        const double kernel = fb.xm[mn] * theta[kl] - fb.xn[mn] * phi[kl];
        const double kernel_tp =
            fb.xm[mn] * (theta[kl] + eps) - fb.xn[mn] * phi[kl];
        const double kernel_tm =
            fb.xm[mn] * (theta[kl] - eps) - fb.xn[mn] * phi[kl];
        const double kernel_zp =
            fb.xm[mn] * theta[kl] - fb.xn[mn] * (phi[kl] + eps);
        const double kernel_zm =
            fb.xm[mn] * theta[kl] - fb.xn[mn] * (phi[kl] - eps);

        const double kernel_tp_zp =
            fb.xm[mn] * (theta[kl] + eps) - fb.xn[mn] * (phi[kl] + eps);
        const double kernel_tp_zm =
            fb.xm[mn] * (theta[kl] + eps) - fb.xn[mn] * (phi[kl] - eps);
        const double kernel_tm_zp =
            fb.xm[mn] * (theta[kl] - eps) - fb.xn[mn] * (phi[kl] + eps);
        const double kernel_tm_zm =
            fb.xm[mn] * (theta[kl] - eps) - fb.xn[mn] * (phi[kl] - eps);

        refR_tp[kl] += surface_geometry_mockup.rmnc[mn] * cos(kernel_tp);
        refR_tm[kl] += surface_geometry_mockup.rmnc[mn] * cos(kernel_tm);
        refR_zp[kl] += surface_geometry_mockup.rmnc[mn] * cos(kernel_zp);
        refR_zm[kl] += surface_geometry_mockup.rmnc[mn] * cos(kernel_zm);
        refR_tp_zp[kl] += surface_geometry_mockup.rmnc[mn] * cos(kernel_tp_zp);
        refR_tp_zm[kl] += surface_geometry_mockup.rmnc[mn] * cos(kernel_tp_zm);
        refR_tm_zp[kl] += surface_geometry_mockup.rmnc[mn] * cos(kernel_tm_zp);
        refR_tm_zm[kl] += surface_geometry_mockup.rmnc[mn] * cos(kernel_tm_zm);
        refRu[kl] +=
            surface_geometry_mockup.rmnc[mn] * sin(kernel) * (-fb.xm[mn]);
        refRv[kl] += surface_geometry_mockup.rmnc[mn] * sin(kernel) * fb.xn[mn];
        refRuu[kl] += surface_geometry_mockup.rmnc[mn] * cos(kernel) *
                      (-fb.xm[mn] * fb.xm[mn]);
        refRuv[kl] += surface_geometry_mockup.rmnc[mn] * cos(kernel) *
                      fb.xn[mn] * fb.xm[mn];
        refRvv[kl] += surface_geometry_mockup.rmnc[mn] * cos(kernel) *
                      (-fb.xn[mn] * fb.xn[mn]);

        refZ_tp[kl] += surface_geometry_mockup.zmns[mn] * sin(kernel_tp);
        refZ_tm[kl] += surface_geometry_mockup.zmns[mn] * sin(kernel_tm);
        refZ_zp[kl] += surface_geometry_mockup.zmns[mn] * sin(kernel_zp);
        refZ_zm[kl] += surface_geometry_mockup.zmns[mn] * sin(kernel_zm);
        refZ_tp_zp[kl] += surface_geometry_mockup.zmns[mn] * sin(kernel_tp_zp);
        refZ_tp_zm[kl] += surface_geometry_mockup.zmns[mn] * sin(kernel_tp_zm);
        refZ_tm_zp[kl] += surface_geometry_mockup.zmns[mn] * sin(kernel_tm_zp);
        refZ_tm_zm[kl] += surface_geometry_mockup.zmns[mn] * sin(kernel_tm_zm);
        refZu[kl] += surface_geometry_mockup.zmns[mn] * cos(kernel) * fb.xm[mn];
        refZv[kl] +=
            surface_geometry_mockup.zmns[mn] * cos(kernel) * (-fb.xn[mn]);
        refZuu[kl] += surface_geometry_mockup.zmns[mn] * sin(kernel) *
                      (-fb.xm[mn] * fb.xm[mn]);
        refZuv[kl] += surface_geometry_mockup.zmns[mn] * sin(kernel) *
                      fb.xn[mn] * fb.xm[mn];
        refZvv[kl] += surface_geometry_mockup.zmns[mn] * sin(kernel) *
                      (-fb.xn[mn] * fb.xn[mn]);

        if (s.lasym) {
          refR_tp[kl] += surface_geometry_mockup.rmns[mn] * sin(kernel_tp);
          refR_tm[kl] += surface_geometry_mockup.rmns[mn] * sin(kernel_tm);
          refR_zp[kl] += surface_geometry_mockup.rmns[mn] * sin(kernel_zp);
          refR_zm[kl] += surface_geometry_mockup.rmns[mn] * sin(kernel_zm);
          refR_tp_zp[kl] +=
              surface_geometry_mockup.rmns[mn] * sin(kernel_tp_zp);
          refR_tp_zm[kl] +=
              surface_geometry_mockup.rmns[mn] * sin(kernel_tp_zm);
          refR_tm_zp[kl] +=
              surface_geometry_mockup.rmns[mn] * sin(kernel_tm_zp);
          refR_tm_zm[kl] +=
              surface_geometry_mockup.rmns[mn] * sin(kernel_tm_zm);
          refRu[kl] +=
              surface_geometry_mockup.rmns[mn] * cos(kernel) * (-fb.xm[mn]);
          refRv[kl] +=
              surface_geometry_mockup.rmns[mn] * cos(kernel) * fb.xn[mn];
          refRuu[kl] += surface_geometry_mockup.rmns[mn] * sin(kernel) *
                        (-fb.xm[mn] * fb.xm[mn]);
          refRuv[kl] += surface_geometry_mockup.rmns[mn] * sin(kernel) *
                        fb.xn[mn] * fb.xm[mn];
          refRvv[kl] += surface_geometry_mockup.rmns[mn] * sin(kernel) *
                        (-fb.xn[mn] * fb.xn[mn]);

          refZ_tp[kl] += surface_geometry_mockup.zmnc[mn] * cos(kernel_tp);
          refZ_tm[kl] += surface_geometry_mockup.zmnc[mn] * cos(kernel_tm);
          refZ_zp[kl] += surface_geometry_mockup.zmnc[mn] * cos(kernel_zp);
          refZ_zm[kl] += surface_geometry_mockup.zmnc[mn] * cos(kernel_zm);
          refZ_tp_zp[kl] +=
              surface_geometry_mockup.zmnc[mn] * cos(kernel_tp_zp);
          refZ_tp_zm[kl] +=
              surface_geometry_mockup.zmnc[mn] * cos(kernel_tp_zm);
          refZ_tm_zp[kl] +=
              surface_geometry_mockup.zmnc[mn] * cos(kernel_tm_zp);
          refZ_tm_zm[kl] +=
              surface_geometry_mockup.zmnc[mn] * cos(kernel_tm_zm);
          refZu[kl] +=
              surface_geometry_mockup.zmnc[mn] * sin(kernel) * fb.xm[mn];
          refZv[kl] +=
              surface_geometry_mockup.zmnc[mn] * sin(kernel) * (-fb.xn[mn]);
          refZuu[kl] += surface_geometry_mockup.zmnc[mn] * cos(kernel) *
                        (-fb.xm[mn] * fb.xm[mn]);
          refZuv[kl] += surface_geometry_mockup.zmnc[mn] * cos(kernel) *
                        fb.xn[mn] * fb.xm[mn];
          refZvv[kl] += surface_geometry_mockup.zmnc[mn] * cos(kernel) *
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
      EXPECT_TRUE(IsCloseRelAbs(refR[kl], sg.r1b[kl], tolerance))
          << "at k=" << k << " l=" << l;

      // Z
      EXPECT_TRUE(IsCloseRelAbs(refZ[kl], sg.z1b[kl], tolerance))
          << "at k=" << k << " l=" << l;
    }  // k
  }    // l

  // b) finite-differences derivatives vs. analytical derivatives from inv-DFT:
  // should be roughly the same
  for (int l = 0; l < s.nThetaReduced; ++l) {
    for (int k = 0; k < s.nZeta; ++k) {
      const int kl = l * s.nZeta + k;

      // dR/dTheta
      double ru_approx = (refR_tp[kl] - refR_tm[kl]) / (2.0 * eps);
      EXPECT_TRUE(IsCloseRelAbs(ru_approx, sg.rub[kl], fdTol))
          << "at k=" << k << " l=" << l;

      // dR/dZeta
      double rv_approx = (refR_zp[kl] - refR_zm[kl]) / (2.0 * eps);
      EXPECT_TRUE(IsCloseRelAbs(rv_approx, sg.rvb[kl], fdTol))
          << "at k=" << k << " l=" << l;

      // ----------------

      // dZ/dTheta
      double zu_approx = (refZ_tp[kl] - refZ_tm[kl]) / (2.0 * eps);
      EXPECT_TRUE(IsCloseRelAbs(zu_approx, sg.zub[kl], fdTol))
          << "at k=" << k << " l=" << l;

      // dZ/dZeta
      double zv_approx = (refZ_zp[kl] - refZ_zm[kl]) / (2.0 * eps);
      EXPECT_TRUE(IsCloseRelAbs(zv_approx, sg.zvb[kl], fdTol))
          << "at k=" << k << " l=" << l;

      // --------------

      // d^2R/dTheta^2
      double ruu_approx =
          (refR_tp[kl] - 2.0 * refR[kl] + refR_tm[kl]) / (eps * eps);
      EXPECT_TRUE(IsCloseRelAbs(ruu_approx, sg.ruu[kl], sqrt(fdTol)))
          << "at k=" << k << " l=" << l;

      // d^2R/(dTheta dZeta)
      double ruv_approx =
          (refR_tp_zp[kl] + refR_tm_zm[kl] - refR_tp_zm[kl] - refR_tm_zp[kl]) /
          (4.0 * eps * eps);
      EXPECT_TRUE(IsCloseRelAbs(ruv_approx, sg.ruv[kl], sqrt(fdTol)))
          << "at k=" << k << " l=" << l;

      // d^2R/dZeta^2
      double rvv_approx =
          (refR_zp[kl] - 2.0 * refR[kl] + refR_zm[kl]) / (eps * eps);
      EXPECT_TRUE(IsCloseRelAbs(rvv_approx, sg.rvv[kl], sqrt(fdTol)))
          << "at k=" << k << " l=" << l;

      // ----------------

      // d^2Z/dTheta^2
      double zuu_approx =
          (refZ_tp[kl] - 2.0 * refZ[kl] + refZ_tm[kl]) / (eps * eps);
      EXPECT_TRUE(IsCloseRelAbs(zuu_approx, sg.zuu[kl], sqrt(fdTol)))
          << "at k=" << k << " l=" << l;

      // d^2Z/(dTheta dZeta)
      double zuv_approx =
          (refZ_tp_zp[kl] + refZ_tm_zm[kl] - refZ_tp_zm[kl] - refZ_tm_zp[kl]) /
          (4.0 * eps * eps);
      EXPECT_TRUE(IsCloseRelAbs(zuv_approx, sg.zuv[kl], sqrt(fdTol)))
          << "at k=" << k << " l=" << l;

      // d^2Z/dZeta^2
      double zvv_approx =
          (refZ_zp[kl] - 2.0 * refZ[kl] + refZ_zm[kl]) / (eps * eps);
      EXPECT_TRUE(IsCloseRelAbs(zvv_approx, sg.zvv[kl], sqrt(fdTol)))
          << "at k=" << k << " l=" << l;
    }  // k
  }    // l

  // c) inv-DFT vs. reference inv-DFT: should be exactly the same
  for (int l = 0; l < s.nThetaReduced; ++l) {
    for (int k = 0; k < s.nZeta; ++k) {
      const int kl = l * s.nZeta + k;

      // dR/dTheta
      EXPECT_TRUE(IsCloseRelAbs(refRu[kl], sg.rub[kl], tolerance))
          << "at k=" << k << " l=" << l;

      // dR/dZeta
      EXPECT_TRUE(IsCloseRelAbs(refRv[kl], sg.rvb[kl], tolerance))
          << "at k=" << k << " l=" << l;

      // -----------------

      // dZ/dTheta
      EXPECT_TRUE(IsCloseRelAbs(refZu[kl], sg.zub[kl], tolerance))
          << "at k=" << k << " l=" << l;

      // dZ/dZeta
      EXPECT_TRUE(IsCloseRelAbs(refZv[kl], sg.zvb[kl], tolerance))
          << "at k=" << k << " l=" << l;

      // -----------------

      // d^2R/dTheta^2
      EXPECT_TRUE(IsCloseRelAbs(refRuu[kl], sg.ruu[kl], tolerance))
          << "at k=" << k << " l=" << l;

      // d^2R/(dTheta dZeta)
      EXPECT_TRUE(IsCloseRelAbs(refRuv[kl], sg.ruv[kl], tolerance))
          << "at k=" << k << " l=" << l;

      // d^2R/dZeta^2
      EXPECT_TRUE(IsCloseRelAbs(refRvv[kl], sg.rvv[kl], tolerance))
          << "at k=" << k << " l=" << l;

      // -----------------

      // d^2Z/dTheta^2
      EXPECT_TRUE(IsCloseRelAbs(refZuu[kl], sg.zuu[kl], tolerance))
          << "at k=" << k << " l=" << l;

      // d^2Z/(dTheta dZeta)
      EXPECT_TRUE(IsCloseRelAbs(refZuv[kl], sg.zuv[kl], tolerance))
          << "at k=" << k << " l=" << l;

      // d^2Z/dZeta^2
      EXPECT_TRUE(IsCloseRelAbs(refZvv[kl], sg.zvv[kl], tolerance))
          << "at k=" << k << " l=" << l;
    }  // k
  }    // l
}

TEST(TestSurfaceGeometry, CheckDerivedQuantities) {
  double tolerance = 1.0e-12;

  SurfaceGeometryMockup surface_geometry_mockup =
      SurfaceGeometryMockup::InitializeFromFile();

  const Sizes& s = surface_geometry_mockup.s;
  const FourierBasisFastToroidal& fb = surface_geometry_mockup.fb;
  const SurfaceGeometry& sg = surface_geometry_mockup.sg;

  // form vector-values quantities as vectors and perform actual dot-products
  // for verification

  // reference inv-DFT
  std::vector<double> refR(s.nThetaEven * s.nZeta);
  std::vector<double> refZ(s.nThetaEven * s.nZeta);

  std::vector<double> refRu(s.nZnT);
  std::vector<double> refRv(s.nZnT);
  std::vector<double> refRuu(s.nZnT);
  std::vector<double> refRuv(s.nZnT);
  std::vector<double> refRvv(s.nZnT);

  std::vector<double> refZu(s.nZnT);
  std::vector<double> refZv(s.nZnT);
  std::vector<double> refZuu(s.nZnT);
  std::vector<double> refZuv(s.nZnT);
  std::vector<double> refZvv(s.nZnT);

  std::vector<double> theta(s.nThetaEven * s.nZeta);
  std::vector<double> phi(s.nThetaEven * s.nZeta);

  const double omega_theta = 2.0 * M_PI / s.nThetaEven;
  const double omega_phi = 2.0 * M_PI / (s.nfp * s.nZeta);

  for (int l = 0; l < s.nThetaEven; ++l) {
    for (int k = 0; k < s.nZeta; ++k) {
      const int kl = l * s.nZeta + k;

      theta[kl] = omega_theta * l;
      phi[kl] = omega_phi * k;

      refR[kl] = 0.0;
      refZ[kl] = 0.0;
      for (int mn = 0; mn < s.mnmax; ++mn) {
        const double kernel = fb.xm[mn] * theta[kl] - fb.xn[mn] * phi[kl];
        refR[kl] += surface_geometry_mockup.rmnc[mn] * cos(kernel);
        refZ[kl] += surface_geometry_mockup.zmns[mn] * sin(kernel);
        if (s.lasym) {
          refR[kl] += surface_geometry_mockup.rmns[mn] * sin(kernel);
          refZ[kl] += surface_geometry_mockup.zmnc[mn] * cos(kernel);
        }  // lasym
      }    // mn
    }      // k
  }        // l

  for (int l = 0; l < s.nThetaReduced; ++l) {
    for (int k = 0; k < s.nZeta; ++k) {
      const int kl = l * s.nZeta + k;

      refRu[kl] = 0.0;
      refRv[kl] = 0.0;
      refRuu[kl] = 0.0;
      refRuv[kl] = 0.0;
      refRvv[kl] = 0.0;

      refZu[kl] = 0.0;
      refZv[kl] = 0.0;
      refZuu[kl] = 0.0;
      refZuv[kl] = 0.0;
      refZvv[kl] = 0.0;

      for (int mn = 0; mn < s.mnmax; ++mn) {
        const double kernel = fb.xm[mn] * theta[kl] - fb.xn[mn] * phi[kl];

        refRu[kl] +=
            surface_geometry_mockup.rmnc[mn] * sin(kernel) * (-fb.xm[mn]);
        refRv[kl] += surface_geometry_mockup.rmnc[mn] * sin(kernel) * fb.xn[mn];
        refRuu[kl] += surface_geometry_mockup.rmnc[mn] * cos(kernel) *
                      (-fb.xm[mn] * fb.xm[mn]);
        refRuv[kl] += surface_geometry_mockup.rmnc[mn] * cos(kernel) *
                      fb.xn[mn] * fb.xm[mn];
        refRvv[kl] += surface_geometry_mockup.rmnc[mn] * cos(kernel) *
                      (-fb.xn[mn] * fb.xn[mn]);

        refZu[kl] += surface_geometry_mockup.zmns[mn] * cos(kernel) * fb.xm[mn];
        refZv[kl] +=
            surface_geometry_mockup.zmns[mn] * cos(kernel) * (-fb.xn[mn]);
        refZuu[kl] += surface_geometry_mockup.zmns[mn] * sin(kernel) *
                      (-fb.xm[mn] * fb.xm[mn]);
        refZuv[kl] += surface_geometry_mockup.zmns[mn] * sin(kernel) *
                      fb.xn[mn] * fb.xm[mn];
        refZvv[kl] += surface_geometry_mockup.zmns[mn] * sin(kernel) *
                      (-fb.xn[mn] * fb.xn[mn]);

        if (s.lasym) {
          refRu[kl] +=
              surface_geometry_mockup.rmns[mn] * cos(kernel) * (-fb.xm[mn]);
          refRv[kl] +=
              surface_geometry_mockup.rmns[mn] * cos(kernel) * fb.xn[mn];
          refRuu[kl] += surface_geometry_mockup.rmns[mn] * sin(kernel) *
                        (-fb.xm[mn] * fb.xm[mn]);
          refRuv[kl] += surface_geometry_mockup.rmns[mn] * sin(kernel) *
                        fb.xn[mn] * fb.xm[mn];
          refRvv[kl] += surface_geometry_mockup.rmns[mn] * sin(kernel) *
                        (-fb.xn[mn] * fb.xn[mn]);

          refZu[kl] +=
              surface_geometry_mockup.zmnc[mn] * sin(kernel) * fb.xm[mn];
          refZv[kl] +=
              surface_geometry_mockup.zmnc[mn] * sin(kernel) * (-fb.xn[mn]);
          refZuu[kl] += surface_geometry_mockup.zmnc[mn] * cos(kernel) *
                        (-fb.xm[mn] * fb.xm[mn]);
          refZuv[kl] += surface_geometry_mockup.zmnc[mn] * cos(kernel) *
                        fb.xn[mn] * fb.xm[mn];
          refZvv[kl] += surface_geometry_mockup.zmnc[mn] * cos(kernel) *
                        (-fb.xn[mn] * fb.xn[mn]);
        }  // lasym
      }    // mn
    }      // k
  }        // l

  // guu == g_{theta,theta}
  for (int l = 0; l < s.nThetaReduced; ++l) {
    for (int k = 0; k < s.nZeta; ++k) {
      const int kl = l * s.nZeta + k;

      const std::array<double, 3> dXdTheta = {
          refRu[kl] * sg.cos_phi[k], refRu[kl] * sg.sin_phi[k], refZu[kl]};

      const double ref_guu = dot3x3(dXdTheta, dXdTheta);

      EXPECT_TRUE(IsCloseRelAbs(ref_guu, sg.guu[kl], tolerance))
          << "at k=" << k << " l=" << l;
    }  // k
  }    // l

  // 2 guv == 2 g_{theta,zeta}
  for (int l = 0; l < s.nThetaReduced; ++l) {
    for (int k = 0; k < s.nZeta; ++k) {
      const int kl = l * s.nZeta + k;

      const std::array<double, 3> dXdTheta = {
          refRu[kl] * sg.cos_phi[k], refRu[kl] * sg.sin_phi[k], refZu[kl]};

      const std::array<double, 3> dXdPhi = {
          refRv[kl] * sg.cos_phi[k] - refR[kl] * sg.sin_phi[k],
          refRv[kl] * sg.sin_phi[k] + refR[kl] * sg.cos_phi[k], refZv[kl]};

      const double ref_2guv = 2.0 * dot3x3(dXdTheta, dXdPhi) / s.nfp;

      EXPECT_TRUE(IsCloseRelAbs(ref_2guv, sg.guv[kl], tolerance))
          << "at k=" << k << " l=" << l;
    }  // k
  }    // l

  // gvv == g_{zeta,zeta}
  for (int l = 0; l < s.nThetaReduced; ++l) {
    for (int k = 0; k < s.nZeta; ++k) {
      const int kl = l * s.nZeta + k;

      const std::array<double, 3> dXdPhi = {
          refRv[kl] * sg.cos_phi[k] - refR[kl] * sg.sin_phi[k],
          refRv[kl] * sg.sin_phi[k] + refR[kl] * sg.cos_phi[k], refZv[kl]};

      const double ref_gvv = dot3x3(dXdPhi, dXdPhi) / (s.nfp * s.nfp);

      EXPECT_TRUE(IsCloseRelAbs(ref_gvv, sg.gvv[kl], tolerance))
          << "at k=" << k << " l=" << l;
    }  // k
  }    // l

  // components of normal vector N (snr, snv, snz) and drv
  for (int l = 0; l < s.nThetaReduced; ++l) {
    for (int k = 0; k < s.nZeta; ++k) {
      const int kl = l * s.nZeta + k;

      // need to compute cross product in cylindrical coordinates
      // to get cylindrical components of normal vector
      const std::array<double, 3> dXdTheta_cyl = {refRu[kl], 0.0, refZu[kl]};

      const std::array<double, 3> dXdPhi_cyl = {refRv[kl], refR[kl], refZv[kl]};

      std::array<double, 3> N_cyl = {0.0, 0.0, 0.0};

      // -(X_u cross X_v) == (X_v cross X_u)
      cross3x3(/*m_result=*/N_cyl, dXdPhi_cyl, dXdTheta_cyl);

      // include sign of Jacobian
      N_cyl[0] *= surface_geometry_mockup.signOfJacobian;
      N_cyl[1] *= surface_geometry_mockup.signOfJacobian;
      N_cyl[2] *= surface_geometry_mockup.signOfJacobian;

      const double ref_drv = -(refR[kl] * N_cyl[0] + refZ[kl] * N_cyl[2]);

      EXPECT_TRUE(IsCloseRelAbs(N_cyl[0], sg.snr[kl], tolerance))
          << "at k=" << k << " l=" << l;
      EXPECT_TRUE(IsCloseRelAbs(N_cyl[1], sg.snv[kl], tolerance))
          << "at k=" << k << " l=" << l;
      EXPECT_TRUE(IsCloseRelAbs(N_cyl[2], sg.snz[kl], tolerance))
          << "at k=" << k << " l=" << l;

      EXPECT_TRUE(IsCloseRelAbs(ref_drv, sg.drv[kl], tolerance))
          << "at k=" << k << " l=" << l;
    }  // k
  }    // l

  // A, B, C (==auu, auv, avv)
  for (int l = 0; l < s.nThetaReduced; ++l) {
    for (int k = 0; k < s.nZeta; ++k) {
      const int kl = l * s.nZeta + k;

      const std::array<double, 3> d2XdTheta2 = {refRuu[kl], 0.0, refZuu[kl]};

      const std::array<double, 3> d2XdThetaDZeta = {refRuv[kl], refRu[kl],
                                                    refZuv[kl]};

      const std::array<double, 3> d2XdZeta2 = {refRvv[kl] - refR[kl],
                                               2 * refRv[kl], refZvv[kl]};

      // already tested above
      const std::array<double, 3> N_cyl = {sg.snr[kl], sg.snv[kl], sg.snz[kl]};

      const double ref_auu = dot3x3(d2XdTheta2, N_cyl) / 2;
      const double ref_auv = dot3x3(d2XdThetaDZeta, N_cyl) / s.nfp;
      const double ref_avv = dot3x3(d2XdZeta2, N_cyl) / (2 * s.nfp * s.nfp);

      EXPECT_TRUE(IsCloseRelAbs(ref_auu, sg.auu[kl], tolerance))
          << "at k=" << k << " l=" << l;
      EXPECT_TRUE(IsCloseRelAbs(ref_auv, sg.auv[kl], tolerance))
          << "at k=" << k << " l=" << l;
      EXPECT_TRUE(IsCloseRelAbs(ref_avv, sg.avv[kl], tolerance))
          << "at k=" << k << " l=" << l;
    }  // k
  }    // l

  // rzb2, rcosuv, rsinuv: full surface
  for (int l = 0; l < s.nThetaEven; ++l) {
    for (int k = 0; k < s.nZeta; ++k) {
      const int kl = l * s.nZeta + k;

      const double ref_rzb2 = refR[kl] * refR[kl] + refZ[kl] * refZ[kl];

      const double ref_x = refR[kl] * sg.cos_phi[k];
      const double ref_y = refR[kl] * sg.sin_phi[k];

      EXPECT_TRUE(IsCloseRelAbs(ref_rzb2, sg.rzb2[kl], tolerance))
          << "at k=" << k << " l=" << l;

      EXPECT_TRUE(IsCloseRelAbs(ref_x, sg.rcosuv[kl], tolerance))
          << "at k=" << k << " l=" << l;
      EXPECT_TRUE(IsCloseRelAbs(ref_y, sg.rsinuv[kl], tolerance))
          << "at k=" << k << " l=" << l;
    }  // k
  }    // l
}

}  // namespace vmecpp
