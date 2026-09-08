// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
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
        "vmecpp_large_cpp_tests/test_data/%s/vac1n_surface/"
        "vac1n_surface_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, number_of_iterations,
        vmec.get_num_eqsolve_retries(), data_source_.identifier);

    std::ifstream ifs_vac1n_surface(filename);
    ASSERT_TRUE(ifs_vac1n_surface.is_open());
    json vac1n_surface = json::parse(ifs_vac1n_surface);

    for (int thread_id = 0; thread_id < vmec.vac_num_threads_; ++thread_id) {
      const Nestor& n = static_cast<const Nestor&>(*vmec.fb_vac_[thread_id]);
      const TangentialPartitioning& tp = *vmec.tp_vac_[thread_id];
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
      }  // fullUpdate
    }  // thread_id
  }
}  // CheckSurfaceGeometry

INSTANTIATE_TEST_SUITE_P(TestSurfaceGeometry, SurfaceGeometryTest,
                         Values(DataSource{.identifier = "cth_like_free_bdy",
                                           .tolerance = 1.0e-12,
                                           .iter2_to_test = {53, 54}}));

}  // namespace vmecpp
