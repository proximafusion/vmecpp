// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/output_quantities/output_quantities.h"

#include <netcdf.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "absl/strings/str_format.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "util/file_io/file_io.h"
#include "util/netcdf_io/netcdf_io.h"
#include "util/testing/numerical_comparison_lib.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/vmec/vmec/vmec.h"

using nlohmann::json;

using file_io::ReadFile;
using netcdf_io::NetcdfReadArray1D;
using netcdf_io::NetcdfReadArray2D;
using netcdf_io::NetcdfReadBool;
using netcdf_io::NetcdfReadChar;
using netcdf_io::NetcdfReadDouble;
using netcdf_io::NetcdfReadInt;
using netcdf_io::NetcdfReadString;
using testing::IsCloseRelAbs;

using ::testing::ElementsAreArray;
using ::testing::TestWithParam;
using ::testing::Values;

namespace fs = std::filesystem;

namespace vmecpp {

// used to specify case-specific tolerances
struct DataSource {
  std::string identifier;
  double tolerance = 0.0;
};

class GatherDataFromThreadsTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(GatherDataFromThreadsTest, CheckGatherDataFromThreads) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  Vmec vmec(*vmec_indata);
  const Sizes& s = vmec.s_;
  const FlowControl& fc = vmec.fc_;

  // run until convergence
  bool reached_checkpoint = vmec.run(VmecCheckpoint::BCOVAR_FILEOUT).value();
  ASSERT_FALSE(reached_checkpoint);

  filename = absl::StrFormat(
      "vmecpp_large_cpp_tests/test_data/%s/bcovar_fileout/"
      "bcovar_fileout_%05d_000000_01.%s.json",
      data_source_.identifier, fc.ns, data_source_.identifier);

  std::ifstream ifs_bcovar_fileout(filename);
  ASSERT_TRUE(ifs_bcovar_fileout.is_open())
      << "failed to open reference file: " << filename;
  json bcovar_fileout = json::parse(ifs_bcovar_fileout);

  const OutputQuantities& output_quantities = vmec.output_quantities_;

  for (int jH = 0; jH < fc.ns - 1; ++jH) {
    for (int k = 0; k < s.nZeta; ++k) {
      for (int l = 0; l < s.nThetaEff; ++l) {
        const int idx_kl = (jH * s.nZeta + k) * s.nThetaEff + l;

        EXPECT_TRUE(IsCloseRelAbs(
            bcovar_fileout["lv_e"][jH + 1][k][l],
            output_quantities.vmec_internal_results.bsupu(idx_kl), tolerance));
        EXPECT_TRUE(IsCloseRelAbs(
            bcovar_fileout["lu_e"][jH + 1][k][l],
            output_quantities.vmec_internal_results.bsupv(idx_kl), tolerance));

        EXPECT_TRUE(IsCloseRelAbs(
            bcovar_fileout["bsubu_e"][jH + 1][k][l],
            output_quantities.vmec_internal_results.bsubu(idx_kl), tolerance));
        EXPECT_TRUE(IsCloseRelAbs(
            bcovar_fileout["bsubv_e"][jH + 1][k][l],
            output_quantities.vmec_internal_results.bsubv(idx_kl), tolerance));
      }  // l
    }  // k

    EXPECT_TRUE(IsCloseRelAbs(bcovar_fileout["fpsi"][jH],
                              output_quantities.vmec_internal_results.bvcoH[jH],
                              tolerance));
  }  // jH
}  // CheckGatherDataFromThreads

// Same as the test above, but we check the matrix elements in bsupu with 2D
// index access instead of linear access. This can go wrong (even if all other
// tests pass) in case output_quantities.cc mixes up the storage order of the
// Eigen matrix (ask me how I know).
TEST_P(GatherDataFromThreadsTest, CheckMatrixElementOrder) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  Vmec vmec(*vmec_indata);
  const Sizes& s = vmec.s_;
  const FlowControl& fc = vmec.fc_;

  // run until convergence
  bool reached_checkpoint = vmec.run(VmecCheckpoint::BCOVAR_FILEOUT).value();
  ASSERT_FALSE(reached_checkpoint);

  filename = absl::StrFormat(
      "vmecpp_large_cpp_tests/test_data/%s/bcovar_fileout/"
      "bcovar_fileout_%05d_000000_01.%s.json",
      data_source_.identifier, fc.ns, data_source_.identifier);

  std::ifstream ifs_bcovar_fileout(filename);
  ASSERT_TRUE(ifs_bcovar_fileout.is_open())
      << "failed to open reference file: " << filename;
  json bcovar_fileout = json::parse(ifs_bcovar_fileout);

  const OutputQuantities& output_quantities = vmec.output_quantities_;

  for (int jH = 0; jH < fc.ns - 1; ++jH) {
    for (int k = 0; k < s.nZeta; ++k) {
      for (int l = 0; l < s.nThetaEff; ++l) {
        const int kl = k * s.nThetaEff + l;

        EXPECT_TRUE(IsCloseRelAbs(
            bcovar_fileout["lv_e"][jH + 1][k][l],
            output_quantities.vmec_internal_results.bsupu(jH, kl), tolerance));
      }  // l
    }  // k
  }  // jH
}  // CheckMatrixElementOrder

INSTANTIATE_TEST_SUITE_P(
    TestOutputQuantities, GatherDataFromThreadsTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 1.0e-12},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 1.0e-12},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 1.0e-12},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 1.0e-12},
           DataSource{.identifier = "cma", .tolerance = 5.0e-11},
           DataSource{.identifier = "cth_like_free_bdy",
                      .tolerance = 5.0e-11}));

class BSSRoutineOutputsTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(BSSRoutineOutputsTest, CheckBSSRoutineOutputs) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  Vmec vmec(*vmec_indata);
  const Sizes& s = vmec.s_;
  const FlowControl& fc = vmec.fc_;

  // run until convergence
  bool reached_checkpoint = vmec.run(VmecCheckpoint::BSS).value();
  ASSERT_FALSE(reached_checkpoint);

  filename = absl::StrFormat(
      "vmecpp_large_cpp_tests/test_data/%s/bss/bss_%05d_000000_01.%s.json",
      data_source_.identifier, fc.ns, data_source_.identifier);

  std::ifstream ifs_bss(filename);
  ASSERT_TRUE(ifs_bss.is_open())
      << "failed to open reference file: " << filename;
  json bss = json::parse(ifs_bss);

  const OutputQuantities& output_quantities = vmec.output_quantities_;

  for (int jH = 0; jH < fc.ns - 1; ++jH) {
    for (int k = 0; k < s.nZeta; ++k) {
      for (int l = 0; l < s.nThetaEff; ++l) {
        const int idx_kl = (jH * s.nZeta + k) * s.nThetaEff + l;

        EXPECT_TRUE(IsCloseRelAbs(
            bss["rv12"][jH + 1][k][l],
            output_quantities.remaining_metric.rv12(idx_kl), tolerance));
        EXPECT_TRUE(IsCloseRelAbs(
            bss["zv12"][jH + 1][k][l],
            output_quantities.remaining_metric.zv12(idx_kl), tolerance));

        EXPECT_TRUE(IsCloseRelAbs(
            bss["rs12"][jH + 1][k][l],
            output_quantities.remaining_metric.rs12(idx_kl), tolerance));
        EXPECT_TRUE(IsCloseRelAbs(
            bss["zs12"][jH + 1][k][l],
            output_quantities.remaining_metric.zs12(idx_kl), tolerance));

        EXPECT_TRUE(IsCloseRelAbs(
            bss["gsu"][jH + 1][k][l],
            output_quantities.remaining_metric.gsu(idx_kl), tolerance));
        EXPECT_TRUE(IsCloseRelAbs(
            bss["gsv"][jH + 1][k][l],
            output_quantities.remaining_metric.gsv(idx_kl), tolerance));

        EXPECT_TRUE(IsCloseRelAbs(
            bss["bsubs"][jH + 1][k][l],
            output_quantities.bsubs_half.bsubs_half(idx_kl), tolerance));

        EXPECT_TRUE(IsCloseRelAbs(bss["br"][jH + 1][k][l],
                                  output_quantities.b_cylindrical.b_r(idx_kl),
                                  tolerance));
        EXPECT_TRUE(IsCloseRelAbs(bss["bphi"][jH + 1][k][l],
                                  output_quantities.b_cylindrical.b_phi(idx_kl),
                                  tolerance));
        EXPECT_TRUE(IsCloseRelAbs(bss["bz"][jH + 1][k][l],
                                  output_quantities.b_cylindrical.b_z(idx_kl),
                                  tolerance));
      }  // l
    }  // k
  }  // jH
}  // CheckBSSRoutineOutputs

INSTANTIATE_TEST_SUITE_P(
    TestOutputQuantities, BSSRoutineOutputsTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 5.0e-11},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 5.0e-11},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 5.0e-13},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 5.0e-13},
           DataSource{.identifier = "cma", .tolerance = 5.0e-11},
           DataSource{.identifier = "cth_like_free_bdy",
                      .tolerance = 2.0e-11}));

class LowpassFilterBSubsSTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(LowpassFilterBSubsSTest, CheckLowpassFilterBSubsS) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  Vmec vmec(*vmec_indata);
  const Sizes& s = vmec.s_;
  const FlowControl& fc = vmec.fc_;

  // run until convergence
  bool reached_checkpoint =
      vmec.run(VmecCheckpoint::LOWPASS_BCOVARIANT).value();
  ASSERT_FALSE(reached_checkpoint);

  filename = absl::StrFormat(
      "vmecpp_large_cpp_tests/test_data/%s/jxbforce_bsub_lowpass/"
      "jxbforce_bsub_lowpass_%05d_000000_01.%s.json",
      data_source_.identifier, fc.ns, data_source_.identifier);

  std::ifstream ifs_jxbforce_bsub_lowpass(filename);
  ASSERT_TRUE(ifs_jxbforce_bsub_lowpass.is_open())
      << "failed to open reference file: " << filename;
  json jxbforce_bsub_lowpass = json::parse(ifs_jxbforce_bsub_lowpass);

  const OutputQuantities& output_quantities = vmec.output_quantities_;

  for (int jH = 0; jH < fc.ns - 1; ++jH) {
    for (int k = 0; k < s.nZeta; ++k) {
      for (int l = 0; l < s.nThetaEff; ++l) {
        const int idx_kl = (jH * s.nZeta + k) * s.nThetaEff + l;

        EXPECT_TRUE(IsCloseRelAbs(
            jxbforce_bsub_lowpass["bsubu_e"][jH + 1][k][l],
            output_quantities.vmec_internal_results.bsubu(idx_kl), tolerance));
        EXPECT_TRUE(IsCloseRelAbs(
            jxbforce_bsub_lowpass["bsubv_e"][jH + 1][k][l],
            output_quantities.vmec_internal_results.bsubv(idx_kl), tolerance));

        EXPECT_TRUE(IsCloseRelAbs(
            jxbforce_bsub_lowpass["bsubuv"][jH + 1][k][l],
            output_quantities.covariant_b_derivatives.bsubuv(idx_kl),
            tolerance));
        EXPECT_TRUE(IsCloseRelAbs(
            jxbforce_bsub_lowpass["bsubvu"][jH + 1][k][l],
            output_quantities.covariant_b_derivatives.bsubvu(idx_kl),
            tolerance));
      }  // l
    }  // k
  }  // jH

  // bsubs was only available on the iterior full-grid points during
  // filtering/derivative computation
  for (int jF = 1; jF < fc.ns - 1; ++jF) {
    for (int k = 0; k < s.nZeta; ++k) {
      for (int l = 0; l < s.nThetaEff; ++l) {
        const int idx_kl = (jF * s.nZeta + k) * s.nThetaEff + l;

        EXPECT_TRUE(IsCloseRelAbs(
            jxbforce_bsub_lowpass["bsubsu_e"][jF][k][l],
            output_quantities.covariant_b_derivatives.bsubsu(idx_kl),
            tolerance));
        EXPECT_TRUE(IsCloseRelAbs(
            jxbforce_bsub_lowpass["bsubsv_e"][jF][k][l],
            output_quantities.covariant_b_derivatives.bsubsv(idx_kl),
            tolerance));
      }  // l
    }  // k
  }  // jH
}  // CheckLowpassFilterBSubsS

INSTANTIATE_TEST_SUITE_P(
    TestOutputQuantities, LowpassFilterBSubsSTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 1.0e-12},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 1.0e-12},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 1.0e-12},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 1.0e-12},
           DataSource{.identifier = "cma", .tolerance = 1.0e-10},
           DataSource{.identifier = "cth_like_free_bdy",
                      .tolerance = 2.0e-11}));

class ExtrapolateBSubsSTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(ExtrapolateBSubsSTest, CheckExtrapolateBSubsS) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  Vmec vmec(*vmec_indata);
  const Sizes& s = vmec.s_;
  const FlowControl& fc = vmec.fc_;

  // run until convergence
  bool reached_checkpoint = vmec.run(VmecCheckpoint::EXTRAPOLATE_BSUBS).value();
  ASSERT_FALSE(reached_checkpoint);

  filename = absl::StrFormat(
      "vmecpp_large_cpp_tests/test_data/%s/jxbout/"
      "jxbout_%05d_000000_01.%s.json",
      data_source_.identifier, fc.ns, data_source_.identifier);

  std::ifstream ifs_jxbout(filename);
  ASSERT_TRUE(ifs_jxbout.is_open())
      << "failed to open reference file: " << filename;
  json jxbout = json::parse(ifs_jxbout);

  const OutputQuantities& output_quantities = vmec.output_quantities_;

  for (int jF = 1; jF < fc.ns - 1; ++jF) {
    for (int k = 0; k < s.nZeta; ++k) {
      for (int l = 0; l < s.nThetaEff; ++l) {
        const int idx_kl = (jF * s.nZeta + k) * s.nThetaEff + l;

        EXPECT_TRUE(IsCloseRelAbs(
            jxbout["bsubs3"][jF][k][l],
            output_quantities.bsubs_full.bsubs_full(idx_kl), tolerance));
      }  // l
    }  // k
  }  // jH
}  // CheckExtrapolateBSubsS

INSTANTIATE_TEST_SUITE_P(
    TestOutputQuantities, ExtrapolateBSubsSTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 1.0e-12},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 1.0e-12},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 2.0e-14},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 5.0e-15},
           DataSource{.identifier = "cma", .tolerance = 1.0e-11},
           DataSource{.identifier = "cth_like_free_bdy",
                      .tolerance = 5.0e-12}));

class JxBOutputContentsTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(JxBOutputContentsTest, CheckJxBOutputContents) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  Vmec vmec(*vmec_indata);
  const Sizes& s = vmec.s_;
  const FlowControl& fc = vmec.fc_;

  // run until convergence
  bool reached_checkpoint = vmec.run(VmecCheckpoint::JXBOUT).value();
  ASSERT_FALSE(reached_checkpoint);

  filename = absl::StrFormat(
      "vmecpp_large_cpp_tests/test_data/%s/jxbout/"
      "jxbout_%05d_000000_01.%s.json",
      data_source_.identifier, fc.ns, data_source_.identifier);

  std::ifstream ifs_jxbout(filename);
  ASSERT_TRUE(ifs_jxbout.is_open())
      << "failed to open reference file: " << filename;
  json jxbout = json::parse(ifs_jxbout);

  const OutputQuantities& output_quantities = vmec.output_quantities_;

  for (int jF = 0; jF < fc.ns; ++jF) {
    for (int k = 0; k < s.nZeta; ++k) {
      for (int l = 0; l < s.nThetaEff; ++l) {
        const int idx_kl = (jF * s.nZeta + k) * s.nThetaEff + l;

        // NOTE: catastrophic cancellation leads to a bad mismatch in itheta and
        // izeta
        EXPECT_TRUE(IsCloseRelAbs(jxbout["itheta"][jF][k][l],
                                  output_quantities.jxbout.itheta(idx_kl),
                                  tolerance));
        EXPECT_TRUE(IsCloseRelAbs(jxbout["izeta"][jF][k][l],
                                  output_quantities.jxbout.izeta(idx_kl),
                                  tolerance));
        EXPECT_TRUE(IsCloseRelAbs(jxbout["bdotk"][jF][k][l],
                                  output_quantities.jxbout.bdotk(idx_kl),
                                  tolerance));

        EXPECT_TRUE(IsCloseRelAbs(jxbout["jsupu3"][jF][k][l],
                                  output_quantities.jxbout.jsupu3(idx_kl),
                                  tolerance));
        EXPECT_TRUE(IsCloseRelAbs(jxbout["jsupv3"][jF][k][l],
                                  output_quantities.jxbout.jsupv3(idx_kl),
                                  tolerance));
      }  // l
    }  // k
  }  // jF

  for (int jF = 0; jF < fc.ns; ++jF) {
    EXPECT_TRUE(IsCloseRelAbs(jxbout["amaxfor"][jF],
                              output_quantities.jxbout.amaxfor[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(jxbout["aminfor"][jF],
                              output_quantities.jxbout.aminfor[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(jxbout["avforce"][jF],
                              output_quantities.jxbout.avforce[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(jxbout["pprim"][jF],
                              output_quantities.jxbout.pprim[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(jxbout["jdotb"][jF],
                              output_quantities.jxbout.jdotb[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(jxbout["bdotb"][jF],
                              output_quantities.jxbout.bdotb[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(jxbout["bdotgradv"][jF],
                              output_quantities.jxbout.bdotgradv[jF],
                              tolerance));
    EXPECT_TRUE(IsCloseRelAbs(jxbout["jpar2"][jF],
                              output_quantities.jxbout.jpar2[jF], tolerance));

    // catastrophic cancellation
    // TODO(jons): Can we make this more accurate?
    // FIXME(jons): This looks SO MUCH off that there actually could be an error
    // still present...
    EXPECT_TRUE(IsCloseRelAbs(jxbout["jperp2"][jF],
                              output_quantities.jxbout.jperp2[jF], 0.2));
  }  // jF

  // The loop in jxbforce.f90:594 goes over js=2,ns1,
  // which means that the last half-grid point is not touched.
  for (int jH = 0; jH < fc.ns - 2; ++jH) {
    for (int k = 0; k < s.nZeta; ++k) {
      for (int l = 0; l < s.nThetaEff; ++l) {
        const int idx_kl = (jH * s.nZeta + k) * s.nThetaEff + l;

        // catastrophic cancellation
        EXPECT_TRUE(IsCloseRelAbs(jxbout["jsups3"][jH + 1][k][l],
                                  output_quantities.jxbout.jsups3(idx_kl),
                                  5.0e-3));

        EXPECT_TRUE(IsCloseRelAbs(jxbout["bsubu3"][jH + 1][k][l],
                                  output_quantities.jxbout.bsubu3(idx_kl),
                                  tolerance));
        EXPECT_TRUE(IsCloseRelAbs(jxbout["bsubv3"][jH + 1][k][l],
                                  output_quantities.jxbout.bsubv3(idx_kl),
                                  tolerance));
      }  // l
    }  // k
  }  // jF
}  // CheckJxBOutputContents

// TODO(jons): Clarify below guess.
// I suspect these are so bad because J x B is close to 0
// in case of an equilibrium with small toroidal current.
// cth_like_fixed_bdy has a large toroidal current,
// so I suspect that J x B is more well-defined in that case...
INSTANTIATE_TEST_SUITE_P(
    TestOutputQuantities, JxBOutputContentsTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 2.0e-5},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 2.0e-5},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 1.0e-5},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 1.0e-5},
           DataSource{.identifier = "cma", .tolerance = 1.0e-4},
           DataSource{.identifier = "cth_like_free_bdy", .tolerance = 1.0e-5}));

class MercierStabilityTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(MercierStabilityTest, CheckMercierStability) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  Vmec vmec(*vmec_indata);
  const Sizes& s = vmec.s_;
  const FlowControl& fc = vmec.fc_;

  // run until convergence
  bool reached_checkpoint = vmec.run(VmecCheckpoint::MERCIER).value();
  ASSERT_FALSE(reached_checkpoint);

  filename = absl::StrFormat(
      "vmecpp_large_cpp_tests/test_data/%s/mercier/"
      "mercier_%05d_000000_01.%s.json",
      data_source_.identifier, fc.ns, data_source_.identifier);

  std::ifstream ifs_mercier(filename);
  ASSERT_TRUE(ifs_mercier.is_open())
      << "failed to open reference file: " << filename;
  json mercier = json::parse(ifs_mercier);

  const OutputQuantities& output_quantities = vmec.output_quantities_;

  const MercierStabilityIntermediateQuantities& mercier_intermediate =
      vmec.output_quantities_.mercier_intermediate;

  EXPECT_EQ(output_quantities.vmec_internal_results.sign_of_jacobian,
            mercier["sign_jac"]);

  for (int jF = 1; jF < fc.ns - 1; ++jF) {
    EXPECT_TRUE(IsCloseRelAbs(mercier["phip_real"][jF],
                              mercier_intermediate.phip_realF[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(mercier["sj"][jF], mercier_intermediate.s[jF],
                              tolerance));
  }  // jF

  for (int jH = 0; jH < fc.ns - 1; ++jH) {
    EXPECT_TRUE(IsCloseRelAbs(mercier["vp_real"][jH],
                              mercier_intermediate.vp_real[jH], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(mercier["torcur"][jH],
                              mercier_intermediate.torcur[jH], tolerance));
  }  // jH

  for (int jF = 1; jF < fc.ns - 1; ++jF) {
    EXPECT_TRUE(IsCloseRelAbs(mercier["shear"][jF - 1],
                              mercier_intermediate.shear[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(mercier["vpp"][jF - 1],
                              mercier_intermediate.vpp[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(mercier["presp"][jF - 1],
                              mercier_intermediate.d_pressure_d_s[jF],
                              tolerance));
    EXPECT_TRUE(IsCloseRelAbs(mercier["ip"][jF - 1],
                              mercier_intermediate.d_toroidal_current_d_s[jF],
                              tolerance));
  }  // jF

  for (int jF = 1; jF < fc.ns - 1; ++jF) {
    for (int l = 0; l < s.nThetaEff; ++l) {
      for (int k = 0; k < s.nZeta; ++k) {
        const int index_full = (jF * s.nZeta + k) * s.nThetaEff + l;

        EXPECT_TRUE(IsCloseRelAbs(mercier["gsqrt_full"][jF - 1][k][l],
                                  mercier_intermediate.gsqrt_full(index_full),
                                  tolerance));
        EXPECT_TRUE(IsCloseRelAbs(mercier["bdotj"][jF][k][l],
                                  mercier_intermediate.bdotj(index_full),
                                  tolerance));
        EXPECT_TRUE(IsCloseRelAbs(mercier["gpp"][jF - 1][k][l],
                                  mercier_intermediate.gpp(index_full),
                                  tolerance));
        EXPECT_TRUE(IsCloseRelAbs(mercier["b2"][jF][k][l],
                                  mercier_intermediate.b2(index_full),
                                  tolerance));
      }  // k
    }  // l
  }  // jF

  for (int jF = 1; jF < fc.ns - 1; ++jF) {
    EXPECT_TRUE(IsCloseRelAbs(mercier["tpp"][jF - 1],
                              mercier_intermediate.tpp[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(mercier["tbb"][jF - 1],
                              mercier_intermediate.tbb[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(mercier["tjb"][jF - 1],
                              mercier_intermediate.tjb[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(mercier["tjj"][jF - 1],
                              mercier_intermediate.tjj[jF], tolerance));
  }  // jF

  const MercierFileContents& mercier_file_contents =
      vmec.output_quantities_.mercier;

  // TODO(jons): check the first table in the Mercier output file

  // second table in Mercier output file
  for (int jF = 1; jF < fc.ns - 1; ++jF) {
    EXPECT_TRUE(IsCloseRelAbs(mercier["Dshear"][jF - 1],
                              mercier_file_contents.Dshear[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(mercier["Dcurr"][jF - 1],
                              mercier_file_contents.Dcurr[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(mercier["Dwell"][jF - 1],
                              mercier_file_contents.Dwell[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(mercier["Dgeod"][jF - 1],
                              mercier_file_contents.Dgeod[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(mercier["DMerc"][jF - 1],
                              mercier_file_contents.DMerc[jF], tolerance));
  }  // jF
}  // CheckMercierStability

INSTANTIATE_TEST_SUITE_P(
    TestOutputQuantities, MercierStabilityTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 1.0e-9},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 1.0e-9},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 1.0e-9},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 1.0e-9},
           DataSource{.identifier = "cma", .tolerance = 5.0e-9},
           DataSource{.identifier = "cth_like_free_bdy", .tolerance = 2.0e-9}));

class Threed1FirstTableTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(Threed1FirstTableTest, CheckThreed1FirstTable) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  Vmec vmec(*vmec_indata);
  const FlowControl& fc = vmec.fc_;

  // run until convergence
  bool reached_checkpoint =
      vmec.run(VmecCheckpoint::THREED1_FIRST_TABLE).value();
  ASSERT_FALSE(reached_checkpoint);

  filename = absl::StrFormat(
      "vmecpp_large_cpp_tests/test_data/%s/threed1_firstTable/"
      "threed1_firstTable_%05d_000000_01.%s.json",
      data_source_.identifier, fc.ns, data_source_.identifier);

  std::ifstream ifs_threed1_firstTable(filename);
  ASSERT_TRUE(ifs_threed1_firstTable.is_open())
      << "failed to open reference file: " << filename;
  json threed1_firstTable = json::parse(ifs_threed1_firstTable);

  const OutputQuantities& output_quantities = vmec.output_quantities_;

  const Threed1FirstTableIntermediate& threed1_first_table_intermediate =
      output_quantities.threed1_first_table_intermediate;
  const Threed1FirstTable& threed1_first_table =
      output_quantities.threed1_first_table;

  for (int jH = 0; jH < fc.ns - 1; ++jH) {
    EXPECT_TRUE(IsCloseRelAbs(threed1_firstTable["beta_vol"][jH],
                              threed1_first_table_intermediate.beta_vol[jH],
                              tolerance));
    EXPECT_TRUE(IsCloseRelAbs(threed1_firstTable["overr"][jH],
                              threed1_first_table_intermediate.overr[jH],
                              tolerance));
  }  // jH

  EXPECT_TRUE(IsCloseRelAbs(threed1_firstTable["betaxis"],
                            threed1_first_table_intermediate.beta_axis,
                            tolerance));

  for (int jF = 0; jF < fc.ns; ++jF) {
    EXPECT_TRUE(IsCloseRelAbs(threed1_firstTable["presf"][jF],
                              threed1_first_table_intermediate.presf[jF],
                              tolerance));
    EXPECT_TRUE(IsCloseRelAbs(threed1_firstTable["phipf_loc"][jF],
                              threed1_first_table_intermediate.phipf_loc[jF],
                              tolerance));
    EXPECT_TRUE(IsCloseRelAbs(threed1_firstTable["phi1"][jF],
                              threed1_first_table_intermediate.phi1[jF],
                              tolerance));
    EXPECT_TRUE(IsCloseRelAbs(threed1_firstTable["chi1"][jF],
                              threed1_first_table_intermediate.chi1[jF],
                              tolerance));
    EXPECT_TRUE(IsCloseRelAbs(threed1_firstTable["chi"][jF],
                              threed1_first_table_intermediate.chi[jF],
                              tolerance));

    EXPECT_TRUE(IsCloseRelAbs(threed1_firstTable["iotaf"][jF],
                              threed1_first_table.iota[jF], tolerance));

    EXPECT_TRUE(IsCloseRelAbs(threed1_firstTable["specw"][jF],
                              threed1_first_table.spectral_width[jF],
                              tolerance));

    EXPECT_TRUE(IsCloseRelAbs(threed1_firstTable["equif"][jF],
                              threed1_first_table_intermediate.equif[jF],
                              tolerance));
    EXPECT_TRUE(IsCloseRelAbs(threed1_firstTable["equif"][jF],
                              threed1_first_table.radial_force[jF], tolerance));

    EXPECT_TRUE(IsCloseRelAbs(threed1_firstTable["bucof"][jF],
                              threed1_first_table_intermediate.bucof[jF],
                              tolerance));
    EXPECT_TRUE(IsCloseRelAbs(threed1_firstTable["bucof"][jF],
                              threed1_first_table.buco_full[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(threed1_firstTable["bvcof"][jF],
                              threed1_first_table_intermediate.bvcof[jF],
                              tolerance));
    EXPECT_TRUE(IsCloseRelAbs(threed1_firstTable["bvcof"][jF],
                              threed1_first_table.bvco_full[jF], tolerance));

    EXPECT_TRUE(IsCloseRelAbs(threed1_firstTable["jcurv"][jF],
                              threed1_first_table_intermediate.jcurv[jF],
                              tolerance));
    EXPECT_TRUE(IsCloseRelAbs(threed1_firstTable["jcuru"][jF],
                              threed1_first_table_intermediate.jcuru[jF],
                              tolerance));

    EXPECT_TRUE(IsCloseRelAbs(threed1_firstTable["presgrad"][jF],
                              threed1_first_table_intermediate.presgrad[jF],
                              tolerance));
    EXPECT_TRUE(IsCloseRelAbs(threed1_firstTable["vpphi"][jF],
                              threed1_first_table_intermediate.vpphi[jF],
                              tolerance));

    EXPECT_TRUE(IsCloseRelAbs(threed1_firstTable["jdotb"][jF],
                              threed1_first_table.j_dot_b[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(threed1_firstTable["bdotb"][jF],
                              threed1_first_table.b_dot_b[jF], tolerance));
  }  // jF
}  // CheckThreed1FirstTable

INSTANTIATE_TEST_SUITE_P(
    TestOutputQuantities, Threed1FirstTableTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 1.0e-9},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 1.0e-9},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 1.0e-8},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 1.0e-8},
           DataSource{.identifier = "cma", .tolerance = 1.0e-6},
           DataSource{.identifier = "cth_like_free_bdy", .tolerance = 2.0e-8}));

class Threed1GeometricMagneticQuantitiesTest
    : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(Threed1GeometricMagneticQuantitiesTest,
       CheckThreed1GeometricMagneticQuantities) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  Vmec vmec(*vmec_indata);
  const Sizes& s = vmec.s_;
  const FlowControl& fc = vmec.fc_;

  // run until convergence
  bool reached_checkpoint = vmec.run(VmecCheckpoint::THREED1_GEOMAG).value();
  ASSERT_FALSE(reached_checkpoint);

  filename = absl::StrFormat(
      "vmecpp_large_cpp_tests/test_data/%s/threed1_geomag/"
      "threed1_geomag_%05d_000000_01.%s.json",
      data_source_.identifier, fc.ns, data_source_.identifier);

  std::ifstream ifs_threed1_geomag(filename);
  ASSERT_TRUE(ifs_threed1_geomag.is_open())
      << "failed to open reference file: " << filename;
  json threed1_geomag = json::parse(ifs_threed1_geomag);

  const OutputQuantities& output_quantities = vmec.output_quantities_;

  const Threed1GeometricAndMagneticQuantitiesIntermediate& intermediate =
      output_quantities.threed1_geometric_magnetic_intermediate;
  const Threed1GeometricAndMagneticQuantities& result =
      output_quantities.threed1_geometric_magnetic;

  EXPECT_TRUE(
      IsCloseRelAbs(threed1_geomag["anorm"], intermediate.anorm, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(threed1_geomag["vnorm"], intermediate.vnorm, tolerance));
  EXPECT_TRUE(IsCloseRelAbs(threed1_geomag["toroidal_flux"],
                            result.toroidal_flux, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(threed1_geomag["circum_p"], result.circum_p, tolerance));

  for (int k = 0; k < s.nZeta; ++k) {
    for (int l = 0; l < s.nThetaEff; ++l) {
      const int kl = k * s.nThetaEff + l;

      EXPECT_TRUE(IsCloseRelAbs(threed1_geomag["surf_area"][k][l],
                                intermediate.surf_area[kl], tolerance));
    }  // k
  }  // l
  EXPECT_TRUE(IsCloseRelAbs(threed1_geomag["surf_area_p"], result.surf_area_p,
                            tolerance));

  EXPECT_TRUE(
      IsCloseRelAbs(threed1_geomag["volume_p"], result.volume_p, tolerance));
  EXPECT_TRUE(IsCloseRelAbs(threed1_geomag["cross_area_p"], result.cross_area_p,
                            tolerance));

  EXPECT_TRUE(
      IsCloseRelAbs(threed1_geomag["Rmajor_p"], result.Rmajor_p, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(threed1_geomag["Aminor_p"], result.Aminor_p, tolerance));

  EXPECT_TRUE(
      IsCloseRelAbs(threed1_geomag["aspect"], result.aspect, tolerance));

  EXPECT_TRUE(
      IsCloseRelAbs(threed1_geomag["kappa_p"], result.kappa_p, tolerance));

  // intermediate:
  // TODO(jons): rcenin (not used anywhere?)
  // TODO(jons): aminr2in (not used anywhere?)
  // TODO(jons): bminz2in (not used anywhere?)
  // TODO(jons): bminz2 (not used anywhere?)

  EXPECT_TRUE(IsCloseRelAbs(threed1_geomag["rcen"], result.rcen, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(threed1_geomag["aminr1"], result.aminr1, tolerance));

  EXPECT_TRUE(
      IsCloseRelAbs(threed1_geomag["sump"], intermediate.sump, tolerance));
  EXPECT_TRUE(IsCloseRelAbs(threed1_geomag["pavg"], result.pavg, tolerance));

  EXPECT_TRUE(IsCloseRelAbs(threed1_geomag["delphid_exact"],
                            intermediate.delphid_exact, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(threed1_geomag["musubi"], intermediate.musubi, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(threed1_geomag["rshaf1"], intermediate.rshaf1, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(threed1_geomag["rshaf2"], intermediate.rshaf2, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(threed1_geomag["rshaf"], intermediate.rshaf, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(threed1_geomag["fpsi0"], intermediate.fpsi0, tolerance));
  EXPECT_TRUE(IsCloseRelAbs(threed1_geomag["b0"], result.b0, tolerance));

  for (int k = 0; k < s.nZeta; ++k) {
    for (int l = 0; l < s.nThetaEff; ++l) {
      const int kl = k * s.nThetaEff + l;

      EXPECT_TRUE(IsCloseRelAbs(threed1_geomag["redge"][k][l],
                                intermediate.redge[kl], tolerance));
      EXPECT_TRUE(IsCloseRelAbs(threed1_geomag["phat"][k][l],
                                intermediate.phat[kl], tolerance));
    }  // k
  }  // l

  EXPECT_TRUE(
      IsCloseRelAbs(threed1_geomag["rmax_surf"], result.rmax_surf, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(threed1_geomag["rmin_surf"], result.rmin_surf, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(threed1_geomag["zmax_surf"], result.zmax_surf, tolerance));

  EXPECT_TRUE(IsCloseRelAbs(threed1_geomag["bmin_1_ns"],
                            result.bmin((fc.ns - 2) * s.nThetaReduced + 0),
                            tolerance));
  EXPECT_TRUE(IsCloseRelAbs(threed1_geomag["bmax_1_ns"],
                            result.bmax((fc.ns - 2) * s.nThetaReduced + 0),
                            tolerance));
  EXPECT_TRUE(IsCloseRelAbs(
      threed1_geomag["bmin_ntheta2_ns"],
      result.bmin((fc.ns - 2) * s.nThetaReduced + (s.nThetaReduced - 1)),
      tolerance));
  EXPECT_TRUE(IsCloseRelAbs(
      threed1_geomag["bmax_ntheta2_ns"],
      result.bmax((fc.ns - 2) * s.nThetaReduced + (s.nThetaReduced - 1)),
      tolerance));

  EXPECT_TRUE(
      IsCloseRelAbs(threed1_geomag["waist"][0], result.waist[0], tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(threed1_geomag["height"][0], result.height[0], tolerance));
  if (s.ntor > 1) {
    EXPECT_TRUE(
        IsCloseRelAbs(threed1_geomag["waist"][1], result.waist[1], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(threed1_geomag["height"][1], result.height[1],
                              tolerance));
  }

  EXPECT_TRUE(IsCloseRelAbs(threed1_geomag["sumbtot"], intermediate.sumbtot,
                            tolerance));
  EXPECT_TRUE(IsCloseRelAbs(threed1_geomag["sumbtor"], intermediate.sumbtor,
                            tolerance));
  EXPECT_TRUE(IsCloseRelAbs(threed1_geomag["sumbpol"], intermediate.sumbpol,
                            tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(threed1_geomag["sump20"], intermediate.sump20, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(threed1_geomag["sump2"], intermediate.sump2, tolerance));

  EXPECT_TRUE(
      IsCloseRelAbs(threed1_geomag["betapol"], result.betapol, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(threed1_geomag["betatot"], result.betatot, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(threed1_geomag["betator"], result.betator, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(threed1_geomag["VolAvgB"], result.VolAvgB, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(threed1_geomag["IonLarmor"], result.IonLarmor, tolerance));

  for (int jF = 1; jF < fc.ns - 1; ++jF) {
    EXPECT_TRUE(IsCloseRelAbs(threed1_geomag["jPS2"][jF - 1],
                              intermediate.jPS2[jF], tolerance));
  }  // jF
  // TODO(jons): Is this catastrophic cancellation again?
  EXPECT_TRUE(IsCloseRelAbs(threed1_geomag["s2"], intermediate.s2, 0.2));
  EXPECT_TRUE(
      IsCloseRelAbs(threed1_geomag["jpar_perp"], result.jpar_perp, 0.15));
  EXPECT_TRUE(
      IsCloseRelAbs(threed1_geomag["jparPS_perp"], result.jparPS_perp, 0.1));

  for (int jF = 0; jF < fc.ns; ++jF) {
    EXPECT_TRUE(
        IsCloseRelAbs(threed1_geomag["psi"][jF], result.psi[jF], tolerance));
  }  // jF

  for (int nplanes = 0; nplanes < 2; ++nplanes) {
    for (int jF = 0; jF < fc.ns; ++jF) {
      EXPECT_TRUE(IsCloseRelAbs(threed1_geomag["ygeo"][nplanes][jF],
                                result.ygeo[nplanes * fc.ns + jF], tolerance));
    }
    for (int jF = 1; jF < fc.ns; ++jF) {
      EXPECT_TRUE(IsCloseRelAbs(threed1_geomag["yinden"][nplanes][jF - 1],
                                result.yinden[nplanes * fc.ns + jF],
                                tolerance));
      EXPECT_TRUE(IsCloseRelAbs(threed1_geomag["yellip"][nplanes][jF - 1],
                                result.yellip[nplanes * fc.ns + jF],
                                tolerance));
      EXPECT_TRUE(IsCloseRelAbs(threed1_geomag["ytrian"][nplanes][jF - 1],
                                result.ytrian[nplanes * fc.ns + jF],
                                tolerance));
      EXPECT_TRUE(IsCloseRelAbs(threed1_geomag["yshift"][nplanes][jF - 1],
                                result.yshift[nplanes * fc.ns + jF],
                                tolerance));
    }
  }  // nplanes
}  // CheckThreed1GeometricMagneticQuantities

INSTANTIATE_TEST_SUITE_P(
    TestOutputQuantities, Threed1GeometricMagneticQuantitiesTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 1.0e-5},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 1.0e-5},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 5.0e-9},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 5.0e-9},
           DataSource{.identifier = "cma", .tolerance = 1.0e-6},
           DataSource{.identifier = "cth_like_free_bdy", .tolerance = 1.0e-6}));

class Threed1VolumetricsTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(Threed1VolumetricsTest, CheckThreed1Volumetrics) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  Vmec vmec(*vmec_indata);

  const FlowControl& fc = vmec.fc_;

  // run until convergence
  bool reached_checkpoint =
      vmec.run(VmecCheckpoint::THREED1_VOLUMETRICS).value();
  ASSERT_FALSE(reached_checkpoint);

  filename = absl::StrFormat(
      "vmecpp_large_cpp_tests/test_data/%s/threed1_volquant/"
      "threed1_volquant_%05d_000000_01.%s.json",
      data_source_.identifier, fc.ns, data_source_.identifier);

  std::ifstream ifs_threed1_volquant(filename);
  ASSERT_TRUE(ifs_threed1_volquant.is_open())
      << "failed to open reference file: " << filename;
  json threed1_volquant = json::parse(ifs_threed1_volquant);

  const OutputQuantities& output_quantities = vmec.output_quantities_;

  const Threed1Volumetrics& result = output_quantities.threed1_volumetrics;

  EXPECT_TRUE(
      IsCloseRelAbs(threed1_volquant["int_p"], result.int_p, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(threed1_volquant["avg_p"], result.avg_p, tolerance));

  EXPECT_TRUE(
      IsCloseRelAbs(threed1_volquant["int_bpol"], result.int_bpol, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(threed1_volquant["avg_bpol"], result.avg_bpol, tolerance));

  EXPECT_TRUE(
      IsCloseRelAbs(threed1_volquant["int_btor"], result.int_btor, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(threed1_volquant["avg_btor"], result.avg_btor, tolerance));

  EXPECT_TRUE(
      IsCloseRelAbs(threed1_volquant["int_modb"], result.int_modb, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(threed1_volquant["avg_modb"], result.avg_modb, tolerance));

  EXPECT_TRUE(
      IsCloseRelAbs(threed1_volquant["int_ekin"], result.int_ekin, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(threed1_volquant["avg_ekin"], result.avg_ekin, tolerance));
}  // CheckThreed1Volumetrics

INSTANTIATE_TEST_SUITE_P(
    TestOutputQuantities, Threed1VolumetricsTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 1.0e-12},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 1.0e-12},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 1.0e-12},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 1.0e-12},
           DataSource{.identifier = "cma", .tolerance = 1.0e-12},
           DataSource{.identifier = "cth_like_free_bdy",
                      .tolerance = 1.0e-10}));

class Threed1AxisTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(Threed1AxisTest, CheckThreed1Axis) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  Vmec vmec(*vmec_indata);
  const Sizes& s = vmec.s_;
  const FlowControl& fc = vmec.fc_;

  bool reached_checkpoint = vmec.run(VmecCheckpoint::THREED1_AXIS).value();
  ASSERT_FALSE(reached_checkpoint);

  filename = absl::StrFormat(
      "vmecpp_large_cpp_tests/test_data/%s/threed1_axis/"
      "threed1_axis_%05d_000000_01.%s.json",
      data_source_.identifier, fc.ns, data_source_.identifier);

  std::ifstream ifs_threed1_axis(filename);
  ASSERT_TRUE(ifs_threed1_axis.is_open())
      << "failed to open reference file: " << filename;
  json threed1_axis = json::parse(ifs_threed1_axis);

  const OutputQuantities& output_quantities = vmec.output_quantities_;

  const Threed1AxisGeometry& axis = output_quantities.threed1_axis;

  for (int n = 0; n <= s.ntor; ++n) {
    EXPECT_TRUE(IsCloseRelAbs(threed1_axis["rax_symm"][n], axis.raxis_symm[n],
                              tolerance));
    EXPECT_TRUE(IsCloseRelAbs(threed1_axis["zax_symm"][n], axis.zaxis_symm[n],
                              tolerance));
    if (s.lasym) {
      EXPECT_TRUE(IsCloseRelAbs(threed1_axis["rax_asym"][n], axis.raxis_asym[n],
                                tolerance));
      EXPECT_TRUE(IsCloseRelAbs(threed1_axis["zax_asym"][n], axis.zaxis_asym[n],
                                tolerance));
    }
  }  // n
}  // CheckThreed1Axis

INSTANTIATE_TEST_SUITE_P(
    TestOutputQuantities, Threed1AxisTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 1.0e-12},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 1.0e-12},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 1.0e-12},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 1.0e-12},
           DataSource{.identifier = "cma", .tolerance = 1.0e-11},
           DataSource{.identifier = "cth_like_free_bdy",
                      .tolerance = 2.0e-11}));

class Threed1BetasTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(Threed1BetasTest, CheckThreed1Betas) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  Vmec vmec(*vmec_indata);
  const FlowControl& fc = vmec.fc_;

  // run until convergence
  bool reached_checkpoint = vmec.run(VmecCheckpoint::THREED1_BETAS).value();
  ASSERT_FALSE(reached_checkpoint);

  filename = absl::StrFormat(
      "vmecpp_large_cpp_tests/test_data/%s/threed1_beta/"
      "threed1_beta_%05d_000000_01.%s.json",
      data_source_.identifier, fc.ns, data_source_.identifier);

  std::ifstream ifs_threed1_beta(filename);
  ASSERT_TRUE(ifs_threed1_beta.is_open())
      << "failed to open reference file: " << filename;
  json threed1_beta = json::parse(ifs_threed1_beta);

  const OutputQuantities& output_quantities = vmec.output_quantities_;

  const Threed1Betas& result = output_quantities.threed1_betas;

  EXPECT_TRUE(
      IsCloseRelAbs(threed1_beta["betatot"], result.betatot, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(threed1_beta["betapol"], result.betapol, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(threed1_beta["betator"], result.betator, tolerance));

  EXPECT_TRUE(IsCloseRelAbs(threed1_beta["rbtor"], result.rbtor, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(threed1_beta["betaxis"], result.betaxis, tolerance));
  EXPECT_TRUE(IsCloseRelAbs(threed1_beta["betstr"], result.betstr, tolerance));
}  // CheckThreed1Betas

INSTANTIATE_TEST_SUITE_P(
    TestOutputQuantities, Threed1BetasTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 1.0e-12},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 1.0e-12},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 1.0e-12},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 1.0e-12},
           DataSource{.identifier = "cma", .tolerance = 1.0e-12},
           DataSource{.identifier = "cth_like_free_bdy",
                      .tolerance = 1.0e-11}));

class Threed1ShafranovIntegralsTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(Threed1ShafranovIntegralsTest, CheckThreed1ShafranovIntegrals) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  Vmec vmec(*vmec_indata);
  const FlowControl& fc = vmec.fc_;

  // run until convergence
  bool reached_checkpoint =
      vmec.run(VmecCheckpoint::THREED1_SHAFRANOV_INTEGRALS).value();
  ASSERT_FALSE(reached_checkpoint);

  filename = absl::StrFormat(
      "vmecpp_large_cpp_tests/test_data/%s/threed1_shafrint/"
      "threed1_shafrint_%05d_000000_01.%s.json",
      data_source_.identifier, fc.ns, data_source_.identifier);

  std::ifstream ifs_threed1_shafrint(filename);
  ASSERT_TRUE(ifs_threed1_shafrint.is_open())
      << "failed to open reference file: " << filename;
  json threed1_shafrint = json::parse(ifs_threed1_shafrint);

  const OutputQuantities& output_quantities = vmec.output_quantities_;

  const Threed1ShafranovIntegrals& result =
      output_quantities.threed1_shafranov_integrals;

  EXPECT_TRUE(IsCloseRelAbs(threed1_shafrint["scaling_ratio"],
                            result.scaling_ratio, tolerance));

  EXPECT_TRUE(IsCloseRelAbs(threed1_shafrint["rlao"], result.r_lao, tolerance));
  EXPECT_TRUE(IsCloseRelAbs(threed1_shafrint["flao"], result.f_lao, tolerance));
  EXPECT_TRUE(IsCloseRelAbs(threed1_shafrint["fgeo"], result.f_geo, tolerance));

  EXPECT_TRUE(
      IsCloseRelAbs(threed1_shafrint["smaleli"], result.smaleli, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(threed1_shafrint["betai"], result.betai, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(threed1_shafrint["musubi"], result.musubi, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(threed1_shafrint["lambda"], result.lambda, tolerance));

  EXPECT_TRUE(IsCloseRelAbs(threed1_shafrint["s11"], result.s11, tolerance));
  EXPECT_TRUE(IsCloseRelAbs(threed1_shafrint["s12"], result.s12, tolerance));
  EXPECT_TRUE(IsCloseRelAbs(threed1_shafrint["s13"], result.s13, tolerance));
  EXPECT_TRUE(IsCloseRelAbs(threed1_shafrint["s2"], result.s2, tolerance));
  EXPECT_TRUE(IsCloseRelAbs(threed1_shafrint["s3"], result.s3, tolerance));

  EXPECT_TRUE(
      IsCloseRelAbs(threed1_shafrint["delta1"], result.delta1, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(threed1_shafrint["delta2"], result.delta2, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(threed1_shafrint["delta3"], result.delta3, tolerance));
}  // CheckThreed1GeometricMagneticQuantities

INSTANTIATE_TEST_SUITE_P(
    TestOutputQuantities, Threed1ShafranovIntegralsTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 1.0e-11},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 1.0e-11},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 5.0e-11},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 5.0e-11},
           DataSource{.identifier = "cma", .tolerance = 5.0e-11},
           DataSource{.identifier = "cth_like_free_bdy", .tolerance = 5.0e-5})
    // NOTE: vacuum_b_phi likely largest influence here!
);

class CurrentDensityTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(CurrentDensityTest, CheckCurrentDensityFourierCoefficients) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  Vmec vmec(*vmec_indata);

  // run until convergence
  bool reached_checkpoint = vmec.run().value();
  ASSERT_FALSE(reached_checkpoint);

  const WOutFileContents& wout = vmec.output_quantities_.wout;

  // Read reference currumnc/currvmnc from Fortran wout file
  std::string ref_wout_filename = absl::StrFormat(
      "vmecpp_large_cpp_tests/test_data/wout_%s.nc", data_source_.identifier);

  int ncid = 0;
  ASSERT_EQ(nc_open(ref_wout_filename.c_str(), NC_NOWRITE, &ncid), NC_NOERR)
      << "Failed to open reference wout file: " << ref_wout_filename;

  // Read 2D arrays: Fortran stores as (ns, mnmax_nyq), C++ as (mnmax_nyq, ns)
  const std::vector<std::vector<double>> ref_currumnc =
      NetcdfReadArray2D(ncid, "currumnc").value();
  const std::vector<std::vector<double>> ref_currvmnc =
      NetcdfReadArray2D(ncid, "currvmnc").value();
  nc_close(ncid);

  const int ns = static_cast<int>(ref_currumnc.size());
  ASSERT_EQ(ns, wout.ns);
  ASSERT_GT(ns, 0);
  const int mnmax_nyq = static_cast<int>(ref_currumnc[0].size());
  ASSERT_EQ(mnmax_nyq, wout.mnmax_nyq);

  for (int jF = 0; jF < ns; ++jF) {
    for (int mn = 0; mn < mnmax_nyq; ++mn) {
      // ref_currumnc[jF][mn] is Fortran ordering (ns, mnmax_nyq)
      // wout.currumnc(mn, jF) is C++ ordering (mnmax_nyq, ns)
      EXPECT_TRUE(
          IsCloseRelAbs(ref_currumnc[jF][mn], wout.currumnc(mn, jF), tolerance))
          << "currumnc mismatch at jF=" << jF << " mn=" << mn
          << " ref=" << ref_currumnc[jF][mn]
          << " test=" << wout.currumnc(mn, jF);
      EXPECT_TRUE(
          IsCloseRelAbs(ref_currvmnc[jF][mn], wout.currvmnc(mn, jF), tolerance))
          << "currvmnc mismatch at jF=" << jF << " mn=" << mn
          << " ref=" << ref_currvmnc[jF][mn]
          << " test=" << wout.currvmnc(mn, jF);
    }  // mn
  }  // jF
}  // CheckCurrentDensityFourierCoefficients

// NOTE: Tolerances are larger than for other output quantities because
// the current density is computed via finite differences of the covariant B
// field Fourier coefficients, which amplifies the differences between the
// Fortran and C++ equilibria.
INSTANTIATE_TEST_SUITE_P(
    TestOutputQuantities, CurrentDensityTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 1.0e-4},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 1.0e-4},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 1.0e-4},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 1.0e-4},
           DataSource{.identifier = "cma", .tolerance = 1.0e-4}));

}  // namespace vmecpp
