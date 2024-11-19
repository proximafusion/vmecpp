#include "vmecpp/vmec/radial_profiles/radial_profiles.h"

#include <fstream>
#include <string>

#include "vmecpp/vmec/vmec_constants/vmec_constants.h"

#ifdef _OPENMP
#include <omp.h>
#endif  // _OPENMP

#include "absl/strings/str_format.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "util/file_io/file_io.h"
#include "util/testing/numerical_comparison_lib.h"
#include "vmecpp/common/flow_control/flow_control.h"
#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/vmec/boundaries/boundaries.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace vmecpp {

namespace {
using nlohmann::json;

using file_io::ReadFile;
using testing::IsCloseRelAbs;

using ::testing::TestWithParam;
using ::testing::Values;
}  // namespace

// used to specify tolerance depending on test case
struct DataSource {
  std::string identifier;
  double tolerance = 0.0;
};

TEST(TestRadialProfiles, CheckSolovevSingleThreaded) {
  double tolerance = DBL_EPSILON / 2;

  absl::StatusOr<std::string> indata_json =
      ReadFile("vmecpp/test_data/solovev.json");
  ASSERT_TRUE(indata_json.ok());

  absl::StatusOr<VmecINDATA> vmec_indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  // force single-threaded execution
  Vmec vmec(*vmec_indata, /*max_threads=*/1);

  const Sizes& s = vmec.s_;
  const FlowControl& fc = vmec.fc_;
  const VmecConstants& vmec_consts = vmec.constants_;

  bool reached_checkpoint =
      vmec.run(VmecCheckpoint::RADIAL_PROFILES_EVAL, 1).value();
  ASSERT_TRUE(reached_checkpoint);

  std::string filename = absl::StrFormat(
      "vmecpp/test_data/solovev/profil1d/"
      "profil1d_%05d_000001_%02d.solovev.json",
      fc.ns, vmec.get_num_eqsolve_retries());
  std::ifstream ifs_profil1d(filename);
  ASSERT_TRUE(ifs_profil1d.is_open());
  json profil1d = json::parse(ifs_profil1d);

  filename = absl::StrFormat(
      "vmecpp/test_data/solovev/profil3d/"
      "profil3d_%05d_000001_%02d.solovev.json",
      fc.ns, vmec.get_num_eqsolve_retries());
  std::ifstream ifs_profil3d(filename);
  ASSERT_TRUE(ifs_profil3d.is_open());
  json profil3d = json::parse(ifs_profil3d);

  const RadialProfiles& p = *(vmec.p_[0]);

  EXPECT_TRUE(
      IsCloseRelAbs(profil1d["torflux_edge"], p.maxToroidalFlux, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(profil1d["polflux_edge"], p.maxPoloidalFlux, tolerance));
  EXPECT_TRUE(IsCloseRelAbs(profil1d["r00"], vmec_indata->rbc[0], tolerance));
  EXPECT_TRUE(IsCloseRelAbs(profil1d["lamscale"], vmec_consts.lamscale,
                            tolerance));  // now in HandoverStorage!
  EXPECT_TRUE(IsCloseRelAbs(profil1d["currv"], p.currv, tolerance));
  EXPECT_TRUE(IsCloseRelAbs(profil1d["Itor"], p.Itor, tolerance));

  // half-grid profiles
  for (int jH = 0; jH < fc.ns - 1; ++jH) {
    EXPECT_TRUE(
        IsCloseRelAbs(profil1d["shalf"][jH + 1], p.sqrtSH[jH], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(profil1d["phips"][jH + 1], p.phipH[jH], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(profil1d["chips"][jH], p.chipH[jH], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(profil1d["iotas"][jH], p.iotaH[jH], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(profil1d["icurv"][jH], p.currH[jH], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(profil1d["mass"][jH], p.massH[jH], tolerance));

    EXPECT_TRUE(IsCloseRelAbs(profil1d["sp"][jH + 1], p.sp[jH], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(profil1d["sm"][jH + 1], p.sm[jH], tolerance));
  }

  // full-grid profiles
  for (int jF = 0; jF < fc.ns; ++jF) {
    EXPECT_TRUE(IsCloseRelAbs(profil1d["sqrts"][jF], p.sqrtSF[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(profil1d["phipf"][jF], p.phipF[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(profil1d["chipf"][jF], p.chipF[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(profil1d["iotaf"][jF], p.iotaF[jF], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(profil1d["bdamp"][jF], p.radialBlending[jF], tolerance));
  }

  // scalxc is in profil3d output in educational_VMEC,
  // but is computed in RadialProfiles in VMEC++, so test it here.
  for (int jF = 0; jF < fc.ns; ++jF) {
    for (int n = 0; n < s.ntor + 1; ++n) {
      for (int m = 0; m < s.mpol; ++m) {
        if (m % 2 == 0) {  // m is even
          EXPECT_TRUE(IsCloseRelAbs(
              profil3d["scalxc"][jF][n][m],
              p.scalxc[(jF - vmec.r_[0]->nsMinF1) * 2 + m_evn], tolerance));
        } else {  // m is odd
          EXPECT_TRUE(IsCloseRelAbs(
              profil3d["scalxc"][jF][n][m],
              p.scalxc[(jF - vmec.r_[0]->nsMinF1) * 2 + m_odd], tolerance));
        }
      }  // m
    }    // n
  }
}  // CheckSolovevSingleThreaded

class RadialProfilesTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(RadialProfilesTest, CheckRadialProfiles) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  absl::StatusOr<VmecINDATA> vmec_indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  Vmec vmec(*vmec_indata);
  const Sizes& s = vmec.s_;
  const FlowControl& fc = vmec.fc_;
  const VmecConstants& vmec_consts = vmec.constants_;

  // run to SETUP_INITIAL_STATE in order to also get scalxc computed
  bool reached_checkpoint =
      vmec.run(VmecCheckpoint::SETUP_INITIAL_STATE, 1).value();
  ASSERT_TRUE(reached_checkpoint);

  filename = absl::StrFormat(
      "vmecpp/test_data/%s/profil1d/profil1d_%05d_000001_%02d.%s.json",
      data_source_.identifier, fc.ns, vmec.get_num_eqsolve_retries(),
      data_source_.identifier);

  std::ifstream ifs_profil1d(filename);
  ASSERT_TRUE(ifs_profil1d.is_open());
  json profil1d = json::parse(ifs_profil1d);

  // TODO(jons): load data for current ns value, once this test runs for all
  // entries in ns_array
  filename = absl::StrFormat(
      "vmecpp/test_data/%s/profil3d/profil3d_%05d_000001_%02d.%s.json",
      data_source_.identifier, fc.ns, vmec.get_num_eqsolve_retries(),
      data_source_.identifier);
  std::ifstream ifs_profil3d(filename);
  ASSERT_TRUE(ifs_profil3d.is_open());
  json profil3d = json::parse(ifs_profil3d);

  for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
    const RadialPartitioning& r = *vmec.r_[thread_id];
    const RadialProfiles& p = *(vmec.p_[thread_id]);

    const int nsMinH = r.nsMinH;
    const int nsMaxH = r.nsMaxH;

    const int nsMinF1 = r.nsMinF1;
    const int nsMaxF1 = r.nsMaxF1;

    EXPECT_TRUE(
        IsCloseRelAbs(profil1d["torflux_edge"], p.maxToroidalFlux, tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(profil1d["polflux_edge"], p.maxPoloidalFlux, tolerance));
    EXPECT_TRUE(IsCloseRelAbs(
        profil1d["r00"], vmec_indata->rbc[vmec_indata->ntor + 0], tolerance));
    // now in HandoverStorage!
    EXPECT_TRUE(
        IsCloseRelAbs(profil1d["lamscale"], vmec_consts.lamscale, tolerance));
    EXPECT_TRUE(IsCloseRelAbs(profil1d["currv"], p.currv, tolerance));
    EXPECT_TRUE(IsCloseRelAbs(profil1d["Itor"], p.Itor, tolerance));

    // half-grid profiles
    for (int jH = nsMinH; jH < nsMaxH; ++jH) {
      EXPECT_TRUE(IsCloseRelAbs(profil1d["shalf"][jH + 1],
                                p.sqrtSH[jH - nsMinH], tolerance));
      EXPECT_TRUE(IsCloseRelAbs(profil1d["phips"][jH + 1], p.phipH[jH - nsMinH],
                                tolerance));
      EXPECT_TRUE(IsCloseRelAbs(profil1d["chips"][jH], p.chipH[jH - nsMinH],
                                tolerance));
      EXPECT_TRUE(IsCloseRelAbs(profil1d["iotas"][jH], p.iotaH[jH - nsMinH],
                                tolerance));
      EXPECT_TRUE(IsCloseRelAbs(profil1d["icurv"][jH], p.currH[jH - nsMinH],
                                tolerance));
      EXPECT_TRUE(
          IsCloseRelAbs(profil1d["mass"][jH], p.massH[jH - nsMinH], tolerance));

      EXPECT_TRUE(
          IsCloseRelAbs(profil1d["sp"][jH + 1], p.sp[jH - nsMinH], tolerance));
      EXPECT_TRUE(
          IsCloseRelAbs(profil1d["sm"][jH + 1], p.sm[jH - nsMinH], tolerance));
    }

    // full-grid profiles
    for (int jF = nsMinF1; jF < nsMaxF1; ++jF) {
      EXPECT_TRUE(IsCloseRelAbs(profil1d["sqrts"][jF], p.sqrtSF[jF - nsMinF1],
                                tolerance));
      EXPECT_TRUE(IsCloseRelAbs(profil1d["phipf"][jF], p.phipF[jF - nsMinF1],
                                tolerance));
      EXPECT_TRUE(IsCloseRelAbs(profil1d["chipf"][jF], p.chipF[jF - nsMinF1],
                                tolerance));
      EXPECT_TRUE(IsCloseRelAbs(profil1d["iotaf"][jF], p.iotaF[jF - nsMinF1],
                                tolerance));
      EXPECT_TRUE(IsCloseRelAbs(profil1d["bdamp"][jF],
                                p.radialBlending[jF - nsMinF1], tolerance));
    }

    // scalxc is in profil3d output in educational_VMEC,
    // but is computed in RadialProfiles in VMEC++, so test it here.
    for (int jF = nsMinF1; jF < nsMaxF1; ++jF) {
      for (int n = 0; n < s.ntor + 1; ++n) {
        for (int m = 0; m < s.mpol; ++m) {
          if (m % 2 == 0) {
            // m is even
            EXPECT_TRUE(IsCloseRelAbs(profil3d["scalxc"][jF][n][m],
                                      p.scalxc[(jF - nsMinF1) * 2 + m_evn],
                                      tolerance));
          } else {
            // m is odd
            EXPECT_TRUE(IsCloseRelAbs(profil3d["scalxc"][jF][n][m],
                                      p.scalxc[(jF - nsMinF1) * 2 + m_odd],
                                      tolerance));
          }
        }  // m
      }    // n
    }
  }  // thread_id
}  // CheckSolovevMultiThreaded

INSTANTIATE_TEST_SUITE_P(
    TestRadialProfiles, RadialProfilesTest,
    Values(DataSource{.identifier = "solovev", .tolerance = DBL_EPSILON / 2},
           DataSource{.identifier = "solovev_analytical", .tolerance = 1.0e-15},
           DataSource{.identifier = "solovev_no_axis",
                      .tolerance = DBL_EPSILON / 2},
           DataSource{.identifier = "cth_like_fixed_bdy",
                      .tolerance = DBL_EPSILON},
           DataSource{.identifier = "cma", .tolerance = DBL_EPSILON},
           DataSource{.identifier = "cth_like_free_bdy",
                      .tolerance = DBL_EPSILON}));

}  // namespace vmecpp
