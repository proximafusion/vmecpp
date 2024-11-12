#include "vmecpp/free_boundary/external_magnetic_field/external_magnetic_field.h"

#include <fstream>
#include <string>
#include <vector>

#include "absl/strings/str_format.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "util/file_io/file_io.h"
#include "util/testing/numerical_comparison_lib.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/util/util.h"
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

class ExternalMagneticFieldTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(ExternalMagneticFieldTest, CheckExternalMagneticField) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  for (int number_of_iterations : data_source_.iter2_to_test) {
    Vmec vmec(*vmec_indata);
    const Sizes& s = vmec.s_;
    const FlowControl& fc = vmec.fc_;

    bool reached_checkpoint =
        vmec.run(VmecCheckpoint::VAC1_BEXTERN, number_of_iterations).value();
    ASSERT_TRUE(reached_checkpoint);

    filename = absl::StrFormat(
        "vmecpp/test_data/%s/vac1n_bextern/"
        "vac1n_bextern_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, number_of_iterations,
        vmec.get_num_eqsolve_retries(), data_source_.identifier);

    std::ifstream ifs_vac1n_bextern(filename);
    ASSERT_TRUE(ifs_vac1n_bextern.is_open());
    json vac1n_bextern = json::parse(ifs_vac1n_bextern);

    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const FreeBoundaryBase& n = *vmec.fb_[thread_id];
      const TangentialPartitioning& tp = *vmec.tp_[thread_id];
      const ExternalMagneticField& ef = n.GetExternalMagneticField();

      // current along magnetic axis in Amperes
      EXPECT_TRUE(IsCloseRelAbs(vac1n_bextern["axis_current"], ef.axis_current,
                                tolerance));

      // axis geometry
      for (int k = 0; k < s.nZeta * s.nfp + 1; ++k) {
        EXPECT_TRUE(IsCloseRelAbs(vac1n_bextern["xpts_axis"][0][k],
                                  ef.axisXYZ[k * 3 + 0], tolerance));
        EXPECT_TRUE(IsCloseRelAbs(vac1n_bextern["xpts_axis"][1][k],
                                  ef.axisXYZ[k * 3 + 1], tolerance));
        EXPECT_TRUE(IsCloseRelAbs(vac1n_bextern["xpts_axis"][2][k],
                                  ef.axisXYZ[k * 3 + 2], tolerance));
      }  // k

      // local tangential partition
      for (int kl = tp.ztMin; kl < tp.ztMax; ++kl) {
        const int l = kl / s.nZeta;
        const int k = kl % s.nZeta;

        const int klRel = kl - tp.ztMin;

        // test mgrid contribution on its own
        EXPECT_TRUE(IsCloseRelAbs(vac1n_bextern["mgrid_brad"][k][l],
                                  ef.interpBr[klRel], tolerance));
        EXPECT_TRUE(IsCloseRelAbs(vac1n_bextern["mgrid_bphi"][k][l],
                                  ef.interpBp[klRel], tolerance));
        EXPECT_TRUE(IsCloseRelAbs(vac1n_bextern["mgrid_bz"][k][l],
                                  ef.interpBz[klRel], tolerance));

        // test axis current contribution on its own
        EXPECT_TRUE(
            IsCloseRelAbs(static_cast<double>(vac1n_bextern["brad"][k][l]) -
                              ef.interpBr[klRel],
                          ef.curtorBr[klRel], tolerance));
        EXPECT_TRUE(
            IsCloseRelAbs(static_cast<double>(vac1n_bextern["bphi"][k][l]) -
                              ef.interpBp[klRel],
                          ef.curtorBp[klRel], tolerance));
        EXPECT_TRUE(IsCloseRelAbs(
            static_cast<double>(vac1n_bextern["bz"][k][l]) - ef.interpBz[klRel],
            ef.curtorBz[klRel], tolerance));

        // add contributions together for testing sum (that is what is output
        // from bextern in educational_VMEC)
        double fullBr = ef.interpBr[klRel] + ef.curtorBr[klRel];
        double fullBp = ef.interpBp[klRel] + ef.curtorBp[klRel];
        double fullBz = ef.interpBz[klRel] + ef.curtorBz[klRel];

        // test full field
        EXPECT_TRUE(
            IsCloseRelAbs(vac1n_bextern["brad"][k][l], fullBr, tolerance));
        EXPECT_TRUE(
            IsCloseRelAbs(vac1n_bextern["bphi"][k][l], fullBp, tolerance));
        EXPECT_TRUE(
            IsCloseRelAbs(vac1n_bextern["bz"][k][l], fullBz, tolerance));

        // test quantities derived from full field
        EXPECT_TRUE(IsCloseRelAbs(vac1n_bextern["bexu"][k][l], ef.bSubU[klRel],
                                  tolerance));
        EXPECT_TRUE(IsCloseRelAbs(vac1n_bextern["bexv"][k][l], ef.bSubV[klRel],
                                  tolerance));
        EXPECT_TRUE(IsCloseRelAbs(vac1n_bextern["bexn"][k][l], ef.bDotN[klRel],
                                  tolerance));

        // bexni is not tested here, since it is computed on-the-fly where
        // needed.
      }  // kl
    }    // thread_id
  }
}  // CheckExternalMagneticField

// NOTE: tolerance is a little more loose here, since Fortran VMEC still uses
// the bsc_t old Biot-Savart module, whereas VMEC++ uses ABSCAB. (The mgrid file
// can be computed using ABSCAB already via //makegrid/makegrid_standalone.)
INSTANTIATE_TEST_SUITE_P(TestExternalMagneticField, ExternalMagneticFieldTest,
                         Values(DataSource{.identifier = "cth_like_free_bdy",
                                           .tolerance = 1.0e-10,
                                           .iter2_to_test = {53, 54}}));

}  // namespace vmecpp
