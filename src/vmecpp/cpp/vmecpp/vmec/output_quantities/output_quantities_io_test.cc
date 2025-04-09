// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include <filesystem>
#include <memory>
#include <string>
#include <utility>  // std::move

#include "H5Cpp.h"
#include "gtest/gtest.h"
#include "util/file_io/file_io.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/output_quantities/output_quantities.h"
#include "vmecpp/vmec/output_quantities/test_helpers.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace fs = std::filesystem;

class OutputQuantitiesIO : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    const std::string filename = "vmecpp/test_data/cth_like_fixed_bdy.json";
    const absl::StatusOr<std::string> indata_json = file_io::ReadFile(filename);
    EXPECT_TRUE(indata_json.ok());

    const auto indata = vmecpp::VmecINDATA::FromJson(*indata_json);
    EXPECT_TRUE(indata.ok());

    auto maybe_oq = vmecpp::run(*indata);
    EXPECT_TRUE(maybe_oq.ok());

    cth_output_quantities_ =
        std::make_unique<vmecpp::OutputQuantities>(std::move(*maybe_oq));
  }

  static std::unique_ptr<vmecpp::OutputQuantities> cth_output_quantities_;
};

std::unique_ptr<vmecpp::OutputQuantities>
    OutputQuantitiesIO::cth_output_quantities_ = nullptr;

TEST_F(OutputQuantitiesIO, VmecInternalResults) {
  const std::string fname = "test_outputquantitiesio_vmecinternalresults.h5";
  H5::H5File h5file(fname, H5F_ACC_TRUNC);

  const auto& internalres = cth_output_quantities_->vmec_internal_results;
  const auto status1 = internalres.WriteTo(h5file);
  ASSERT_TRUE(status1.ok());

  vmecpp::VmecInternalResults internalres_fromfile;
  const auto status2 =
      vmecpp::VmecInternalResults::LoadInto(internalres_fromfile, h5file);
  ASSERT_TRUE(status2.ok());

  EXPECT_EQ(internalres.sign_of_jacobian,
            internalres_fromfile.sign_of_jacobian);
  EXPECT_EQ(internalres.num_full, internalres_fromfile.num_full);
  EXPECT_EQ(internalres.num_half, internalres_fromfile.num_half);
  EXPECT_EQ(internalres.nZnT_reduced, internalres_fromfile.nZnT_reduced);
  EXPECT_EQ(internalres.sqrtSH, internalres_fromfile.sqrtSH);
  EXPECT_EQ(internalres.sqrtSF, internalres_fromfile.sqrtSF);
  EXPECT_EQ(internalres.sm, internalres_fromfile.sm);
  EXPECT_EQ(internalres.sp, internalres_fromfile.sp);
  EXPECT_EQ(internalres.phipF, internalres_fromfile.phipF);
  EXPECT_EQ(internalres.chipF, internalres_fromfile.chipF);
  EXPECT_EQ(internalres.phipH, internalres_fromfile.phipH);
  EXPECT_EQ(internalres.phiF, internalres_fromfile.phiF);
  EXPECT_EQ(internalres.iotaF, internalres_fromfile.iotaF);
  EXPECT_EQ(internalres.spectral_width, internalres_fromfile.spectral_width);
  EXPECT_EQ(internalres.bvcoH, internalres_fromfile.bvcoH);
  EXPECT_EQ(internalres.dVdsH, internalres_fromfile.dVdsH);
  EXPECT_EQ(internalres.massH, internalres_fromfile.massH);
  EXPECT_EQ(internalres.presH, internalres_fromfile.presH);
  EXPECT_EQ(internalres.iotaH, internalres_fromfile.iotaH);
  EXPECT_EQ(internalres.rmncc, internalres_fromfile.rmncc);
  EXPECT_EQ(internalres.rmnss, internalres_fromfile.rmnss);
  EXPECT_EQ(internalres.rmnsc, internalres_fromfile.rmnsc);
  EXPECT_EQ(internalres.rmncs, internalres_fromfile.rmncs);
  EXPECT_EQ(internalres.zmnsc, internalres_fromfile.zmnsc);
  EXPECT_EQ(internalres.zmncs, internalres_fromfile.zmncs);
  EXPECT_EQ(internalres.zmncc, internalres_fromfile.zmncc);
  EXPECT_EQ(internalres.zmnss, internalres_fromfile.zmnss);
  EXPECT_EQ(internalres.lmnsc, internalres_fromfile.lmnsc);
  EXPECT_EQ(internalres.lmncs, internalres_fromfile.lmncs);
  EXPECT_EQ(internalres.lmncc, internalres_fromfile.lmncc);
  EXPECT_EQ(internalres.lmnss, internalres_fromfile.lmnss);
  EXPECT_EQ(internalres.r_e, internalres_fromfile.r_e);
  EXPECT_EQ(internalres.r_o, internalres_fromfile.r_o);
  EXPECT_EQ(internalres.z_e, internalres_fromfile.z_e);
  EXPECT_EQ(internalres.z_o, internalres_fromfile.z_o);
  EXPECT_EQ(internalres.ru_e, internalres_fromfile.ru_e);
  EXPECT_EQ(internalres.ru_o, internalres_fromfile.ru_o);
  EXPECT_EQ(internalres.zu_e, internalres_fromfile.zu_e);
  EXPECT_EQ(internalres.zu_o, internalres_fromfile.zu_o);
  EXPECT_EQ(internalres.rv_e, internalres_fromfile.rv_e);
  EXPECT_EQ(internalres.rv_o, internalres_fromfile.rv_o);
  EXPECT_EQ(internalres.zv_e, internalres_fromfile.zv_e);
  EXPECT_EQ(internalres.zv_o, internalres_fromfile.zv_o);
  EXPECT_EQ(internalres.ruFull, internalres_fromfile.ruFull);
  EXPECT_EQ(internalres.zuFull, internalres_fromfile.zuFull);
  EXPECT_EQ(internalres.r12, internalres_fromfile.r12);
  EXPECT_EQ(internalres.ru12, internalres_fromfile.ru12);
  EXPECT_EQ(internalres.zu12, internalres_fromfile.zu12);
  EXPECT_EQ(internalres.rs, internalres_fromfile.rs);
  EXPECT_EQ(internalres.zs, internalres_fromfile.zs);
  EXPECT_EQ(internalres.gsqrt, internalres_fromfile.gsqrt);
  EXPECT_EQ(internalres.guu, internalres_fromfile.guu);
  EXPECT_EQ(internalres.guv, internalres_fromfile.guv);
  EXPECT_EQ(internalres.gvv, internalres_fromfile.gvv);
  EXPECT_EQ(internalres.bsupu, internalres_fromfile.bsupu);
  EXPECT_EQ(internalres.bsupv, internalres_fromfile.bsupv);
  EXPECT_EQ(internalres.bsubu, internalres_fromfile.bsubu);
  EXPECT_EQ(internalres.bsubv, internalres_fromfile.bsubv);
  EXPECT_EQ(internalres.bsubvF, internalres_fromfile.bsubvF);
  EXPECT_EQ(internalres.total_pressure, internalres_fromfile.total_pressure);
  EXPECT_EQ(internalres.currv, internalres_fromfile.currv);
}

TEST_F(OutputQuantitiesIO, WOut) {
  const std::string fname = "test_outputquantitiesio_wout.h5";
  H5::H5File h5file(fname, H5F_ACC_TRUNC);

  const auto& wout = cth_output_quantities_->wout;
  const auto status1 = wout.WriteTo(h5file);
  ASSERT_TRUE(status1.ok());

  vmecpp::WOutFileContents wout_fromfile;
  const auto status2 =
      vmecpp::WOutFileContents::LoadInto(wout_fromfile, h5file);
  ASSERT_TRUE(status2.ok());

  CheckWOutEquality(wout, wout_fromfile);
}

TEST_F(OutputQuantitiesIO, OutputQuantities) {
  // create directory for our test
  const fs::path test_dir = std::filesystem::temp_directory_path() /
                            ("vmecpp_tests_" + std::to_string(getpid()));
  std::error_code err;
  fs::create_directory(test_dir, err);
  ASSERT_FALSE(err) << "Could not create test directory " << test_dir
                    << ", error was: " << err.message();

  // write out...
  const fs::path fname =
      test_dir / "test_outputquantitiesio_outputquantities.h5";
  const absl::Status s = cth_output_quantities_->Save(fname);
  ASSERT_TRUE(s.ok()) << s;

  // ...and read back
  const absl::StatusOr<vmecpp::OutputQuantities> maybe_oq =
      vmecpp::OutputQuantities::Load(fname);
  ASSERT_TRUE(maybe_oq.ok()) << maybe_oq.status();

  // check that the contents that we read back are the same as the ones we wrote
  const auto& read_oq = maybe_oq.value();

  EXPECT_EQ(read_oq.vmec_internal_results,
            cth_output_quantities_->vmec_internal_results);
  EXPECT_EQ(read_oq.remaining_metric, cth_output_quantities_->remaining_metric);
  EXPECT_EQ(read_oq.b_cylindrical, cth_output_quantities_->b_cylindrical);
  EXPECT_EQ(read_oq.bsubs_half, cth_output_quantities_->bsubs_half);
  EXPECT_EQ(read_oq.bsubs_full, cth_output_quantities_->bsubs_full);
  EXPECT_EQ(read_oq.covariant_b_derivatives,
            cth_output_quantities_->covariant_b_derivatives);
  EXPECT_EQ(read_oq.jxbout, cth_output_quantities_->jxbout);
  EXPECT_EQ(read_oq.mercier_intermediate,
            cth_output_quantities_->mercier_intermediate);
  EXPECT_EQ(read_oq.mercier, cth_output_quantities_->mercier);
  EXPECT_EQ(read_oq.threed1_first_table_intermediate,
            cth_output_quantities_->threed1_first_table_intermediate);
  EXPECT_EQ(read_oq.threed1_first_table,
            cth_output_quantities_->threed1_first_table);
  EXPECT_EQ(read_oq.threed1_geometric_magnetic_intermediate,
            cth_output_quantities_->threed1_geometric_magnetic_intermediate);
  EXPECT_EQ(read_oq.threed1_geometric_magnetic,
            cth_output_quantities_->threed1_geometric_magnetic);
  EXPECT_EQ(read_oq.threed1_volumetrics,
            cth_output_quantities_->threed1_volumetrics);
  EXPECT_EQ(read_oq.threed1_axis, cth_output_quantities_->threed1_axis);
  EXPECT_EQ(read_oq.threed1_betas, cth_output_quantities_->threed1_betas);
  EXPECT_EQ(read_oq.threed1_shafranov_integrals,
            cth_output_quantities_->threed1_shafranov_integrals);

  CheckWOutEquality(read_oq.wout, cth_output_quantities_->wout);
}
