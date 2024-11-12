#include "vmecpp/vmec/vmec/vmec.h"

#include <fstream>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "vmecpp/common/flow_control/flow_control.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/fourier_geometry/fourier_geometry.h"
#include "vmecpp/vmec/handover_storage/handover_storage.h"
#include "vmecpp/vmec/output_quantities/output_quantities.h"
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"

#ifdef _OPENMP
#include <omp.h>
#endif  // _OPENMP

#include "absl/strings/str_format.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "util/file_io/file_io.h"
#include "util/testing/numerical_comparison_lib.h"
#include "vmecpp/common/util/util.h"

using nlohmann::json;

using file_io::ReadFile;
using testing::IsCloseRelAbs;

using ::testing::DoubleNear;
using ::testing::ElementsAreArray;
using ::testing::Pointwise;
using ::testing::TestWithParam;
using ::testing::Values;

using vmecpp::FlowControl;
using vmecpp::HandoverStorage;
using vmecpp::RadialPartitioning;
using vmecpp::Sizes;
using vmecpp::Vmec;
using vmecpp::VmecCheckpoint;
using vmecpp::VmecINDATA;

namespace fs = std::filesystem;

// used to specify case-specific tolerances
// and which iterations to test
struct DataSource {
  std::string identifier;
  double tolerance = 0.0;
  std::vector<int> iter2_to_test = {1, 2};
};

class PrintoutTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(PrintoutTest, CheckPrintout) {
  const double tolerance = data_source_.tolerance;

  static constexpr int kR = 0;
  static constexpr int kZ = 1;
  static constexpr int kL = 2;

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
    const HandoverStorage& h = vmec.h_;

    bool reached_checkpoint =
        vmec.run(VmecCheckpoint::PRINTOUT, number_of_iterations).value();
    ASSERT_TRUE(reached_checkpoint);

    filename = absl::StrFormat(
        "vmecpp/test_data/%s/printout/printout_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, number_of_iterations,
        vmec.get_num_eqsolve_retries(), data_source_.identifier);

    std::ifstream ifs_printout(filename);
    ASSERT_TRUE(ifs_printout.is_open())
        << "failed to open reference file: " << filename;
    json printout = json::parse(ifs_printout);

    EXPECT_TRUE(IsCloseRelAbs(printout["betav"],
                              h.thermalEnergy / h.magneticEnergy, tolerance));

    // perform testing outside of multi-threaded region to avoid overlapping
    // error messages
    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const RadialPartitioning& radial_partitioning = *vmec.r_[thread_id];

      if (radial_partitioning.nsMaxF1 == fc.ns) {
        EXPECT_TRUE(IsCloseRelAbs(printout["delbsq"],
                                  vmec.m_[thread_id]->get_delbsq(), tolerance));
      }

      const int nsMinF1 = radial_partitioning.nsMinF1;
      const int nsMaxF1 = radial_partitioning.nsMaxF1;

      for (int idx_basis = 0; idx_basis < s.num_basis; ++idx_basis) {
        for (int jF = nsMinF1; jF < nsMaxF1; ++jF) {
          for (int n = 0; n < s.ntor + 1; ++n) {
            for (int m = 0; m < s.mpol; ++m) {
              EXPECT_TRUE(IsCloseRelAbs(
                  printout["gc"][kR][idx_basis][jF][n][m],
                  vmec.physical_x_backup_[thread_id]->GetXcElement(
                      kR, idx_basis, jF, n, m),
                  tolerance));
              EXPECT_TRUE(IsCloseRelAbs(
                  printout["gc"][kZ][idx_basis][jF][n][m],
                  vmec.physical_x_backup_[thread_id]->GetXcElement(
                      kZ, idx_basis, jF, n, m),
                  tolerance));
              EXPECT_TRUE(IsCloseRelAbs(
                  printout["gc"][kL][idx_basis][jF][n][m],
                  vmec.physical_x_backup_[thread_id]->GetXcElement(
                      kL, idx_basis, jF, n, m),
                  tolerance));
            }  // m
          }    // n
        }      // jF
      }        // idx_basis

      for (int jF = nsMinF1; jF < nsMaxF1; ++jF) {
        EXPECT_TRUE(IsCloseRelAbs(
            printout["specw"][jF],
            vmec.p_[thread_id]->spectral_width[jF - nsMinF1], tolerance));
      }  // jF
    }    // thread_id

    EXPECT_TRUE(IsCloseRelAbs(printout["avm"], h.VolumeAveragedSpectralWidth(),
                              tolerance));
  }
}  // CheckPrintout

INSTANTIATE_TEST_SUITE_P(
    TestVmec, PrintoutTest,
    Values(DataSource{.identifier = "solovev",
                      .tolerance = 5.0e-16,
                      .iter2_to_test = {1}},
           DataSource{.identifier = "solovev_no_axis",
                      .tolerance = 5.0e-16,
                      .iter2_to_test = {1}},
           DataSource{.identifier = "cth_like_fixed_bdy",
                      .tolerance = 1.0e-14,
                      .iter2_to_test = {1}},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 1.0e-14,
                      .iter2_to_test = {1}},
           DataSource{
               .identifier = "cma", .tolerance = 5.0e-13, .iter2_to_test = {1}},
           DataSource{.identifier = "cth_like_free_bdy",
                      .tolerance = 1.0e-14,
                      .iter2_to_test = {1}}));

class EvolveTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(EvolveTest, CheckEvolve) {
  const double tolerance = data_source_.tolerance;

  static constexpr int kR = 0;
  static constexpr int kZ = 1;
  static constexpr int kL = 2;

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
        vmec.run(VmecCheckpoint::EVOLVE, number_of_iterations).value();
    ASSERT_TRUE(reached_checkpoint);

    filename = absl::StrFormat(
        "vmecpp/test_data/%s/evolve/evolve_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, number_of_iterations,
        vmec.get_num_eqsolve_retries(), data_source_.identifier);

    std::ifstream ifs_evolve(filename);
    ASSERT_TRUE(ifs_evolve.is_open())
        << "failed to open reference file: " << filename;
    json evolve = json::parse(ifs_evolve);

    // perform testing outside of multi-threaded region to avoid overlapping
    // error messages
    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const RadialPartitioning& radial_partitioning = *vmec.r_[thread_id];

      const int nsMinF = radial_partitioning.nsMinF;
      const int nsMaxFIncludingLcfs = radial_partitioning.nsMaxFIncludingLcfs;

      for (int idx_basis = 0; idx_basis < s.num_basis; ++idx_basis) {
        for (int jF = nsMinF; jF < nsMaxFIncludingLcfs; ++jF) {
          for (int n = 0; n < s.ntor + 1; ++n) {
            for (int m = 0; m < s.mpol; ++m) {
              EXPECT_TRUE(
                  IsCloseRelAbs(evolve["gc"][kR][idx_basis][jF][n][m],
                                vmec.decomposed_f_[thread_id]->GetXcElement(
                                    kR, idx_basis, jF, n, m),
                                tolerance));
              EXPECT_TRUE(
                  IsCloseRelAbs(evolve["gc"][kZ][idx_basis][jF][n][m],
                                vmec.decomposed_f_[thread_id]->GetXcElement(
                                    kZ, idx_basis, jF, n, m),
                                tolerance));
              EXPECT_TRUE(
                  IsCloseRelAbs(evolve["gc"][kL][idx_basis][jF][n][m],
                                vmec.decomposed_f_[thread_id]->GetXcElement(
                                    kL, idx_basis, jF, n, m),
                                tolerance));

              EXPECT_TRUE(
                  IsCloseRelAbs(evolve["xcdot_after"][kR][idx_basis][jF][n][m],
                                vmec.decomposed_v_[thread_id]->GetXcElement(
                                    kR, idx_basis, jF, n, m),
                                tolerance));
              EXPECT_TRUE(
                  IsCloseRelAbs(evolve["xcdot_after"][kZ][idx_basis][jF][n][m],
                                vmec.decomposed_v_[thread_id]->GetXcElement(
                                    kZ, idx_basis, jF, n, m),
                                tolerance));
              EXPECT_TRUE(
                  IsCloseRelAbs(evolve["xcdot_after"][kL][idx_basis][jF][n][m],
                                vmec.decomposed_v_[thread_id]->GetXcElement(
                                    kL, idx_basis, jF, n, m),
                                tolerance));

              EXPECT_TRUE(
                  IsCloseRelAbs(evolve["xc_after"][kR][idx_basis][jF][n][m],
                                vmec.decomposed_x_[thread_id]->GetXcElement(
                                    kR, idx_basis, jF, n, m),
                                tolerance));
              EXPECT_TRUE(
                  IsCloseRelAbs(evolve["xc_after"][kZ][idx_basis][jF][n][m],
                                vmec.decomposed_x_[thread_id]->GetXcElement(
                                    kZ, idx_basis, jF, n, m),
                                tolerance));
              EXPECT_TRUE(
                  IsCloseRelAbs(evolve["xc_after"][kL][idx_basis][jF][n][m],
                                vmec.decomposed_x_[thread_id]->GetXcElement(
                                    kL, idx_basis, jF, n, m),
                                tolerance));
            }  // m
          }    // n
        }      // jF
      }        // idx_basis
    }          // thread_id
  }
}  // CheckEvolve

INSTANTIATE_TEST_SUITE_P(
    TestVmec, EvolveTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 1.0e-15},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 1.0e-15},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 2.0e-14},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 2.0e-14},
           DataSource{
               .identifier = "cma", .tolerance = 5.0e-13, .iter2_to_test = {1}},
           DataSource{.identifier = "cth_like_free_bdy",
                      .tolerance = 5.0e-13,
                      .iter2_to_test = {1, 2, 53, 54}}));

class MultigridResultTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(MultigridResultTest, CheckMultigridResult) {
  const double tolerance = data_source_.tolerance;

  static constexpr int kR = 0;
  static constexpr int kZ = 1;
  static constexpr int kL = 2;

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
  // TODO(jons): need to limit for first multi-grid step for now ...
  bool reached_checkpoint = vmec.run(VmecCheckpoint::NONE, INT_MAX, 1).value();
  ASSERT_FALSE(reached_checkpoint);

  // Here, we implicitly test for correct number of iterations until convergence
  // by constructing the filename based on the number of iterations VMEC++ took.
  filename = absl::StrFormat(
      "vmecpp/test_data/%s/multigrid_result/"
      "multigrid_result_%05d_%06d_%02d.%s.json",
      data_source_.identifier, fc.ns, vmec.get_iter2(), vmec.get_jacob_off(),
      data_source_.identifier);

  std::ifstream ifs_multigrid_result(filename);
  ASSERT_TRUE(ifs_multigrid_result.is_open())
      << "failed to open reference file: " << filename;
  json multigrid_result = json::parse(ifs_multigrid_result);

  // perform testing outside of multi-threaded region to avoid overlapping error
  // messages
  for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
    const RadialPartitioning& radial_partitioning = *vmec.r_[thread_id];

    const int nsMinF = radial_partitioning.nsMinF;
    const int nsMaxFIncludingLcfs = radial_partitioning.nsMaxFIncludingLcfs;

    for (int idx_basis = 0; idx_basis < s.num_basis; ++idx_basis) {
      for (int jF = nsMinF; jF < nsMaxFIncludingLcfs; ++jF) {
        for (int n = 0; n < s.ntor + 1; ++n) {
          for (int m = 0; m < s.mpol; ++m) {
            EXPECT_TRUE(
                IsCloseRelAbs(multigrid_result["xc"][kR][idx_basis][jF][n][m],
                              vmec.decomposed_x_[thread_id]->GetXcElement(
                                  kR, idx_basis, jF, n, m),
                              tolerance));
            EXPECT_TRUE(
                IsCloseRelAbs(multigrid_result["xc"][kZ][idx_basis][jF][n][m],
                              vmec.decomposed_x_[thread_id]->GetXcElement(
                                  kZ, idx_basis, jF, n, m),
                              tolerance));
            EXPECT_TRUE(
                IsCloseRelAbs(multigrid_result["xc"][kL][idx_basis][jF][n][m],
                              vmec.decomposed_x_[thread_id]->GetXcElement(
                                  kL, idx_basis, jF, n, m),
                              tolerance));
          }  // m
        }    // n
      }      // jF
    }        // idx_basis
  }          // thread_id
}  // CheckMultigridResult

INSTANTIATE_TEST_SUITE_P(
    TestVmec, MultigridResultTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 5.0e-15},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 5.0e-15},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 1.0e-13},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 1.0e-13},
           DataSource{.identifier = "cma", .tolerance = 1.0e-11},
           DataSource{.identifier = "cth_like_free_bdy",
                      .tolerance = 1.0e-11}));

class InterpTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(InterpTest, CheckInterp) {
  const double tolerance = data_source_.tolerance;

  static constexpr int kR = 0;
  static constexpr int kZ = 1;
  static constexpr int kL = 2;

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
  bool reached_checkpoint = vmec.run(VmecCheckpoint::INTERP).value();
  ASSERT_TRUE(reached_checkpoint);

  // Here, we implicitly test for correct number of iterations until convergence
  // by constructing the filename based on the number of iterations VMEC++ took.

  filename = absl::StrFormat(
      "vmecpp/test_data/%s/interp/interp_%05d_%06d_%02d.%s.json",
      data_source_.identifier, fc.ns, 1, 1, data_source_.identifier);

  std::ifstream ifs_interp(filename);
  ASSERT_TRUE(ifs_interp.is_open())
      << "failed to open reference file: " << filename;
  json interp = json::parse(ifs_interp);

  EXPECT_THAT(vmec.sj, ElementsAreArray(interp["sj"]));
  EXPECT_THAT(vmec.js1, ElementsAreArray(interp["js1"]));
  EXPECT_THAT(vmec.js2, ElementsAreArray(interp["js2"]));
  EXPECT_THAT(vmec.s1, ElementsAreArray(interp["s1"]));
  EXPECT_THAT(vmec.xint, ElementsAreArray(interp["xint"]));

  // test previous xc
  std::size_t num_threads_old = vmec.old_xc_scaled_.size();
  for (std::size_t thread_id = 0; thread_id < num_threads_old; ++thread_id) {
    const RadialPartitioning& radial_partitioning = *vmec.old_r_[thread_id];

    const int nsMinF = radial_partitioning.nsMinF;
    const int nsMaxFIncludingLcfs = radial_partitioning.nsMaxFIncludingLcfs;

    for (int idx_basis = 0; idx_basis < s.num_basis; ++idx_basis) {
      for (int jF = nsMinF; jF < nsMaxFIncludingLcfs; ++jF) {
        for (int n = 0; n < s.ntor + 1; ++n) {
          for (int m = 0; m < s.mpol; ++m) {
            EXPECT_TRUE(
                IsCloseRelAbs(interp["xold"][kR][idx_basis][jF][n][m],
                              vmec.old_xc_scaled_[thread_id]->GetXcElement(
                                  kR, idx_basis, jF, n, m),
                              tolerance));
            EXPECT_TRUE(
                IsCloseRelAbs(interp["xold"][kZ][idx_basis][jF][n][m],
                              vmec.old_xc_scaled_[thread_id]->GetXcElement(
                                  kZ, idx_basis, jF, n, m),
                              tolerance));
            EXPECT_TRUE(
                IsCloseRelAbs(interp["xold"][kL][idx_basis][jF][n][m],
                              vmec.old_xc_scaled_[thread_id]->GetXcElement(
                                  kL, idx_basis, jF, n, m),
                              tolerance));
          }  // m
        }    // n
      }      // jF
    }        // idx_basis
  }          // thread_id

  // perform testing outside of multi-threaded region to avoid overlapping error
  // messages
  for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
    const RadialPartitioning& radial_partitioning = *vmec.r_[thread_id];

    const int nsMinF1 = radial_partitioning.nsMinF1;
    const int nsMaxF1 = radial_partitioning.nsMaxF1;

    for (int idx_basis = 0; idx_basis < s.num_basis; ++idx_basis) {
      for (int jF = nsMinF1; jF < nsMaxF1; ++jF) {
        for (int n = 0; n < s.ntor + 1; ++n) {
          for (int m = 0; m < s.mpol; ++m) {
            const int m_parity = m % 2;
            const int scal_index = (jF - nsMinF1) * 2 + m_parity;
            EXPECT_TRUE(IsCloseRelAbs(interp["scalxc"][kR][idx_basis][jF][n][m],
                                      vmec.p_[thread_id]->scalxc[scal_index],
                                      tolerance));

            EXPECT_TRUE(
                IsCloseRelAbs(interp["xnew"][kR][idx_basis][jF][n][m],
                              vmec.decomposed_x_[thread_id]->GetXcElement(
                                  kR, idx_basis, jF, n, m),
                              tolerance));
            EXPECT_TRUE(
                IsCloseRelAbs(interp["xnew"][kZ][idx_basis][jF][n][m],
                              vmec.decomposed_x_[thread_id]->GetXcElement(
                                  kZ, idx_basis, jF, n, m),
                              tolerance));
            EXPECT_TRUE(
                IsCloseRelAbs(interp["xnew"][kL][idx_basis][jF][n][m],
                              vmec.decomposed_x_[thread_id]->GetXcElement(
                                  kL, idx_basis, jF, n, m),
                              tolerance));
          }  // m
        }    // n
      }      // jF
    }        // idx_basis
  }          // thread_id
}  // CheckInterp

INSTANTIATE_TEST_SUITE_P(
    TestVmec, InterpTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 1.0e-14},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 1.0e-14},
           DataSource{.identifier = "cma", .tolerance = 5.0e-11}));

TEST(TestVmec, CheckErrorOnNonConvergence) {
  // make sure VMEC++ reports an error if the run couldn't converge
  const std::string filename = "vmecpp/test_data/cth_like_fixed_bdy.json";
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  absl::StatusOr<VmecINDATA> indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(indata.ok());

  indata->niter_array[0] = 1;

  Vmec vmec(*indata);
  const absl::StatusOr<bool> status = vmec.run();

  CHECK(!status.ok());
  CHECK_EQ(status.status().message(), "VMEC++ did not converge");
}

TEST(TestVmec, CheckInMemoryMgrid) {
  // test the constructor that takes an in-memory mgrid

  // LOAD INDATA FILE
  const std::string filename = "vmecpp/test_data/cth_like_free_bdy.json";
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  absl::StatusOr<VmecINDATA> maybe_indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(maybe_indata.ok());
  VmecINDATA& indata = maybe_indata.value();

  // LOAD COILS FILE
  const std::string coils_filename = "vmecpp/test_data/coils.cth_like";
  const auto maybe_magnetic_configuration =
      magnetics::ImportMagneticConfigurationFromCoilsFile(coils_filename);
  ASSERT_TRUE(maybe_magnetic_configuration.ok());
  const auto& magnetic_configuration = *maybe_magnetic_configuration;

  // load makegrid params
  const auto maybe_makegrid_params = makegrid::ImportMakegridParametersFromFile(
      "vmecpp/test_data/makegrid_parameters_cth_like.json");
  ASSERT_TRUE(maybe_makegrid_params.ok());
  const auto& makegrid_params = *maybe_makegrid_params;

  // compute magnetic field response tables
  const auto maybe_magnetic_response_table =
      makegrid::ComputeMagneticFieldResponseTable(makegrid_params,
                                                  magnetic_configuration);
  ASSERT_TRUE(maybe_magnetic_response_table.ok());
  const auto& magnetic_response_table = *maybe_magnetic_response_table;

  // RUNS
  // using the mgrid file on disk
  // NOTE: we assume the mgrid file was produced with our C++ version of
  // makegrid. If it's re-generated using a different makegrid implementation,
  // this test might fail.
  const auto original_output = vmecpp::run(indata);
  ASSERT_TRUE(original_output.ok());

  // using the in-memory mgrid
  const auto output_with_inmemory_mgrid =
      vmecpp::run(indata, magnetic_response_table);
  ASSERT_TRUE(output_with_inmemory_mgrid.ok());

  // compare wout contents
  vmecpp::CompareWOut(output_with_inmemory_mgrid->wout, original_output->wout,
                      /*tolerance=*/1e-7);
}
