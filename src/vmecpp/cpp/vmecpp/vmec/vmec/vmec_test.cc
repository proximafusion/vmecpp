// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
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
#include "vmecpp/vmec/output_quantities/test_helpers.h"
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"

#ifdef _OPENMP
#include <omp.h>
#endif  // _OPENMP

#include "absl/strings/match.h"
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

TEST(TestVmec, CheckErrorOnNonConvergence) {
  // make sure VMEC++ reports an error if the run couldn't converge
  const std::string filename = "vmecpp/test_data/cth_like_fixed_bdy.json";
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  absl::StatusOr<VmecINDATA> indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(indata.ok());

  // allow only 1 iteration - not enough to let VMEC converge
  indata->niter_array[0] = 1;

  auto maybe_vmec = Vmec::FromIndata(*indata);
  ASSERT_TRUE(maybe_vmec.ok());
  Vmec& vmec = **maybe_vmec;

  const absl::StatusOr<bool> status = vmec.run();

  CHECK(!status.ok());
  CHECK(
      absl::StrContains(status.status().message(), "VMEC++ did not converge"));
}  // CheckErrorOnNonConvergence

TEST(TestVmec, CheckNoErrorOnNonConvergenceIfDesired) {
  // make sure VMEC++ returns the outputs without an error
  // if explicitly instructed to do so
  const std::string filename = "vmecpp/test_data/cth_like_fixed_bdy.json";
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  absl::StatusOr<VmecINDATA> indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(indata.ok());

  // allow only 1 iteration - not enough to let VMEC converge
  indata->niter_array[0] = 1;

  // instruct VMEC++ to return its outputs, even if it did not converge
  indata->return_outputs_even_if_not_converged = true;

  auto maybe_vmec = Vmec::FromIndata(*indata);
  ASSERT_TRUE(maybe_vmec.ok());
  Vmec& vmec = **maybe_vmec;

  const absl::StatusOr<bool> status = vmec.run();

  CHECK(status.ok());
}  // CheckNoErrorOnNonConvergenceIfDesired

TEST(TestVmec, CheckFromIndataReturnsErrorForInvalidMgridPath) {
  // Verify that FromIndata returns an error status (rather than throwing)
  // when a free-boundary run specifies a non-existent mgrid file.
  const std::string filename = "vmecpp/test_data/cth_like_free_bdy.json";
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  absl::StatusOr<VmecINDATA> maybe_indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(maybe_indata.ok());
  VmecINDATA& indata = maybe_indata.value();

  indata.mgrid_file = "/does/not/exist/mgrid.nc";

  auto maybe_vmec = Vmec::FromIndata(indata);
  EXPECT_FALSE(maybe_vmec.ok());
}  // CheckFromIndataReturnsErrorForInvalidMgridPath

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
  CompareWOut(output_with_inmemory_mgrid->wout, original_output->wout,
              /*tolerance=*/1e-7);
}  // CheckInMemoryMgrid

// A stellarator-symmetric, axisymmetric equilibrium (solovev) must converge to
// the same result whether run with lasym=false or with lasym=true and zero
// antisymmetric content. This exercises the 2D non-stellarator-symmetric
// inverse/forward DFTs, symrzl, and symforce against the known symmetric
// answer: the antisymmetric pieces stay zero, and symrzl / symforce must
// reconstruct the full poloidal range so the converged wout is unchanged.
TEST(TestVmec, LasymAxisymmetricDegeneratesToSymmetric) {
  const std::string filename = "vmecpp/test_data/solovev.json";
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  absl::StatusOr<VmecINDATA> indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(indata.ok());
  ASSERT_FALSE(indata->lasym);

  // symmetric baseline
  const auto symmetric_output = vmecpp::run(*indata);
  ASSERT_TRUE(symmetric_output.ok());

  // same equilibrium, run through the lasym code path with zero antisymmetric
  // boundary and axis coefficients
  VmecINDATA asym_indata = *indata;
  asym_indata.lasym = true;
  auto zero_rbs = indata->rbc;
  zero_rbs.setZero();
  auto zero_zbc = indata->zbs;
  zero_zbc.setZero();
  asym_indata.rbs = zero_rbs;
  asym_indata.zbc = zero_zbc;
  auto zero_raxis_s = indata->raxis_c;
  zero_raxis_s.setZero();
  auto zero_zaxis_c = indata->zaxis_s;
  zero_zaxis_c.setZero();
  asym_indata.raxis_s = zero_raxis_s;
  asym_indata.zaxis_c = zero_zaxis_c;

  const auto asymmetric_output = vmecpp::run(asym_indata);
  ASSERT_TRUE(asymmetric_output.ok());

  // CompareWOut cannot be used directly because it requires identical lasym
  // flags, so compare the converged physics: the key scalar quantities and the
  // symmetric Fourier coefficients must match, and the antisymmetric
  // coefficients of the lasym run must stay at zero.
  const auto& sym = symmetric_output->wout;
  const auto& asym = asymmetric_output->wout;
  const double kTol = 1.0e-10;

  ASSERT_EQ(asym.ns, sym.ns);

  // Integrated scalars, including b0 (it agrees to ~1e-14, not the 1e-3 a
  // previous normalization gap suggested).
  EXPECT_TRUE(IsCloseRelAbs(sym.wb, asym.wb, kTol)) << "wb";
  EXPECT_TRUE(IsCloseRelAbs(sym.wp, asym.wp, kTol)) << "wp";
  EXPECT_TRUE(IsCloseRelAbs(sym.volume, asym.volume, kTol)) << "volume";
  EXPECT_TRUE(IsCloseRelAbs(sym.aspect, asym.aspect, kTol)) << "aspect";
  EXPECT_TRUE(IsCloseRelAbs(sym.betatotal, asym.betatotal, kTol))
      << "betatotal";
  EXPECT_TRUE(IsCloseRelAbs(sym.b0, asym.b0, kTol)) << "b0";
  EXPECT_TRUE(IsCloseRelAbs(sym.Aminor_p, asym.Aminor_p, kTol)) << "Aminor_p";
  EXPECT_TRUE(IsCloseRelAbs(sym.Rmajor_p, asym.Rmajor_p, kTol)) << "Rmajor_p";

  // Peak-normalized max difference between two coefficient arrays.
  auto rel_max = [](const auto& a, const auto& b) -> double {
    const double peak = a.cwiseAbs().maxCoeff();
    return (a - b).cwiseAbs().maxCoeff() / (peak > 0.0 ? peak : 1.0);
  };
  // Max magnitude of an array that must vanish, normalized by its symmetric
  // companion's peak.
  auto rel_zero = [](const auto& a, const auto& companion) -> double {
    const double peak = companion.cwiseAbs().maxCoeff();
    return a.cwiseAbs().maxCoeff() / (peak > 0.0 ? peak : 1.0);
  };

  // Symmetric geometry and the derived Nyquist-grid spectra (field magnitude,
  // Jacobian, covariant/contravariant field, and B_s) reproduce the symmetric
  // run to machine precision. These are the arrays a lasym output-normalization
  // error would have doubled.
  EXPECT_LT(rel_max(sym.rmnc, asym.rmnc), kTol) << "rmnc";
  EXPECT_LT(rel_max(sym.zmns, asym.zmns), kTol) << "zmns";
  EXPECT_LT(rel_max(sym.lmns_full, asym.lmns_full), kTol) << "lmns_full";
  EXPECT_LT(rel_max(sym.bmnc, asym.bmnc), kTol) << "bmnc";
  EXPECT_LT(rel_max(sym.gmnc, asym.gmnc), kTol) << "gmnc";
  EXPECT_LT(rel_max(sym.bsubumnc, asym.bsubumnc), kTol) << "bsubumnc";
  EXPECT_LT(rel_max(sym.bsubvmnc, asym.bsubvmnc), kTol) << "bsubvmnc";
  EXPECT_LT(rel_max(sym.bsupumnc, asym.bsupumnc), kTol) << "bsupumnc";
  EXPECT_LT(rel_max(sym.bsupvmnc, asym.bsupvmnc), kTol) << "bsupvmnc";
  EXPECT_LT(rel_max(sym.bsubsmns, asym.bsubsmns), kTol) << "bsubsmns";

  // currumnc/currvmnc are radial finite differences of bsubu/vmnc and so carry
  // more cancellation noise; a looser but still tight bound.
  EXPECT_LT(rel_max(sym.currumnc, asym.currumnc), 1.0e-8) << "currumnc";
  EXPECT_LT(rel_max(sym.currvmnc, asym.currvmnc), 1.0e-8) << "currvmnc";

  // The lasym run carries no asymmetry, so every antisymmetric coefficient set
  // vanishes to machine precision.
  EXPECT_LT(rel_zero(asym.rmns, sym.rmnc), kTol) << "rmns";
  EXPECT_LT(rel_zero(asym.zmnc, sym.zmns), kTol) << "zmnc";
  EXPECT_LT(rel_zero(asym.bmns, sym.bmnc), kTol) << "bmns";
  EXPECT_LT(rel_zero(asym.gmns, sym.gmnc), kTol) << "gmns";
  EXPECT_LT(rel_zero(asym.bsubumns, sym.bsubumnc), kTol) << "bsubumns";
  EXPECT_LT(rel_zero(asym.bsubvmns, sym.bsubvmnc), kTol) << "bsubvmns";
  EXPECT_LT(rel_zero(asym.bsupumns, sym.bsupumnc), kTol) << "bsupumns";
  EXPECT_LT(rel_zero(asym.bsupvmns, sym.bsupvmnc), kTol) << "bsupvmns";
  EXPECT_LT(rel_zero(asym.bsubsmnc, sym.bsubsmns), kTol) << "bsubsmnc";
}  // LasymAxisymmetricDegeneratesToSymmetric

// ---- the (lasym, lthreed, ncurr, lfreeb) matrix ----------------------------
//
// vmecpp_large_cpp_tests/test_data/README.md asks for every combination of
// free boundary, constrained current, three-dimensionality and
// non-stellarator-symmetry to be covered. The cases below fill in four
// fixed-boundary cells that had none. Each was run through educational_VMEC
// (VMEC 8.52) to convergence with no Jacobian resets, and the scalars here are
// that run's output.

namespace {

struct FlagCombinationCase {
  const char* filename;
  bool lasym;
  bool lthreed;
  int ncurr;
  double aspect;
  double volume;
  double Rmajor_p;
  double Aminor_p;
  double rbtor;
  double rbtor0;
  double volavgB;
};

void CheckFlagCombinationAgainstEducationalVmec(const FlagCombinationCase& c) {
  absl::StatusOr<std::string> indata_json = ReadFile(c.filename);
  ASSERT_TRUE(indata_json.ok());
  absl::StatusOr<VmecINDATA> indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(indata.ok());

  // The case is only worth anything if it really sits in the cell it claims.
  ASSERT_EQ(indata->lasym, c.lasym);
  ASSERT_EQ(indata->ntor > 0, c.lthreed);
  ASSERT_EQ(indata->ncurr, c.ncurr);
  ASSERT_FALSE(indata->lfreeb);

  const auto output = vmecpp::run(*indata);
  ASSERT_TRUE(output.ok()) << output.status();
  const auto& w = output->wout;

  const double tol = 1.0e-5;
  EXPECT_TRUE(IsCloseRelAbs(c.aspect, w.aspect, tol)) << "aspect=" << w.aspect;
  EXPECT_TRUE(IsCloseRelAbs(c.volume, w.volume, tol)) << "volume=" << w.volume;
  EXPECT_TRUE(IsCloseRelAbs(c.Rmajor_p, w.Rmajor_p, tol))
      << "Rmajor=" << w.Rmajor_p;
  EXPECT_TRUE(IsCloseRelAbs(c.Aminor_p, w.Aminor_p, tol))
      << "Aminor=" << w.Aminor_p;
  EXPECT_TRUE(IsCloseRelAbs(c.rbtor, w.rbtor, tol)) << "rbtor=" << w.rbtor;
  EXPECT_TRUE(IsCloseRelAbs(c.rbtor0, w.rbtor0, tol)) << "rbtor0=" << w.rbtor0;
  EXPECT_TRUE(IsCloseRelAbs(c.volavgB, w.volavgB, tol))
      << "volavgB=" << w.volavgB;

  EXPECT_EQ(w.lasym, c.lasym);
  if (c.lasym) {
    EXPECT_GT(w.rmns.cwiseAbs().maxCoeff(), 1.0e-3) << "rmns must be non-zero";
    EXPECT_GT(w.zmnc.cwiseAbs().maxCoeff(), 1.0e-3) << "zmnc must be non-zero";
  }
}

// With ncurr = 0 the rotational transform is prescribed rather than solved for,
// so the half-grid profile has to be the ai polynomial evaluated at the
// half-grid flux positions. Index 0 of iotas is unused.
void CheckPrescribedIotaProfile(const vmecpp::WOutFileContents& w,
                                const Eigen::VectorXd& ai) {
  ASSERT_GT(w.ns, 1);
  ASSERT_EQ(w.iotas.size(), w.ns);
  const double delta_s = 1.0 / (w.ns - 1.0);
  for (int j = 1; j < w.ns; ++j) {
    const double s = (j - 0.5) * delta_s;
    double expected = 0.0;
    for (Eigen::Index i = ai.size() - 1; i >= 0; --i) {
      expected = s * expected + ai[i];
    }
    EXPECT_NEAR(w.iotas[j], expected, 1.0e-12) << "iotas at j=" << j;
  }
}

}  // namespace

// lasym = F, lthreed = T, ncurr = 0, lfreeb = F.
// Every other three-dimensional case in the suite constrains the current, so
// this is the only place the constrained-iota path runs in 3D.
TEST(TestVmec, ThreeDimensionalConstrainedIotaMatchesEducationalVmec) {
  const FlagCombinationCase kCase = {
      /*filename=*/"vmecpp/test_data/cth_like_fixed_bdy_iota.json",
      /*lasym=*/false,
      /*lthreed=*/true,
      /*ncurr=*/0,
      /*aspect=*/5.48582995348779,
      /*volume=*/0.315396710123312,
      /*Rmajor_p=*/0.783436919078514,
      /*Aminor_p=*/0.142811010498132,
      /*rbtor=*/-0.451317819328199,
      /*rbtor0=*/-0.461019928760768,
      /*volavgB=*/0.5666316420317945};
  CheckFlagCombinationAgainstEducationalVmec(kCase);

  absl::StatusOr<std::string> indata_json = ReadFile(kCase.filename);
  ASSERT_TRUE(indata_json.ok());
  absl::StatusOr<VmecINDATA> indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(indata.ok());
  const auto output = vmecpp::run(*indata);
  ASSERT_TRUE(output.ok()) << output.status();
  CheckPrescribedIotaProfile(output->wout, indata->ai);
}

// lasym = T, lthreed = F, ncurr = 1, lfreeb = F.
TEST(TestVmec, AsymmetricTokamakConstrainedCurrentMatchesEducationalVmec) {
  const FlagCombinationCase kCase = {
      /*filename=*/"vmecpp/test_data/up_down_asym_current.json",
      /*lasym=*/true,
      /*lthreed=*/false,
      /*ncurr=*/1,
      /*aspect=*/10.1,
      /*volume=*/43.0630579228331,
      /*Rmajor_p=*/6.06,
      /*Aminor_p=*/0.6,
      /*rbtor=*/31.9581912180759,
      /*rbtor0=*/32.0674722883524,
      /*volavgB=*/5.29648456676457};
  CheckFlagCombinationAgainstEducationalVmec(kCase);
}

// lasym = T, lthreed = T, ncurr = 1, lfreeb = F.
TEST(TestVmec, AsymmetricStellaratorConstrainedCurrentMatchesEducationalVmec) {
  const FlagCombinationCase kCase = {
      /*filename=*/"vmecpp/test_data/cth_like_fixed_bdy_asym.json",
      /*lasym=*/true,
      /*lthreed=*/true,
      /*ncurr=*/1,
      /*aspect=*/5.51459955489889,
      /*volume=*/0.311959643944142,
      /*Rmajor_p=*/0.783307369710139,
      /*Aminor_p=*/0.142042475054111,
      /*rbtor=*/-0.456903029831871,
      /*rbtor0=*/-0.466702565409973,
      /*volavgB=*/0.573224626419241};
  CheckFlagCombinationAgainstEducationalVmec(kCase);
}

// lasym = T, lthreed = T, ncurr = 0, lfreeb = F.
TEST(TestVmec, AsymmetricStellaratorConstrainedIotaMatchesEducationalVmec) {
  const FlagCombinationCase kCase = {
      /*filename=*/"vmecpp/test_data/cth_like_fixed_bdy_asym_iota.json",
      /*lasym=*/true,
      /*lthreed=*/true,
      /*ncurr=*/0,
      /*aspect=*/5.51459955489889,
      /*volume=*/0.311959643944142,
      /*Rmajor_p=*/0.783307369710139,
      /*Aminor_p=*/0.142042475054111,
      /*rbtor=*/-0.456526156320673,
      /*rbtor0=*/-0.466540212041614,
      /*volavgB=*/0.573164827790394};
  CheckFlagCombinationAgainstEducationalVmec(kCase);
}

// A genuinely up-down-asymmetric tokamak (lasym=true with nonzero rbs): the
// converged equilibrium must match a VMEC 8.52 (educational_VMEC) reference for
// the same input. The reference scalars are taken from threed1.up_down_asym,
// produced by running xvmec on the equivalent INDATA file. This is the
// non-degenerate validation of the antisymmetric physics: the axis is pushed
// off the midplane by the asymmetry, which a symmetric run could not produce.
TEST(TestVmec, LasymAxisymmetricTokamakMatchesEducationalVmec) {
  const std::string filename = "vmecpp/test_data/up_down_asym.json";
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  absl::StatusOr<VmecINDATA> indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(indata.ok());
  ASSERT_TRUE(indata->lasym);

  // This genuinely up-down-asymmetric equilibrium converges to the 1e-11 force
  // tolerance with no accept-unconverged escape hatch, and the converged result
  // is identical across 1..N OpenMP threads. Validate it against the VMEC 8.52
  // (educational_VMEC) reference on three fronts: the integrated scalars, the
  // antisymmetric Fourier content that only an up-down-asymmetric equilibrium
  // produces, and the magnetic axis pushed off the midplane by that asymmetry.
  const auto output = vmecpp::run(*indata);
  ASSERT_TRUE(output.ok());
  const auto& w = output->wout;
  ASSERT_TRUE(w.lasym);

  // educational_VMEC (VMEC 8.52) golden scalars. The reference run converges to
  // the same 1e-11 tolerance; VMEC++ reproduces these to <= 1e-5 (an order
  // tighter than the previous 1e-4 bound), the largest residual being rbtor at
  // ~5e-6.
  const double tol = 1.0e-5;
  EXPECT_TRUE(IsCloseRelAbs(10.100000, w.aspect, tol)) << "aspect=" << w.aspect;
  EXPECT_TRUE(IsCloseRelAbs(43.063058, w.volume, tol)) << "volume=" << w.volume;
  EXPECT_TRUE(IsCloseRelAbs(6.060000, w.Rmajor_p, tol))
      << "Rmajor=" << w.Rmajor_p;
  EXPECT_TRUE(IsCloseRelAbs(0.600000, w.Aminor_p, tol))
      << "Aminor=" << w.Aminor_p;
  EXPECT_TRUE(IsCloseRelAbs(31.958251, w.rbtor, tol)) << "rbtor=" << w.rbtor;
  EXPECT_TRUE(IsCloseRelAbs(32.071235, w.rbtor0, tol)) << "rbtor0=" << w.rbtor0;
  EXPECT_TRUE(IsCloseRelAbs(5.296540, w.volavgB, tol))
      << "volavgB=" << w.volavgB;

  // The field on axis must be the physical on-axis value F/R = rbtor0 / Raxis,
  // not the doubled lasym Nyquist normalization. This guards the tmult = 0.5
  // output normalization: a regression to the doubled value would put b0 at ~2x
  // and fail here.
  ASSERT_GT(w.raxis_cc.size(), 0);
  EXPECT_TRUE(IsCloseRelAbs(w.rbtor0 / w.raxis_cc[0], w.b0, 1.0e-3))
      << "b0=" << w.b0 << " rbtor0/Raxis=" << (w.rbtor0 / w.raxis_cc[0]);

  // Genuine asymmetry: every antisymmetric wout array is sized and carries
  // content. A stellarator-symmetric run leaves all of them empty, so the size
  // and the floor together pin each array to the asymmetric solve. The floors
  // sit one to two orders below the values this equilibrium produces.
  auto expect_asym = [&w](const vmecpp::RowMatrixXd& a, int rows, double floor,
                          const char* name) {
    EXPECT_EQ(a.rows(), rows) << name;
    EXPECT_EQ(a.cols(), w.ns) << name;
    if (a.size() > 0) {
      EXPECT_GT(a.cwiseAbs().maxCoeff(), floor) << name << " must be non-zero";
    }
  };
  expect_asym(w.rmns, w.mnmax, 1.0e-3, "rmns");
  expect_asym(w.zmnc, w.mnmax, 1.0e-3, "zmnc");
  expect_asym(w.lmnc_full, w.mnmax, 1.0e-2, "lmnc_full");
  expect_asym(w.lmnc, w.mnmax, 1.0e-2, "lmnc");
  expect_asym(w.gmns, w.mnmax_nyq, 1.0e-2, "gmns");
  expect_asym(w.bmns, w.mnmax_nyq, 1.0e-2, "bmns");
  expect_asym(w.bsubumns, w.mnmax_nyq, 1.0e-2, "bsubumns");
  // B_zeta is the near-vacuum 1/R field of a tokamak and so is almost up-down
  // symmetric; its antisymmetric half is four orders below the rest.
  expect_asym(w.bsubvmns, w.mnmax_nyq, 1.0e-6, "bsubvmns");
  expect_asym(w.bsubsmnc, w.mnmax_nyq, 1.0e-3, "bsubsmnc");
  expect_asym(w.bsubsmnc_full, w.mnmax_nyq, 1.0e-3, "bsubsmnc_full");
  expect_asym(w.bsupumns, w.mnmax_nyq, 1.0e-2, "bsupumns");
  expect_asym(w.bsupvmns, w.mnmax_nyq, 1.0e-3, "bsupvmns");
  expect_asym(w.currumns, w.mnmax_nyq, 1.0, "currumns");
  expect_asym(w.currvmns, w.mnmax_nyq, 1.0, "currvmns");

  // raxis_cs multiplies sin(n * zeta) and this case has ntor = 0, so it is
  // structurally zero rather than merely small.
  ASSERT_EQ(w.raxis_cs.size(), 1);
  EXPECT_EQ(w.raxis_cs[0], 0.0) << "raxis_cs is sin(0 * zeta)";

  // The asymmetry pushes the magnetic axis off the midplane: zaxis_cc is the
  // cos(n*zeta) (here n=0) antisymmetric axis amplitude, exactly zero for a
  // symmetric equilibrium. educational_VMEC places it at Z = 0.078295; VMEC++
  // agrees to better than 1%.
  ASSERT_GT(w.zaxis_cc.size(), 0);
  EXPECT_GT(w.zaxis_cc[0], 0.05) << "axis must be off-midplane";
  EXPECT_TRUE(IsCloseRelAbs(0.0782953, w.zaxis_cc[0], 5.0e-2))
      << "zaxis_cc[0]=" << w.zaxis_cc[0];
}  // LasymAxisymmetricTokamakMatchesEducationalVmec

// A stellarator-symmetric 3D equilibrium (cth_like_fixed_bdy) run through the
// lasym path with zero antisymmetric content must reproduce the symmetric
// equilibrium. This exercises the 3D antisymmetric inverse/forward DFTs and the
// toroidal (zeta) reflection in symrzl / symforce, the 3D analog of the
// axisymmetric degenerate test above.
TEST(TestVmec, Lasym3DDegeneratesToSymmetric) {
  const std::string filename = "vmecpp/test_data/cth_like_fixed_bdy.json";
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  absl::StatusOr<VmecINDATA> indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(indata.ok());
  ASSERT_FALSE(indata->lasym);

  const auto symmetric_output = vmecpp::run(*indata);
  ASSERT_TRUE(symmetric_output.ok());

  VmecINDATA asym_indata = *indata;
  asym_indata.lasym = true;
  auto zero_rbs = indata->rbc;
  zero_rbs.setZero();
  auto zero_zbc = indata->zbs;
  zero_zbc.setZero();
  asym_indata.rbs = zero_rbs;
  asym_indata.zbc = zero_zbc;
  auto zero_raxis_s = indata->raxis_c;
  zero_raxis_s.setZero();
  auto zero_zaxis_c = indata->zaxis_s;
  zero_zaxis_c.setZero();
  asym_indata.raxis_s = zero_raxis_s;
  asym_indata.zaxis_c = zero_zaxis_c;

  const auto asymmetric_output = vmecpp::run(asym_indata);
  ASSERT_TRUE(asymmetric_output.ok());

  const auto& sym = symmetric_output->wout;
  const auto& asym = asymmetric_output->wout;
  const double tol = 1.0e-9;
  ASSERT_EQ(asym.ns, sym.ns);
  EXPECT_TRUE(IsCloseRelAbs(sym.wb, asym.wb, tol)) << "wb";
  EXPECT_TRUE(IsCloseRelAbs(sym.volume, asym.volume, tol)) << "volume";
  EXPECT_TRUE(IsCloseRelAbs(sym.aspect, asym.aspect, tol)) << "aspect";
  EXPECT_TRUE(IsCloseRelAbs(sym.betatotal, asym.betatotal, tol)) << "betatotal";
  EXPECT_TRUE(IsCloseRelAbs(sym.b0, asym.b0, 1.0e-3)) << "b0";
  EXPECT_TRUE(IsCloseRelAbs(sym.Aminor_p, asym.Aminor_p, tol)) << "Aminor_p";
  EXPECT_TRUE(IsCloseRelAbs(sym.Rmajor_p, asym.Rmajor_p, tol)) << "Rmajor_p";

  // The converged Fourier geometry must reproduce the symmetric run; the
  // antisymmetric arrays remain at the null-space noise floor.
  EXPECT_LT((asym.rmnc - sym.rmnc).cwiseAbs().maxCoeff(), 1.0e-9) << "rmnc";
  EXPECT_LT((asym.zmns - sym.zmns).cwiseAbs().maxCoeff(), 1.0e-9) << "zmns";
}  // Lasym3DDegeneratesToSymmetric

// Free-boundary 3D analog: the non-stellarator-symmetric (lasym=true) path with
// zero antisymmetric content must reproduce the symmetric free-boundary
// equilibrium. This exercises the antisymmetric free-boundary NESTOR vacuum
// solver over the full poloidal range: the singular-integral boundary source,
// the cos-basis matrix projection, and the real-space vacuum field used for the
// free-boundary pressure balance.
TEST(TestVmec, Lasym3DFreeBoundaryDegeneratesToSymmetric) {
  const std::string filename = "vmecpp/test_data/cth_like_free_bdy.json";
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  absl::StatusOr<VmecINDATA> indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(indata.ok());
  ASSERT_FALSE(indata->lasym);
  ASSERT_TRUE(indata->lfreeb);

  const auto symmetric_output = vmecpp::run(*indata);
  ASSERT_TRUE(symmetric_output.ok());

  VmecINDATA asym_indata = *indata;
  asym_indata.lasym = true;
  auto zero_rbs = indata->rbc;
  zero_rbs.setZero();
  auto zero_zbc = indata->zbs;
  zero_zbc.setZero();
  asym_indata.rbs = zero_rbs;
  asym_indata.zbc = zero_zbc;
  auto zero_raxis_s = indata->raxis_c;
  zero_raxis_s.setZero();
  auto zero_zaxis_c = indata->zaxis_s;
  zero_zaxis_c.setZero();
  asym_indata.raxis_s = zero_raxis_s;
  asym_indata.zaxis_c = zero_zaxis_c;

  const auto asymmetric_output = vmecpp::run(asym_indata);
  ASSERT_TRUE(asymmetric_output.ok());

  const auto& sym = symmetric_output->wout;
  const auto& asym = asymmetric_output->wout;
  // The free-boundary lasym path runs the vacuum solve through the doubled
  // (sin/cos) response matrix and the symrzl full-range fold, which accumulate
  // more round-off than the fixed-boundary degenerate case (1e-9): the
  // degenerate equilibrium reproduces the symmetric one to ~5e-9. The tolerance
  // is 1e-7, matching the free-boundary mgrid comparison in CheckInMemoryMgrid
  // and three orders below the genuinely-asymmetric physics validation (1e-4).
  const double tol = 1.0e-7;
  ASSERT_EQ(asym.ns, sym.ns);
  EXPECT_TRUE(IsCloseRelAbs(sym.wb, asym.wb, tol)) << "wb";
  EXPECT_TRUE(IsCloseRelAbs(sym.volume, asym.volume, tol)) << "volume";
  EXPECT_TRUE(IsCloseRelAbs(sym.aspect, asym.aspect, tol)) << "aspect";
  EXPECT_TRUE(IsCloseRelAbs(sym.betatotal, asym.betatotal, tol)) << "betatotal";
  EXPECT_TRUE(IsCloseRelAbs(sym.b0, asym.b0, 1.0e-3)) << "b0";
  EXPECT_TRUE(IsCloseRelAbs(sym.Aminor_p, asym.Aminor_p, tol)) << "Aminor_p";
  EXPECT_TRUE(IsCloseRelAbs(sym.Rmajor_p, asym.Rmajor_p, tol)) << "Rmajor_p";

  // The converged Fourier geometry must reproduce the symmetric run; the
  // antisymmetric arrays remain at the null-space noise floor.
  EXPECT_LT((asym.rmnc - sym.rmnc).cwiseAbs().maxCoeff(), tol) << "rmnc";
  EXPECT_LT((asym.zmns - sym.zmns).cwiseAbs().maxCoeff(), tol) << "zmns";
}  // Lasym3DFreeBoundaryDegeneratesToSymmetric

// A genuinely up-down-asymmetric free-boundary equilibrium validated against
// educational_VMEC (VMEC 8.52). The stellarator-symmetric cth_like external
// field (mgrid_cth_like) is perturbed by a small up-down-asymmetry-breaking
// vacuum field built from the flux psi = c (R^4 Z - 4/3 R^2 Z^3 - Rc^2 R^2 Z),
// which is divergence- and curl-free (Delta* psi = 0) and a shaping (not
// vertical-shift) term, so the free-boundary equilibrium stays vertically
// stable. The same perturbed mgrid (mgrid_cth_like_asym) was run through xvmec;
// the reference scalars below are from that run. This is the non-degenerate
// validation of the antisymmetric NESTOR free-boundary path: the vacuum solver
// runs with genuinely non-zero antisymmetric content.
TEST(TestVmec, ZeroMaximumMultiGridStepIsRejected) {
  // Without the guard the multigrid loop runs no steps at all and run()
  // returns false with an OK status, so nothing is solved and nothing says so.
  const absl::StatusOr<std::string> indata_json =
      ReadFile("vmecpp/test_data/solovev.json");
  ASSERT_TRUE(indata_json.ok());
  const absl::StatusOr<VmecINDATA> indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(indata.ok());

  Vmec vmec(*indata);
  const absl::StatusOr<bool> reached =
      vmec.run(VmecCheckpoint::NONE, INT_MAX, /*maximum_multi_grid_step=*/0);
  ASSERT_FALSE(reached.ok());
  EXPECT_EQ(reached.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(TestVmec, LasymFreeBoundaryMatchesEducationalVmec) {
  const std::string filename = "vmecpp/test_data/cth_like_free_bdy_asym.json";
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  absl::StatusOr<VmecINDATA> indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(indata.ok());
  ASSERT_TRUE(indata->lasym);
  ASSERT_TRUE(indata->lfreeb);

  // The antisymmetric free-boundary force residual limit-cycles near the 1e-8
  // tolerance; accept the equilibrium at that level (as the axisymmetric
  // reference test does) and validate the physics against the reference
  // scalars.
  indata->return_outputs_even_if_not_converged = true;

  const auto output = vmecpp::run(*indata);
  ASSERT_TRUE(output.ok());
  const auto& w = output->wout;
  ASSERT_TRUE(w.lasym);

  // educational_VMEC (VMEC 8.52) golden scalars for the identical perturbed
  // mgrid.
  const double tol = 1.0e-4;
  EXPECT_TRUE(IsCloseRelAbs(5.4351302689, w.aspect, tol))
      << "aspect=" << w.aspect;
  EXPECT_TRUE(IsCloseRelAbs(0.3073676511, w.volume, tol))
      << "volume=" << w.volume;
  EXPECT_TRUE(IsCloseRelAbs(0.7719386349, w.Rmajor_p, tol))
      << "Rmajor=" << w.Rmajor_p;
  EXPECT_TRUE(IsCloseRelAbs(0.1420276234, w.Aminor_p, tol))
      << "Aminor=" << w.Aminor_p;
  EXPECT_TRUE(IsCloseRelAbs(0.0018738865, w.betatotal, tol))
      << "beta=" << w.betatotal;
  EXPECT_TRUE(IsCloseRelAbs(-0.4512430727, w.rbtor, tol))
      << "rbtor=" << w.rbtor;
  EXPECT_TRUE(IsCloseRelAbs(0.5742222261, w.volavgB, tol))
      << "volavgB=" << w.volavgB;

  // Genuine asymmetry: the antisymmetric Fourier content is clearly non-zero.
  EXPECT_GT(w.rmns.cwiseAbs().maxCoeff(), 1.0e-4) << "rmns must be non-zero";
  EXPECT_GT(w.zmnc.cwiseAbs().maxCoeff(), 1.0e-4) << "zmnc must be non-zero";
}  // LasymFreeBoundaryMatchesEducationalVmec

// A multi-grid free-boundary equilibrium (cth_like_free_bdy with an added grid
// step) must converge. The free-boundary (Nestor) solver, together with its
// accumulated vacuum response matrix and right-hand side, is kept in memory
// across the multi-grid steps (reproducing Fortran VMEC's persistent vacuum
// state), so this also exercises that the reused solver state stays valid
// across a grid-size change. The step-by-step agreement against the Fortran
// reference is exercised in vmecpp_large_cpp_tests.
TEST(TestVmec, MultiGridFreeBoundary) {
  const std::string filename =
      "vmecpp/test_data/cth_like_free_bdy_multigrid.json";
  const absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(indata.ok());
  ASSERT_EQ(indata->ns_array.size(), 2u);

  const auto output = vmecpp::run(*indata, std::nullopt, 1);
  ASSERT_TRUE(output.ok());

  // Regression guard for issue #330/#640 and other changes to the multigrid
  // convergence path. 344 with the historical unbalanced stage entry; 321
  // since the vacuum state is seeded across multigrid transitions (the
  // second stage enters force-balanced instead of kicking the boundary).
  EXPECT_EQ(output->wout.niter, 321);
}  // MultiGridFreeBoundary
