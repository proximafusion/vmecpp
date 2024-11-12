
#include <algorithm>
#include <fstream>
#include <string>

#include "vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.h"

#ifdef _OPENMP
#include <omp.h>
#endif  // _OPENMP

#include "absl/strings/str_format.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "util/file_io/file_io.h"
#include "util/testing/numerical_comparison_lib.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/vmec/analytical_solovev_model/analytical_solovev_model.h"
#include "vmecpp/vmec/vmec/vmec.h"

using nlohmann::json;

using file_io::ReadFile;
using testing::IsCloseRelAbs;

using ::testing::TestWithParam;
using ::testing::Values;

namespace vmecpp {

class IdealMhdModelTestground {
 public:
  explicit IdealMhdModelTestground(const VmecINDATA &indata)
      : ivac(0),
        fc(indata.lfreeb, indata.delt,
           static_cast<int>(indata.ns_array.size()) + 1),
        s(indata),
        t(&s),
        h(&s),
        p(&rp, &h, &indata, &fc, signOfJacobian, pDamp),
        b(&s, &t, signOfJacobian),
        ls(&s),
        m(&fc, &s, &t, &p, &b, &constants, &ls, &h, &rp, nullptr,
          signOfJacobian, nvacskip, &ivac),
        decomposed_x(&s, &rp, indata.ns_array[0]),
        physical_x(&s, &rp, indata.ns_array[0]),
        decomposed_f(&s, &rp, indata.ns_array[0]),
        physical_f(&s, &rp, indata.ns_array[0]) {
    fc.ns = indata.ns_array[0];
    fc.deltaS = 1.0 / (fc.ns - 1.0);

    rp.adjustRadialPartitioning(num_threads, thread_id, fc.ns, indata.lfreeb,
                                false);

    h.allocate(rp, fc.ns);

    p.setupInputProfiles();

    fc.haveToFlipTheta = b.setupFromIndata(indata);

    m.setFromINDATA(indata.ncurr, indata.gamma, indata.tcon0);

    decomposed_x.setZero();

    p.evalRadialProfiles(fc.haveToFlipTheta, thread_id, constants);
    constants.lamscale = sqrt(constants.rmsPhiP * fc.deltaS);
  }  // constructor

  void PerformUpdate(VmecCheckpoint checkpoint = VmecCheckpoint::NONE,
                     int maximum_iterations = INT_MAX) {
    bool need_restart = false;
    int last_preconditioner_update = 0;
    int last_full_update_nestor = 0;
    const int iter1 = 0;
    const int iter2 = 0;

    absl::StatusOr<bool> maybe_reached_checkpoint = m.update(
        decomposed_x, physical_x, h, decomposed_f, physical_f, need_restart,
        last_preconditioner_update, last_full_update_nestor, rp, fc, thread_id,
        iter1, iter2, checkpoint, maximum_iterations);

    ASSERT_TRUE(maybe_reached_checkpoint.ok())
        << maybe_reached_checkpoint.status().message();
    const bool reached_checkpoint = maybe_reached_checkpoint.value();
    if (checkpoint == VmecCheckpoint::NONE) {
      EXPECT_FALSE(reached_checkpoint);
    } else {
      EXPECT_TRUE(reached_checkpoint);
    }
  }  // PerformUpdate

  static constexpr int num_threads = 1;
  static constexpr int thread_id = 0;

  static constexpr int signOfJacobian = -1;
  static constexpr double pDamp = 0.05;

  static constexpr int nvacskip = 0;
  int ivac;

  FlowControl fc;
  Sizes s;
  FourierBasisFastPoloidal t;
  RadialPartitioning rp;
  HandoverStorage h;
  RadialProfiles p;
  Boundaries b;
  VmecConstants constants;
  ThreadLocalStorage ls;
  IdealMhdModel m;
  FourierGeometry decomposed_x;
  FourierGeometry physical_x;
  FourierForces decomposed_f;
  FourierForces physical_f;
};  // IdealMhdModelTestground

TEST(TestVmecAgainstAnalyticalSolovev, CheckModelSetup) {
  static constexpr double kTolerance = 1.0e-3;

  AnalyticalSolovevModel model;
  IdealMhdModelTestground tg(model.GetIndata());

  // Note: In next, actual test, intercept here and inject
  // geometry directly from AnalyticalSolovevModel.
  // -> This test checks that the rest of the setup works.
  tg.decomposed_x.interpFromBoundaryAndAxis(tg.t, tg.b, tg.p);

  tg.PerformUpdate();

  EXPECT_TRUE(IsCloseRelAbs(1.57e-1, tg.fc.fsqr, kTolerance));
  EXPECT_TRUE(IsCloseRelAbs(3.82e-3, tg.fc.fsqz, kTolerance));
  EXPECT_TRUE(IsCloseRelAbs(5.99e-2, tg.fc.fsql, kTolerance));
}  // CheckModelSetup

TEST(TestVmecAgainstAnalyticalSolovev, CheckEquilibriumFourierGeometry) {
  static constexpr double kTolerance = 1.0e-3;

  AnalyticalSolovevModel model;
  IdealMhdModelTestground tg(model.GetIndata());

  model.IntoFourierGeometry(tg.s, tg.t, tg.rp, tg.p, tg.constants,
                            /*m_fourier_geometry=*/tg.decomposed_x);

  tg.PerformUpdate();

  // Note that these forces are a lot lower than with the VMEC initial guess,
  // but nowhere near where VMEC pushes them after a few iterations,
  // even though we feed the perfect equilibrium geometry to VMEC.
  // -> Something to do for @jons...
  EXPECT_TRUE(IsCloseRelAbs(1.93e-4, tg.fc.fsqr, kTolerance));
  EXPECT_TRUE(IsCloseRelAbs(3.17e-4, tg.fc.fsqz, kTolerance));
  EXPECT_TRUE(IsCloseRelAbs(3.09e-9, tg.fc.fsql, kTolerance));
}  // CheckEquilibriumFourierGeometry

TEST(TestVmecAgainstAnalyticalSolovev,
     CheckGeometryAgainstAnalyticalEquilibrium) {
  // full-grid quantities
  static constexpr double kToleranceF = 5.0e-8;

  // half-grid quantities
  static constexpr double kToleranceH = 1.0e-4;

  // radial derivatives and Jacobian ingredients
  static constexpr double kToleranceJac = 5.0e-4;

  AnalyticalSolovevModel model;
  IdealMhdModelTestground tg(model.GetIndata());

  model.IntoFourierGeometry(tg.s, tg.t, tg.rp, tg.p, tg.constants,
                            /*m_fourier_geometry=*/tg.decomposed_x);

  tg.PerformUpdate(VmecCheckpoint::JACOBIAN, 0);

  const double delta_theta = 2.0 * M_PI / tg.s.nThetaEven;

  // full-grid geometry from inverse-DFT
  for (int jF = tg.rp.nsMinF1; jF < tg.rp.nsMaxF1; ++jF) {
    const double rhoF = tg.p.sqrtSF[jF - tg.rp.nsMinF1];
    for (int l = 0; l < tg.s.nThetaEff; ++l) {
      const int idx_kl = (jF - tg.rp.nsMinF1) * tg.s.nThetaEff + l;
      const double theta = l * delta_theta;

      const double r = tg.m.r1_e[idx_kl] + rhoF * tg.m.r1_o[idx_kl];
      EXPECT_TRUE(IsCloseRelAbs(model.R(rhoF, theta), r, kToleranceF));

      const double z = tg.m.z1_e[idx_kl] + rhoF * tg.m.z1_o[idx_kl];
      EXPECT_TRUE(IsCloseRelAbs(model.Z(rhoF, theta), z, kToleranceF));

      const double ru = tg.m.ru_e[idx_kl] + rhoF * tg.m.ru_o[idx_kl];
      EXPECT_TRUE(IsCloseRelAbs(model.DRDTheta(rhoF, theta), ru, kToleranceF));

      const double zu = tg.m.zu_e[idx_kl] + rhoF * tg.m.zu_o[idx_kl];
      EXPECT_TRUE(IsCloseRelAbs(model.DZDTheta(rhoF, theta), zu, kToleranceF));
    }  // l
  }    // jF

  // half-grid geometry from Jacobian computation
  for (int jH = tg.rp.nsMinH; jH < tg.rp.nsMaxH; ++jH) {
    const int jFi = jH;
    const int jFo = jH + 1;

    const double rhoH = tg.p.sqrtSH[jH - tg.rp.nsMinH];
    for (int l = 0; l < tg.s.nThetaEff; ++l) {
      const int idx_kl = (jH - tg.rp.nsMinH) * tg.s.nThetaEff + l;
      const int idx_kl_i = (jFi - tg.rp.nsMinF1) * tg.s.nThetaEff + l;
      const int idx_kl_o = (jFo - tg.rp.nsMinF1) * tg.s.nThetaEff + l;
      const double theta = l * delta_theta;

      EXPECT_TRUE(
          IsCloseRelAbs(model.R(rhoH, theta), tg.m.r12[idx_kl], kToleranceH));
      EXPECT_TRUE(IsCloseRelAbs(model.DRDTheta(rhoH, theta), tg.m.ru12[idx_kl],
                                kToleranceH));
      EXPECT_TRUE(IsCloseRelAbs(model.DZDTheta(rhoH, theta), tg.m.zu12[idx_kl],
                                kToleranceH));

      const double r1_o = 0.5 * (tg.m.r1_o[idx_kl_o] + tg.m.r1_o[idx_kl_i]);
      const double rs = tg.m.rs[idx_kl] + 0.5 / rhoH * r1_o;
      EXPECT_TRUE(IsCloseRelAbs(model.DRDS(rhoH, theta), rs, kToleranceJac));

      const double z1_o = 0.5 * (tg.m.z1_o[idx_kl_o] + tg.m.z1_o[idx_kl_i]);
      const double zs = tg.m.zs[idx_kl] + 0.5 / rhoH * z1_o;
      EXPECT_TRUE(IsCloseRelAbs(model.DZDS(rhoH, theta), zs, kToleranceJac));

      // In IdealMHDModel, sqrtG is only computed along with the metric
      // coefficients, but we don't want to run the code that far yet.
      const double sqrtG = tg.m.r12[idx_kl] * tg.m.tau[idx_kl];

      // Note that the Jacobian in the analytical model is for rho as radial
      // coordinate, whereas VMEC uses s = rho^2 (for the iota=2 choice of
      // beta_2) as radial coordinate. Hence, we need to apply the chain rule
      // factor 1/(2 * rho) to the analytical Jacobian.
      EXPECT_TRUE(IsCloseRelAbs(model.Jacobian(rhoH) * 0.5 / rhoH, sqrtG,
                                kToleranceJac));
    }  // l
  }    // jH
}  // CheckGeometryAgainstAnalyticalEquilibrium

}  // namespace vmecpp
