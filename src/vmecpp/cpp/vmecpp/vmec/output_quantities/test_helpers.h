// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_OUTPUT_QUANTITIES_TEST_HELPERS_H_
#define VMECPP_VMEC_OUTPUT_QUANTITIES_TEST_HELPERS_H_

#include "gtest/gtest.h"
#include "util/testing/numerical_comparison_lib.h"
#include "vmecpp/vmec/output_quantities/output_quantities.h"

// check for equality of two `WOutFileContents` using the Google test framework
inline void CheckWOutEquality(const vmecpp::WOutFileContents& wout1,
                              const vmecpp::WOutFileContents& wout2) {
  EXPECT_EQ(wout1.version_, wout2.version_);
  EXPECT_EQ(wout1.signgs, wout2.signgs);
  EXPECT_EQ(wout1.gamma, wout2.gamma);
  EXPECT_EQ(wout1.pcurr_type, wout2.pcurr_type);
  EXPECT_EQ(wout1.pmass_type, wout2.pmass_type);
  EXPECT_EQ(wout1.piota_type, wout2.piota_type);
  EXPECT_EQ(wout1.am, wout2.am);
  EXPECT_EQ(wout1.ac, wout2.ac);
  EXPECT_EQ(wout1.ai, wout2.ai);
  EXPECT_EQ(wout1.am_aux_s, wout2.am_aux_s);
  EXPECT_EQ(wout1.am_aux_f, wout2.am_aux_f);
  EXPECT_EQ(wout1.ac_aux_s, wout2.ac_aux_s);
  EXPECT_EQ(wout1.ac_aux_f, wout2.ac_aux_f);
  EXPECT_EQ(wout1.ai_aux_s, wout2.ai_aux_s);
  EXPECT_EQ(wout1.ai_aux_f, wout2.ai_aux_f);
  EXPECT_EQ(wout1.nfp, wout2.nfp);
  EXPECT_EQ(wout1.mpol, wout2.mpol);
  EXPECT_EQ(wout1.ntor, wout2.ntor);
  EXPECT_EQ(wout1.lasym, wout2.lasym);
  EXPECT_EQ(wout1.ns, wout2.ns);
  EXPECT_EQ(wout1.ftolv, wout2.ftolv);
  EXPECT_EQ(wout1.niter, wout2.niter);
  EXPECT_EQ(wout1.lfreeb, wout2.lfreeb);
  EXPECT_EQ(wout1.mgrid_file, wout2.mgrid_file);
  EXPECT_EQ(wout1.extcur, wout2.extcur);
  EXPECT_EQ(wout1.mgrid_mode, wout2.mgrid_mode);
  EXPECT_EQ(wout1.wb, wout2.wb);
  EXPECT_EQ(wout1.wp, wout2.wp);
  EXPECT_EQ(wout1.rmax_surf, wout2.rmax_surf);
  EXPECT_EQ(wout1.rmin_surf, wout2.rmin_surf);
  EXPECT_EQ(wout1.zmax_surf, wout2.zmax_surf);
  EXPECT_EQ(wout1.mnmax, wout2.mnmax);
  EXPECT_EQ(wout1.mnmax_nyq, wout2.mnmax_nyq);
  EXPECT_EQ(wout1.ier_flag, wout2.ier_flag);
  EXPECT_EQ(wout1.aspect, wout2.aspect);
  EXPECT_EQ(wout1.betatotal, wout2.betatotal);
  EXPECT_EQ(wout1.betapol, wout2.betapol);
  EXPECT_EQ(wout1.betator, wout2.betator);
  EXPECT_EQ(wout1.betaxis, wout2.betaxis);
  EXPECT_EQ(wout1.b0, wout2.b0);
  EXPECT_EQ(wout1.rbtor0, wout2.rbtor0);
  EXPECT_EQ(wout1.rbtor, wout2.rbtor);
  EXPECT_EQ(wout1.IonLarmor, wout2.IonLarmor);
  EXPECT_EQ(wout1.volavgB, wout2.volavgB);
  EXPECT_EQ(wout1.ctor, wout2.ctor);
  EXPECT_EQ(wout1.Aminor_p, wout2.Aminor_p);
  EXPECT_EQ(wout1.Rmajor_p, wout2.Rmajor_p);
  EXPECT_EQ(wout1.volume, wout2.volume);
  EXPECT_EQ(wout1.fsqr, wout2.fsqr);
  EXPECT_EQ(wout1.fsqz, wout2.fsqz);
  EXPECT_EQ(wout1.fsql, wout2.fsql);
  EXPECT_EQ(wout1.iotaf, wout2.iotaf);
  EXPECT_EQ(wout1.q_factor, wout2.q_factor);
  EXPECT_EQ(wout1.presf, wout2.presf);
  EXPECT_EQ(wout1.phi, wout2.phi);
  EXPECT_EQ(wout1.phipf, wout2.phipf);
  EXPECT_EQ(wout1.chi, wout2.chi);
  EXPECT_EQ(wout1.chipf, wout2.chipf);
  EXPECT_EQ(wout1.jcuru, wout2.jcuru);
  EXPECT_EQ(wout1.jcurv, wout2.jcurv);
  EXPECT_EQ(wout1.iotas, wout2.iotas);
  EXPECT_EQ(wout1.mass, wout2.mass);
  EXPECT_EQ(wout1.pres, wout2.pres);
  EXPECT_EQ(wout1.beta_vol, wout2.beta_vol);
  EXPECT_EQ(wout1.buco, wout2.buco);
  EXPECT_EQ(wout1.bvco, wout2.bvco);
  EXPECT_EQ(wout1.vp, wout2.vp);
  EXPECT_EQ(wout1.specw, wout2.specw);
  EXPECT_EQ(wout1.phips, wout2.phips);
  EXPECT_EQ(wout1.over_r, wout2.over_r);
  EXPECT_EQ(wout1.jdotb, wout2.jdotb);
  EXPECT_EQ(wout1.bdotgradv, wout2.bdotgradv);
  EXPECT_EQ(wout1.DMerc, wout2.DMerc);
  EXPECT_EQ(wout1.DShear, wout2.DShear);
  EXPECT_EQ(wout1.DWell, wout2.DWell);
  EXPECT_EQ(wout1.DCurr, wout2.DCurr);
  EXPECT_EQ(wout1.DGeod, wout2.DGeod);
  EXPECT_EQ(wout1.equif, wout2.equif);
  EXPECT_EQ(wout1.curlabel, wout2.curlabel);
  EXPECT_EQ(wout1.potvac, wout2.potvac);
  EXPECT_EQ(wout1.xmpot, wout2.xmpot);
  EXPECT_EQ(wout1.xnpot, wout2.xnpot);
  EXPECT_EQ(wout1.xm, wout2.xm);
  EXPECT_EQ(wout1.xn, wout2.xn);
  EXPECT_EQ(wout1.xm_nyq, wout2.xm_nyq);
  EXPECT_EQ(wout1.xn_nyq, wout2.xn_nyq);
  EXPECT_EQ(wout1.raxis_cc, wout2.raxis_cc);
  EXPECT_EQ(wout1.zaxis_cs, wout2.zaxis_cs);
  EXPECT_EQ(wout1.rmnc, wout2.rmnc);
  EXPECT_EQ(wout1.zmns, wout2.zmns);
  EXPECT_EQ(wout1.lmns_full, wout2.lmns_full);
  EXPECT_EQ(wout1.lmns, wout2.lmns);
  EXPECT_EQ(wout1.gmnc, wout2.gmnc);
  EXPECT_EQ(wout1.bmnc, wout2.bmnc);
  EXPECT_EQ(wout1.bsubumnc, wout2.bsubumnc);
  EXPECT_EQ(wout1.bsubvmnc, wout2.bsubvmnc);
  EXPECT_EQ(wout1.bsubsmns, wout2.bsubsmns);
  EXPECT_EQ(wout1.bsubsmns_full, wout2.bsubsmns_full);
  EXPECT_EQ(wout1.bsupumnc, wout2.bsupumnc);
  EXPECT_EQ(wout1.bsupvmnc, wout2.bsupvmnc);
  EXPECT_EQ(wout1.raxis_cs, wout2.raxis_cs);
  EXPECT_EQ(wout1.zaxis_cc, wout2.zaxis_cc);
  EXPECT_EQ(wout1.rmns, wout2.rmns);
  EXPECT_EQ(wout1.zmnc, wout2.zmnc);
  EXPECT_EQ(wout1.lmnc_full, wout2.lmnc_full);
  EXPECT_EQ(wout1.lmnc, wout2.lmnc);
  EXPECT_EQ(wout1.gmns, wout2.gmns);
  EXPECT_EQ(wout1.bmns, wout2.bmns);
  EXPECT_EQ(wout1.bsubumns, wout2.bsubumns);
  EXPECT_EQ(wout1.bsubvmns, wout2.bsubvmns);
  EXPECT_EQ(wout1.bsubsmnc, wout2.bsubsmnc);
  EXPECT_EQ(wout1.bsubsmnc_full, wout2.bsubsmnc_full);
  EXPECT_EQ(wout1.bsupumns, wout2.bsupumns);
  EXPECT_EQ(wout1.bsupvmns, wout2.bsupvmns);
}

// Compare two `WOutFileContents` against each other, reporting every quantity
// that differs rather than stopping at the first one.
// `current_density_tolerance` applies to jcuru and jcurv alone and falls back
// to `tolerance` when zero.
inline void CompareWOut(const vmecpp::WOutFileContents& test_wout,
                        const vmecpp::WOutFileContents& expected_wout,
                        double tolerance, bool check_equal_niter = true,
                        double current_density_tolerance = 0.0) {
  using testing::IsCloseRelAbs;
  using testing::IsVectorCloseRelAbs;

  // jcuru, jcurv compare looser only if the caller opts in; otherwise
  // tolerance.
  const double current_tolerance =
      current_density_tolerance > 0.0 ? current_density_tolerance : tolerance;
  ASSERT_EQ(test_wout.signgs, expected_wout.signgs);
  ASSERT_EQ(test_wout.gamma, expected_wout.gamma);
  ASSERT_EQ(test_wout.pcurr_type, expected_wout.pcurr_type);
  ASSERT_EQ(test_wout.pmass_type, expected_wout.pmass_type);
  ASSERT_EQ(test_wout.piota_type, expected_wout.piota_type);

  EXPECT_TRUE(IsVectorCloseRelAbs(expected_wout.am, test_wout.am, tolerance));
  EXPECT_TRUE(IsVectorCloseRelAbs(expected_wout.ac, test_wout.ac, tolerance));
  EXPECT_TRUE(IsVectorCloseRelAbs(expected_wout.ai, test_wout.ai, tolerance));

  ASSERT_EQ(test_wout.nfp, expected_wout.nfp);
  ASSERT_EQ(test_wout.mpol, expected_wout.mpol);
  ASSERT_EQ(test_wout.ntor, expected_wout.ntor);
  ASSERT_EQ(test_wout.lasym, expected_wout.lasym);

  ASSERT_EQ(test_wout.ns, expected_wout.ns);
  ASSERT_EQ(test_wout.ftolv, expected_wout.ftolv);
  if (check_equal_niter) {
    ASSERT_EQ(test_wout.niter, expected_wout.niter);
  }

  ASSERT_EQ(test_wout.lfreeb, expected_wout.lfreeb);
  ASSERT_EQ(test_wout.mgrid_mode, expected_wout.mgrid_mode);

  // -------------------
  // scalar quantities

  EXPECT_TRUE(IsCloseRelAbs(expected_wout.wb, test_wout.wb, tolerance));
  EXPECT_TRUE(IsCloseRelAbs(expected_wout.wp, test_wout.wp, tolerance));

  EXPECT_TRUE(
      IsCloseRelAbs(expected_wout.rmax_surf, test_wout.rmax_surf, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(expected_wout.rmin_surf, test_wout.rmin_surf, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(expected_wout.zmax_surf, test_wout.zmax_surf, tolerance));

  ASSERT_EQ(test_wout.mnmax, expected_wout.mnmax);
  ASSERT_EQ(test_wout.mnmax_nyq, expected_wout.mnmax_nyq);

  ASSERT_EQ(test_wout.ier_flag, expected_wout.ier_flag);

  EXPECT_TRUE(IsCloseRelAbs(expected_wout.aspect, test_wout.aspect, tolerance));

  EXPECT_TRUE(
      IsCloseRelAbs(expected_wout.betatotal, test_wout.betatotal, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(expected_wout.betapol, test_wout.betapol, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(expected_wout.betator, test_wout.betator, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(expected_wout.betaxis, test_wout.betaxis, tolerance));

  EXPECT_TRUE(IsCloseRelAbs(expected_wout.b0, test_wout.b0, tolerance));

  EXPECT_TRUE(IsCloseRelAbs(expected_wout.rbtor0, test_wout.rbtor0, tolerance));
  EXPECT_TRUE(IsCloseRelAbs(expected_wout.rbtor, test_wout.rbtor, tolerance));

  EXPECT_TRUE(
      IsCloseRelAbs(expected_wout.IonLarmor, test_wout.IonLarmor, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(expected_wout.volavgB, test_wout.volavgB, tolerance));

  EXPECT_TRUE(IsCloseRelAbs(expected_wout.ctor, test_wout.ctor, tolerance));

  EXPECT_TRUE(
      IsCloseRelAbs(expected_wout.Aminor_p, test_wout.Aminor_p, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(expected_wout.Rmajor_p, test_wout.Rmajor_p, tolerance));
  EXPECT_TRUE(IsCloseRelAbs(expected_wout.volume, test_wout.volume, tolerance));

  EXPECT_TRUE(IsCloseRelAbs(expected_wout.fsqr, test_wout.fsqr, tolerance));
  EXPECT_TRUE(IsCloseRelAbs(expected_wout.fsqz, test_wout.fsqz, tolerance));
  EXPECT_TRUE(IsCloseRelAbs(expected_wout.fsql, test_wout.fsql, tolerance));

  // -------------------
  // one-dimensional array quantities

  const int ns = static_cast<int>(expected_wout.iotaf.size());
  for (int jF = 0; jF < ns; ++jF) {
    EXPECT_TRUE(
        IsCloseRelAbs(expected_wout.iotaf[jF], test_wout.iotaf[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(expected_wout.q_factor[jF],
                              test_wout.q_factor[jF], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(expected_wout.presf[jF], test_wout.presf[jF], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(expected_wout.phi[jF], test_wout.phi[jF], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(expected_wout.chi[jF], test_wout.chi[jF], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(expected_wout.phipf[jF], test_wout.phipf[jF], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(expected_wout.chipf[jF], test_wout.chipf[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(expected_wout.jcuru[jF], test_wout.jcuru[jF],
                              current_tolerance))
        << "jF = " << jF;
    EXPECT_TRUE(IsCloseRelAbs(expected_wout.jcurv[jF], test_wout.jcurv[jF],
                              current_tolerance))
        << "jF = " << jF;
    EXPECT_TRUE(
        IsCloseRelAbs(expected_wout.specw[jF], test_wout.specw[jF], tolerance));
  }  // jF

  for (int jF = 0; jF < ns; ++jF) {
    EXPECT_TRUE(
        IsCloseRelAbs(expected_wout.iotas[jF], test_wout.iotas[jF], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(expected_wout.mass[jF], test_wout.mass[jF], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(expected_wout.pres[jF], test_wout.pres[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(expected_wout.beta_vol[jF],
                              test_wout.beta_vol[jF], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(expected_wout.buco[jF], test_wout.buco[jF], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(expected_wout.bvco[jF], test_wout.bvco[jF], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(expected_wout.vp[jF], test_wout.vp[jF], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(expected_wout.phips[jF], test_wout.phips[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(expected_wout.over_r[jF], test_wout.over_r[jF],
                              tolerance));
  }  // jF

  for (int jF = 0; jF < ns; ++jF) {
    EXPECT_TRUE(
        IsCloseRelAbs(expected_wout.jdotb[jF], test_wout.jdotb[jF], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(expected_wout.bdotb[jF], test_wout.bdotb[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(expected_wout.bdotgradv[jF],
                              test_wout.bdotgradv[jF], tolerance));
  }  // jF

  for (int jF = 0; jF < ns; ++jF) {
    EXPECT_TRUE(
        IsCloseRelAbs(expected_wout.DMerc[jF], test_wout.DMerc[jF], tolerance))
        << "jF = " << jF;
    EXPECT_TRUE(IsCloseRelAbs(expected_wout.DShear[jF], test_wout.DShear[jF],
                              tolerance))
        << "jF = " << jF;
    EXPECT_TRUE(
        IsCloseRelAbs(expected_wout.DWell[jF], test_wout.DWell[jF], tolerance))
        << "jF = " << jF;
    EXPECT_TRUE(
        IsCloseRelAbs(expected_wout.DCurr[jF], test_wout.DCurr[jF], tolerance))
        << "jF = " << jF;
    EXPECT_TRUE(
        IsCloseRelAbs(expected_wout.DGeod[jF], test_wout.DGeod[jF], tolerance))
        << "jF = " << jF;
  }  // jF

  for (int jF = 0; jF < ns; ++jF) {
    EXPECT_TRUE(
        IsCloseRelAbs(expected_wout.equif[jF], test_wout.equif[jF], tolerance))
        << "jF = " << jF;
  }

  // -------------------
  // mode numbers for Fourier coefficient arrays below

  for (int mn = 0; mn < test_wout.mnmax; ++mn) {
    EXPECT_EQ(test_wout.xm[mn], expected_wout.xm[mn]);
    EXPECT_EQ(test_wout.xn[mn], expected_wout.xn[mn]);
  }  // mn

  for (int mn_nyq = 0; mn_nyq < test_wout.mnmax_nyq; ++mn_nyq) {
    EXPECT_EQ(test_wout.xm_nyq[mn_nyq], expected_wout.xm_nyq[mn_nyq]);
    EXPECT_EQ(test_wout.xn_nyq[mn_nyq], expected_wout.xn_nyq[mn_nyq]);
  }  // mn_nyq

  // -------------------
  // stellarator-symmetric Fourier coefficients

  for (int n = 0; n <= test_wout.ntor; ++n) {
    EXPECT_TRUE(IsCloseRelAbs(expected_wout.raxis_cc[n], test_wout.raxis_cc[n],
                              tolerance));
    EXPECT_TRUE(IsCloseRelAbs(expected_wout.zaxis_cs[n], test_wout.zaxis_cs[n],
                              tolerance));
  }  // n

  for (int jF = 0; jF < ns; ++jF) {
    for (int mn = 0; mn < test_wout.mnmax; ++mn) {
      EXPECT_TRUE(IsCloseRelAbs(expected_wout.rmnc(mn, jF),
                                test_wout.rmnc(mn, jF), tolerance))
          << "jF = " << jF << " mn = " << mn;
      EXPECT_TRUE(IsCloseRelAbs(expected_wout.zmns(mn, jF),
                                test_wout.zmns(mn, jF), tolerance))
          << "jF = " << jF << " mn = " << mn;
    }  // mn
  }  // jF

  for (int jF = 0; jF < ns; ++jF) {
    for (int mn = 0; mn < test_wout.mnmax; ++mn) {
      EXPECT_TRUE(IsCloseRelAbs(expected_wout.lmns(mn, jF),
                                test_wout.lmns(mn, jF), tolerance))
          << "jF = " << jF << " mn = " << mn;
    }  // mn
  }  // jF

  for (int jF = 0; jF < ns; ++jF) {
    for (int mn_nyq = 0; mn_nyq < test_wout.mnmax_nyq; ++mn_nyq) {
      EXPECT_TRUE(IsCloseRelAbs(expected_wout.gmnc(mn_nyq, jF),
                                test_wout.gmnc(mn_nyq, jF), tolerance))
          << "jF = " << jF << " mn_nyq = " << mn_nyq;
      EXPECT_TRUE(IsCloseRelAbs(expected_wout.bmnc(mn_nyq, jF),
                                test_wout.bmnc(mn_nyq, jF), tolerance))
          << "jF = " << jF << " mn_nyq = " << mn_nyq;
      EXPECT_TRUE(IsCloseRelAbs(expected_wout.bsubumnc(mn_nyq, jF),
                                test_wout.bsubumnc(mn_nyq, jF), tolerance))
          << "jF = " << jF << " mn_nyq = " << mn_nyq;
      EXPECT_TRUE(IsCloseRelAbs(expected_wout.bsubvmnc(mn_nyq, jF),
                                test_wout.bsubvmnc(mn_nyq, jF), tolerance))
          << "jF = " << jF << " mn_nyq = " << mn_nyq;
      EXPECT_TRUE(IsCloseRelAbs(expected_wout.bsubsmns(mn_nyq, jF),
                                test_wout.bsubsmns(mn_nyq, jF), tolerance))
          << "jF = " << jF << " mn_nyq = " << mn_nyq;
      EXPECT_TRUE(IsCloseRelAbs(expected_wout.bsupumnc(mn_nyq, jF),
                                test_wout.bsupumnc(mn_nyq, jF), tolerance))
          << "jF = " << jF << " mn_nyq = " << mn_nyq;
      EXPECT_TRUE(IsCloseRelAbs(expected_wout.bsupvmnc(mn_nyq, jF),
                                test_wout.bsupvmnc(mn_nyq, jF), tolerance))
          << "jF = " << jF << " mn_nyq = " << mn_nyq;
      // Current density coefficients are computed via finite differences
      // of covariant B Fourier coefficients, which amplifies differences.
      // Skip when base tolerance is very loose (e.g. hot restart comparisons
      // where bsub fields already differ by ~10%), as finite differencing
      // amplifies those to >100% for current density, making comparison
      // meaningless. Also skip axis/edge extrapolation points and cases
      // where the arrays are empty (e.g. loaded from old HDF5 files).
      if (expected_wout.currumnc.size() > 0 && test_wout.currumnc.size() > 0 &&
          jF > 0 && jF < ns - 1 && tolerance < 1.0e-2) {
        const double curr_tol = std::max(tolerance * 10.0, 1.0e-4);
        EXPECT_TRUE(IsCloseRelAbs(expected_wout.currumnc(mn_nyq, jF),
                                  test_wout.currumnc(mn_nyq, jF), curr_tol))
            << "jF = " << jF << " mn_nyq = " << mn_nyq;
        EXPECT_TRUE(IsCloseRelAbs(expected_wout.currvmnc(mn_nyq, jF),
                                  test_wout.currvmnc(mn_nyq, jF), curr_tol))
            << "jF = " << jF << " mn_nyq = " << mn_nyq;
      }
    }  // mn_nyq
  }  // jF

  // -------------------
  // non-stellarator-symmetric Fourier coefficients

  if (test_wout.lasym) {
    for (int n = 0; n <= test_wout.ntor; ++n) {
      EXPECT_TRUE(IsCloseRelAbs(expected_wout.raxis_cs[n],
                                test_wout.raxis_cs[n], tolerance));
      EXPECT_TRUE(IsCloseRelAbs(expected_wout.zaxis_cc[n],
                                test_wout.zaxis_cc[n], tolerance));
    }  // n

    for (int jF = 0; jF < ns; ++jF) {
      for (int mn = 0; mn < test_wout.mnmax; ++mn) {
        EXPECT_TRUE(IsCloseRelAbs(expected_wout.rmns(mn, jF),
                                  test_wout.rmns(mn, jF), tolerance))
            << "jF = " << jF << " mn = " << mn;
        EXPECT_TRUE(IsCloseRelAbs(expected_wout.zmnc(mn, jF),
                                  test_wout.zmnc(mn, jF), tolerance))
            << "jF = " << jF << " mn = " << mn;
      }  // mn
    }  // jF

    for (int jF = 0; jF < ns; ++jF) {
      for (int mn = 0; mn < test_wout.mnmax; ++mn) {
        EXPECT_TRUE(IsCloseRelAbs(expected_wout.lmnc(mn, jF),
                                  test_wout.lmnc(mn, jF), tolerance))
            << "jF = " << jF << " mn = " << mn;
      }  // mn
    }  // jF

    for (int jF = 0; jF < ns; ++jF) {
      for (int mn_nyq = 0; mn_nyq < test_wout.mnmax_nyq; ++mn_nyq) {
        EXPECT_TRUE(IsCloseRelAbs(expected_wout.gmns(mn_nyq, jF),
                                  test_wout.gmns(mn_nyq, jF), tolerance))
            << "jF = " << jF << " mn_nyq = " << mn_nyq;
        EXPECT_TRUE(IsCloseRelAbs(expected_wout.bmns(mn_nyq, jF),
                                  test_wout.bmns(mn_nyq, jF), tolerance))
            << "jF = " << jF << " mn_nyq = " << mn_nyq;
        EXPECT_TRUE(IsCloseRelAbs(expected_wout.bsubumns(mn_nyq, jF),
                                  test_wout.bsubumns(mn_nyq, jF), tolerance))
            << "jF = " << jF << " mn_nyq = " << mn_nyq;
        EXPECT_TRUE(IsCloseRelAbs(expected_wout.bsubvmns(mn_nyq, jF),
                                  test_wout.bsubvmns(mn_nyq, jF), tolerance))
            << "jF = " << jF << " mn_nyq = " << mn_nyq;
        EXPECT_TRUE(IsCloseRelAbs(expected_wout.bsubsmnc(mn_nyq, jF),
                                  test_wout.bsubsmnc(mn_nyq, jF), tolerance))
            << "jF = " << jF << " mn_nyq = " << mn_nyq;
        EXPECT_TRUE(IsCloseRelAbs(expected_wout.bsupumns(mn_nyq, jF),
                                  test_wout.bsupumns(mn_nyq, jF), tolerance))
            << "jF = " << jF << " mn_nyq = " << mn_nyq;
        EXPECT_TRUE(IsCloseRelAbs(expected_wout.bsupvmns(mn_nyq, jF),
                                  test_wout.bsupvmns(mn_nyq, jF), tolerance))
            << "jF = " << jF << " mn_nyq = " << mn_nyq;
        // See comment above on currumnc/currvmnc for why these are skipped
        // when the base tolerance is loose or the arrays are empty.
        if (expected_wout.currumns.size() > 0 &&
            test_wout.currumns.size() > 0 && jF > 0 && jF < ns - 1 &&
            tolerance < 1.0e-2) {
          const double curr_tol = std::max(tolerance * 10.0, 1.0e-4);
          EXPECT_TRUE(IsCloseRelAbs(expected_wout.currumns(mn_nyq, jF),
                                    test_wout.currumns(mn_nyq, jF), curr_tol))
              << "jF = " << jF << " mn_nyq = " << mn_nyq;
          EXPECT_TRUE(IsCloseRelAbs(expected_wout.currvmns(mn_nyq, jF),
                                    test_wout.currvmns(mn_nyq, jF), curr_tol))
              << "jF = " << jF << " mn_nyq = " << mn_nyq;
        }
      }  // mn_nyq
    }  // jF
  }
}

#endif  // VMECPP_VMEC_OUTPUT_QUANTITIES_TEST_HELPERS_H_"
