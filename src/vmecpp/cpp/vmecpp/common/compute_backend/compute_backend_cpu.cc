// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include "vmecpp/common/compute_backend/compute_backend_cpu.h"

#include "vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.h"

namespace vmecpp {

void ComputeBackendCpu::FourierToReal(const FourierGeometry& physical_x,
                                      const std::vector<double>& xmpq,
                                      const RadialPartitioning& rp,
                                      const Sizes& s,
                                      const RadialProfiles& profiles,
                                      const FourierBasisFastPoloidal& fb,
                                      RealSpaceGeometry& m_geometry) {
  // Delegate to the existing CPU implementation.
  FourierToReal3DSymmFastPoloidal(physical_x, xmpq, rp, s, profiles, fb,
                                  m_geometry);
}

void ComputeBackendCpu::ForcesToFourier(const RealSpaceForces& forces,
                                        const std::vector<double>& xmpq,
                                        const RadialPartitioning& rp,
                                        const FlowControl& fc, const Sizes& s,
                                        const FourierBasisFastPoloidal& fb,
                                        VacuumPressureState vacuum_pressure_state,
                                        FourierForces& m_physical_forces) {
  // Delegate to the existing CPU implementation.
  ForcesToFourier3DSymmFastPoloidal(forces, xmpq, rp, fc, s, fb,
                                    vacuum_pressure_state, m_physical_forces);
}

bool ComputeBackendCpu::ComputeJacobian(const JacobianInput& input,
                                        const RadialPartitioning& rp,
                                        const Sizes& s,
                                        JacobianOutput& m_output) {
  // Constant: 1/4 from d(sHalf)/ds and 1/2 from interpolation.
  constexpr double dSHalfDsInterp = 0.25;

  double minTau = 0.0;
  double maxTau = 0.0;

  // Temporary storage for "inside" values at each poloidal point.
  std::vector<double> r1e_i(s.nZnT), r1o_i(s.nZnT);
  std::vector<double> z1e_i(s.nZnT), z1o_i(s.nZnT);
  std::vector<double> rue_i(s.nZnT), ruo_i(s.nZnT);
  std::vector<double> zue_i(s.nZnT), zuo_i(s.nZnT);

  // Initialize with first full-grid surface.
  const int j0 = rp.nsMinF1;
  for (int kl = 0; kl < s.nZnT; ++kl) {
    r1e_i[kl] = input.r1_e[(j0 - rp.nsMinF1) * s.nZnT + kl];
    r1o_i[kl] = input.r1_o[(j0 - rp.nsMinF1) * s.nZnT + kl];
    z1e_i[kl] = input.z1_e[(j0 - rp.nsMinF1) * s.nZnT + kl];
    z1o_i[kl] = input.z1_o[(j0 - rp.nsMinF1) * s.nZnT + kl];
    rue_i[kl] = input.ru_e[(j0 - rp.nsMinF1) * s.nZnT + kl];
    ruo_i[kl] = input.ru_o[(j0 - rp.nsMinF1) * s.nZnT + kl];
    zue_i[kl] = input.zu_e[(j0 - rp.nsMinF1) * s.nZnT + kl];
    zuo_i[kl] = input.zu_o[(j0 - rp.nsMinF1) * s.nZnT + kl];
  }

  for (int jH = rp.nsMinH; jH < rp.nsMaxH; ++jH) {
    const double sqrtSH = input.sqrtSH[jH - rp.nsMinH];

    for (int kl = 0; kl < s.nZnT; ++kl) {
      // Load "outside" values from next full-grid surface.
      const double r1e_o = input.r1_e[(jH + 1 - rp.nsMinF1) * s.nZnT + kl];
      const double r1o_o = input.r1_o[(jH + 1 - rp.nsMinF1) * s.nZnT + kl];
      const double z1e_o = input.z1_e[(jH + 1 - rp.nsMinF1) * s.nZnT + kl];
      const double z1o_o = input.z1_o[(jH + 1 - rp.nsMinF1) * s.nZnT + kl];
      const double rue_o = input.ru_e[(jH + 1 - rp.nsMinF1) * s.nZnT + kl];
      const double ruo_o = input.ru_o[(jH + 1 - rp.nsMinF1) * s.nZnT + kl];
      const double zue_o = input.zu_e[(jH + 1 - rp.nsMinF1) * s.nZnT + kl];
      const double zuo_o = input.zu_o[(jH + 1 - rp.nsMinF1) * s.nZnT + kl];

      const int iHalf = (jH - rp.nsMinH) * s.nZnT + kl;

      // R on half-grid.
      m_output.r12[iHalf] =
          0.5 * ((r1e_i[kl] + r1e_o) + sqrtSH * (r1o_i[kl] + r1o_o));

      // dR/dTheta on half-grid.
      m_output.ru12[iHalf] =
          0.5 * ((rue_i[kl] + rue_o) + sqrtSH * (ruo_i[kl] + ruo_o));

      // dZ/dTheta on half-grid.
      m_output.zu12[iHalf] =
          0.5 * ((zue_i[kl] + zue_o) + sqrtSH * (zuo_i[kl] + zuo_o));

      // dR/ds on half-grid.
      m_output.rs[iHalf] =
          ((r1e_o - r1e_i[kl]) + sqrtSH * (r1o_o - r1o_i[kl])) / input.deltaS;

      // dZ/ds on half-grid.
      m_output.zs[iHalf] =
          ((z1e_o - z1e_i[kl]) + sqrtSH * (z1o_o - z1o_i[kl])) / input.deltaS;

      // sqrt(g)/R (tau) on half-grid.
      const double tau1 =
          m_output.ru12[iHalf] * m_output.zs[iHalf] -
          m_output.rs[iHalf] * m_output.zu12[iHalf];
      const double tau2 = ruo_o * z1o_o + ruo_i[kl] * z1o_i[kl] -
                          zuo_o * r1o_o - zuo_i[kl] * r1o_i[kl] +
                          (rue_o * z1o_o + rue_i[kl] * z1o_i[kl] -
                           zue_o * r1o_o - zue_i[kl] * r1o_i[kl]) /
                              sqrtSH;
      const double tau_val = tau1 + dSHalfDsInterp * tau2;

      if (tau_val < minTau || minTau == 0.0) {
        minTau = tau_val;
      }
      if (tau_val > maxTau || maxTau == 0.0) {
        maxTau = tau_val;
      }

      m_output.tau[iHalf] = tau_val;

      // Hand over "outside" to "inside" for next radial iteration.
      r1e_i[kl] = r1e_o;
      r1o_i[kl] = r1o_o;
      z1e_i[kl] = z1e_o;
      z1o_i[kl] = z1o_o;
      rue_i[kl] = rue_o;
      ruo_i[kl] = ruo_o;
      zue_i[kl] = zue_o;
      zuo_i[kl] = zuo_o;
    }
  }

  // Return true if bad Jacobian detected (sign change).
  return (minTau * maxTau < 0.0);
}

void ComputeBackendCpu::ComputeMetricElements(const MetricInput& input,
                                              const RadialPartitioning& rp,
                                              const Sizes& s,
                                              MetricOutput& m_output) {
  // Temporary storage for "inside" values.
  std::vector<double> r1e_i(s.nZnT), r1o_i(s.nZnT);
  std::vector<double> rue_i(s.nZnT), ruo_i(s.nZnT);
  std::vector<double> zue_i(s.nZnT), zuo_i(s.nZnT);
  std::vector<double> rve_i(s.nZnT), rvo_i(s.nZnT);
  std::vector<double> zve_i(s.nZnT), zvo_i(s.nZnT);

  // Initialize with first full-grid surface.
  const int j0 = rp.nsMinF1;
  for (int kl = 0; kl < s.nZnT; ++kl) {
    r1e_i[kl] = input.r1_e[(j0 - rp.nsMinF1) * s.nZnT + kl];
    r1o_i[kl] = input.r1_o[(j0 - rp.nsMinF1) * s.nZnT + kl];
    rue_i[kl] = input.ru_e[(j0 - rp.nsMinF1) * s.nZnT + kl];
    ruo_i[kl] = input.ru_o[(j0 - rp.nsMinF1) * s.nZnT + kl];
    zue_i[kl] = input.zu_e[(j0 - rp.nsMinF1) * s.nZnT + kl];
    zuo_i[kl] = input.zu_o[(j0 - rp.nsMinF1) * s.nZnT + kl];
    if (input.lthreed) {
      rve_i[kl] = input.rv_e[(j0 - rp.nsMinF1) * s.nZnT + kl];
      rvo_i[kl] = input.rv_o[(j0 - rp.nsMinF1) * s.nZnT + kl];
      zve_i[kl] = input.zv_e[(j0 - rp.nsMinF1) * s.nZnT + kl];
      zvo_i[kl] = input.zv_o[(j0 - rp.nsMinF1) * s.nZnT + kl];
    }
  }

  double sF_i = input.sqrtSF[rp.nsMinH - rp.nsMinF1] *
                input.sqrtSF[rp.nsMinH - rp.nsMinF1];

  for (int jH = rp.nsMinH; jH < rp.nsMaxH; ++jH) {
    const double sF_o =
        input.sqrtSF[jH + 1 - rp.nsMinF1] * input.sqrtSF[jH + 1 - rp.nsMinF1];
    const double sqrtSH = input.sqrtSH[jH - rp.nsMinH];

    for (int kl = 0; kl < s.nZnT; ++kl) {
      const int iHalf = (jH - rp.nsMinH) * s.nZnT + kl;

      // Jacobian gsqrt = tau * R.
      m_output.gsqrt[iHalf] = input.tau[iHalf] * input.r12[iHalf];

      // Load "outside" values.
      const double r1e_o = input.r1_e[(jH + 1 - rp.nsMinF1) * s.nZnT + kl];
      const double r1o_o = input.r1_o[(jH + 1 - rp.nsMinF1) * s.nZnT + kl];
      const double rue_o = input.ru_e[(jH + 1 - rp.nsMinF1) * s.nZnT + kl];
      const double ruo_o = input.ru_o[(jH + 1 - rp.nsMinF1) * s.nZnT + kl];
      const double zue_o = input.zu_e[(jH + 1 - rp.nsMinF1) * s.nZnT + kl];
      const double zuo_o = input.zu_o[(jH + 1 - rp.nsMinF1) * s.nZnT + kl];

      // g_{\theta,\theta}.
      m_output.guu[iHalf] =
          0.5 * ((rue_i[kl] * rue_i[kl] + zue_i[kl] * zue_i[kl]) +
                 (rue_o * rue_o + zue_o * zue_o) +
                 sF_i * (ruo_i[kl] * ruo_i[kl] + zuo_i[kl] * zuo_i[kl]) +
                 sF_o * (ruo_o * ruo_o + zuo_o * zuo_o)) +
          sqrtSH * ((rue_i[kl] * ruo_i[kl] + zue_i[kl] * zuo_i[kl]) +
                    (rue_o * ruo_o + zue_o * zuo_o));

      // g_{\zeta,\zeta} (base term: R^2).
      m_output.gvv[iHalf] =
          0.5 * (r1e_i[kl] * r1e_i[kl] + r1e_o * r1e_o +
                 sF_i * r1o_i[kl] * r1o_i[kl] + sF_o * r1o_o * r1o_o) +
          sqrtSH * (r1e_i[kl] * r1o_i[kl] + r1e_o * r1o_o);

      if (input.lthreed) {
        const double rve_o = input.rv_e[(jH + 1 - rp.nsMinF1) * s.nZnT + kl];
        const double rvo_o = input.rv_o[(jH + 1 - rp.nsMinF1) * s.nZnT + kl];
        const double zve_o = input.zv_e[(jH + 1 - rp.nsMinF1) * s.nZnT + kl];
        const double zvo_o = input.zv_o[(jH + 1 - rp.nsMinF1) * s.nZnT + kl];

        // g_{\theta,\zeta}.
        m_output.guv[iHalf] =
            0.5 * ((rue_i[kl] * rve_i[kl] + zue_i[kl] * zve_i[kl]) +
                   (rue_o * rve_o + zue_o * zve_o) +
                   sF_i * (ruo_i[kl] * rvo_i[kl] + zuo_i[kl] * zvo_i[kl]) +
                   sF_o * (ruo_o * rvo_o + zuo_o * zvo_o) +
                   sqrtSH * ((rue_i[kl] * rvo_i[kl] + zue_i[kl] * zvo_i[kl]) +
                             (rue_o * rvo_o + zue_o * zvo_o) +
                             (rve_i[kl] * ruo_i[kl] + zve_i[kl] * zuo_i[kl]) +
                             (rve_o * ruo_o + zve_o * zuo_o)));

        // Add 3D contribution to g_{\zeta,\zeta}.
        m_output.gvv[iHalf] +=
            0.5 * ((rve_i[kl] * rve_i[kl] + zve_i[kl] * zve_i[kl]) +
                   (rve_o * rve_o + zve_o * zve_o) +
                   sF_i * (rvo_i[kl] * rvo_i[kl] + zvo_i[kl] * zvo_i[kl]) +
                   sF_o * (rvo_o * rvo_o + zvo_o * zvo_o)) +
            sqrtSH * ((rve_i[kl] * rvo_i[kl] + zve_i[kl] * zvo_i[kl]) +
                      (rve_o * rvo_o + zve_o * zvo_o));

        // Hand over for 3D arrays.
        rve_i[kl] = rve_o;
        rvo_i[kl] = rvo_o;
        zve_i[kl] = zve_o;
        zvo_i[kl] = zvo_o;
      }

      // Hand over "outside" to "inside".
      r1e_i[kl] = r1e_o;
      r1o_i[kl] = r1o_o;
      rue_i[kl] = rue_o;
      ruo_i[kl] = ruo_o;
      zue_i[kl] = zue_o;
      zuo_i[kl] = zuo_o;
    }

    sF_i = sF_o;
  }
}

void ComputeBackendCpu::ComputeBContra(const BContraInput& input,
                                       const RadialPartitioning& rp,
                                       const Sizes& s,
                                       BContraOutput& m_output) {
  // Temporary storage for "inside" lambda values.
  std::vector<double> lue_i(s.nZnT), luo_i(s.nZnT);
  std::vector<double> lve_i(s.nZnT), lvo_i(s.nZnT);

  // Unnormalize lambda and add phi' for first radial location.
  const int j0 = rp.nsMinH;
  for (int kl = 0; kl < s.nZnT; ++kl) {
    input.lu_e[(j0 - rp.nsMinF1) * s.nZnT + kl] *= input.lamscale;
    input.lu_o[(j0 - rp.nsMinF1) * s.nZnT + kl] *= input.lamscale;
    if (input.lthreed) {
      input.lv_e[(j0 - rp.nsMinF1) * s.nZnT + kl] *= input.lamscale;
      input.lv_o[(j0 - rp.nsMinF1) * s.nZnT + kl] *= input.lamscale;
    }

    // Add phi' to d(lambda)/d(theta).
    input.lu_e[(j0 - rp.nsMinF1) * s.nZnT + kl] += input.phipF[j0 - rp.nsMinF1];

    lue_i[kl] = input.lu_e[(j0 - rp.nsMinF1) * s.nZnT + kl];
    luo_i[kl] = input.lu_o[(j0 - rp.nsMinF1) * s.nZnT + kl];
    if (input.lthreed) {
      lve_i[kl] = input.lv_e[(j0 - rp.nsMinF1) * s.nZnT + kl];
      lvo_i[kl] = input.lv_o[(j0 - rp.nsMinF1) * s.nZnT + kl];
    }
  }

  // Main loop: compute B^theta and B^zeta.
  for (int jH = rp.nsMinH; jH < rp.nsMaxH; ++jH) {
    const double sqrtSH = input.sqrtSH[jH - rp.nsMinH];

    for (int kl = 0; kl < s.nZnT; ++kl) {
      const int iHalf = (jH - rp.nsMinH) * s.nZnT + kl;

      // Unnormalize next full-grid location.
      input.lu_e[(jH + 1 - rp.nsMinF1) * s.nZnT + kl] *= input.lamscale;
      input.lu_o[(jH + 1 - rp.nsMinF1) * s.nZnT + kl] *= input.lamscale;
      if (input.lthreed) {
        input.lv_e[(jH + 1 - rp.nsMinF1) * s.nZnT + kl] *= input.lamscale;
        input.lv_o[(jH + 1 - rp.nsMinF1) * s.nZnT + kl] *= input.lamscale;
      }

      // Add phi'.
      input.lu_e[(jH + 1 - rp.nsMinF1) * s.nZnT + kl] +=
          input.phipF[jH + 1 - rp.nsMinH];

      const double lue_o = input.lu_e[(jH + 1 - rp.nsMinF1) * s.nZnT + kl];
      const double luo_o = input.lu_o[(jH + 1 - rp.nsMinF1) * s.nZnT + kl];

      if (input.lthreed) {
        const double lve_o = input.lv_e[(jH + 1 - rp.nsMinF1) * s.nZnT + kl];
        const double lvo_o = input.lv_o[(jH + 1 - rp.nsMinF1) * s.nZnT + kl];

        m_output.bsupu[iHalf] =
            0.5 * ((lve_i[kl] + lve_o) + sqrtSH * (lvo_i[kl] + lvo_o)) /
            input.gsqrt[iHalf];

        lve_i[kl] = lve_o;
        lvo_i[kl] = lvo_o;
      } else {
        m_output.bsupu[iHalf] = 0.0;
      }

      m_output.bsupv[iHalf] =
          0.5 * ((lue_i[kl] + lue_o) + sqrtSH * (luo_i[kl] + luo_o)) /
          input.gsqrt[iHalf];

      lue_i[kl] = lue_o;
      luo_i[kl] = luo_o;
    }
  }

  // Handle iota/current constraint.
  if (input.ncurr == 1) {
    // Constrained toroidal current profile.
    for (int jH = rp.nsMinH; jH < rp.nsMaxH; ++jH) {
      double jvPlasma = 0.0;
      double avg_guu_gsqrt = 0.0;
      for (int kl = 0; kl < s.nZnT; ++kl) {
        const int iHalf = (jH - rp.nsMinH) * s.nZnT + kl;
        const int l = kl % s.nThetaEff;
        if (input.lthreed) {
          jvPlasma +=
              (input.guu[iHalf] * m_output.bsupu[iHalf] +
               input.guv[iHalf] * m_output.bsupv[iHalf]) *
              input.wInt[l];
        } else {
          jvPlasma += input.guu[iHalf] * m_output.bsupu[iHalf] * input.wInt[l];
        }
        avg_guu_gsqrt +=
            input.guu[iHalf] / input.gsqrt[iHalf] * input.wInt[l];
      }

      if (avg_guu_gsqrt != 0.0) {
        m_output.chipH[jH - rp.nsMinH] =
            (input.currH[jH - rp.nsMinH] - jvPlasma) / avg_guu_gsqrt;
      }

      if (input.phipH[jH - rp.nsMinH] != 0.0) {
        m_output.iotaH[jH - rp.nsMinH] =
            m_output.chipH[jH - rp.nsMinH] / input.phipH[jH - rp.nsMinH];
      }
    }
  } else {
    // Constrained iota profile.
    for (int jH = rp.nsMinH; jH < rp.nsMaxH; ++jH) {
      m_output.iotaH[jH - rp.nsMinH] = input.iotaH_in[jH - rp.nsMinH];
      m_output.chipH[jH - rp.nsMinH] =
          input.iotaH_in[jH - rp.nsMinH] * input.phipH[jH - rp.nsMinH];
    }
  }

  // Interpolate chipF and iotaF.
  for (int jFi = rp.nsMinFi; jFi < rp.nsMaxFi; ++jFi) {
    m_output.chipF[jFi - rp.nsMinF1] =
        0.5 * (m_output.chipH[jFi - rp.nsMinH] +
               m_output.chipH[jFi - 1 - rp.nsMinH]);
    m_output.iotaF[jFi - rp.nsMinF1] =
        0.5 * (m_output.iotaH[jFi - rp.nsMinH] +
               m_output.iotaH[jFi - 1 - rp.nsMinH]);
  }

  // Add chip'/gsqrt to B^theta.
  for (int jH = rp.nsMinH; jH < rp.nsMaxH; ++jH) {
    for (int kl = 0; kl < s.nZnT; ++kl) {
      const int iHalf = (jH - rp.nsMinH) * s.nZnT + kl;
      m_output.bsupu[iHalf] +=
          m_output.chipH[jH - rp.nsMinH] / input.gsqrt[iHalf];
    }
  }
}

void ComputeBackendCpu::ComputeMHDForces(const MHDForcesInput& input,
                                         const RadialPartitioning& rp,
                                         const Sizes& s,
                                         MHDForcesOutput& m_output) {
  int jMaxRZ = std::min(rp.nsMaxF, input.ns - 1);
  if (input.lfreeb) {
    jMaxRZ = std::min(rp.nsMaxF, input.ns);
  }

  // Temporary storage for "inside" values.
  std::vector<double> P_i(s.nZnT, 0.0), rup_i(s.nZnT, 0.0), zup_i(s.nZnT, 0.0);
  std::vector<double> rsp_i(s.nZnT, 0.0), zsp_i(s.nZnT, 0.0);
  std::vector<double> taup_i(s.nZnT, 0.0);
  std::vector<double> gbubu_i(s.nZnT, 0.0), gbubv_i(s.nZnT, 0.0);
  std::vector<double> gbvbv_i(s.nZnT, 0.0);

  double sqrtSHi = 1.0;
  if (rp.nsMinF > 0) {
    const int j0 = rp.nsMinH;
    for (int kl = 0; kl < s.nZnT; ++kl) {
      const int iHalf = (j0 - rp.nsMinH) * s.nZnT + kl;
      P_i[kl] = input.r12[iHalf] * input.totalPressure[iHalf];
      rup_i[kl] = input.ru12[iHalf] * P_i[kl];
      zup_i[kl] = input.zu12[iHalf] * P_i[kl];
      rsp_i[kl] = input.rs[iHalf] * P_i[kl];
      zsp_i[kl] = input.zs[iHalf] * P_i[kl];
      taup_i[kl] = input.tau[iHalf] * input.totalPressure[iHalf];
      gbubu_i[kl] = input.gsqrt[iHalf] * input.bsupu[iHalf] * input.bsupu[iHalf];
      gbubv_i[kl] = input.gsqrt[iHalf] * input.bsupu[iHalf] * input.bsupv[iHalf];
      gbvbv_i[kl] = input.gsqrt[iHalf] * input.bsupv[iHalf] * input.bsupv[iHalf];
    }
    sqrtSHi = input.sqrtSH[j0 - rp.nsMinH];
  }

  std::vector<double> P_o(s.nZnT), rup_o(s.nZnT), zup_o(s.nZnT);
  std::vector<double> rsp_o(s.nZnT), zsp_o(s.nZnT), taup_o(s.nZnT);
  std::vector<double> gbubu_o(s.nZnT), gbubv_o(s.nZnT), gbvbv_o(s.nZnT);

  for (int jF = rp.nsMinF; jF < jMaxRZ; ++jF) {
    const double sFull =
        input.sqrtSF[jF - rp.nsMinF1] * input.sqrtSF[jF - rp.nsMinF1];
    double sqrtSHo = 1.0;
    if (jF < rp.nsMaxH) {
      sqrtSHo = input.sqrtSH[jF - rp.nsMinH];
    }

    if (jF < rp.nsMaxH) {
      const int iHalf_base = (jF - rp.nsMinH) * s.nZnT;
      for (int kl = 0; kl < s.nZnT; ++kl) {
        const int iHalf = iHalf_base + kl;
        P_o[kl] = input.r12[iHalf] * input.totalPressure[iHalf];
        rup_o[kl] = input.ru12[iHalf] * P_o[kl];
        zup_o[kl] = input.zu12[iHalf] * P_o[kl];
        rsp_o[kl] = input.rs[iHalf] * P_o[kl];
        zsp_o[kl] = input.zs[iHalf] * P_o[kl];
        taup_o[kl] = input.tau[iHalf] * input.totalPressure[iHalf];
        gbubu_o[kl] =
            input.gsqrt[iHalf] * input.bsupu[iHalf] * input.bsupu[iHalf];
        gbubv_o[kl] =
            input.gsqrt[iHalf] * input.bsupu[iHalf] * input.bsupv[iHalf];
        gbvbv_o[kl] =
            input.gsqrt[iHalf] * input.bsupv[iHalf] * input.bsupv[iHalf];
      }
    } else {
      std::fill(P_o.begin(), P_o.end(), 0.0);
      std::fill(rup_o.begin(), rup_o.end(), 0.0);
      std::fill(zup_o.begin(), zup_o.end(), 0.0);
      std::fill(rsp_o.begin(), rsp_o.end(), 0.0);
      std::fill(zsp_o.begin(), zsp_o.end(), 0.0);
      std::fill(taup_o.begin(), taup_o.end(), 0.0);
      std::fill(gbubu_o.begin(), gbubu_o.end(), 0.0);
      std::fill(gbubv_o.begin(), gbubv_o.end(), 0.0);
      std::fill(gbvbv_o.begin(), gbvbv_o.end(), 0.0);
    }

    for (int kl = 0; kl < s.nZnT; ++kl) {
      const int idx_g = (jF - rp.nsMinF1) * s.nZnT + kl;
      const int idx_f = (jF - rp.nsMinF) * s.nZnT + kl;

      // A_R force (even).
      m_output.armn_e[idx_f] =
          (zup_o[kl] - zup_i[kl]) / input.deltaS +
          0.5 * (taup_o[kl] + taup_i[kl]) -
          0.5 * (gbvbv_o[kl] + gbvbv_i[kl]) * input.r1_e[idx_g] -
          0.5 * (gbvbv_o[kl] * sqrtSHo + gbvbv_i[kl] * sqrtSHi) *
              input.r1_o[idx_g];

      // A_R force (odd).
      m_output.armn_o[idx_f] =
          (zup_o[kl] * sqrtSHo - zup_i[kl] * sqrtSHi) / input.deltaS -
          0.25 * (P_o[kl] / sqrtSHo + P_i[kl] / sqrtSHi) * input.zu_e[idx_g] -
          0.25 * (P_o[kl] + P_i[kl]) * input.zu_o[idx_g] +
          0.5 * (taup_o[kl] * sqrtSHo + taup_i[kl] * sqrtSHi) -
          0.5 * (gbvbv_o[kl] * sqrtSHo + gbvbv_i[kl] * sqrtSHi) *
              input.r1_e[idx_g] -
          0.5 * (gbvbv_o[kl] + gbvbv_i[kl]) * input.r1_o[idx_g] * sFull;

      // A_Z force (even).
      m_output.azmn_e[idx_f] = -(rup_o[kl] - rup_i[kl]) / input.deltaS;

      // A_Z force (odd).
      m_output.azmn_o[idx_f] =
          -(rup_o[kl] * sqrtSHo - rup_i[kl] * sqrtSHi) / input.deltaS +
          0.25 * (P_o[kl] / sqrtSHo + P_i[kl] / sqrtSHi) * input.ru_e[idx_g] +
          0.25 * (P_o[kl] + P_i[kl]) * input.ru_o[idx_g];

      // B_R force (even).
      m_output.brmn_e[idx_f] =
          0.5 * (zsp_o[kl] + zsp_i[kl]) +
          0.25 * (P_o[kl] / sqrtSHo + P_i[kl] / sqrtSHi) * input.z1_o[idx_g] -
          0.5 * (gbubu_o[kl] + gbubu_i[kl]) * input.ru_e[idx_g] -
          0.5 * (gbubu_o[kl] * sqrtSHo + gbubu_i[kl] * sqrtSHi) *
              input.ru_o[idx_g];

      // B_R force (odd).
      m_output.brmn_o[idx_f] =
          0.5 * (zsp_o[kl] * sqrtSHo + zsp_i[kl] * sqrtSHi) +
          0.25 * (P_o[kl] + P_i[kl]) * input.z1_o[idx_g] -
          0.5 * (gbubu_o[kl] * sqrtSHo + gbubu_i[kl] * sqrtSHi) *
              input.ru_e[idx_g] -
          0.5 * (gbubu_o[kl] + gbubu_i[kl]) * input.ru_o[idx_g] * sFull;

      // B_Z force (even).
      m_output.bzmn_e[idx_f] =
          -0.5 * (rsp_o[kl] + rsp_i[kl]) -
          0.25 * (P_o[kl] / sqrtSHo + P_i[kl] / sqrtSHi) * input.r1_o[idx_g] -
          0.5 * (gbubu_o[kl] + gbubu_i[kl]) * input.zu_e[idx_g] -
          0.5 * (gbubu_o[kl] * sqrtSHo + gbubu_i[kl] * sqrtSHi) *
              input.zu_o[idx_g];

      // B_Z force (odd).
      m_output.bzmn_o[idx_f] =
          -0.5 * (rsp_o[kl] * sqrtSHo + rsp_i[kl] * sqrtSHi) -
          0.25 * (P_o[kl] + P_i[kl]) * input.r1_o[idx_g] -
          0.5 * (gbubu_o[kl] * sqrtSHo + gbubu_i[kl] * sqrtSHi) *
              input.zu_e[idx_g] -
          0.5 * (gbubu_o[kl] + gbubu_i[kl]) * input.zu_o[idx_g] * sFull;

      if (input.lthreed) {
        // 3D contributions to B_R force.
        m_output.brmn_e[idx_f] +=
            -0.5 * (gbubv_o[kl] + gbubv_i[kl]) * input.rv_e[idx_g] -
            0.5 * (gbubv_o[kl] * sqrtSHo + gbubv_i[kl] * sqrtSHi) *
                input.rv_o[idx_g];
        m_output.brmn_o[idx_f] +=
            -0.5 * (gbubv_o[kl] * sqrtSHo + gbubv_i[kl] * sqrtSHi) *
                input.rv_e[idx_g] -
            0.5 * (gbubv_o[kl] + gbubv_i[kl]) * input.rv_o[idx_g] * sFull;

        // 3D contributions to B_Z force.
        m_output.bzmn_e[idx_f] +=
            -0.5 * (gbubv_o[kl] + gbubv_i[kl]) * input.zv_e[idx_g] -
            0.5 * (gbubv_o[kl] * sqrtSHo + gbubv_i[kl] * sqrtSHi) *
                input.zv_o[idx_g];
        m_output.bzmn_o[idx_f] +=
            -0.5 * (gbubv_o[kl] * sqrtSHo + gbubv_i[kl] * sqrtSHi) *
                input.zv_e[idx_g] -
            0.5 * (gbubv_o[kl] + gbubv_i[kl]) * input.zv_o[idx_g] * sFull;

        // C_R force (even).
        m_output.crmn_e[idx_f] =
            0.5 * (gbubv_o[kl] + gbubv_i[kl]) * input.ru_e[idx_g] +
            0.5 * (gbubv_o[kl] * sqrtSHo + gbubv_i[kl] * sqrtSHi) *
                input.ru_o[idx_g] +
            0.5 * (gbvbv_o[kl] + gbvbv_i[kl]) * input.rv_e[idx_g] +
            0.5 * (gbvbv_o[kl] * sqrtSHo + gbvbv_i[kl] * sqrtSHi) *
                input.rv_o[idx_g];

        // C_R force (odd).
        m_output.crmn_o[idx_f] =
            0.5 * (gbubv_o[kl] * sqrtSHo + gbubv_i[kl] * sqrtSHi) *
                input.ru_e[idx_g] +
            0.5 * (gbubv_o[kl] + gbubv_i[kl]) * input.ru_o[idx_g] * sFull +
            0.5 * (gbvbv_o[kl] * sqrtSHo + gbvbv_i[kl] * sqrtSHi) *
                input.rv_e[idx_g] +
            0.5 * (gbvbv_o[kl] + gbvbv_i[kl]) * input.rv_o[idx_g] * sFull;

        // C_Z force (even).
        m_output.czmn_e[idx_f] =
            0.5 * (gbubv_o[kl] + gbubv_i[kl]) * input.zu_e[idx_g] +
            0.5 * (gbubv_o[kl] * sqrtSHo + gbubv_i[kl] * sqrtSHi) *
                input.zu_o[idx_g] +
            0.5 * (gbvbv_o[kl] + gbvbv_i[kl]) * input.zv_e[idx_g] +
            0.5 * (gbvbv_o[kl] * sqrtSHo + gbvbv_i[kl] * sqrtSHi) *
                input.zv_o[idx_g];

        // C_Z force (odd).
        m_output.czmn_o[idx_f] =
            0.5 * (gbubv_o[kl] * sqrtSHo + gbubv_i[kl] * sqrtSHi) *
                input.zu_e[idx_g] +
            0.5 * (gbubv_o[kl] + gbubv_i[kl]) * input.zu_o[idx_g] * sFull +
            0.5 * (gbvbv_o[kl] * sqrtSHo + gbvbv_i[kl] * sqrtSHi) *
                input.zv_e[idx_g] +
            0.5 * (gbvbv_o[kl] + gbvbv_i[kl]) * input.zv_o[idx_g] * sFull;
      }
    }

    // Shift to next point.
    P_i = P_o;
    rup_i = rup_o;
    zup_i = zup_o;
    rsp_i = rsp_o;
    zsp_i = zsp_o;
    taup_i = taup_o;
    gbubu_i = gbubu_o;
    gbubv_i = gbubv_o;
    gbvbv_i = gbvbv_o;
    sqrtSHi = sqrtSHo;
  }
}

}  // namespace vmecpp
