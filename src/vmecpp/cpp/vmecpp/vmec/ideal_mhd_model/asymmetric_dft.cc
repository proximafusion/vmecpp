// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.h"

#include <array>
#include <cstddef>
#include <vector>

#include "vmecpp/vmec/ideal_mhd_model/force_symmetry.h"

namespace vmecpp {
namespace {

struct ReducedGeometryBuffers {
  explicit ReducedGeometryBuffers(std::size_t size)
      : r1_e(size, 0.0),
        r1_o(size, 0.0),
        ru_e(size, 0.0),
        ru_o(size, 0.0),
        rv_e(size, 0.0),
        rv_o(size, 0.0),
        z1_e(size, 0.0),
        z1_o(size, 0.0),
        zu_e(size, 0.0),
        zu_o(size, 0.0),
        zv_e(size, 0.0),
        zv_o(size, 0.0),
        lu_e(size, 0.0),
        lu_o(size, 0.0),
        lv_e(size, 0.0),
        lv_o(size, 0.0) {}

  std::vector<double> r1_e;
  std::vector<double> r1_o;
  std::vector<double> ru_e;
  std::vector<double> ru_o;
  std::vector<double> rv_e;
  std::vector<double> rv_o;
  std::vector<double> z1_e;
  std::vector<double> z1_o;
  std::vector<double> zu_e;
  std::vector<double> zu_o;
  std::vector<double> zv_e;
  std::vector<double> zv_o;
  std::vector<double> lu_e;
  std::vector<double> lu_o;
  std::vector<double> lv_e;
  std::vector<double> lv_o;
};

struct ReducedConstraintBuffers {
  explicit ReducedConstraintBuffers(std::size_t size)
      : rCon_e(size, 0.0),
        rCon_o(size, 0.0),
        zCon_e(size, 0.0),
        zCon_o(size, 0.0) {}

  std::vector<double> rCon_e;
  std::vector<double> rCon_o;
  std::vector<double> zCon_e;
  std::vector<double> zCon_o;
};

std::size_t ReducedIndex(int j, int k, int l, int n_zeta, int n_theta_reduced) {
  return ((static_cast<std::size_t>(j) * n_zeta) + k) * n_theta_reduced + l;
}

std::size_t FullIndex(int j, int k, int l, int n_zeta, int n_theta_eff) {
  return ((static_cast<std::size_t>(j) * n_zeta) + k) * n_theta_eff + l;
}

std::span<double> SelectBuffer(std::vector<double>& even, std::vector<double>& odd,
                               bool is_even) {
  return is_even ? std::span<double>(even) : std::span<double>(odd);
}

std::span<const double> SelectBuffer(const std::vector<double>& even,
                                     const std::vector<double>& odd,
                                     bool is_even) {
  return is_even ? std::span<const double>(even)
                 : std::span<const double>(odd);
}

void ReflectGeometrySecondHalf(std::span<double> sym, std::span<const double> asym,
                               const Sizes& s) {
  const int num_surfaces = static_cast<int>(sym.size()) / s.nZnT;
  for (int j = 0; j < num_surfaces; ++j) {
    for (int k = 0; k < s.nZeta; ++k) {
      const int k_ref = (s.nZeta - k) % s.nZeta;
      for (int l = s.nThetaReduced; l < s.nThetaEff; ++l) {
        const int l_ref = s.nThetaEff - l;
        const std::size_t dst = FullIndex(j, k, l, s.nZeta, s.nThetaEff);
        const std::size_t src = FullIndex(j, k_ref, l_ref, s.nZeta, s.nThetaEff);
        const std::size_t asym_src =
            ReducedIndex(j, k_ref, l_ref, s.nZeta, s.nThetaReduced);
        sym[dst] = sym[src] - asym[asym_src];
      }
    }
  }
}

void ReflectDerivativeSecondHalf(std::span<double> sym, std::span<const double> asym,
                                 const Sizes& s, double sym_sign,
                                 double asym_sign) {
  const int num_surfaces = static_cast<int>(sym.size()) / s.nZnT;
  for (int j = 0; j < num_surfaces; ++j) {
    for (int k = 0; k < s.nZeta; ++k) {
      const int k_ref = (s.nZeta - k) % s.nZeta;
      for (int l = s.nThetaReduced; l < s.nThetaEff; ++l) {
        const int l_ref = s.nThetaEff - l;
        const std::size_t dst = FullIndex(j, k, l, s.nZeta, s.nThetaEff);
        const std::size_t src = FullIndex(j, k_ref, l_ref, s.nZeta, s.nThetaEff);
        const std::size_t asym_src =
            ReducedIndex(j, k_ref, l_ref, s.nZeta, s.nThetaReduced);
        sym[dst] = sym_sign * sym[src] + asym_sign * asym[asym_src];
      }
    }
  }
}

void AddFirstHalf(std::span<double> sym, std::span<const double> asym,
                  const Sizes& s) {
  const int num_surfaces = static_cast<int>(sym.size()) / s.nZnT;
  for (int j = 0; j < num_surfaces; ++j) {
    for (int k = 0; k < s.nZeta; ++k) {
      for (int l = 0; l < s.nThetaReduced; ++l) {
        const std::size_t dst = FullIndex(j, k, l, s.nZeta, s.nThetaEff);
        const std::size_t src =
            ReducedIndex(j, k, l, s.nZeta, s.nThetaReduced);
        sym[dst] += asym[src];
      }
    }
  }
}

}  // namespace

void FourierToReal3DAsymmFastPoloidal(const FourierGeometry& physical_x,
                                      const Eigen::VectorXd& xmpq,
                                      const RadialPartitioning& r,
                                      const Sizes& s,
                                      const RadialProfiles& rp,
                                      const FourierBasisFastPoloidal& fb,
                                      RealSpaceGeometry& m_geometry) {
  FourierToReal3DSymmFastPoloidal(physical_x, xmpq, r, s, rp, fb, m_geometry);

  const std::size_t local_geometry_size =
      static_cast<std::size_t>(r.nsMaxF1 - r.nsMinF1) * s.nZeta * s.nThetaReduced;
  ReducedGeometryBuffers asym(local_geometry_size);

  const std::size_t local_constraint_size =
      static_cast<std::size_t>(r.nsMaxFIncludingLcfs - r.nsMinF) * s.nZeta *
      s.nThetaReduced;
  ReducedConstraintBuffers asym_con(local_constraint_size);

  for (int jF = r.nsMinF1; jF < r.nsMaxF1; ++jF) {
    const int local_surface = jF - r.nsMinF1;
    for (int m = 0; m < s.mpol; ++m) {
      const bool m_even = m % 2 == 0;
      const int j_min = (m == 0 || m == 1) ? 0 : 1;
      if (jF < j_min) {
        continue;
      }

      const int idx_ml_base = m * s.nThetaReduced;
      const double con_factor =
          m_even ? xmpq[m] : xmpq[m] * rp.sqrtSF[jF - r.nsMinF1];

      auto r1 = SelectBuffer(asym.r1_e, asym.r1_o, m_even);
      auto ru = SelectBuffer(asym.ru_e, asym.ru_o, m_even);
      auto rv = SelectBuffer(asym.rv_e, asym.rv_o, m_even);
      auto z1 = SelectBuffer(asym.z1_e, asym.z1_o, m_even);
      auto zu = SelectBuffer(asym.zu_e, asym.zu_o, m_even);
      auto zv = SelectBuffer(asym.zv_e, asym.zv_o, m_even);
      auto lu = SelectBuffer(asym.lu_e, asym.lu_o, m_even);
      auto lv = SelectBuffer(asym.lv_e, asym.lv_o, m_even);

      std::span<double> r_con;
      std::span<double> z_con;
      const bool has_constraint_surface = jF < r.nsMaxFIncludingLcfs;
      if (has_constraint_surface) {
        r_con = SelectBuffer(asym_con.rCon_e, asym_con.rCon_o, m_even);
        z_con = SelectBuffer(asym_con.zCon_e, asym_con.zCon_o, m_even);
      }

      for (int k = 0; k < s.nZeta; ++k) {
        const int idx_kn_base = k * (s.nnyq2 + 1);
        const int idx_mn_base = ((jF - r.nsMinF1) * s.mpol + m) * (s.ntor + 1);

        auto cosnv_seg = fb.cosnv.segment(idx_kn_base, s.ntor + 1);
        auto sinnv_seg = fb.sinnv.segment(idx_kn_base, s.ntor + 1);
        auto sinnvn_seg = fb.sinnvn.segment(idx_kn_base, s.ntor + 1);
        auto cosnvn_seg = fb.cosnvn.segment(idx_kn_base, s.ntor + 1);

        auto rmnsc_seg = Eigen::Map<const Eigen::VectorXd>(
            physical_x.rmnsc.data() + idx_mn_base, s.ntor + 1);
        auto rmncs_seg = Eigen::Map<const Eigen::VectorXd>(
            physical_x.rmncs.data() + idx_mn_base, s.ntor + 1);
        auto zmncc_seg = Eigen::Map<const Eigen::VectorXd>(
            physical_x.zmncc.data() + idx_mn_base, s.ntor + 1);
        auto zmnss_seg = Eigen::Map<const Eigen::VectorXd>(
            physical_x.zmnss.data() + idx_mn_base, s.ntor + 1);
        auto lmncc_seg = Eigen::Map<const Eigen::VectorXd>(
            physical_x.lmncc.data() + idx_mn_base, s.ntor + 1);
        auto lmnss_seg = Eigen::Map<const Eigen::VectorXd>(
            physical_x.lmnss.data() + idx_mn_base, s.ntor + 1);

        const double rmksc = rmnsc_seg.dot(cosnv_seg);
        const double rmkcs = rmncs_seg.dot(sinnv_seg);
        const double rmksc_n = rmnsc_seg.dot(sinnvn_seg);
        const double rmkcs_n = rmncs_seg.dot(cosnvn_seg);
        const double zmkcc = zmncc_seg.dot(cosnv_seg);
        const double zmkss = zmnss_seg.dot(sinnv_seg);
        const double zmkss_n = zmnss_seg.dot(cosnvn_seg);
        const double zmkcc_n = zmncc_seg.dot(sinnvn_seg);
        const double lmkcc = lmncc_seg.dot(cosnv_seg);
        const double lmkss = lmnss_seg.dot(sinnv_seg);
        const double lmkss_n = lmnss_seg.dot(cosnvn_seg);
        const double lmkcc_n = lmncc_seg.dot(sinnvn_seg);

        for (int l = 0; l < s.nThetaReduced; ++l) {
          const int idx_ml = idx_ml_base + l;
          const std::size_t idx_reduced =
              ReducedIndex(local_surface, k, l, s.nZeta, s.nThetaReduced);

          const double sinmu = fb.sinmu[idx_ml];
          const double cosmu = fb.cosmu[idx_ml];
          const double sinmum = fb.sinmum[idx_ml];
          const double cosmum = fb.cosmum[idx_ml];

          r1[idx_reduced] += rmksc * sinmu + rmkcs * cosmu;
          ru[idx_reduced] += rmksc * cosmum + rmkcs * sinmum;
          rv[idx_reduced] += rmksc_n * sinmu + rmkcs_n * cosmu;

          z1[idx_reduced] += zmkss * sinmu + zmkcc * cosmu;
          zu[idx_reduced] += zmkss * cosmum + zmkcc * sinmum;
          zv[idx_reduced] += zmkss_n * sinmu + zmkcc_n * cosmu;

          lu[idx_reduced] += lmkss * cosmum + lmkcc * sinmum;
          lv[idx_reduced] -= lmkss_n * sinmu + lmkcc_n * cosmu;

          if (has_constraint_surface) {
            const int local_constraint_surface = jF - r.nsMinF;
            const std::size_t idx_constraint = ReducedIndex(
                local_constraint_surface, k, l, s.nZeta, s.nThetaReduced);
            r_con[idx_constraint] +=
                (rmksc * sinmu + rmkcs * cosmu) * con_factor;
            z_con[idx_constraint] +=
                (zmkss * sinmu + zmkcc * cosmu) * con_factor;
          }
        }
      }
    }
  }

  ReflectGeometrySecondHalf(m_geometry.r1_e, asym.r1_e, s);
  ReflectGeometrySecondHalf(m_geometry.r1_o, asym.r1_o, s);
  ReflectDerivativeSecondHalf(m_geometry.ru_e, asym.ru_e, s, -1.0, 1.0);
  ReflectDerivativeSecondHalf(m_geometry.ru_o, asym.ru_o, s, -1.0, 1.0);
  ReflectDerivativeSecondHalf(m_geometry.rv_e, asym.rv_e, s, -1.0, 1.0);
  ReflectDerivativeSecondHalf(m_geometry.rv_o, asym.rv_o, s, -1.0, 1.0);
  ReflectDerivativeSecondHalf(m_geometry.z1_e, asym.z1_e, s, -1.0, 1.0);
  ReflectDerivativeSecondHalf(m_geometry.z1_o, asym.z1_o, s, -1.0, 1.0);
  ReflectDerivativeSecondHalf(m_geometry.zu_e, asym.zu_e, s, 1.0, -1.0);
  ReflectDerivativeSecondHalf(m_geometry.zu_o, asym.zu_o, s, 1.0, -1.0);
  ReflectDerivativeSecondHalf(m_geometry.zv_e, asym.zv_e, s, 1.0, -1.0);
  ReflectDerivativeSecondHalf(m_geometry.zv_o, asym.zv_o, s, 1.0, -1.0);
  ReflectDerivativeSecondHalf(m_geometry.lu_e, asym.lu_e, s, 1.0, -1.0);
  ReflectDerivativeSecondHalf(m_geometry.lu_o, asym.lu_o, s, 1.0, -1.0);
  ReflectDerivativeSecondHalf(m_geometry.lv_e, asym.lv_e, s, 1.0, -1.0);
  ReflectDerivativeSecondHalf(m_geometry.lv_o, asym.lv_o, s, 1.0, -1.0);

  AddFirstHalf(m_geometry.r1_e, asym.r1_e, s);
  AddFirstHalf(m_geometry.r1_o, asym.r1_o, s);
  AddFirstHalf(m_geometry.ru_e, asym.ru_e, s);
  AddFirstHalf(m_geometry.ru_o, asym.ru_o, s);
  AddFirstHalf(m_geometry.rv_e, asym.rv_e, s);
  AddFirstHalf(m_geometry.rv_o, asym.rv_o, s);
  AddFirstHalf(m_geometry.z1_e, asym.z1_e, s);
  AddFirstHalf(m_geometry.z1_o, asym.z1_o, s);
  AddFirstHalf(m_geometry.zu_e, asym.zu_e, s);
  AddFirstHalf(m_geometry.zu_o, asym.zu_o, s);
  AddFirstHalf(m_geometry.zv_e, asym.zv_e, s);
  AddFirstHalf(m_geometry.zv_o, asym.zv_o, s);
  AddFirstHalf(m_geometry.lu_e, asym.lu_e, s);
  AddFirstHalf(m_geometry.lu_o, asym.lu_o, s);
  AddFirstHalf(m_geometry.lv_e, asym.lv_e, s);
  AddFirstHalf(m_geometry.lv_o, asym.lv_o, s);

  std::vector<double> total_r_con(local_constraint_size);
  std::vector<double> total_z_con(local_constraint_size);
  for (std::size_t i = 0; i < local_constraint_size; ++i) {
    total_r_con[i] = asym_con.rCon_e[i] + asym_con.rCon_o[i];
    total_z_con[i] = asym_con.zCon_e[i] + asym_con.zCon_o[i];
  }
  ReflectGeometrySecondHalf(m_geometry.rCon, total_r_con, s);
  ReflectDerivativeSecondHalf(m_geometry.zCon, total_z_con, s, -1.0, 1.0);
  AddFirstHalf(m_geometry.rCon, total_r_con, s);
  AddFirstHalf(m_geometry.zCon, total_z_con, s);
}

void ForcesToFourier3DAsymmFastPoloidal(
    const RealSpaceForces& d, const Eigen::VectorXd& xmpq,
    const RadialPartitioning& rp, const FlowControl& fc, const Sizes& s,
    const FourierBasisFastPoloidal& fb,
    VacuumPressureState vacuum_pressure_state,
    FourierForces& m_physical_forces) {
  m_physical_forces.setZero();

#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

  int jMaxRZ = std::min(rp.nsMaxF, fc.ns - 1);
  if (fc.lfreeb &&
      (vacuum_pressure_state == VacuumPressureState::kInitialized ||
       vacuum_pressure_state == VacuumPressureState::kActive)) {
    jMaxRZ = std::min(rp.nsMaxF, fc.ns);
  }

  const std::size_t nrzt_reduced =
      static_cast<std::size_t>(rp.nsMaxF - rp.nsMinF) * s.nZeta *
      s.nThetaReduced;
  const std::size_t nrzt_reduced_including_boundary =
      static_cast<std::size_t>(rp.nsMaxFIncludingLcfs - rp.nsMinF) * s.nZeta *
      s.nThetaReduced;

  std::vector<double> armn_sym_e(nrzt_reduced), armn_sym_o(nrzt_reduced);
  std::vector<double> armn_asym_e(nrzt_reduced), armn_asym_o(nrzt_reduced);
  std::vector<double> brmn_sym_e(nrzt_reduced), brmn_sym_o(nrzt_reduced);
  std::vector<double> brmn_asym_e(nrzt_reduced), brmn_asym_o(nrzt_reduced);
  std::vector<double> crmn_sym_e(nrzt_reduced), crmn_sym_o(nrzt_reduced);
  std::vector<double> crmn_asym_e(nrzt_reduced), crmn_asym_o(nrzt_reduced);
  std::vector<double> azmn_sym_e(nrzt_reduced), azmn_sym_o(nrzt_reduced);
  std::vector<double> azmn_asym_e(nrzt_reduced), azmn_asym_o(nrzt_reduced);
  std::vector<double> bzmn_sym_e(nrzt_reduced), bzmn_sym_o(nrzt_reduced);
  std::vector<double> bzmn_asym_e(nrzt_reduced), bzmn_asym_o(nrzt_reduced);
  std::vector<double> czmn_sym_e(nrzt_reduced), czmn_sym_o(nrzt_reduced);
  std::vector<double> czmn_asym_e(nrzt_reduced), czmn_asym_o(nrzt_reduced);
  std::vector<double> frcon_sym_e(nrzt_reduced), frcon_sym_o(nrzt_reduced);
  std::vector<double> frcon_asym_e(nrzt_reduced), frcon_asym_o(nrzt_reduced);
  std::vector<double> fzcon_sym_e(nrzt_reduced), fzcon_sym_o(nrzt_reduced);
  std::vector<double> fzcon_asym_e(nrzt_reduced), fzcon_asym_o(nrzt_reduced);
  std::vector<double> blmn_sym_e(nrzt_reduced_including_boundary),
      blmn_sym_o(nrzt_reduced_including_boundary);
  std::vector<double> blmn_asym_e(nrzt_reduced_including_boundary),
      blmn_asym_o(nrzt_reduced_including_boundary);
  std::vector<double> clmn_sym_e(nrzt_reduced_including_boundary),
      clmn_sym_o(nrzt_reduced_including_boundary);
  std::vector<double> clmn_asym_e(nrzt_reduced_including_boundary),
      clmn_asym_o(nrzt_reduced_including_boundary);

  DecomposeForceComponent(s, d.armn_e, ReflectionParity::kStandard, armn_sym_e,
                          armn_asym_e);
  DecomposeForceComponent(s, d.armn_o, ReflectionParity::kStandard, armn_sym_o,
                          armn_asym_o);
  DecomposeForceComponent(s, d.brmn_e, ReflectionParity::kReversed, brmn_sym_e,
                          brmn_asym_e);
  DecomposeForceComponent(s, d.brmn_o, ReflectionParity::kReversed, brmn_sym_o,
                          brmn_asym_o);
  DecomposeForceComponent(s, d.crmn_e, ReflectionParity::kReversed, crmn_sym_e,
                          crmn_asym_e);
  DecomposeForceComponent(s, d.crmn_o, ReflectionParity::kReversed, crmn_sym_o,
                          crmn_asym_o);
  DecomposeForceComponent(s, d.azmn_e, ReflectionParity::kReversed, azmn_sym_e,
                          azmn_asym_e);
  DecomposeForceComponent(s, d.azmn_o, ReflectionParity::kReversed, azmn_sym_o,
                          azmn_asym_o);
  DecomposeForceComponent(s, d.bzmn_e, ReflectionParity::kStandard, bzmn_sym_e,
                          bzmn_asym_e);
  DecomposeForceComponent(s, d.bzmn_o, ReflectionParity::kStandard, bzmn_sym_o,
                          bzmn_asym_o);
  DecomposeForceComponent(s, d.czmn_e, ReflectionParity::kStandard, czmn_sym_e,
                          czmn_asym_e);
  DecomposeForceComponent(s, d.czmn_o, ReflectionParity::kStandard, czmn_sym_o,
                          czmn_asym_o);
  DecomposeForceComponent(s, d.frcon_e, ReflectionParity::kStandard,
                          frcon_sym_e, frcon_asym_e);
  DecomposeForceComponent(s, d.frcon_o, ReflectionParity::kStandard,
                          frcon_sym_o, frcon_asym_o);
  DecomposeForceComponent(s, d.fzcon_e, ReflectionParity::kReversed,
                          fzcon_sym_e, fzcon_asym_e);
  DecomposeForceComponent(s, d.fzcon_o, ReflectionParity::kReversed,
                          fzcon_sym_o, fzcon_asym_o);
  DecomposeForceComponent(s, d.blmn_e, ReflectionParity::kStandard, blmn_sym_e,
                          blmn_asym_e);
  DecomposeForceComponent(s, d.blmn_o, ReflectionParity::kStandard, blmn_sym_o,
                          blmn_asym_o);
  DecomposeForceComponent(s, d.clmn_e, ReflectionParity::kStandard, clmn_sym_e,
                          clmn_asym_e);
  DecomposeForceComponent(s, d.clmn_o, ReflectionParity::kStandard, clmn_sym_o,
                          clmn_asym_o);

  for (int jF = rp.nsMinF; jF < jMaxRZ; ++jF) {
    const int num_m = jF == 0 ? 1 : s.mpol;
    const int local_surface = jF - rp.nsMinF;
    for (int m = 0; m < num_m; ++m) {
      const bool m_even = m % 2 == 0;
      const auto& armn_asym = m_even ? armn_asym_e : armn_asym_o;
      const auto& brmn_asym = m_even ? brmn_asym_e : brmn_asym_o;
      const auto& crmn_asym = m_even ? crmn_asym_e : crmn_asym_o;
      const auto& azmn_asym = m_even ? azmn_asym_e : azmn_asym_o;
      const auto& bzmn_asym = m_even ? bzmn_asym_e : bzmn_asym_o;
      const auto& czmn_asym = m_even ? czmn_asym_e : czmn_asym_o;
      const auto& frcon_asym = m_even ? frcon_asym_e : frcon_asym_o;
      const auto& fzcon_asym = m_even ? fzcon_asym_e : fzcon_asym_o;

      for (int k = 0; k < s.nZeta; ++k) {
        std::array<double, 12> work = {};
        for (int l = 0; l < s.nThetaReduced; ++l) {
          const std::size_t idx_jkl =
              ReducedIndex(local_surface, k, l, s.nZeta, s.nThetaReduced);
          const int idx_ml = m * s.nThetaReduced + l;

          const double temp_r = armn_asym[idx_jkl] + xmpq[m] * frcon_asym[idx_jkl];
          const double temp_z = azmn_asym[idx_jkl] + xmpq[m] * fzcon_asym[idx_jkl];

          work[2] += temp_r * fb.sinmui[idx_ml] + brmn_asym[idx_jkl] * fb.cosmumi[idx_ml];
          work[4] += temp_z * fb.cosmui[idx_ml] + bzmn_asym[idx_jkl] * fb.sinmumi[idx_ml];

          work[0] += temp_r * fb.cosmui[idx_ml] + brmn_asym[idx_jkl] * fb.sinmumi[idx_ml];
          work[1] -= crmn_asym[idx_jkl] * fb.cosmui[idx_ml];
          work[3] -= crmn_asym[idx_jkl] * fb.sinmui[idx_ml];
          work[5] -= czmn_asym[idx_jkl] * fb.cosmui[idx_ml];
          work[6] += temp_z * fb.sinmui[idx_ml] + bzmn_asym[idx_jkl] * fb.cosmumi[idx_ml];
          work[7] -= czmn_asym[idx_jkl] * fb.sinmui[idx_ml];
        }

        const int idx_kn_base = k * (s.nnyq2 + 1);
        const int idx_mn_base = ((jF - rp.nsMinF) * s.mpol + m) * (s.ntor + 1);
        for (int n = 0; n <= s.ntor; ++n) {
          const int idx_mn = idx_mn_base + n;
          const int idx_kn = idx_kn_base + n;
          m_physical_forces.frsc[idx_mn] +=
              work[2] * fb.cosnv[idx_kn] + work[3] * fb.sinnvn[idx_kn];
          m_physical_forces.fzcc[idx_mn] +=
              work[4] * fb.cosnv[idx_kn] + work[5] * fb.sinnvn[idx_kn];
          m_physical_forces.frcs[idx_mn] +=
              work[0] * fb.sinnv[idx_kn] + work[1] * fb.cosnvn[idx_kn];
          m_physical_forces.fzss[idx_mn] +=
              work[6] * fb.sinnv[idx_kn] + work[7] * fb.cosnvn[idx_kn];
        }
      }
    }
  }

  for (int jF = std::max(1, rp.nsMinF); jF < rp.nsMaxFIncludingLcfs; ++jF) {
    const int local_surface = jF - rp.nsMinF;
    for (int m = 0; m < s.mpol; ++m) {
      const bool m_even = m % 2 == 0;
      const auto& blmn_asym = m_even ? blmn_asym_e : blmn_asym_o;
      const auto& clmn_asym = m_even ? clmn_asym_e : clmn_asym_o;
      for (int k = 0; k < s.nZeta; ++k) {
        std::array<double, 4> work = {};
        for (int l = 0; l < s.nThetaReduced; ++l) {
          const std::size_t idx_jkl =
              ReducedIndex(local_surface, k, l, s.nZeta, s.nThetaReduced);
          const int idx_ml = m * s.nThetaReduced + l;
          work[0] += blmn_asym[idx_jkl] * fb.sinmumi[idx_ml];
          work[1] -= clmn_asym[idx_jkl] * fb.cosmui[idx_ml];
          work[2] += blmn_asym[idx_jkl] * fb.cosmumi[idx_ml];
          work[3] -= clmn_asym[idx_jkl] * fb.sinmui[idx_ml];
        }

        const int idx_kn_base = k * (s.nnyq2 + 1);
        const int idx_mn_base = ((jF - rp.nsMinF) * s.mpol + m) * (s.ntor + 1);
        for (int n = 0; n <= s.ntor; ++n) {
          const int idx_mn = idx_mn_base + n;
          const int idx_kn = idx_kn_base + n;
          m_physical_forces.flcc[idx_mn] +=
              work[0] * fb.cosnv[idx_kn] + work[1] * fb.sinnvn[idx_kn];
          m_physical_forces.flss[idx_mn] +=
              work[2] * fb.sinnv[idx_kn] + work[3] * fb.cosnvn[idx_kn];
        }
      }
    }
  }
}

}  // namespace vmecpp
