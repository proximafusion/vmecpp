#include "vmecpp_tools/dft_bench/ideal_mhd_data.h"

#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"

dft_bench::IdealMHDData::IdealMHDData(const vmecpp::Sizes& s,
                                      const vmecpp::RadialPartitioning& r)
    : ivac(-1) {
  const int nrzt = s.nZnT * (r.nsMaxF - r.nsMinF);
  const int nrztIncludingBoundary = s.nZnT * (r.nsMaxFIncludingLcfs - r.nsMinF);

  armn_e.resize(nrzt);
  armn_e.setZero();
  armn_o.resize(nrzt);
  armn_o.setZero();
  azmn_e.resize(nrzt);
  azmn_e.setZero();
  azmn_o.resize(nrzt);
  azmn_o.setZero();

  blmn_e.resize(nrztIncludingBoundary);
  blmn_e.setZero();
  blmn_o.resize(nrztIncludingBoundary);
  blmn_o.setZero();
  brmn_e.resize(nrzt);
  brmn_e.setZero();
  brmn_o.resize(nrzt);
  brmn_o.setZero();
  bzmn_e.resize(nrzt);
  bzmn_e.setZero();
  bzmn_o.resize(nrzt);
  bzmn_o.setZero();

  clmn_e.resize(nrztIncludingBoundary);
  clmn_e.setZero();
  clmn_o.resize(nrztIncludingBoundary);
  clmn_o.setZero();
  crmn_e.resize(nrzt);
  crmn_e.setZero();
  crmn_o.resize(nrzt);
  crmn_o.setZero();
  czmn_e.resize(nrzt);
  czmn_e.setZero();
  czmn_o.resize(nrzt);
  czmn_o.setZero();

  frcon_e.resize(nrzt);
  frcon_e.setZero();
  frcon_o.resize(nrzt);
  frcon_o.setZero();
  fzcon_e.resize(nrzt);
  fzcon_e.setZero();
  fzcon_o.resize(nrzt);
  fzcon_o.setZero();

  xmpq.resize(s.mpol);
  for (int m = 0; m < s.mpol; ++m) {
    xmpq[m] = m * (m - 1);
  }

  const int nrzt1 = s.nZnT * (r.nsMaxF1 - r.nsMinF1);

  r1_e.resize(nrzt1);
  r1_e.setZero();
  r1_o.resize(nrzt1);
  r1_o.setZero();
  ru_e.resize(nrzt1);
  ru_e.setZero();
  ru_o.resize(nrzt1);
  ru_o.setZero();
  rv_e.resize(nrzt1);
  rv_e.setZero();
  rv_o.resize(nrzt1);
  rv_o.setZero();
  z1_e.resize(nrzt1);
  z1_e.setZero();
  z1_o.resize(nrzt1);
  z1_o.setZero();
  zu_e.resize(nrzt1);
  zu_e.setZero();
  zu_o.resize(nrzt1);
  zu_o.setZero();
  zv_e.resize(nrzt1);
  zv_e.setZero();
  zv_o.resize(nrzt1);
  zv_o.setZero();

  lu_e.resize(nrzt1);
  lu_e.setZero();
  lu_o.resize(nrzt1);
  lu_o.setZero();
  lv_e.resize(nrzt1);
  lv_e.setZero();
  lv_o.resize(nrzt1);
  lv_o.setZero();

  rCon.resize(nrztIncludingBoundary);
  rCon.setZero();
  zCon.resize(nrztIncludingBoundary);
  zCon.setZero();
}
