#ifndef VMECPP_TOOLS_DFT_BENCH_IDEAL_MHD_DATA_H_
#define VMECPP_TOOLS_DFT_BENCH_IDEAL_MHD_DATA_H_

#include <Eigen/Dense>

#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"

namespace dft_bench {

// Same data that is directly stored in a IdealMHDModel
// Note that the constructor gives vectors the correct size,
// but contents are simply set to zeros.
struct IdealMHDData {
  IdealMHDData(const vmecpp::Sizes& s, const vmecpp::RadialPartitioning& r);

  // used by ForcesToFourier
  Eigen::VectorXd armn_e;
  Eigen::VectorXd armn_o;
  Eigen::VectorXd azmn_e;
  Eigen::VectorXd azmn_o;
  Eigen::VectorXd blmn_e;
  Eigen::VectorXd blmn_o;
  Eigen::VectorXd brmn_e;
  Eigen::VectorXd brmn_o;
  Eigen::VectorXd bzmn_e;
  Eigen::VectorXd bzmn_o;
  Eigen::VectorXd clmn_e;
  Eigen::VectorXd clmn_o;
  Eigen::VectorXd crmn_e;
  Eigen::VectorXd crmn_o;
  Eigen::VectorXd czmn_e;
  Eigen::VectorXd czmn_o;
  Eigen::VectorXd frcon_e;
  Eigen::VectorXd frcon_o;
  Eigen::VectorXd fzcon_e;
  Eigen::VectorXd fzcon_o;

  // [mpol] xmpq[m] = f(m) = m*(m - 1) in constraint force
  Eigen::VectorXd xmpq;

  // used by FourierToReal
  Eigen::VectorXd r1_e;
  Eigen::VectorXd r1_o;
  Eigen::VectorXd ru_e;
  Eigen::VectorXd ru_o;
  Eigen::VectorXd rv_e;
  Eigen::VectorXd rv_o;
  Eigen::VectorXd z1_e;
  Eigen::VectorXd z1_o;
  Eigen::VectorXd zu_e;
  Eigen::VectorXd zu_o;
  Eigen::VectorXd zv_e;
  Eigen::VectorXd zv_o;
  Eigen::VectorXd lu_e;
  Eigen::VectorXd lu_o;
  Eigen::VectorXd lv_e;
  Eigen::VectorXd lv_o;
  Eigen::VectorXd rCon;
  Eigen::VectorXd zCon;

  int ivac;
};

}  // namespace dft_bench

#endif  // VMECPP_TOOLS_DFT_BENCH_IDEAL_MHD_DATA_H_
