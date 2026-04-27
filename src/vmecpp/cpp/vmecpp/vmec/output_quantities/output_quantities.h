// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_OUTPUT_QUANTITIES_OUTPUT_QUANTITIES_H_
#define VMECPP_VMEC_OUTPUT_QUANTITIES_OUTPUT_QUANTITIES_H_

#include <Eigen/Dense>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "H5Cpp.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/util/real_type.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/handover_storage/handover_storage.h"
#include "vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.h"
#include "vmecpp/vmec/vmec_constants/vmec_constants.h"

namespace vmecpp {

// This is the data from inside VMEC, gathered from all threads,
// that form the basis of computing the output quantities.
struct VmecInternalResults {
  int sign_of_jacobian;

  // total number of full-grid points
  int num_full;

  // total number of half-grid points
  int num_half;

  // nZeta * nThetaReduced: always one half-period (for DFTs)
  int nZnT_reduced;

  Eigen::Matrix<real_t, Eigen::Dynamic, 1> sqrtSH;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> sqrtSF;

  Eigen::Matrix<real_t, Eigen::Dynamic, 1> sm;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> sp;

  // [ns] radial derivative of enclosed toroidal magnetic flux on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> phipF;

  // [ns] radial derivative of enclosed poloidal magnetic flux on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> chipF;

  // [ns - 1] radial derivative of enclosed toroidal magnetic flux on half-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> phipH;

  // [ns - 1] radial derivative of enclosed poloidal magnetic flux on half-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> chipH;

  // [ns - 1] enclosed current profile on half-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> currH;

  // [ns] enclosed toroidal magnetic flux on full-grid; computed in
  // RecomputeToroidalFlux here!
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> phiF;

  // [ns] rotational transform on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> iotaF;

  // [ns] surface-averaged spectral width profile on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> spectral_width;

  // enclosed poloidal current on half-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> bvcoH;

  // [num_half] d(volume)/ds on half-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> dVdsH;

  // [num_half] mass profile on half-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> massH;

  // [num_half] kinetic pressure on half-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> presH;

  // [num_half] rotational transform profile on half-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> iotaH;

  // -------------------
  // state vector
  // (num_full, mnsize)
  RowMatrixXr rmncc;
  RowMatrixXr rmnss;
  RowMatrixXr rmnsc;
  RowMatrixXr rmncs;

  RowMatrixXr zmnsc;
  RowMatrixXr zmncs;
  RowMatrixXr zmncc;
  RowMatrixXr zmnss;

  RowMatrixXr lmnsc;
  RowMatrixXr lmncs;
  RowMatrixXr lmncc;
  RowMatrixXr lmnss;

  // -------------------
  // from inv-DFTs

  // (num_full, nZnT)
  RowMatrixXr r_e;
  RowMatrixXr r_o;
  RowMatrixXr z_e;
  RowMatrixXr z_o;

  // dX/dTheta for R, Z on full-grid
  RowMatrixXr ru_e;
  RowMatrixXr ru_o;
  RowMatrixXr zu_e;
  RowMatrixXr zu_o;

  // dX/dZeta for R, Z on full-grid
  RowMatrixXr rv_e;
  RowMatrixXr rv_o;
  RowMatrixXr zv_e;
  RowMatrixXr zv_o;

  // -------------------
  // from even-m and odd-m contributions

  RowMatrixXr ruFull;
  RowMatrixXr zuFull;

  // -------------------
  // from Jacobian calculation

  // R on half-grid
  // (num_half, nZnT)
  RowMatrixXr r12;

  // dX/dTheta for R, Z on the half-grid
  RowMatrixXr ru12;
  RowMatrixXr zu12;

  // parts of dX/ds for R, Z on half-grid from Jacobian calculation
  RowMatrixXr rs;
  RowMatrixXr zs;

  // Jacobian on half-grid
  // (num_half, nZnT)
  RowMatrixXr gsqrt;

  // -------------------
  // metric elements

  RowMatrixXr guu;
  RowMatrixXr guv;
  RowMatrixXr gvv;

  // -------------------
  // magnetic field

  // contravariant magnetic field components on half-grid
  // (num_half, nZnT)
  RowMatrixXr bsupu;
  RowMatrixXr bsupv;

  // covariant magnetic field components on half-grid
  // (num_half, nZnT)
  RowMatrixXr bsubu;
  RowMatrixXr bsubv;

  // covariant magnetic field components on full-grid from lambda force
  // (num_full, nZnT)
  RowMatrixXr bsubvF;

  // (|B|^2 + mu_0 p) on half-grid
  // (num_half, nZnT)
  RowMatrixXr total_pressure;

  // -------------------
  // (more or less) directly from input data

  // mu_0 * curtor (from INDATA) -> prescribed toroidal current in A
  real_t currv;

  bool operator==(const VmecInternalResults&) const = default;
  bool operator!=(const VmecInternalResults& o) const { return !(*this == o); }

  // Write object to the specified HDF5 file, under key this->H5key.
  absl::Status WriteTo(H5::H5File& file) const;

  // Load contents of `from_file` into the specified instance.
  // The file is expected to have the same schema as the one produced by
  // WriteTo.
  static absl::Status LoadInto(VmecInternalResults& m_obj,
                               H5::H5File& from_file);

  static constexpr char H5key[] = "/vmec_internal_results";
};

struct PoloidalCurrentToFixBSubV {
  // [numHalf] poloidal current on half-grid
  Eigen::VectorXd poloidal_current_deviation;
};

struct RemainingMetric {
  // -------------------
  // specifically for bss and cylindrical components of B

  // dX/dZeta for R, Z on the half-grid
  // (num_half, nZnT)
  RowMatrixXr rv12;
  // (num_full, nZnT)
  RowMatrixXr zv12;

  // full dX/ds for R, Z on half-grid
  // (num_half, nZnT)
  RowMatrixXr rs12;
  // (num_full, nZnT)
  RowMatrixXr zs12;

  // metric elements on half-grid
  // (num_half, nZnT)
  RowMatrixXr gsu;
  // (num_full, nZnT)
  RowMatrixXr gsv;

  bool operator==(const RemainingMetric&) const = default;
  bool operator!=(const RemainingMetric& o) const { return !(*this == o); }

  // Write object to the specified HDF5 file, under key this->H5key.
  absl::Status WriteTo(H5::H5File& file) const;

  // Load contents of `from_file` into the specified instance.
  // The file is expected to have the same schema as the one produced by
  // WriteTo.
  static absl::Status LoadInto(RemainingMetric& m_obj, H5::H5File& from_file);

  static constexpr char H5key[] = "/remaining_metric";
};

struct CylindricalComponentsOfB {
  // cylindrical components of magnetic field on half-grid
  // (num_half, nZnT)
  RowMatrixXr b_r;
  RowMatrixXr b_phi;
  RowMatrixXr b_z;

  bool operator==(const CylindricalComponentsOfB&) const = default;
  bool operator!=(const CylindricalComponentsOfB& o) const {
    return !(*this == o);
  }

  // Write object to the specified HDF5 file, under key this->H5key.
  absl::Status WriteTo(H5::H5File& file) const;

  // Load contents of `from_file` into the specified instance.
  // The file is expected to have the same schema as the one produced by
  // WriteTo.
  static absl::Status LoadInto(CylindricalComponentsOfB& m_obj,
                               H5::H5File& from_file);

  static constexpr char H5key[] = "/cylindrical_components_of_b";
};

struct BSubSHalf {
  //  covariant magnetic field component on half-grid
  // (num_half, nZnT)
  RowMatrixXr bsubs_half;

  bool operator==(const BSubSHalf&) const = default;
  bool operator!=(const BSubSHalf& o) const { return !(*this == o); }

  // Write object to the specified HDF5 file, under key this->H5key.
  absl::Status WriteTo(H5::H5File& file) const;

  // Load contents of `from_file` into the specified instance.
  // The file is expected to have the same schema as the one produced by
  // WriteTo.
  static absl::Status LoadInto(BSubSHalf& m_obj, H5::H5File& from_file);

  static constexpr char H5key[] = "/bsubs_half";
};

struct BSubSFull {
  // covariant magnetic field component on full-grid
  // (num_full * nZnT)
  RowMatrixXr bsubs_full;

  bool operator==(const BSubSFull&) const = default;
  bool operator!=(const BSubSFull& o) const { return !(*this == o); }

  // Write object to the specified HDF5 file, under key this->H5key.
  absl::Status WriteTo(H5::H5File& file) const;

  // Load contents of `from_file` into the specified instance.
  // The file is expected to have the same schema as the one produced by
  // WriteTo.
  static absl::Status LoadInto(BSubSFull& m_obj, H5::H5File& from_file);

  static constexpr char H5key[] = "/bsubs_full";
};

struct SymmetryDecomposedCovariantB {
  // stellarator-symmetric B_s
  // (num_full, nZnT_reduced)
  RowMatrixXr bsubs_s;

  // non-stellarator-symmetric B_s
  // (num_full, nZnT_reduced)
  RowMatrixXr bsubs_a;

  // stellarator-symmetric B_theta
  // (num_half, nZnT_reduced)
  RowMatrixXr bsubu_s;

  // non-stellarator-symmetric B_theta
  // (num_half, nZnT_reduced)
  RowMatrixXr bsubu_a;

  //  stellarator-symmetric B_zeta
  // (num_half, nZnT_reduced)
  RowMatrixXr bsubv_s;

  // non-stellarator-symmetric B_zeta
  // [num_half x nZnT_reduced]
  RowMatrixXr bsubv_a;
};

struct CovariantBDerivatives {
  // d(B_s)/dTheta
  // (num_full, nZnT)
  RowMatrixXr bsubsu;

  // d(B_s)/dZeta
  // (num_full, nZnT)
  RowMatrixXr bsubsv;

  // d(B_theta)/dZeta
  // (num_half, nZnT)
  RowMatrixXr bsubuv;

  // d(B_zeta)/dTheta
  // (num_half, nZnT)
  RowMatrixXr bsubvu;

  bool operator==(const CovariantBDerivatives&) const = default;
  bool operator!=(const CovariantBDerivatives& o) const {
    return !(*this == o);
  }

  // Write object to the specified HDF5 file, under key this->H5key.
  absl::Status WriteTo(H5::H5File& file) const;

  // Load contents of `from_file` into the specified instance.
  // The file is expected to have the same schema as the one produced by
  // WriteTo.
  static absl::Status LoadInto(CovariantBDerivatives& m_obj,
                               H5::H5File& from_file);

  static constexpr char H5key[] = "/covariant_b_derivatives";
};

struct JxBOutFileContents {
  // (num_full, nZnT)
  RowMatrixXr itheta;
  RowMatrixXr izeta;
  RowMatrixXr bdotk;

  Eigen::Matrix<real_t, Eigen::Dynamic, 1> amaxfor;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> aminfor;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> avforce;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> pprim;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> jdotb;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> bdotb;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> bdotgradv;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> jpar2;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> jperp2;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> phin;

  // (num_full, nZnT)
  RowMatrixXr jsupu3;
  RowMatrixXr jsupv3;

  // (num_half, nZnT)
  RowMatrixXr jsups3;

  // (num_full, nZnT)
  RowMatrixXr bsupu3;
  RowMatrixXr bsupv3;
  RowMatrixXr jcrossb;
  RowMatrixXr jxb_gradp;
  RowMatrixXr jdotb_sqrtg;
  RowMatrixXr sqrtg3;

  // (num_half, nZnT)
  RowMatrixXr bsubu3;
  RowMatrixXr bsubv3;

  // (num_full, nZnT)
  RowMatrixXr bsubs3;

  bool operator==(const JxBOutFileContents&) const = default;
  bool operator!=(const JxBOutFileContents& o) const { return !(*this == o); }

  // Write object to the specified HDF5 file, under key this->H5key.
  absl::Status WriteTo(H5::H5File& file) const;

  // Load contents of `from_file` into the specified instance.
  // The file is expected to have the same schema as the one produced by
  // WriteTo.
  static absl::Status LoadInto(JxBOutFileContents& m_obj,
                               H5::H5File& from_file);

  static constexpr char H5key[] = "/jxbout";
};

struct MercierStabilityIntermediateQuantities {
  // normalized toroidal flux on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> s;

  // magnetic shear == radial derivative of iota on full grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> shear;

  // magnetic well == d^2V/ds^2 on full grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> vpp;

  // radial derivative of kinetic pressure on full grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> d_pressure_d_s;

  // d(I_tor)/ds on full grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> d_toroidal_current_d_s;

  // real, physical d(phi)/ds on half-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> phip_realH;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> phip_realF;

  // dV/d(PHI) on half mesh
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> vp_real;

  // toroidal current on half-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> torcur;

  // Jacobian on full-grid
  // (num_full, nZnT)
  RowMatrixXr gsqrt_full;

  // B \cdot j on full-grid
  // (num_full, nZnT)
  RowMatrixXr bdotj;

  // 1.0 / gpp on full-grid
  // (num_full, nZnT)
  // TODO(jons): figure out what this really is
  RowMatrixXr gpp;

  // |B|^2 on half grid
  // (num_half, nZnT)
  RowMatrixXr b2;

  // <1/B**2> on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> tpp;

  // <b*b/|grad-phi|**3> on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> tbb;

  // <j*b/|grad-phi|**3>
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> tjb;

  // <(j*b)2/b**2*|grad-phi|**3>
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> tjj;

  bool operator==(const MercierStabilityIntermediateQuantities&) const =
      default;
  bool operator!=(const MercierStabilityIntermediateQuantities& o) const {
    return !(*this == o);
  }

  // Write object to the specified HDF5 file, under key this->H5key.
  absl::Status WriteTo(H5::H5File& file) const;

  // Load contents of `from_file` into the specified instance.
  // The file is expected to have the same schema as the one produced by
  // WriteTo.
  static absl::Status LoadInto(MercierStabilityIntermediateQuantities& m_obj,
                               H5::H5File& from_file);

  static constexpr char H5key[] = "/mercier_intermediate";
};

struct MercierFileContents {
  // normalized toroidal flux on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> s;

  // -------------------

  // toroidal magnetic flux on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> toroidal_flux;

  // rotational transform on full grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> iota;

  // magnetic shear == radial derivative of iota on full grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> shear;

  // dV/ds on full grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> d_volume_d_s;

  // magnetic well == d^2V/ds^2 on full grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> well;

  // I_tor on full grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> toroidal_current;

  // d(I_tor)/ds on full grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> d_toroidal_current_d_s;

  // kinetic pressure on full grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> pressure;

  // radial derivative of kinetic pressure on full grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> d_pressure_d_s;

  // -------------------

  // Mercier criterion on full grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> DMerc;

  // shear contribution to Mercier criterion on full grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> Dshear;

  // magnetic well contribution to Mercier criterion on full grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> Dwell;

  // toroidal current contribution to Mercier criterion on full grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> Dcurr;

  // geodesic curvature contribution to Mercier criterion on full grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> Dgeod;

  bool operator==(const MercierFileContents&) const = default;
  bool operator!=(const MercierFileContents& o) const { return !(*this == o); }

  // Write object to the specified HDF5 file, under key this->H5key.
  absl::Status WriteTo(H5::H5File& file) const;

  // Load contents of `from_file` into the specified instance.
  // The file is expected to have the same schema as the one produced by
  // WriteTo.
  static absl::Status LoadInto(MercierFileContents& m_obj,
                               H5::H5File& from_file);

  static constexpr char H5key[] = "/mercier";
};

struct Threed1FirstTableIntermediate {
  // ready-to-integrate Jacobian on half-grid
  // (num_half, nZnT)
  RowMatrixXr tau;

  // [num_half] surface-averaged beta profile
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> beta_vol;

  // [num_half] <tau / R> / V'
  // TODO(jons): figure out what this really is
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> overr;

  // plasma beta on magnetic axis
  real_t beta_axis;

  // [num_full] pressure on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> presf;

  // [num_full] phip * 2 * pi * sign_of_jacobian on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> phipf_loc;

  // [num_full] toroidal flux profile on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> phi1;

  // [num_full] poloidal flux profile on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> chi1;

  // [num_full] 2 pi * poloidal flux profile on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> chi;

  Eigen::Matrix<real_t, Eigen::Dynamic, 1> bvcoH;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> bucoH;

  // [num_full] toroidal current density profile on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> jcurv;

  // [num_full] poloidal current density profile on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> jcuru;

  // [num_full] radial derivative of pressure on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> presgrad;

  // [num_full] dV/d(phi) on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> vpphi;

  // [num_full] radial force balance residual on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> equif;

  // [num_full] toroidal current profile on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> bucof;

  // [num_full] poloidal current profile on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> bvcof;

  bool operator==(const Threed1FirstTableIntermediate&) const = default;
  bool operator!=(const Threed1FirstTableIntermediate& o) const {
    return !(*this == o);
  }

  // Write object to the specified HDF5 file, under key this->H5key.
  absl::Status WriteTo(H5::H5File& file) const;

  // Load contents of `from_file` into the specified instance.
  // The file is expected to have the same schema as the one produced by
  // WriteTo.
  static absl::Status LoadInto(Threed1FirstTableIntermediate& m_obj,
                               H5::H5File& from_file);

  static constexpr char H5key[] = "/threed1_first_table_intermediate";
};

struct Threed1FirstTable {
  // [num_full] S: normalized toroidal flux on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> s;

  // [num_full] <RADIAL FORCE>: radial force balance residual on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> radial_force;

  // [num_full] TOROIDAL FLUX: toroidal flux profile on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> toroidal_flux;

  // [num_full] IOTA: rotational transform profile on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> iota;

  // [num_full] <JSUPU>: surface-averaged poloidal current density on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> avg_jsupu;

  // [num_full] <JSUPV>: surface-averaged toroidal current density on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> avg_jsupv;

  // [num_full] d(VOL)/d(PHI): differential volume on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> d_volume_d_phi;

  // [num_full] d(PRES)/d(PHI): radial derivative of pressure on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> d_pressure_d_phi;

  // [num_full] <M>: surface-averaged spectral width profile on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> spectral_width;

  // [num_full] PRESF: pressure on full-grid in Pa (no mu_0!)
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> pressure;

  // [num_full] <BSUBU>: toroidal current profile on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> buco_full;

  // [num_full] <BSUBV>: poloidal current profile on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> bvco_full;

  // [num_full] <J.B>: parallel current density profile on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> j_dot_b;

  // [num_full] <B.B>: <|B|^2> profile on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> b_dot_b;

  bool operator==(const Threed1FirstTable&) const = default;
  bool operator!=(const Threed1FirstTable& o) const { return !(*this == o); }

  // Write object to the specified HDF5 file, under key this->H5key.
  absl::Status WriteTo(H5::H5File& file) const;

  // Load contents of `from_file` into the specified instance.
  // The file is expected to have the same schema as the one produced by
  // WriteTo.
  static absl::Status LoadInto(Threed1FirstTable& m_obj, H5::H5File& from_file);

  static constexpr char H5key[] = "/threed1_first_table";
};

struct Threed1GeometricAndMagneticQuantitiesIntermediate {
  real_t anorm;
  real_t vnorm;

  // differential surface area element |dS|, already with poloidal integration
  // weights
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> surf_area;
  real_t circumference_sum;

  real_t rcenin;
  real_t aminr2in;
  real_t bminz2in;
  real_t bminz2;

  real_t sump;

  Eigen::Matrix<real_t, Eigen::Dynamic, 1> btor_vac;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> btor1;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> dbtor;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> phat;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> redge;

  real_t delphid_exact;
  real_t musubi;
  real_t rshaf1;
  real_t rshaf2;
  real_t rshaf;

  real_t fpsi0;

  real_t sumbtot;
  real_t sumbtor;
  real_t sumbpol;
  real_t sump20;
  real_t sump2;

  Eigen::Matrix<real_t, Eigen::Dynamic, 1> jPS2;
  real_t jpar_perp_sum;
  real_t jparPS_perp_sum;
  real_t s2;

  real_t fac;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> r3v;

  bool operator==(
      const Threed1GeometricAndMagneticQuantitiesIntermediate&) const = default;
  bool operator!=(
      const Threed1GeometricAndMagneticQuantitiesIntermediate& o) const {
    return !(*this == o);
  }

  // Write object to the specified HDF5 file, under key this->H5key.
  absl::Status WriteTo(H5::H5File& file) const;

  // Load contents of `from_file` into the specified instance.
  // The file is expected to have the same schema as the one produced by
  // WriteTo.
  static absl::Status LoadInto(
      Threed1GeometricAndMagneticQuantitiesIntermediate& obj,
      H5::H5File& from_file);

  static constexpr char H5key[] =
      "/threed1_geometric_and_magnetic_quantities_intermediate";
};

struct Threed1GeometricAndMagneticQuantities {
  real_t toroidal_flux;

  real_t circum_p;
  real_t surf_area_p;

  real_t cross_area_p;
  real_t volume_p;

  real_t Rmajor_p;
  real_t Aminor_p;
  real_t aspect;

  real_t kappa_p;
  real_t rcen;

  // volume-averaged minor radius
  real_t aminr1;

  real_t pavg;
  real_t factor;

  real_t b0;

  real_t rmax_surf;
  real_t rmin_surf;
  real_t zmax_surf;

  // (num_half, nThetaReduced)
  RowMatrixXr bmin;
  // (num_half, nThetaReduced)
  RowMatrixXr bmax;

  Eigen::Matrix<real_t, Eigen::Dynamic, 1> waist;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> height;

  real_t betapol;
  real_t betatot;
  real_t betator;
  real_t VolAvgB;
  real_t IonLarmor;

  real_t jpar_perp;
  real_t jparPS_perp;

  // net toroidal current in A
  real_t toroidal_current;

  real_t rbtor;
  real_t rbtor0;

  // poloidal magnetic flux on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> psi;

  // Geometric minor radius
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> ygeo;

  // Geometric indentation
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> yinden;

  // Geometric ellipticity
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> yellip;

  // Geometric triangularity
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> ytrian;

  // Geometric shift measured from magnetic axis
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> yshift;

  Eigen::Matrix<real_t, Eigen::Dynamic, 1> loc_jpar_perp;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> loc_jparPS_perp;

  bool operator==(const Threed1GeometricAndMagneticQuantities&) const = default;
  bool operator!=(const Threed1GeometricAndMagneticQuantities& o) const {
    return !(*this == o);
  }

  // Write object to the specified HDF5 file, under key this->H5key.
  absl::Status WriteTo(H5::H5File& file) const;

  // Load contents of `from_file` into the specified instance.
  // The file is expected to have the same schema as the one produced by
  // WriteTo.
  static absl::Status LoadInto(Threed1GeometricAndMagneticQuantities& m_obj,
                               H5::H5File& from_file);

  static constexpr char H5key[] = "/threed1_geometric_and_magnetic_quantities";
};

// Volume Integrals (Joules) and Volume Averages (Pascals)
struct Threed1Volumetrics {
  real_t int_p;
  real_t avg_p;

  real_t int_bpol;
  real_t avg_bpol;

  real_t int_btor;
  real_t avg_btor;

  real_t int_modb;
  real_t avg_modb;

  real_t int_ekin;
  real_t avg_ekin;

  bool operator==(const Threed1Volumetrics&) const = default;
  bool operator!=(const Threed1Volumetrics& o) const { return !(*this == o); }

  // Write object to the specified HDF5 file, under key this->H5key.
  absl::Status WriteTo(H5::H5File& file) const;

  // Load contents of `from_file` into the specified instance.
  // The file is expected to have the same schema as the one produced by
  // WriteTo.
  static absl::Status LoadInto(Threed1Volumetrics& m_obj,
                               H5::H5File& from_file);

  static constexpr char H5key[] = "/threed1_volumetrics";
};

// geometry of the magnetic axis, as written to the threed1 file by Fortran VMEC
struct Threed1AxisGeometry {
  // [ntor + 1] stellarator-symmetric Fourier coefficients of axis: R * cos(n *
  // zeta)
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> raxis_symm;

  // [ntor + 1] stellarator-symmetric Fourier coefficients of axis: Z * sin(n *
  // zeta)
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> zaxis_symm;

  // [ntor + 1] non-stellarator-symmetric Fourier coefficients of axis: R *
  // sin(n * zeta)
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> raxis_asym;

  // [ntor + 1] non-stellarator-symmetric Fourier coefficients of axis: Z *
  // cos(n * zeta)
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> zaxis_asym;

  bool operator==(const Threed1AxisGeometry&) const = default;
  bool operator!=(const Threed1AxisGeometry& o) const { return !(*this == o); }

  // Write object to the specified HDF5 file, under key this->H5key.
  absl::Status WriteTo(H5::H5File& file) const;

  // Load contents of `from_file` into the specified instance.
  // The file is expected to have the same schema as the one produced by
  // WriteTo.
  static absl::Status LoadInto(Threed1AxisGeometry& m_obj,
                               H5::H5File& from_file);

  static constexpr char H5key[] = "/threed1_axis_geometry";
};

// beta values from volume averages over plasma
struct Threed1Betas {
  // beta total
  real_t betatot;

  // beta poloidal
  real_t betapol;

  // beta toroidal
  real_t betator;

  // R * Btor-vac
  real_t rbtor;

  // Peak Beta (on axis)
  real_t betaxis;

  // Beta-star
  real_t betstr;

  bool operator==(const Threed1Betas&) const = default;
  bool operator!=(const Threed1Betas& o) const { return !(*this == o); }

  // Write object to the specified HDF5 file, under key this->H5key.
  absl::Status WriteTo(H5::H5File& file) const;

  // Load contents of `from_file` into the specified instance.
  // The file is expected to have the same schema as the one produced by
  // WriteTo.
  static absl::Status LoadInto(Threed1Betas& m_obj, H5::H5File& from_file);

  static constexpr char H5key[] = "/threed1_betas";
};

// Shafranov Surface Integrals
//
// Ref: S. P. Hirshman, Phys. Fluids B, 5, (1993) 3119
//
// Note: s1 = S1/2, s2 = S2/2, where s1,s2 are the Shafranov definitions,
//       and s3 = S3/2, where S3 is Lao's definition.
//
// The quantity lsubi gives the ratio of volume poloidal field energy
// to the field energy estimated from the surface integral in Eq. (8).
struct Threed1ShafranovIntegrals {
  real_t scaling_ratio;

  real_t r_lao;
  real_t f_lao;
  real_t f_geo;

  real_t smaleli;
  real_t betai;
  real_t musubi;
  real_t lambda;

  real_t s11;
  real_t s12;
  real_t s13;
  real_t s2;
  real_t s3;

  real_t delta1;
  real_t delta2;
  real_t delta3;

  bool operator==(const Threed1ShafranovIntegrals&) const = default;
  bool operator!=(const Threed1ShafranovIntegrals& o) const {
    return !(*this == o);
  }

  // Write object to the specified HDF5 file, under key this->H5key.
  absl::Status WriteTo(H5::H5File& file) const;

  // Load contents of `from_file` into the specified instance.
  // The file is expected to have the same schema as the one produced by
  // WriteTo.
  static absl::Status LoadInto(Threed1ShafranovIntegrals& m_obj,
                               H5::H5File& from_file);

  static constexpr char H5key[] = "/threed1_shafranov_integrals";
};

struct WOutFileContents {
  // -------------------
  // copy of input data

  // version identifier for this VMEC implementation
  real_t version_;

  std::string input_extension;

  // sign of Jacobian between cylindrical and flux coordinates; hardcoded to -1
  int signgs;

  // adiabatic index
  real_t gamma;

  // parameterization identifier for toroidal current profile
  std::string pcurr_type;

  // parameterization identifier for mass/pressure profile
  std::string pmass_type;

  // parameterization identifier for iota profile
  std::string piota_type;

  // pressure profile coefficients
  // interpretation depends on value of `pmass_type`
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> am;

  // toroidal current profile coefficients
  // interpretation depends on value of `pcurr_type`
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> ac;

  // iota profile coefficients
  // interpretation depends on value of `piota_type`
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> ai;

  // knots for discrete mass/pressure profile
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> am_aux_s;

  // values for discrete mass/pressure profile
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> am_aux_f;

  // knots for discrete toroidal current profile
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> ac_aux_s;

  // values for discrete toroidal current profile
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> ac_aux_f;

  // knots for discrete iota profile
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> ai_aux_s;

  // values for discrete iota profile
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> ai_aux_f;

  // number of toroidal field periods
  int nfp;

  // poloidal Fourier resolution; m = 0, 1, ..., (mpol-1)
  int mpol;

  // toroidal Fourier resolution; n = -ntor, ..., -1, 0, 1, ..., ntor
  int ntor;

  // flag to indicate a non-stellarator-symmetric case
  bool lasym = false;

  // flag to indicate whether iota or safety factor (q) is used as input
  bool lrfp = false;

  // final radial resolution
  // == number of flux surfaces
  // == number of full-grid points
  int ns;

  // requested force tolerance level for convergence
  double ftolv;

  // iterations required until convergence
  int niter;

  // flag to indicate a free-boundary run
  bool lfreeb = false;

  // path to file which contains the magnetic field response tables
  std::string mgrid_file;

  // number of external coil currents
  int nextcur;

  // external coil currents
  Eigen::VectorXd extcur;

  // 'R': mgrid file contains un-normalized response tables
  // 'S': mgrid file contains response tables normalized to unit currents
  std::string mgrid_mode;

  // -------------------
  // scalar quantities

  // magnetic energy
  real_t wb;

  // thermal energy
  real_t wp;

  // maximum R of LCFS
  real_t rmax_surf;

  // minimum R of LCFS
  real_t rmin_surf;

  // maximum |Z| of LCFS
  real_t zmax_surf;

  // number of Fourier coefficients in state vector (R, Z, lambda)
  int mnmax;

  // number of Fourier coefficients for derived quantities, Nyquist-extended
  // (sqrt(g), |B|, co- and contravariant B components, currents)
  int mnmax_nyq;

  int ier_flag;

  real_t aspect;

  real_t betatotal;
  real_t betapol;
  real_t betator;
  real_t betaxis;

  real_t b0;

  real_t rbtor0;
  real_t rbtor;

  real_t IonLarmor;
  real_t volavgB;

  real_t ctor;

  real_t Aminor_p;
  real_t Rmajor_p;
  real_t volume;

  real_t fsqr;
  real_t fsqz;
  real_t fsql;

  // Number of "time steps" of the force relaxation that were actually required
  // to achieve convergence. (How many of the maximum niter steps we ended up
  // using.)
  int itfsq;

  // -------------------
  // one-dimensional array quantities

  // full-grid: rotational_transform
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> iotaf;

  // full-grid: 1 / iota (where iota != 0)
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> q_factor;

  // full-grid: pressure in Pa
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> presf;

  // full-grid: enclosed toroidal magnetic flux (phi) in Vs
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> phi;

  // full-grid: toroidal flux differential (phi-prime)
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> phipf;

  // full-grid: enclosed poloidal magnetic flux (chi) in Vs
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> chi;

  // full-grid: poloidal flux differential (chi-prime)
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> chipf;

  Eigen::Matrix<real_t, Eigen::Dynamic, 1> jcuru;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> jcurv;

  // Convergence quantities (one entry per time step)
  // Force residual at each iteration
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> fsqt;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> force_residual_r;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> force_residual_z;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> force_residual_lambda;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> delbsq;
  Eigen::VectorXi restart_reason_timetrace;

  // Gradient of the energy at each iteration
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> wdot;

  // ---------

  Eigen::Matrix<real_t, Eigen::Dynamic, 1> iotas;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> mass;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> pres;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> beta_vol;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> buco;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> bvco;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> vp;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> specw;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> phips;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> over_r;

  Eigen::Matrix<real_t, Eigen::Dynamic, 1> jdotb;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> bdotb;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> bdotgradv;

  Eigen::Matrix<real_t, Eigen::Dynamic, 1> DMerc;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> DShear;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> DWell;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> DCurr;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> DGeod;

  Eigen::Matrix<real_t, Eigen::Dynamic, 1> equif;

  std::vector<std::string> curlabel;

  // currently unused
  Eigen::VectorXd potvac;

  // -------------------
  // mode numbers for Fourier coefficient arrays below

  Eigen::VectorXi xm;
  Eigen::VectorXi xn;
  Eigen::VectorXi xm_nyq;
  Eigen::VectorXi xn_nyq;

  // -------------------
  // stellarator-symmetric Fourier coefficients

  Eigen::Matrix<real_t, Eigen::Dynamic, 1> raxis_cc;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> zaxis_cs;

  // full-grid: R
  RowMatrixXr rmnc;

  // full-grid: Z
  RowMatrixXr zmns;

  // full-grid: lambda
  // NOTE: new with respect to Fortran VMEC
  RowMatrixXr lmns_full;

  // half-grid: lambda
  RowMatrixXr lmns;

  // half-grid: Jacobian
  RowMatrixXr gmnc;

  // half-grid: |B|
  RowMatrixXr bmnc;

  // half-grid: covariant B_\theta
  RowMatrixXr bsubumnc;

  // half-grid: covariant B_\zeta
  RowMatrixXr bsubvmnc;

  // half-grid: covariant B_s
  RowMatrixXr bsubsmns;

  // full-grid: covariant B_s
  // NOTE: new with respect to Fortran VMEC
  RowMatrixXr bsubsmns_full;

  // half-grid: contravariant B^\theta
  RowMatrixXr bsupumnc;

  // half-grid: contravariant B^\zeta
  RowMatrixXr bsupvmnc;

  // full-grid: sqrt(g) * J^\theta, Fourier coefficients (cos)
  RowMatrixXr currumnc;

  // full-grid: sqrt(g) * J^\zeta, Fourier coefficients (cos)
  RowMatrixXr currvmnc;

  // -------------------
  // non-stellarator-symmetric Fourier coefficients

  Eigen::Matrix<real_t, Eigen::Dynamic, 1> raxis_cs;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> zaxis_cc;

  // full-grid: R
  RowMatrixXr rmns;

  // full-grid: Z
  RowMatrixXr zmnc;

  // full-grid: lambda
  // NOTE: new with respect to Fortran VMEC
  RowMatrixXr lmnc_full;

  // half-grid: lambda
  RowMatrixXr lmnc;

  // half-grid: Jacobian
  RowMatrixXr gmns;

  // half-grid: |B|
  RowMatrixXr bmns;

  // half-grid: covariant B_\theta
  RowMatrixXr bsubumns;

  // half-grid: covariant B_\zeta
  RowMatrixXr bsubvmns;

  // half-grid: covariant B_s
  RowMatrixXr bsubsmnc;

  // full-grid: covariant B_s
  // NOTE: new with respect to Fortran VMEC
  RowMatrixXr bsubsmnc_full;

  // half-grid: contravariant B^\theta
  RowMatrixXr bsupumns;

  // half-grid: contravariant B^\zeta
  RowMatrixXr bsupvmns;

  // full-grid: sqrt(g) * J^\theta, Fourier coefficients (sin)
  RowMatrixXr currumns;

  // full-grid: sqrt(g) * J^\zeta, Fourier coefficients (sin)
  RowMatrixXr currvmns;

  bool operator==(const WOutFileContents&) const = default;
  bool operator!=(const WOutFileContents& o) const { return !(*this == o); }

  // Write object to the specified HDF5 file, under key this->H5key.
  absl::Status WriteTo(H5::H5File& file) const;

  // Load contents of `from_file` into the specified instance.
  // The file is expected to have the same schema as the one produced by
  // WriteTo.
  static absl::Status LoadInto(WOutFileContents& m_obj, H5::H5File& from_file);

  static constexpr char H5key[] = "/wout";
};

// Output quantities from VMEC++
// that would normally end up in the various output file(s).
struct OutputQuantities {
  VmecInternalResults vmec_internal_results;
  RemainingMetric remaining_metric;
  CylindricalComponentsOfB b_cylindrical;
  BSubSHalf bsubs_half;
  BSubSFull bsubs_full;
  CovariantBDerivatives covariant_b_derivatives;
  JxBOutFileContents jxbout;
  MercierStabilityIntermediateQuantities mercier_intermediate;
  MercierFileContents mercier;
  Threed1FirstTableIntermediate threed1_first_table_intermediate;
  Threed1FirstTable threed1_first_table;
  Threed1GeometricAndMagneticQuantitiesIntermediate
      threed1_geometric_magnetic_intermediate;
  Threed1GeometricAndMagneticQuantities threed1_geometric_magnetic;
  Threed1Volumetrics threed1_volumetrics;
  Threed1AxisGeometry threed1_axis;
  Threed1Betas threed1_betas;
  Threed1ShafranovIntegrals threed1_shafranov_integrals;
  WOutFileContents wout;
  VmecINDATA indata;

  bool operator==(const OutputQuantities&) const = default;
  bool operator!=(const OutputQuantities& o) const { return !(*this == o); }

  // Write the output quantities to the HDF5 file at the specified path.
  // If a file already exists, it is overwritten.
  absl::Status Save(const std::filesystem::path& path) const;

  // Return a OutputQuantities instance populated with the contents of the
  // specified HDF5 file. The file is expected to have the same schema as the
  // one produced by OutputQuantities::Save.
  static absl::StatusOr<OutputQuantities> Load(
      const std::filesystem::path& path);
};

// Compute the output quantities of VMEC++.
// With respect to Fortran VMEC, this is equivalent to the fileout subroutine,
// but without the actual file writing routines.
OutputQuantities ComputeOutputQuantities(
    int sign_of_jacobian, const VmecINDATA& indata, const Sizes& s,
    const FlowControl& fc, const VmecConstants& constants,
    const FourierBasisFastPoloidal& t, const HandoverStorage& h,
    const std::string& mgrid_mode,
    const std::vector<std::unique_ptr<RadialPartitioning> >&
        radial_partitioning,
    const std::vector<std::unique_ptr<FourierGeometry> >& decomposed_x,
    const std::vector<std::unique_ptr<IdealMhdModel> >& models_from_threads,
    const std::vector<std::unique_ptr<RadialProfiles> >& radial_profiles,
    const VmecCheckpoint& checkpoint, VacuumPressureState vacuum_pressure_state,
    VmecStatus vmec_status, int iter2);

// gather data from all threads into the main thread
VmecInternalResults GatherDataFromThreads(
    int sign_of_jacobian, const Sizes& s, const FlowControl& fc,
    const VmecConstants& constants,
    const std::vector<std::unique_ptr<RadialPartitioning> >&
        radial_partitioning,
    const std::vector<std::unique_ptr<FourierGeometry> >& decomposed_x,
    const std::vector<std::unique_ptr<IdealMhdModel> >& models_from_threads,
    const std::vector<std::unique_ptr<RadialProfiles> >& radial_profiles);

// mesh blending for B_zeta back to half-grid
void MeshBledingBSubZeta(const Sizes& s, const FlowControl& fc,
                         VmecInternalResults& m_vmec_internal_results);

PoloidalCurrentToFixBSubV ComputePoloidalCurrentToFixBSubV(
    const Sizes& s, const VmecInternalResults& vmec_internal_results);

// ADJUST <bsubvh> AFTER MESH-BLENDING
void FixupPoloidalCurrent(
    const Sizes& s,
    const PoloidalCurrentToFixBSubV& poloidal_current_to_fix_bsubv,
    VmecInternalResults& m_vmec_internal_results);

// re-compute the enclosed toroidal flux from its derivative by quadrature
void RecomputeToroidalFlux(const FlowControl& fc,
                           VmecInternalResults& m_vmec_internal_results);

RemainingMetric ComputeRemainingMetric(
    const Sizes& s, const VmecInternalResults& vmec_internal_results);

// compute the cylindrical components of B
CylindricalComponentsOfB BCylindricalComponents(
    const Sizes& s, const VmecInternalResults& vmec_internal_results,
    const RemainingMetric& remaining_metric);

// compute B_s on half-grid
BSubSHalf ComputeBSubSOnHalfGrid(
    const Sizes& s, const VmecInternalResults& vmec_internal_results,
    const RemainingMetric& remaining_metric);

// linear interpolation of B_s onto full-grid
BSubSFull PutBSubSOnFullGrid(const Sizes& s,
                             const VmecInternalResults& vmec_internal_results,
                             const BSubSHalf& bsubs_half);

SymmetryDecomposedCovariantB DecomposeCovariantBBySymmetry(
    const Sizes& s, const VmecInternalResults& vmec_internal_results,
    const BSubSFull& bsubs_full);

// Fourier-low-pass-filter covariant components of magnetic field
CovariantBDerivatives LowPassFilterCovariantB(
    const Sizes& s, const FourierBasisFastPoloidal& t,
    const SymmetryDecomposedCovariantB& decomposed_bcov,
    VmecInternalResults& m_vmec_internal_results);

// extarpolate B_s on full-grid to axis and boundary
void ExtrapolateBSubS(const Sizes& s, const FlowControl& fc,
                      BSubSFull& m_bsubs_full);

JxBOutFileContents ComputeJxBOutputFileContents(
    const Sizes& s, const FlowControl& fc,
    const VmecInternalResults& vmec_internal_results,
    const BSubSFull& bsubs_full,
    const CovariantBDerivatives& covariant_b_derivatives,
    const bool return_outputs_even_if_not_converged, VmecStatus vmec_status);

MercierStabilityIntermediateQuantities ComputeIntermediateMercierQuantities(
    const Sizes& s, const FlowControl& fc,
    const VmecInternalResults& vmec_internal_results,
    const JxBOutFileContents& jxbout);

MercierFileContents ComputeMercierStability(
    const FlowControl& fc, const VmecInternalResults& vmec_internal_results,
    const MercierStabilityIntermediateQuantities& mercier_intermediate);

Threed1FirstTableIntermediate ComputeIntermediateThreed1FirstTableQuantities(
    const Sizes& s, const FlowControl& fc,
    const VmecInternalResults& vmec_internal_results);

Threed1FirstTable ComputeThreed1FirstTable(
    const FlowControl& fc, const VmecInternalResults& vmec_internal_results,
    const JxBOutFileContents& jxbout,
    const Threed1FirstTableIntermediate& threed1_first_table_intermediate);

Threed1GeometricAndMagneticQuantitiesIntermediate
ComputeIntermediateThreed1GeometricMagneticQuantities(
    const Sizes& s, const FlowControl& fc,
    const HandoverStorage& handover_storage,
    const VmecInternalResults& vmec_internal_results,
    const JxBOutFileContents& jxbout,
    const Threed1FirstTableIntermediate& threed1_first_table_intermediate,
    VacuumPressureState vacuum_pressure_state);

Threed1GeometricAndMagneticQuantities ComputeThreed1GeometricMagneticQuantities(
    const Sizes& s, const FlowControl& fc,
    const HandoverStorage& handover_storage,
    const VmecInternalResults& vmec_internal_results,
    const JxBOutFileContents& jxbout,
    const Threed1FirstTableIntermediate& threed1_first_table_intermediate,
    const Threed1GeometricAndMagneticQuantitiesIntermediate&
        threed1_geometric_magnetic_intermediate);

Threed1Volumetrics ComputeThreed1Volumetrics(
    const Threed1GeometricAndMagneticQuantitiesIntermediate&
        threed1_geometric_magnetic_intermediate,
    const Threed1GeometricAndMagneticQuantities& threed1_geomag);

Threed1AxisGeometry ComputeThreed1AxisGeometry(
    const Sizes& s, const FourierBasisFastPoloidal& fourier_basis,
    const VmecInternalResults& vmec_internal_results);

Threed1Betas ComputeThreed1Betas(
    const HandoverStorage& handover_storage,
    const Threed1FirstTableIntermediate& threed1_first_table_intermediate,
    const Threed1GeometricAndMagneticQuantitiesIntermediate&
        threed1_geomag_intermediate,
    const Threed1GeometricAndMagneticQuantities& threed1_geomag);

Threed1ShafranovIntegrals ComputeThreed1ShafranovIntegrals(
    const Sizes& s, const FlowControl& fc,
    const HandoverStorage& handover_storage,
    const VmecInternalResults& vmec_internal_results,
    const Threed1GeometricAndMagneticQuantitiesIntermediate&
        threed1_geometric_magnetic_intermediate,
    const Threed1GeometricAndMagneticQuantities& threed1_geomag,
    VacuumPressureState vacuum_pressure_state);

WOutFileContents ComputeWOutFileContents(
    const VmecINDATA& indata, const Sizes& s, const FourierBasisFastPoloidal& t,
    const FlowControl& fc, const VmecConstants& constants,
    const HandoverStorage& handover_storage, const std::string& mgrid_mode,
    VmecInternalResults& m_vmec_internal_results, const BSubSHalf& bsubs_half,
    const MercierFileContents& mercier, const JxBOutFileContents& jxbout,
    const Threed1FirstTableIntermediate& threed1_first_table_intermediate,
    const Threed1FirstTable& threed1_first_table,
    const Threed1GeometricAndMagneticQuantities& threed1_geomag,
    const Threed1AxisGeometry& threed1_axis, const Threed1Betas& threed1_betas,
    VmecStatus vmec_status, int iter2);

// Compare the contents of a test wout object against a reference wout object,
// exiting with an error in case of mismatches.
// The comparison is performed using the specified tolerance in the "relabs"
// metric.
void CompareWOut(const WOutFileContents& test_wout,
                 const WOutFileContents& expected_wout, real_t tolerance,
                 bool check_equal_niter = true);
}  // namespace vmecpp

#endif  // VMECPP_VMEC_OUTPUT_QUANTITIES_OUTPUT_QUANTITIES_H_
