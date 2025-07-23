// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_OUTPUT_QUANTITIES_OUTPUT_QUANTITIES_H_
#define VMECPP_VMEC_OUTPUT_QUANTITIES_OUTPUT_QUANTITIES_H_

#include <Eigen/Dense>  // VectorXd
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "H5Cpp.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "vmecpp/common/sizes/sizes.h"
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

  Eigen::VectorXd sqrtSH;
  Eigen::VectorXd sqrtSF;

  Eigen::VectorXd sm;
  Eigen::VectorXd sp;

  // [ns] radial derivative of enclosed toroidal magnetic flux on full-grid
  Eigen::VectorXd phipF;

  // [ns] radial derivative of enclosed poloidal magnetic flux on full-grid
  Eigen::VectorXd chipF;

  // [ns - 1] radial derivative of enclosed toroidal magnetic flux on half-grid
  Eigen::VectorXd phipH;

  // [ns - 1] radial derivative of enclosed poloidal magnetic flux on half-grid
  Eigen::VectorXd chipH;

  // [ns - 1] enclosed current profile on half-grid
  Eigen::VectorXd currH;

  // [ns] enclosed toroidal magnetic flux on full-grid; computed in
  // RecomputeToroidalFlux here!
  Eigen::VectorXd phiF;

  // [ns] rotational transform on full-grid
  Eigen::VectorXd iotaF;

  // [ns] surface-averaged spectral width profile on full-grid
  Eigen::VectorXd spectral_width;

  // enclosed poloidal current on half-grid
  Eigen::VectorXd bvcoH;

  // [num_half] d(volume)/ds on half-grid
  Eigen::VectorXd dVdsH;

  // [num_half] mass profile on half-grid
  Eigen::VectorXd massH;

  // [num_half] kinetic pressure on half-grid
  Eigen::VectorXd presH;

  // [num_half] rotational transform profile on half-grid
  Eigen::VectorXd iotaH;

  // -------------------
  // state vector
  // (num_full, mnsize)
  RowMatrixXd rmncc;
  RowMatrixXd rmnss;
  RowMatrixXd rmnsc;
  RowMatrixXd rmncs;

  RowMatrixXd zmnsc;
  RowMatrixXd zmncs;
  RowMatrixXd zmncc;
  RowMatrixXd zmnss;

  RowMatrixXd lmnsc;
  RowMatrixXd lmncs;
  RowMatrixXd lmncc;
  RowMatrixXd lmnss;

  // -------------------
  // from inv-DFTs

  // (num_full, nZnT)
  RowMatrixXd r_e;
  RowMatrixXd r_o;
  RowMatrixXd z_e;
  RowMatrixXd z_o;

  // dX/dTheta for R, Z on full-grid
  RowMatrixXd ru_e;
  RowMatrixXd ru_o;
  RowMatrixXd zu_e;
  RowMatrixXd zu_o;

  // dX/dZeta for R, Z on full-grid
  RowMatrixXd rv_e;
  RowMatrixXd rv_o;
  RowMatrixXd zv_e;
  RowMatrixXd zv_o;

  // -------------------
  // from even-m and odd-m contributions

  RowMatrixXd ruFull;
  RowMatrixXd zuFull;

  // -------------------
  // from Jacobian calculation

  // R on half-grid
  // (num_half, nZnT)
  RowMatrixXd r12;

  // dX/dTheta for R, Z on the half-grid
  RowMatrixXd ru12;
  RowMatrixXd zu12;

  // parts of dX/ds for R, Z on half-grid from Jacobian calculation
  RowMatrixXd rs;
  RowMatrixXd zs;

  // Jacobian on half-grid
  // (num_half, nZnT)
  RowMatrixXd gsqrt;

  // -------------------
  // metric elements

  RowMatrixXd guu;
  RowMatrixXd guv;
  RowMatrixXd gvv;

  // -------------------
  // magnetic field

  // contravariant magnetic field components on half-grid
  // (num_half, nZnT)
  RowMatrixXd bsupu;
  RowMatrixXd bsupv;

  // covariant magnetic field components on half-grid
  // (num_half, nZnT)
  RowMatrixXd bsubu;
  RowMatrixXd bsubv;

  // covariant magnetic field components on full-grid from lambda force
  // (num_full, nZnT)
  RowMatrixXd bsubvF;

  // (|B|^2 + mu_0 p) on half-grid
  // (num_half, nZnT)
  RowMatrixXd total_pressure;

  // -------------------
  // (more or less) directly from input data

  // mu_0 * curtor (from INDATA) -> prescribed toroidal current in A
  double currv;

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
  RowMatrixXd rv12;
  // (num_full, nZnT)
  RowMatrixXd zv12;

  // full dX/ds for R, Z on half-grid
  // (num_half, nZnT)
  RowMatrixXd rs12;
  // (num_full, nZnT)
  RowMatrixXd zs12;

  // metric elements on half-grid
  // (num_half, nZnT)
  RowMatrixXd gsu;
  // (num_full, nZnT)
  RowMatrixXd gsv;

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
  RowMatrixXd b_r;
  RowMatrixXd b_phi;
  RowMatrixXd b_z;

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
  RowMatrixXd bsubs_half;

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
  RowMatrixXd bsubs_full;

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
  RowMatrixXd bsubs_s;

  // non-stellarator-symmetric B_s
  // (num_full, nZnT_reduced)
  RowMatrixXd bsubs_a;

  // stellarator-symmetric B_theta
  // (num_half, nZnT_reduced)
  RowMatrixXd bsubu_s;

  // non-stellarator-symmetric B_theta
  // (num_half, nZnT_reduced)
  RowMatrixXd bsubu_a;

  //  stellarator-symmetric B_zeta
  // (num_half, nZnT_reduced)
  RowMatrixXd bsubv_s;

  // non-stellarator-symmetric B_zeta
  // [num_half x nZnT_reduced]
  RowMatrixXd bsubv_a;
};

struct CovariantBDerivatives {
  // d(B_s)/dTheta
  // (num_full, nZnT)
  RowMatrixXd bsubsu;

  // d(B_s)/dZeta
  // (num_full, nZnT)
  RowMatrixXd bsubsv;

  // d(B_theta)/dZeta
  // (num_half, nZnT)
  RowMatrixXd bsubuv;

  // d(B_zeta)/dTheta
  // (num_half, nZnT)
  RowMatrixXd bsubvu;

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
  RowMatrixXd itheta;
  RowMatrixXd izeta;
  RowMatrixXd bdotk;

  Eigen::VectorXd amaxfor;
  Eigen::VectorXd aminfor;
  Eigen::VectorXd avforce;
  Eigen::VectorXd pprim;
  Eigen::VectorXd jdotb;
  Eigen::VectorXd bdotb;
  Eigen::VectorXd bdotgradv;
  Eigen::VectorXd jpar2;
  Eigen::VectorXd jperp2;
  Eigen::VectorXd phin;

  // (num_full, nZnT)
  RowMatrixXd jsupu3;
  RowMatrixXd jsupv3;

  // (num_half, nZnT)
  RowMatrixXd jsups3;

  // (num_full, nZnT)
  RowMatrixXd bsupu3;
  RowMatrixXd bsupv3;
  RowMatrixXd jcrossb;
  RowMatrixXd jxb_gradp;
  RowMatrixXd jdotb_sqrtg;
  RowMatrixXd sqrtg3;

  // (num_half, nZnT)
  RowMatrixXd bsubu3;
  RowMatrixXd bsubv3;

  // (num_full, nZnT)
  RowMatrixXd bsubs3;

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
  Eigen::VectorXd s;

  // magnetic shear == radial derivative of iota on full grid
  Eigen::VectorXd shear;

  // magnetic well == d^2V/ds^2 on full grid
  Eigen::VectorXd vpp;

  // radial derivative of kinetic pressure on full grid
  Eigen::VectorXd d_pressure_d_s;

  // d(I_tor)/ds on full grid
  Eigen::VectorXd d_toroidal_current_d_s;

  // real, physical d(phi)/ds on half-grid
  Eigen::VectorXd phip_realH;
  Eigen::VectorXd phip_realF;

  // dV/d(PHI) on half mesh
  Eigen::VectorXd vp_real;

  // toroidal current on half-grid
  Eigen::VectorXd torcur;

  // Jacobian on full-grid
  // (num_full, nZnT)
  RowMatrixXd gsqrt_full;

  // B \cdot j on full-grid
  // (num_full, nZnT)
  RowMatrixXd bdotj;

  // 1.0 / gpp on full-grid
  // (num_full, nZnT)
  // TODO(jons): figure out what this really is
  RowMatrixXd gpp;

  // |B|^2 on half grid
  // (num_half, nZnT)
  RowMatrixXd b2;

  // <1/B**2> on full-grid
  Eigen::VectorXd tpp;

  // <b*b/|grad-phi|**3> on full-grid
  Eigen::VectorXd tbb;

  // <j*b/|grad-phi|**3>
  Eigen::VectorXd tjb;

  // <(j*b)2/b**2*|grad-phi|**3>
  Eigen::VectorXd tjj;

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
  Eigen::VectorXd s;

  // -------------------

  // toroidal magnetic flux on full-grid
  Eigen::VectorXd toroidal_flux;

  // rotational transform on full grid
  Eigen::VectorXd iota;

  // magnetic shear == radial derivative of iota on full grid
  Eigen::VectorXd shear;

  // dV/ds on full grid
  Eigen::VectorXd d_volume_d_s;

  // magnetic well == d^2V/ds^2 on full grid
  Eigen::VectorXd well;

  // I_tor on full grid
  Eigen::VectorXd toroidal_current;

  // d(I_tor)/ds on full grid
  Eigen::VectorXd d_toroidal_current_d_s;

  // kinetic pressure on full grid
  Eigen::VectorXd pressure;

  // radial derivative of kinetic pressure on full grid
  Eigen::VectorXd d_pressure_d_s;

  // -------------------

  // Mercier criterion on full grid
  Eigen::VectorXd DMerc;

  // shear contribution to Mercier criterion on full grid
  Eigen::VectorXd Dshear;

  // magnetic well contribution to Mercier criterion on full grid
  Eigen::VectorXd Dwell;

  // toroidal current contribution to Mercier criterion on full grid
  Eigen::VectorXd Dcurr;

  // geodesic curvature contribution to Mercier criterion on full grid
  Eigen::VectorXd Dgeod;

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
  RowMatrixXd tau;

  // [num_half] surface-averaged beta profile
  Eigen::VectorXd beta_vol;

  // [num_half] <tau / R> / V'
  // TODO(jons): figure out what this really is
  Eigen::VectorXd overr;

  // plasma beta on magnetic axis
  double beta_axis;

  // [num_full] pressure on full-grid
  Eigen::VectorXd presf;

  // [num_full] phip * 2 * pi * sign_of_jacobian on full-grid
  Eigen::VectorXd phipf_loc;

  // [num_full] toroidal flux profile on full-grid
  Eigen::VectorXd phi1;

  // [num_full] poloidal flux profile on full-grid
  Eigen::VectorXd chi1;

  // [num_full] 2 pi * poloidal flux profile on full-grid
  Eigen::VectorXd chi;

  Eigen::VectorXd bvcoH;
  Eigen::VectorXd bucoH;

  // [num_full] toroidal current density profile on full-grid
  Eigen::VectorXd jcurv;

  // [num_full] poloidal current density profile on full-grid
  Eigen::VectorXd jcuru;

  // [num_full] radial derivative of pressure on full-grid
  Eigen::VectorXd presgrad;

  // [num_full] dV/d(phi) on full-grid
  Eigen::VectorXd vpphi;

  // [num_full] radial force balance residual on full-grid
  Eigen::VectorXd equif;

  // [num_full] toroidal current profile on full-grid
  Eigen::VectorXd bucof;

  // [num_full] poloidal current profile on full-grid
  Eigen::VectorXd bvcof;

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
  Eigen::VectorXd s;

  // [num_full] <RADIAL FORCE>: radial force balance residual on full-grid
  Eigen::VectorXd radial_force;

  // [num_full] TOROIDAL FLUX: toroidal flux profile on full-grid
  Eigen::VectorXd toroidal_flux;

  // [num_full] IOTA: rotational transform profile on full-grid
  Eigen::VectorXd iota;

  // [num_full] <JSUPU>: surface-averaged poloidal current density on full-grid
  Eigen::VectorXd avg_jsupu;

  // [num_full] <JSUPV>: surface-averaged toroidal current density on full-grid
  Eigen::VectorXd avg_jsupv;

  // [num_full] d(VOL)/d(PHI): differential volume on full-grid
  Eigen::VectorXd d_volume_d_phi;

  // [num_full] d(PRES)/d(PHI): radial derivative of pressure on full-grid
  Eigen::VectorXd d_pressure_d_phi;

  // [num_full] <M>: surface-averaged spectral width profile on full-grid
  Eigen::VectorXd spectral_width;

  // [num_full] PRESF: pressure on full-grid in Pa (no mu_0!)
  Eigen::VectorXd pressure;

  // [num_full] <BSUBU>: toroidal current profile on full-grid
  Eigen::VectorXd buco_full;

  // [num_full] <BSUBV>: poloidal current profile on full-grid
  Eigen::VectorXd bvco_full;

  // [num_full] <J.B>: parallel current density profile on full-grid
  Eigen::VectorXd j_dot_b;

  // [num_full] <B.B>: <|B|^2> profile on full-grid
  Eigen::VectorXd b_dot_b;

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
  double anorm;
  double vnorm;

  // differential surface area element |dS|, already with poloidal integration
  // weights
  Eigen::VectorXd surf_area;
  double circumference_sum;

  double rcenin;
  double aminr2in;
  double bminz2in;
  double bminz2;

  double sump;

  Eigen::VectorXd btor_vac;
  Eigen::VectorXd btor1;
  Eigen::VectorXd dbtor;
  Eigen::VectorXd phat;
  Eigen::VectorXd redge;

  double delphid_exact;
  double musubi;
  double rshaf1;
  double rshaf2;
  double rshaf;

  double fpsi0;

  double sumbtot;
  double sumbtor;
  double sumbpol;
  double sump20;
  double sump2;

  Eigen::VectorXd jPS2;
  double jpar_perp_sum;
  double jparPS_perp_sum;
  double s2;

  double fac;
  Eigen::VectorXd r3v;

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
  double toroidal_flux;

  double circum_p;
  double surf_area_p;

  double cross_area_p;
  double volume_p;

  double Rmajor_p;
  double Aminor_p;
  double aspect;

  double kappa_p;
  double rcen;

  // volume-averaged minor radius
  double aminr1;

  double pavg;
  double factor;

  double b0;

  double rmax_surf;
  double rmin_surf;
  double zmax_surf;

  // (num_half, nThetaReduced)
  RowMatrixXd bmin;
  // (num_half, nThetaReduced)
  RowMatrixXd bmax;

  Eigen::VectorXd waist;
  Eigen::VectorXd height;

  double betapol;
  double betatot;
  double betator;
  double VolAvgB;
  double IonLarmor;

  double jpar_perp;
  double jparPS_perp;

  // net toroidal current in A
  double toroidal_current;

  double rbtor;
  double rbtor0;

  // poloidal magnetic flux on full-grid
  Eigen::VectorXd psi;

  // Geometric minor radius
  Eigen::VectorXd ygeo;

  // Geometric indentation
  Eigen::VectorXd yinden;

  // Geometric ellipticity
  Eigen::VectorXd yellip;

  // Geometric triangularity
  Eigen::VectorXd ytrian;

  // Geometric shift measured from magnetic axis
  Eigen::VectorXd yshift;

  Eigen::VectorXd loc_jpar_perp;
  Eigen::VectorXd loc_jparPS_perp;

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
  double int_p;
  double avg_p;

  double int_bpol;
  double avg_bpol;

  double int_btor;
  double avg_btor;

  double int_modb;
  double avg_modb;

  double int_ekin;
  double avg_ekin;

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
  Eigen::VectorXd raxis_symm;

  // [ntor + 1] stellarator-symmetric Fourier coefficients of axis: Z * sin(n *
  // zeta)
  Eigen::VectorXd zaxis_symm;

  // [ntor + 1] non-stellarator-symmetric Fourier coefficients of axis: R *
  // sin(n * zeta)
  Eigen::VectorXd raxis_asym;

  // [ntor + 1] non-stellarator-symmetric Fourier coefficients of axis: Z *
  // cos(n * zeta)
  Eigen::VectorXd zaxis_asym;

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
  double betatot;

  // beta poloidal
  double betapol;

  // beta toroidal
  double betator;

  // R * Btor-vac
  double rbtor;

  // Peak Beta (on axis)
  double betaxis;

  // Beta-star
  double betstr;

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
  double scaling_ratio;

  double r_lao;
  double f_lao;
  double f_geo;

  double smaleli;
  double betai;
  double musubi;
  double lambda;

  double s11;
  double s12;
  double s13;
  double s2;
  double s3;

  double delta1;
  double delta2;
  double delta3;

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
  std::string version;

  std::string input_extension;

  // sign of Jacobian between cylindrical and flux coordinates; hardcoded to -1
  int sign_of_jacobian;

  // adiabatic index
  double gamma;

  // parameterization identifier for toroidal current profile
  std::string pcurr_type;

  // parameterization identifier for mass/pressure profile
  std::string pmass_type;

  // parameterization identifier for iota profile
  std::string piota_type;

  // pressure profile coefficients
  // interpretation depends on value of `pmass_type`
  Eigen::VectorXd am;

  // toroidal current profile coefficients
  // interpretation depends on value of `pcurr_type`
  Eigen::VectorXd ac;

  // iota profile coefficients
  // interpretation depends on value of `piota_type`
  Eigen::VectorXd ai;

  // knots for discrete mass/pressure profile
  Eigen::VectorXd am_aux_s;

  // values for discrete mass/pressure profile
  Eigen::VectorXd am_aux_f;

  // knots for discrete toroidal current profile
  Eigen::VectorXd ac_aux_s;

  // values for discrete toroidal current profile
  Eigen::VectorXd ac_aux_f;

  // knots for discrete iota profile
  Eigen::VectorXd ai_aux_s;

  // values for discrete iota profile
  Eigen::VectorXd ai_aux_f;

  // number of toroidal field periods
  int nfp;

  // poloidal Fourier resolution; m = 0, 1, ..., (mpol-1)
  int mpol;

  // toroidal Fourier resolution; n = -ntor, ..., -1, 0, 1, ..., ntor
  int ntor;

  // flag to indicate a non-stellarator-symmetric case
  bool lasym = false;

  // final radial resolution
  // == number of flux surfaces
  // == number of full-grid points
  int ns;

  // requested force tolerance level for convergence
  double ftolv;

  // iterations required until convergence
  int maximum_iterations;

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
  double wb;

  // thermal energy
  double wp;

  // maximum R of LCFS
  double rmax_surf;

  // minimum R of LCFS
  double rmin_surf;

  // maximum |Z| of LCFS
  double zmax_surf;

  // number of Fourier coefficients in state vector (R, Z, lambda)
  int mnmax;

  // number of Fourier coefficients for derived quantities, Nyquist-extended
  // (sqrt(g), |B|, co- and contravariant B components, currents)
  int mnmax_nyq;

  int ier_flag;

  double aspect;

  double betatot;
  double betapol;
  double betator;
  double betaxis;

  double b0;

  double rbtor0;
  double rbtor;

  double IonLarmor;
  double VolAvgB;

  double ctor;

  double Aminor_p;
  double Rmajor_p;
  double volume_p;

  double fsqr;
  double fsqz;
  double fsql;

  // Number of "time steps" of the force relaxation that were actually required
  // to achieve convergence. (How many of the maximum niter steps we ended up
  // using.)
  int itfsq;

  // -------------------
  // one-dimensional array quantities

  // full-grid: rotational_transform
  Eigen::VectorXd iota_full;

  // full-grid: 1 / iota (where iota != 0)
  Eigen::VectorXd safety_factor;

  // full-grid: pressure in Pa
  Eigen::VectorXd pressure_full;

  // full-grid: enclosed toroidal magnetic flux (phi) in Vs
  Eigen::VectorXd toroidal_flux;

  // full-grid: toroidal flux differential (phi-prime)
  Eigen::VectorXd phipf;

  // full-grid: enclosed poloidal magnetic flux (chi) in Vs
  Eigen::VectorXd poloidal_flux;

  // full-grid: poloidal flux differential (chi-prime)
  Eigen::VectorXd chipf;

  Eigen::VectorXd jcuru;
  Eigen::VectorXd jcurv;

  // Convergence quantities (one entry per time step)
  // Force residual at each iteration
  Eigen::VectorXd fsqt;
  Eigen::VectorXd force_residual_r;
  Eigen::VectorXd force_residual_z;
  Eigen::VectorXd force_residual_lambda;
  Eigen::VectorXd delbsq;
  Eigen::VectorXi restart_reasons;

  // Gradient of the energy at each iteration
  Eigen::VectorXd wdot;

  // ---------

  Eigen::VectorXd iota_half;
  Eigen::VectorXd mass;
  Eigen::VectorXd pressure_half;
  Eigen::VectorXd beta;
  Eigen::VectorXd buco;
  Eigen::VectorXd bvco;
  Eigen::VectorXd dVds;
  Eigen::VectorXd spectral_width;
  Eigen::VectorXd phips;
  Eigen::VectorXd overr;

  Eigen::VectorXd jdotb;
  Eigen::VectorXd bdotb;
  Eigen::VectorXd bdotgradv;

  Eigen::VectorXd DMerc;
  Eigen::VectorXd Dshear;
  Eigen::VectorXd Dwell;
  Eigen::VectorXd Dcurr;
  Eigen::VectorXd Dgeod;

  Eigen::VectorXd equif;

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

  Eigen::VectorXd raxis_c;
  Eigen::VectorXd zaxis_s;

  // full-grid: R
  RowMatrixXd rmnc;

  // full-grid: Z
  RowMatrixXd zmns;

  // full-grid: lambda
  // NOTE: new with respect to Fortran VMEC
  RowMatrixXd lmns_full;

  // half-grid: lambda
  RowMatrixXd lmns;

  // half-grid: Jacobian
  RowMatrixXd gmnc;

  // half-grid: |B|
  RowMatrixXd bmnc;

  // half-grid: covariant B_\theta
  RowMatrixXd bsubumnc;

  // half-grid: covariant B_\zeta
  RowMatrixXd bsubvmnc;

  // half-grid: covariant B_s
  RowMatrixXd bsubsmns;

  // full-grid: covariant B_s
  // NOTE: new with respect to Fortran VMEC
  RowMatrixXd bsubsmns_full;

  // half-grid: contravariant B^\theta
  RowMatrixXd bsupumnc;

  // half-grid: contravariant B^\zeta
  RowMatrixXd bsupvmnc;

  // -------------------
  // non-stellarator-symmetric Fourier coefficients

  Eigen::VectorXd raxis_s;
  Eigen::VectorXd zaxis_c;

  // full-grid: R
  RowMatrixXd rmns;

  // full-grid: Z
  RowMatrixXd zmnc;

  // full-grid: lambda
  // NOTE: new with respect to Fortran VMEC
  RowMatrixXd lmnc_full;

  // half-grid: lambda
  RowMatrixXd lmnc;

  // half-grid: Jacobian
  RowMatrixXd gmns;

  // half-grid: |B|
  RowMatrixXd bmns;

  // half-grid: covariant B_\theta
  RowMatrixXd bsubumns;

  // half-grid: covariant B_\zeta
  RowMatrixXd bsubvmns;

  // half-grid: covariant B_s
  RowMatrixXd bsubsmnc;

  // full-grid: covariant B_s
  // NOTE: new with respect to Fortran VMEC
  RowMatrixXd bsubsmnc_full;

  // half-grid: contravariant B^\theta
  RowMatrixXd bsupumns;

  // half-grid: contravariant B^\zeta
  RowMatrixXd bsupvmns;

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
                 const WOutFileContents& expected_wout, double tolerance,
                 bool check_equal_maximum_iterations = true);
}  // namespace vmecpp

#endif  // VMECPP_VMEC_OUTPUT_QUANTITIES_OUTPUT_QUANTITIES_H_
