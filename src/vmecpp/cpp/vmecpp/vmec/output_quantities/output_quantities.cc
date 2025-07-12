// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/output_quantities/output_quantities.h"

#include <Eigen/Dense>  // VectorXd
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "H5Cpp.h"
#include "absl/log/check.h"
#include "util/hdf5_io/hdf5_io.h"
#include "util/testing/numerical_comparison_lib.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/vmec/vmec_constants/vmec_constants.h"

using Eigen::VectorXd;
using hdf5_io::ReadH5Dataset;
using hdf5_io::WriteH5Dataset;
using testing::IsCloseRelAbs;
using testing::IsVectorCloseRelAbs;

namespace {
// Return a Eigen version of the input vector if non-empty, else a 1-element
// vector containing val (it's a common initialization pattern used below).
VectorXd NonEmptyVectorOr(const std::vector<double>& vec, const double val) {
  if (!vec.empty()) {
    return vmecpp::ToEigenVector(vec);
  } else {
    return VectorXd::Constant(1, val);
  }
}

}  // namespace

// Shorthands for the calls required to read/write data members from/to HDF5
// files. They assume that `H5key`, `file`, `m_obj` and `from_file` are
// available in the calling context.
#define WRITEMEMBER(x) \
  WriteH5Dataset(x, absl::StrFormat("%s/%s", H5key, #x), file);
#define READMEMBER(x) \
  ReadH5Dataset(m_obj.x, absl::StrFormat("%s/%s", H5key, #x), from_file);

// Write object to the specified HDF5 file, under key "vmecinternalresults".
absl::Status vmecpp::VmecInternalResults::WriteTo(H5::H5File& file) const {
  file.createGroup(H5key);
  WRITEMEMBER(sign_of_jacobian);
  WRITEMEMBER(num_full);
  WRITEMEMBER(num_half);
  WRITEMEMBER(nZnT_reduced);
  WRITEMEMBER(sqrtSH);
  WRITEMEMBER(sqrtSF);
  WRITEMEMBER(sm);
  WRITEMEMBER(sp);
  WRITEMEMBER(phipF);
  WRITEMEMBER(chipF);
  WRITEMEMBER(phipH);
  WRITEMEMBER(chipH);
  WRITEMEMBER(currH);
  WRITEMEMBER(phiF);
  WRITEMEMBER(iotaF);
  WRITEMEMBER(spectral_width);
  WRITEMEMBER(bvcoH);
  WRITEMEMBER(dVdsH);
  WRITEMEMBER(massH);
  WRITEMEMBER(presH);
  WRITEMEMBER(iotaH);
  WRITEMEMBER(rmncc);
  WRITEMEMBER(rmnss);
  WRITEMEMBER(rmnsc);
  WRITEMEMBER(rmncs);
  WRITEMEMBER(zmnsc);
  WRITEMEMBER(zmncs);
  WRITEMEMBER(zmncc);
  WRITEMEMBER(zmnss);
  WRITEMEMBER(lmnsc);
  WRITEMEMBER(lmncs);
  WRITEMEMBER(lmncc);
  WRITEMEMBER(lmnss);
  WRITEMEMBER(r_e);
  WRITEMEMBER(r_o);
  WRITEMEMBER(z_e);
  WRITEMEMBER(z_o);
  WRITEMEMBER(ru_e);
  WRITEMEMBER(ru_o);
  WRITEMEMBER(zu_e);
  WRITEMEMBER(zu_o);
  WRITEMEMBER(rv_e);
  WRITEMEMBER(rv_o);
  WRITEMEMBER(zv_e);
  WRITEMEMBER(zv_o);
  WRITEMEMBER(ruFull);
  WRITEMEMBER(zuFull);
  WRITEMEMBER(r12);
  WRITEMEMBER(ru12);
  WRITEMEMBER(zu12);
  WRITEMEMBER(rs);
  WRITEMEMBER(zs);
  WRITEMEMBER(gsqrt);
  WRITEMEMBER(guu);
  WRITEMEMBER(guv);
  WRITEMEMBER(gvv);
  WRITEMEMBER(bsupu);
  WRITEMEMBER(bsupv);
  WRITEMEMBER(bsubu);
  WRITEMEMBER(bsubv);
  WRITEMEMBER(bsubvF);
  WRITEMEMBER(total_pressure);
  WRITEMEMBER(currv);

  return absl::OkStatus();
}

absl::Status vmecpp::VmecInternalResults::LoadInto(
    vmecpp::VmecInternalResults& m_obj, H5::H5File& from_file) {
  READMEMBER(sign_of_jacobian);
  READMEMBER(num_full);
  READMEMBER(num_half);
  READMEMBER(nZnT_reduced);
  READMEMBER(sqrtSH);
  READMEMBER(sqrtSF);
  READMEMBER(sm);
  READMEMBER(sp);
  READMEMBER(phipF);
  READMEMBER(chipF);
  READMEMBER(phipH);
  // Legacy way of checking for dataset existence
  // TODO(jons) replace with from_file.nameExists when we get a newer HDF5
  // version in pip wheels
  if (H5Lexists(from_file.getId(), absl::StrFormat("%s/chipH", H5key).c_str(),
                0) == 1) {
    READMEMBER(chipH);
  } else {
    m_obj.chipH = Eigen::VectorXd::Zero(m_obj.phipH.size());
  }
  // Legacy way of checking for dataset existence
  if (H5Lexists(from_file.getId(), absl::StrFormat("%s/currH", H5key).c_str(),
                0) == 1) {
    READMEMBER(currH);
  } else {
    m_obj.currH = Eigen::VectorXd::Zero(m_obj.phipH.size());
  }
  READMEMBER(phiF);
  READMEMBER(iotaF);
  READMEMBER(spectral_width);
  READMEMBER(bvcoH);
  READMEMBER(dVdsH);
  READMEMBER(massH);
  READMEMBER(presH);
  READMEMBER(iotaH);
  READMEMBER(rmncc);
  READMEMBER(rmnss);
  READMEMBER(rmnsc);
  READMEMBER(rmncs);
  READMEMBER(zmnsc);
  READMEMBER(zmncs);
  READMEMBER(zmncc);
  READMEMBER(zmnss);
  READMEMBER(lmnsc);
  READMEMBER(lmncs);
  READMEMBER(lmncc);
  READMEMBER(lmnss);
  READMEMBER(r_e);
  READMEMBER(r_o);
  READMEMBER(z_e);
  READMEMBER(z_o);
  READMEMBER(ru_e);
  READMEMBER(ru_o);
  READMEMBER(zu_e);
  READMEMBER(zu_o);
  READMEMBER(rv_e);
  READMEMBER(rv_o);
  READMEMBER(zv_e);
  READMEMBER(zv_o);
  READMEMBER(ruFull);
  READMEMBER(zuFull);
  READMEMBER(r12);
  READMEMBER(ru12);
  READMEMBER(zu12);
  READMEMBER(rs);
  READMEMBER(zs);
  READMEMBER(gsqrt);
  READMEMBER(guu);
  READMEMBER(guv);
  READMEMBER(gvv);
  READMEMBER(bsupu);
  READMEMBER(bsupv);
  READMEMBER(bsubu);
  READMEMBER(bsubv);
  READMEMBER(bsubvF);
  READMEMBER(total_pressure);
  READMEMBER(currv);

  return absl::OkStatus();
}

absl::Status vmecpp::RemainingMetric::WriteTo(H5::H5File& file) const {
  file.createGroup(this->H5key);
  WRITEMEMBER(rv12);
  WRITEMEMBER(zv12);
  WRITEMEMBER(rs12);
  WRITEMEMBER(zs12);
  WRITEMEMBER(gsu);
  WRITEMEMBER(gsv);

  return absl::OkStatus();
}

absl::Status vmecpp::RemainingMetric::LoadInto(RemainingMetric& m_obj,
                                               H5::H5File& from_file) {
  READMEMBER(rv12);
  READMEMBER(zv12);
  READMEMBER(rs12);
  READMEMBER(zs12);
  READMEMBER(gsu);
  READMEMBER(gsv);

  return absl::OkStatus();
}

absl::Status vmecpp::CylindricalComponentsOfB::WriteTo(H5::H5File& file) const {
  file.createGroup(this->H5key);
  WRITEMEMBER(b_r);
  WRITEMEMBER(b_phi);
  WRITEMEMBER(b_z);

  return absl::OkStatus();
}

absl::Status vmecpp::CylindricalComponentsOfB::LoadInto(
    CylindricalComponentsOfB& m_obj, H5::H5File& from_file) {
  READMEMBER(b_r);
  READMEMBER(b_phi);
  READMEMBER(b_z);

  return absl::OkStatus();
}

absl::Status vmecpp::BSubSHalf::WriteTo(H5::H5File& file) const {
  file.createGroup(this->H5key);
  WRITEMEMBER(bsubs_half);
  return absl::OkStatus();
}

absl::Status vmecpp::BSubSHalf::LoadInto(BSubSHalf& m_obj,
                                         H5::H5File& from_file) {
  READMEMBER(bsubs_half);
  return absl::OkStatus();
}

absl::Status vmecpp::BSubSFull::WriteTo(H5::H5File& file) const {
  file.createGroup(this->H5key);
  WRITEMEMBER(bsubs_full);
  return absl::OkStatus();
}

absl::Status vmecpp::BSubSFull::LoadInto(BSubSFull& m_obj,
                                         H5::H5File& from_file) {
  READMEMBER(bsubs_full);
  return absl::OkStatus();
}

absl::Status vmecpp::CovariantBDerivatives::WriteTo(H5::H5File& file) const {
  file.createGroup(this->H5key);
  WRITEMEMBER(bsubsu);
  WRITEMEMBER(bsubsv);
  WRITEMEMBER(bsubuv);
  WRITEMEMBER(bsubvu);

  return absl::OkStatus();
}

absl::Status vmecpp::CovariantBDerivatives::LoadInto(
    CovariantBDerivatives& m_obj, H5::H5File& from_file) {
  READMEMBER(bsubsu);
  READMEMBER(bsubsv);
  READMEMBER(bsubuv);
  READMEMBER(bsubvu);

  return absl::OkStatus();
}

absl::Status vmecpp::JxBOutFileContents::WriteTo(H5::H5File& file) const {
  file.createGroup(this->H5key);
  WRITEMEMBER(itheta);
  WRITEMEMBER(izeta);
  WRITEMEMBER(bdotk);
  WRITEMEMBER(amaxfor);
  WRITEMEMBER(aminfor);
  WRITEMEMBER(avforce);
  WRITEMEMBER(pprim);
  WRITEMEMBER(jdotb);
  WRITEMEMBER(bdotb);
  WRITEMEMBER(bdotgradv);
  WRITEMEMBER(jpar2);
  WRITEMEMBER(jperp2);
  WRITEMEMBER(phin);
  WRITEMEMBER(jsupu3);
  WRITEMEMBER(jsupv3);
  WRITEMEMBER(jsups3);
  WRITEMEMBER(bsupu3);
  WRITEMEMBER(bsupv3);
  WRITEMEMBER(jcrossb);
  WRITEMEMBER(jxb_gradp);
  WRITEMEMBER(jdotb_sqrtg);
  WRITEMEMBER(sqrtg3);
  WRITEMEMBER(bsubu3);
  WRITEMEMBER(bsubv3);
  WRITEMEMBER(bsubs3);

  return absl::OkStatus();
}
absl::Status vmecpp::JxBOutFileContents::LoadInto(JxBOutFileContents& m_obj,
                                                  H5::H5File& from_file) {
  READMEMBER(itheta);
  READMEMBER(izeta);
  READMEMBER(bdotk);
  READMEMBER(amaxfor);
  READMEMBER(aminfor);
  READMEMBER(avforce);
  READMEMBER(pprim);
  READMEMBER(jdotb);
  READMEMBER(bdotb);
  READMEMBER(bdotgradv);
  READMEMBER(jpar2);
  READMEMBER(jperp2);
  READMEMBER(phin);
  READMEMBER(jsupu3);
  READMEMBER(jsupv3);
  READMEMBER(jsups3);
  READMEMBER(bsupu3);
  READMEMBER(bsupv3);
  READMEMBER(jcrossb);
  READMEMBER(jxb_gradp);
  READMEMBER(jdotb_sqrtg);
  READMEMBER(sqrtg3);
  READMEMBER(bsubu3);
  READMEMBER(bsubv3);
  READMEMBER(bsubs3);
  return absl::OkStatus();
}

absl::Status vmecpp::MercierStabilityIntermediateQuantities::WriteTo(
    H5::H5File& file) const {
  file.createGroup(this->H5key);
  WRITEMEMBER(s);
  WRITEMEMBER(shear);
  WRITEMEMBER(vpp);
  WRITEMEMBER(d_pressure_d_s);
  WRITEMEMBER(d_toroidal_current_d_s);
  WRITEMEMBER(phip_realH);
  WRITEMEMBER(phip_realF);
  WRITEMEMBER(vp_real);
  WRITEMEMBER(torcur);
  WRITEMEMBER(gsqrt_full);
  WRITEMEMBER(bdotj);
  WRITEMEMBER(gpp);
  WRITEMEMBER(b2);
  WRITEMEMBER(tpp);
  WRITEMEMBER(tbb);
  WRITEMEMBER(tjb);
  WRITEMEMBER(tjj);

  return absl::OkStatus();
}
absl::Status vmecpp::MercierStabilityIntermediateQuantities::LoadInto(
    MercierStabilityIntermediateQuantities& m_obj, H5::H5File& from_file) {
  READMEMBER(s);
  READMEMBER(shear);
  READMEMBER(vpp);
  READMEMBER(d_pressure_d_s);
  READMEMBER(d_toroidal_current_d_s);
  READMEMBER(phip_realH);
  READMEMBER(phip_realF);
  READMEMBER(vp_real);
  READMEMBER(torcur);
  READMEMBER(gsqrt_full);
  READMEMBER(bdotj);
  READMEMBER(gpp);
  READMEMBER(b2);
  READMEMBER(tpp);
  READMEMBER(tbb);
  READMEMBER(tjb);
  READMEMBER(tjj);
  return absl::OkStatus();
}

absl::Status vmecpp::MercierFileContents::WriteTo(H5::H5File& file) const {
  file.createGroup(this->H5key);
  WRITEMEMBER(s);
  WRITEMEMBER(toroidal_flux);
  WRITEMEMBER(iota);
  WRITEMEMBER(shear);
  WRITEMEMBER(d_volume_d_s);
  WRITEMEMBER(well);
  WRITEMEMBER(toroidal_current);
  WRITEMEMBER(d_toroidal_current_d_s);
  WRITEMEMBER(pressure);
  WRITEMEMBER(d_pressure_d_s);
  WRITEMEMBER(DMerc);
  WRITEMEMBER(Dshear);
  WRITEMEMBER(Dwell);
  WRITEMEMBER(Dcurr);
  WRITEMEMBER(Dgeod);

  return absl::OkStatus();
}
absl::Status vmecpp::MercierFileContents::LoadInto(MercierFileContents& m_obj,
                                                   H5::H5File& from_file) {
  READMEMBER(s);
  READMEMBER(toroidal_flux);
  READMEMBER(iota);
  READMEMBER(shear);
  READMEMBER(d_volume_d_s);
  READMEMBER(well);
  READMEMBER(toroidal_current);
  READMEMBER(d_toroidal_current_d_s);
  READMEMBER(pressure);
  READMEMBER(d_pressure_d_s);
  READMEMBER(DMerc);
  READMEMBER(Dshear);
  READMEMBER(Dwell);
  READMEMBER(Dcurr);
  READMEMBER(Dgeod);
  return absl::OkStatus();
}

absl::Status vmecpp::Threed1FirstTableIntermediate::WriteTo(
    H5::H5File& file) const {
  file.createGroup(this->H5key);
  WRITEMEMBER(tau);
  WRITEMEMBER(beta_vol);
  WRITEMEMBER(overr);
  WRITEMEMBER(beta_axis);
  WRITEMEMBER(presf);
  WRITEMEMBER(phipf_loc);
  WRITEMEMBER(phi1);
  WRITEMEMBER(chi1);
  WRITEMEMBER(chi);
  WRITEMEMBER(bvcoH);
  WRITEMEMBER(bucoH);
  WRITEMEMBER(jcurv);
  WRITEMEMBER(jcuru);
  WRITEMEMBER(presgrad);
  WRITEMEMBER(vpphi);
  WRITEMEMBER(equif);
  WRITEMEMBER(bucof);
  WRITEMEMBER(bvcof);

  return absl::OkStatus();
}
absl::Status vmecpp::Threed1FirstTableIntermediate::LoadInto(
    Threed1FirstTableIntermediate& m_obj, H5::H5File& from_file) {
  READMEMBER(tau);
  READMEMBER(beta_vol);
  READMEMBER(overr);
  READMEMBER(beta_axis);
  READMEMBER(presf);
  READMEMBER(phipf_loc);
  READMEMBER(phi1);
  READMEMBER(chi1);
  READMEMBER(chi);
  READMEMBER(bvcoH);
  READMEMBER(bucoH);
  READMEMBER(jcurv);
  READMEMBER(jcuru);
  READMEMBER(presgrad);
  READMEMBER(vpphi);
  READMEMBER(equif);
  READMEMBER(bucof);
  READMEMBER(bvcof);
  return absl::OkStatus();
}

absl::Status vmecpp::Threed1FirstTable::WriteTo(H5::H5File& file) const {
  file.createGroup(this->H5key);
  WRITEMEMBER(s);
  WRITEMEMBER(radial_force);
  WRITEMEMBER(toroidal_flux);
  WRITEMEMBER(iota);
  WRITEMEMBER(avg_jsupu);
  WRITEMEMBER(avg_jsupv);
  WRITEMEMBER(d_volume_d_phi);
  WRITEMEMBER(d_pressure_d_phi);
  WRITEMEMBER(spectral_width);
  WRITEMEMBER(pressure);
  WRITEMEMBER(buco_full);
  WRITEMEMBER(bvco_full);
  WRITEMEMBER(j_dot_b);
  WRITEMEMBER(b_dot_b);

  return absl::OkStatus();
}
absl::Status vmecpp::Threed1FirstTable::LoadInto(Threed1FirstTable& m_obj,
                                                 H5::H5File& from_file) {
  READMEMBER(s);
  READMEMBER(radial_force);
  READMEMBER(toroidal_flux);
  READMEMBER(iota);
  READMEMBER(avg_jsupu);
  READMEMBER(avg_jsupv);
  READMEMBER(d_volume_d_phi);
  READMEMBER(d_pressure_d_phi);
  READMEMBER(spectral_width);
  READMEMBER(pressure);
  READMEMBER(buco_full);
  READMEMBER(bvco_full);
  READMEMBER(j_dot_b);
  READMEMBER(b_dot_b);
  return absl::OkStatus();
}

absl::Status vmecpp::Threed1GeometricAndMagneticQuantitiesIntermediate::WriteTo(
    H5::H5File& file) const {
  file.createGroup(this->H5key);
  WRITEMEMBER(anorm);
  WRITEMEMBER(vnorm);
  WRITEMEMBER(surf_area);
  WRITEMEMBER(circumference_sum);
  WRITEMEMBER(rcenin);
  WRITEMEMBER(aminr2in);
  WRITEMEMBER(bminz2in);
  WRITEMEMBER(bminz2);
  WRITEMEMBER(sump);
  WRITEMEMBER(btor_vac);
  WRITEMEMBER(btor1);
  WRITEMEMBER(dbtor);
  WRITEMEMBER(phat);
  WRITEMEMBER(redge);
  WRITEMEMBER(delphid_exact);
  WRITEMEMBER(musubi);
  WRITEMEMBER(rshaf1);
  WRITEMEMBER(rshaf2);
  WRITEMEMBER(rshaf);
  WRITEMEMBER(fpsi0);
  WRITEMEMBER(sumbtot);
  WRITEMEMBER(sumbtor);
  WRITEMEMBER(sumbpol);
  WRITEMEMBER(sump20);
  WRITEMEMBER(sump2);
  WRITEMEMBER(jPS2);
  WRITEMEMBER(jpar_perp_sum);
  WRITEMEMBER(jparPS_perp_sum);
  WRITEMEMBER(s2);
  WRITEMEMBER(fac);
  WRITEMEMBER(r3v);

  return absl::OkStatus();
}
absl::Status
vmecpp::Threed1GeometricAndMagneticQuantitiesIntermediate::LoadInto(
    Threed1GeometricAndMagneticQuantitiesIntermediate& m_obj,
    H5::H5File& from_file) {
  READMEMBER(anorm);
  READMEMBER(vnorm);
  READMEMBER(surf_area);
  READMEMBER(circumference_sum);
  READMEMBER(rcenin);
  READMEMBER(aminr2in);
  READMEMBER(bminz2in);
  READMEMBER(bminz2);
  READMEMBER(sump);
  READMEMBER(btor_vac);
  READMEMBER(btor1);
  READMEMBER(dbtor);
  READMEMBER(phat);
  READMEMBER(redge);
  READMEMBER(delphid_exact);
  READMEMBER(musubi);
  READMEMBER(rshaf1);
  READMEMBER(rshaf2);
  READMEMBER(rshaf);
  READMEMBER(fpsi0);
  READMEMBER(sumbtot);
  READMEMBER(sumbtor);
  READMEMBER(sumbpol);
  READMEMBER(sump20);
  READMEMBER(sump2);
  READMEMBER(jPS2);
  READMEMBER(jpar_perp_sum);
  READMEMBER(jparPS_perp_sum);
  READMEMBER(s2);
  READMEMBER(fac);
  READMEMBER(r3v);
  return absl::OkStatus();
}

absl::Status vmecpp::Threed1GeometricAndMagneticQuantities::WriteTo(
    H5::H5File& file) const {
  file.createGroup(this->H5key);
  WRITEMEMBER(toroidal_flux);
  WRITEMEMBER(circum_p);
  WRITEMEMBER(surf_area_p);
  WRITEMEMBER(cross_area_p);
  WRITEMEMBER(volume_p);
  WRITEMEMBER(Rmajor_p);
  WRITEMEMBER(Aminor_p);
  WRITEMEMBER(aspect);
  WRITEMEMBER(kappa_p);
  WRITEMEMBER(rcen);
  WRITEMEMBER(aminr1);
  WRITEMEMBER(pavg);
  WRITEMEMBER(factor);
  WRITEMEMBER(b0);
  WRITEMEMBER(rmax_surf);
  WRITEMEMBER(rmin_surf);
  WRITEMEMBER(zmax_surf);
  WRITEMEMBER(bmin);
  WRITEMEMBER(bmax);
  WRITEMEMBER(waist);
  WRITEMEMBER(height);
  WRITEMEMBER(betapol);
  WRITEMEMBER(betatot);
  WRITEMEMBER(betator);
  WRITEMEMBER(VolAvgB);
  WRITEMEMBER(IonLarmor);
  WRITEMEMBER(jpar_perp);
  WRITEMEMBER(jparPS_perp);
  WRITEMEMBER(toroidal_current);
  WRITEMEMBER(rbtor);
  WRITEMEMBER(rbtor0);
  WRITEMEMBER(psi);
  WRITEMEMBER(ygeo);
  WRITEMEMBER(yinden);
  WRITEMEMBER(yellip);
  WRITEMEMBER(ytrian);
  WRITEMEMBER(yshift);
  WRITEMEMBER(loc_jpar_perp);
  WRITEMEMBER(loc_jparPS_perp);

  return absl::OkStatus();
}
absl::Status vmecpp::Threed1GeometricAndMagneticQuantities::LoadInto(
    Threed1GeometricAndMagneticQuantities& m_obj, H5::H5File& from_file) {
  READMEMBER(toroidal_flux);
  READMEMBER(circum_p);
  READMEMBER(surf_area_p);
  READMEMBER(cross_area_p);
  READMEMBER(volume_p);
  READMEMBER(Rmajor_p);
  READMEMBER(Aminor_p);
  READMEMBER(aspect);
  READMEMBER(kappa_p);
  READMEMBER(rcen);
  READMEMBER(aminr1);
  READMEMBER(pavg);
  READMEMBER(factor);
  READMEMBER(b0);
  READMEMBER(rmax_surf);
  READMEMBER(rmin_surf);
  READMEMBER(zmax_surf);
  READMEMBER(bmin);
  READMEMBER(bmax);
  READMEMBER(waist);
  READMEMBER(height);
  READMEMBER(betapol);
  READMEMBER(betatot);
  READMEMBER(betator);
  READMEMBER(VolAvgB);
  READMEMBER(IonLarmor);
  READMEMBER(jpar_perp);
  READMEMBER(jparPS_perp);
  READMEMBER(toroidal_current);
  READMEMBER(rbtor);
  READMEMBER(rbtor0);
  READMEMBER(psi);
  READMEMBER(ygeo);
  READMEMBER(yinden);
  READMEMBER(yellip);
  READMEMBER(ytrian);
  READMEMBER(yshift);
  READMEMBER(loc_jpar_perp);
  READMEMBER(loc_jparPS_perp);

  return absl::OkStatus();
}

absl::Status vmecpp::Threed1Volumetrics::WriteTo(H5::H5File& file) const {
  file.createGroup(this->H5key);
  WRITEMEMBER(int_p);
  WRITEMEMBER(avg_p);
  WRITEMEMBER(int_bpol);
  WRITEMEMBER(avg_bpol);
  WRITEMEMBER(int_btor);
  WRITEMEMBER(avg_btor);
  WRITEMEMBER(int_modb);
  WRITEMEMBER(avg_modb);
  WRITEMEMBER(int_ekin);
  WRITEMEMBER(avg_ekin);

  return absl::OkStatus();
}

absl::Status vmecpp::Threed1Volumetrics::LoadInto(Threed1Volumetrics& m_obj,
                                                  H5::H5File& from_file) {
  READMEMBER(int_p);
  READMEMBER(avg_p);
  READMEMBER(int_bpol);
  READMEMBER(avg_bpol);
  READMEMBER(int_btor);
  READMEMBER(avg_btor);
  READMEMBER(int_modb);
  READMEMBER(avg_modb);
  READMEMBER(int_ekin);
  READMEMBER(avg_ekin);
  return absl::OkStatus();
}

absl::Status vmecpp::Threed1AxisGeometry::WriteTo(H5::H5File& file) const {
  file.createGroup(this->H5key);
  WRITEMEMBER(raxis_symm);
  WRITEMEMBER(zaxis_symm);
  WRITEMEMBER(raxis_asym);
  WRITEMEMBER(zaxis_asym);

  return absl::OkStatus();
}

absl::Status vmecpp::Threed1AxisGeometry::LoadInto(Threed1AxisGeometry& m_obj,
                                                   H5::H5File& from_file) {
  READMEMBER(raxis_symm);
  READMEMBER(zaxis_symm);
  READMEMBER(raxis_asym);
  READMEMBER(zaxis_asym);
  return absl::OkStatus();
}

absl::Status vmecpp::Threed1Betas::WriteTo(H5::H5File& file) const {
  file.createGroup(this->H5key);
  WRITEMEMBER(betatot);
  WRITEMEMBER(betapol);
  WRITEMEMBER(betator);
  WRITEMEMBER(rbtor);
  WRITEMEMBER(betaxis);
  WRITEMEMBER(betstr);

  return absl::OkStatus();
}
absl::Status vmecpp::Threed1Betas::LoadInto(Threed1Betas& m_obj,
                                            H5::H5File& from_file) {
  READMEMBER(betatot);
  READMEMBER(betapol);
  READMEMBER(betator);
  READMEMBER(rbtor);
  READMEMBER(betaxis);
  READMEMBER(betstr);
  return absl::OkStatus();
}

absl::Status vmecpp::Threed1ShafranovIntegrals::WriteTo(
    H5::H5File& file) const {
  file.createGroup(this->H5key);
  WRITEMEMBER(scaling_ratio);
  WRITEMEMBER(r_lao);
  WRITEMEMBER(f_lao);
  WRITEMEMBER(f_geo);
  WRITEMEMBER(smaleli);
  WRITEMEMBER(betai);
  WRITEMEMBER(musubi);
  WRITEMEMBER(lambda);
  WRITEMEMBER(s11);
  WRITEMEMBER(s12);
  WRITEMEMBER(s13);
  WRITEMEMBER(s2);
  WRITEMEMBER(s3);
  WRITEMEMBER(delta1);
  WRITEMEMBER(delta2);
  WRITEMEMBER(delta3);

  return absl::OkStatus();
}
absl::Status vmecpp::Threed1ShafranovIntegrals::LoadInto(
    Threed1ShafranovIntegrals& m_obj, H5::H5File& from_file) {
  READMEMBER(scaling_ratio);
  READMEMBER(r_lao);
  READMEMBER(f_lao);
  READMEMBER(f_geo);
  READMEMBER(smaleli);
  READMEMBER(betai);
  READMEMBER(musubi);
  READMEMBER(lambda);
  READMEMBER(s11);
  READMEMBER(s12);
  READMEMBER(s13);
  READMEMBER(s2);
  READMEMBER(s3);
  READMEMBER(delta1);
  READMEMBER(delta2);
  READMEMBER(delta3);
  return absl::OkStatus();
}

absl::Status vmecpp::WOutFileContents::WriteTo(H5::H5File& file) const {
  file.createGroup(this->H5key);

  WRITEMEMBER(version);
  // TODO(jurasic) input_extension
  WRITEMEMBER(sign_of_jacobian);
  WRITEMEMBER(gamma);
  WRITEMEMBER(pcurr_type);
  WRITEMEMBER(pmass_type);
  WRITEMEMBER(piota_type);
  WRITEMEMBER(am);
  WRITEMEMBER(ac);
  WRITEMEMBER(ai);
  WRITEMEMBER(am_aux_s);
  WRITEMEMBER(am_aux_f);
  WRITEMEMBER(ac_aux_s);
  WRITEMEMBER(ac_aux_f);
  WRITEMEMBER(ai_aux_s);
  WRITEMEMBER(ai_aux_f);
  WRITEMEMBER(nfp);
  WRITEMEMBER(mpol);
  WRITEMEMBER(ntor);
  WRITEMEMBER(lasym);
  WRITEMEMBER(ns);
  WRITEMEMBER(ftolv);
  WRITEMEMBER(maximum_iterations);
  WRITEMEMBER(lfreeb);
  WRITEMEMBER(mgrid_file);
  WRITEMEMBER(nextcur);
  WRITEMEMBER(extcur);
  WRITEMEMBER(mgrid_mode);
  WRITEMEMBER(wb);
  WRITEMEMBER(wp);
  WRITEMEMBER(rmax_surf);
  WRITEMEMBER(rmin_surf);
  WRITEMEMBER(zmax_surf);
  WRITEMEMBER(mnmax);
  WRITEMEMBER(mnmax_nyq);
  WRITEMEMBER(ier_flag);
  WRITEMEMBER(aspect);
  WRITEMEMBER(betatot);
  WRITEMEMBER(betapol);
  WRITEMEMBER(betator);
  WRITEMEMBER(betaxis);
  WRITEMEMBER(b0);
  WRITEMEMBER(rbtor0);
  WRITEMEMBER(rbtor);
  WRITEMEMBER(IonLarmor);
  WRITEMEMBER(VolAvgB);
  WRITEMEMBER(ctor);
  WRITEMEMBER(Aminor_p);
  WRITEMEMBER(Rmajor_p);
  WRITEMEMBER(volume_p);
  WRITEMEMBER(fsqr);
  WRITEMEMBER(fsqz);
  WRITEMEMBER(fsql);
  WRITEMEMBER(iota_full);
  WRITEMEMBER(safety_factor);
  WRITEMEMBER(pressure_full);
  WRITEMEMBER(toroidal_flux);
  WRITEMEMBER(phipf);
  WRITEMEMBER(poloidal_flux);
  WRITEMEMBER(chipf);
  WRITEMEMBER(jcuru);
  WRITEMEMBER(jcurv);
  WRITEMEMBER(iota_half);
  WRITEMEMBER(mass);
  WRITEMEMBER(pressure_half);
  WRITEMEMBER(beta);
  WRITEMEMBER(buco);
  WRITEMEMBER(bvco);
  WRITEMEMBER(dVds);
  WRITEMEMBER(spectral_width);
  WRITEMEMBER(phips);
  WRITEMEMBER(overr);
  WRITEMEMBER(jdotb);
  // TODO(jurasic) We will deprecate HDF5 soon, regenerate large_cpp_tests
  // reference files with all quantities once that is done
  //  WRITEMEMBER(bdotb);
  WRITEMEMBER(bdotgradv);
  WRITEMEMBER(DMerc);
  WRITEMEMBER(Dshear);
  WRITEMEMBER(Dwell);
  WRITEMEMBER(Dcurr);
  WRITEMEMBER(Dgeod);
  WRITEMEMBER(equif);
  WRITEMEMBER(curlabel);
  WRITEMEMBER(potvac);
  WRITEMEMBER(xm);
  WRITEMEMBER(xn);
  WRITEMEMBER(xm_nyq);
  WRITEMEMBER(xn_nyq);
  WRITEMEMBER(raxis_c);
  WRITEMEMBER(zaxis_s);
  WRITEMEMBER(rmnc);
  WRITEMEMBER(zmns);
  WRITEMEMBER(lmns_full);
  WRITEMEMBER(lmns);
  WRITEMEMBER(gmnc);
  WRITEMEMBER(bmnc);
  WRITEMEMBER(bsubumnc);
  WRITEMEMBER(bsubvmnc);
  WRITEMEMBER(bsubsmns);
  WRITEMEMBER(bsubsmns_full);
  WRITEMEMBER(bsupumnc);
  WRITEMEMBER(bsupvmnc);
  WRITEMEMBER(raxis_s);
  WRITEMEMBER(zaxis_c);
  WRITEMEMBER(rmns);
  WRITEMEMBER(zmnc);
  WRITEMEMBER(lmnc_full);
  WRITEMEMBER(lmnc);
  WRITEMEMBER(gmns);
  WRITEMEMBER(bmns);
  WRITEMEMBER(bsubumns);
  WRITEMEMBER(bsubvmns);
  WRITEMEMBER(bsubsmnc);
  WRITEMEMBER(bsubsmnc_full);
  WRITEMEMBER(bsupumns);
  WRITEMEMBER(bsupvmns);

  return absl::OkStatus();
}

absl::Status vmecpp::WOutFileContents::LoadInto(WOutFileContents& m_obj,
                                                H5::H5File& from_file) {
  READMEMBER(version);
  // TODO(jurasic) input_extension
  READMEMBER(sign_of_jacobian);
  READMEMBER(gamma);
  READMEMBER(pcurr_type);
  READMEMBER(pmass_type);
  READMEMBER(piota_type);
  READMEMBER(am);
  READMEMBER(ac);
  READMEMBER(ai);
  READMEMBER(am_aux_s);
  READMEMBER(am_aux_f);
  READMEMBER(ac_aux_s);
  READMEMBER(ac_aux_f);
  READMEMBER(ai_aux_s);
  READMEMBER(ai_aux_f);
  READMEMBER(nfp);
  READMEMBER(mpol);
  READMEMBER(ntor);
  READMEMBER(lasym);
  READMEMBER(ns);
  READMEMBER(ftolv);
  READMEMBER(maximum_iterations);
  READMEMBER(lfreeb);
  READMEMBER(mgrid_file);
  // Compatibility with HDF5 files that do not have the nextcur and extcur
  // fields yet (before v0.3.3)
  if (m_obj.lfreeb) {
    READMEMBER(nextcur);
    READMEMBER(extcur);
  } else {
    m_obj.nextcur = 0;
    m_obj.extcur = Eigen::Vector<double, 0>::Zero();
  }
  READMEMBER(mgrid_mode);
  READMEMBER(wb);
  READMEMBER(wp);
  READMEMBER(rmax_surf);
  READMEMBER(rmin_surf);
  READMEMBER(zmax_surf);
  READMEMBER(mnmax);
  READMEMBER(mnmax_nyq);
  READMEMBER(ier_flag);
  READMEMBER(aspect);
  READMEMBER(betatot);
  READMEMBER(betapol);
  READMEMBER(betator);
  READMEMBER(betaxis);
  READMEMBER(b0);
  READMEMBER(rbtor0);
  READMEMBER(rbtor);
  READMEMBER(IonLarmor);
  READMEMBER(VolAvgB);
  READMEMBER(ctor);
  READMEMBER(Aminor_p);
  READMEMBER(Rmajor_p);
  READMEMBER(volume_p);
  READMEMBER(fsqr);
  READMEMBER(fsqz);
  READMEMBER(fsql);
  READMEMBER(iota_full);
  READMEMBER(safety_factor);
  READMEMBER(pressure_full);
  READMEMBER(toroidal_flux);
  READMEMBER(phipf);
  READMEMBER(poloidal_flux);
  READMEMBER(chipf);
  READMEMBER(jcuru);
  READMEMBER(jcurv);
  READMEMBER(iota_half);
  READMEMBER(mass);
  READMEMBER(pressure_half);
  READMEMBER(beta);
  READMEMBER(buco);
  READMEMBER(bvco);
  READMEMBER(dVds);
  READMEMBER(spectral_width);
  READMEMBER(phips);
  READMEMBER(overr);
  READMEMBER(jdotb);
  // TODO(jurasic) We will deprecate HDF5 soon, regenerate large_cpp_tests
  // reference files with all quantities once that is done
  //  READMEMBER(bdotb);
  READMEMBER(bdotgradv);
  READMEMBER(DMerc);
  READMEMBER(Dshear);
  READMEMBER(Dwell);
  READMEMBER(Dcurr);
  READMEMBER(Dgeod);
  READMEMBER(equif);
  READMEMBER(curlabel);
  READMEMBER(potvac);
  READMEMBER(xm);
  READMEMBER(xn);
  READMEMBER(xm_nyq);
  READMEMBER(xn_nyq);
  READMEMBER(raxis_c);
  READMEMBER(zaxis_s);
  READMEMBER(rmnc);
  READMEMBER(zmns);
  READMEMBER(lmns_full);
  READMEMBER(lmns);
  READMEMBER(gmnc);
  READMEMBER(bmnc);
  READMEMBER(bsubumnc);
  READMEMBER(bsubvmnc);
  READMEMBER(bsubsmns);
  READMEMBER(bsubsmns_full);
  READMEMBER(bsupumnc);
  READMEMBER(bsupvmnc);
  READMEMBER(raxis_s);
  READMEMBER(zaxis_c);
  READMEMBER(rmns);
  READMEMBER(zmnc);
  READMEMBER(lmnc_full);
  READMEMBER(lmnc);
  READMEMBER(gmns);
  READMEMBER(bmns);
  READMEMBER(bsubumns);
  READMEMBER(bsubvmns);
  READMEMBER(bsubsmnc);
  READMEMBER(bsubsmnc_full);
  READMEMBER(bsupumns);
  READMEMBER(bsupvmns);

  return absl::OkStatus();
}

#undef WRITEMEMBER
#undef READMEMBER

absl::Status vmecpp::OutputQuantities::Save(
    const std::filesystem::path& path) const {
  H5::H5File file(path, H5F_ACC_TRUNC);

  absl::Status status;

  status = vmec_internal_results.WriteTo(file);
  if (!status.ok()) {
    return status;
  }

  status = remaining_metric.WriteTo(file);
  if (!status.ok()) {
    return status;
  }

  status = b_cylindrical.WriteTo(file);
  if (!status.ok()) {
    return status;
  }

  status = bsubs_half.WriteTo(file);
  if (!status.ok()) {
    return status;
  }

  status = bsubs_full.WriteTo(file);
  if (!status.ok()) {
    return status;
  }

  status = covariant_b_derivatives.WriteTo(file);
  if (!status.ok()) {
    return status;
  }

  status = jxbout.WriteTo(file);
  if (!status.ok()) {
    return status;
  }

  status = mercier_intermediate.WriteTo(file);
  if (!status.ok()) {
    return status;
  }

  status = mercier.WriteTo(file);
  if (!status.ok()) {
    return status;
  }

  status = threed1_first_table_intermediate.WriteTo(file);
  if (!status.ok()) {
    return status;
  }

  status = threed1_first_table.WriteTo(file);
  if (!status.ok()) {
    return status;
  }

  status = threed1_geometric_magnetic_intermediate.WriteTo(file);
  if (!status.ok()) {
    return status;
  }

  status = threed1_geometric_magnetic.WriteTo(file);
  if (!status.ok()) {
    return status;
  }

  status = threed1_volumetrics.WriteTo(file);
  if (!status.ok()) {
    return status;
  }

  status = threed1_axis.WriteTo(file);
  if (!status.ok()) {
    return status;
  }

  status = threed1_betas.WriteTo(file);
  if (!status.ok()) {
    return status;
  }

  status = threed1_shafranov_integrals.WriteTo(file);
  if (!status.ok()) {
    return status;
  }

  status = wout.WriteTo(file);
  if (!status.ok()) {
    return status;
  }

  status = indata.WriteTo(file);
  if (!status.ok()) {
    return status;
  }

  return absl::OkStatus();
}

absl::StatusOr<vmecpp::OutputQuantities> vmecpp::OutputQuantities::Load(
    const std::filesystem::path& path) {
  H5::H5File file(path, H5F_ACC_RDONLY);

  OutputQuantities oq;
  absl::Status status;

  status = decltype(oq.vmec_internal_results)::LoadInto(
      oq.vmec_internal_results, file);
  if (!status.ok()) {
    return status;
  }

  status = decltype(oq.remaining_metric)::LoadInto(oq.remaining_metric, file);
  if (!status.ok()) {
    return status;
  }

  status = decltype(oq.b_cylindrical)::LoadInto(oq.b_cylindrical, file);
  if (!status.ok()) {
    return status;
  }

  status = decltype(oq.bsubs_half)::LoadInto(oq.bsubs_half, file);
  if (!status.ok()) {
    return status;
  }

  status = decltype(oq.bsubs_full)::LoadInto(oq.bsubs_full, file);
  if (!status.ok()) {
    return status;
  }

  status = decltype(oq.covariant_b_derivatives)::LoadInto(
      oq.covariant_b_derivatives, file);
  if (!status.ok()) {
    return status;
  }

  status = decltype(oq.jxbout)::LoadInto(oq.jxbout, file);
  if (!status.ok()) {
    return status;
  }

  status = decltype(oq.mercier_intermediate)::LoadInto(oq.mercier_intermediate,
                                                       file);
  if (!status.ok()) {
    return status;
  }

  status = decltype(oq.mercier)::LoadInto(oq.mercier, file);
  if (!status.ok()) {
    return status;
  }

  status = decltype(oq.threed1_first_table_intermediate)::LoadInto(
      oq.threed1_first_table_intermediate, file);
  if (!status.ok()) {
    return status;
  }

  status =
      decltype(oq.threed1_first_table)::LoadInto(oq.threed1_first_table, file);
  if (!status.ok()) {
    return status;
  }

  status = decltype(oq.threed1_geometric_magnetic_intermediate)::LoadInto(
      oq.threed1_geometric_magnetic_intermediate, file);
  if (!status.ok()) {
    return status;
  }

  status = decltype(oq.threed1_geometric_magnetic)::LoadInto(
      oq.threed1_geometric_magnetic, file);
  if (!status.ok()) {
    return status;
  }

  status =
      decltype(oq.threed1_volumetrics)::LoadInto(oq.threed1_volumetrics, file);
  if (!status.ok()) {
    return status;
  }

  status = decltype(oq.threed1_axis)::LoadInto(oq.threed1_axis, file);
  if (!status.ok()) {
    return status;
  }

  status = decltype(oq.threed1_betas)::LoadInto(oq.threed1_betas, file);
  if (!status.ok()) {
    return status;
  }

  status = decltype(oq.threed1_shafranov_integrals)::LoadInto(
      oq.threed1_shafranov_integrals, file);
  if (!status.ok()) {
    return status;
  }

  status = decltype(oq.wout)::LoadInto(oq.wout, file);
  if (!status.ok()) {
    return status;
  }

  status = decltype(oq.indata)::LoadInto(oq.indata, file);
  if (!status.ok()) {
    return status;
  }

  return oq;
}

vmecpp::OutputQuantities vmecpp::ComputeOutputQuantities(
    const int sign_of_jacobian, const VmecINDATA& indata, const Sizes& s,
    const FlowControl& fc, const VmecConstants& constants,
    const FourierBasisFastPoloidal& t, const HandoverStorage& h,
    const std::string& mgrid_mode,
    const std::vector<std::unique_ptr<RadialPartitioning>>& radial_partitioning,
    const std::vector<std::unique_ptr<FourierGeometry>>& decomposed_x,
    const std::vector<std::unique_ptr<IdealMhdModel>>& models_from_threads,
    const std::vector<std::unique_ptr<RadialProfiles>>& radial_profiles,
    const VmecCheckpoint& checkpoint, VacuumPressureState vacuum_pressure_state,
    VmecStatus vmec_status, int iter2) {
  OutputQuantities output_quantities;

  output_quantities.vmec_internal_results = GatherDataFromThreads(
      sign_of_jacobian, s, fc, constants, radial_partitioning, decomposed_x,
      models_from_threads, radial_profiles);

  if (vmec_status == VmecStatus::NORMAL_TERMINATION ||
      vmec_status == VmecStatus::SUCCESSFUL_TERMINATION ||
      indata.return_outputs_even_if_not_converged) {
    MeshBledingBSubZeta(
        s, fc,
        /*m_vmec_internal_results=*/output_quantities.vmec_internal_results);

    const PoloidalCurrentToFixBSubV pctf = ComputePoloidalCurrentToFixBSubV(
        s, output_quantities.vmec_internal_results);
    FixupPoloidalCurrent(
        s, pctf,
        /*m_vmec_internal_results=*/output_quantities.vmec_internal_results);

    if (checkpoint == VmecCheckpoint::BCOVAR_FILEOUT) {
      return output_quantities;  // output_quantities partially uninitialized.
    }

    RecomputeToroidalFlux(
        fc,
        /*m_vmec_internal_results=*/output_quantities.vmec_internal_results);

    // eqfor
    output_quantities.remaining_metric =
        ComputeRemainingMetric(s, output_quantities.vmec_internal_results);
    output_quantities.b_cylindrical =
        BCylindricalComponents(s, output_quantities.vmec_internal_results,
                               output_quantities.remaining_metric);
    output_quantities.bsubs_half =
        ComputeBSubSOnHalfGrid(s, output_quantities.vmec_internal_results,
                               output_quantities.remaining_metric);

    if (checkpoint == VmecCheckpoint::BSS) {
      return output_quantities;  // output_quantities partially uninitialized.
    }

    // -> jxbforce
    output_quantities.bsubs_full =
        PutBSubSOnFullGrid(s, output_quantities.vmec_internal_results,
                           output_quantities.bsubs_half);

    SymmetryDecomposedCovariantB decomposed_bcov =
        DecomposeCovariantBBySymmetry(s,
                                      output_quantities.vmec_internal_results,
                                      output_quantities.bsubs_full);

    output_quantities.covariant_b_derivatives = LowPassFilterCovariantB(
        s, t, decomposed_bcov,
        /*m_vmec_internal_results=*/output_quantities.vmec_internal_results);

    if (checkpoint == VmecCheckpoint::LOWPASS_BCOVARIANT) {
      return output_quantities;  // output_quantities partially uninitialized.
    }

    // TODO(jons): optionally, re-compute B_s to solve radial force balance
    // (lbsubs flag in Fortran VMEC)

    ExtrapolateBSubS(s, fc, /*m_bsubs_full=*/output_quantities.bsubs_full);

    if (checkpoint == VmecCheckpoint::EXTRAPOLATE_BSUBS) {
      return output_quantities;  // output_quantities partially uninitialized.
    }

    output_quantities.jxbout = ComputeJxBOutputFileContents(
        s, fc, output_quantities.vmec_internal_results,
        output_quantities.bsubs_full, output_quantities.covariant_b_derivatives,
        indata.return_outputs_even_if_not_converged, vmec_status);

    if (checkpoint == VmecCheckpoint::JXBOUT) {
      return output_quantities;  // output_quantities partially uninitialized.
    }

    output_quantities.mercier_intermediate =
        ComputeIntermediateMercierQuantities(
            s, fc, output_quantities.vmec_internal_results,
            output_quantities.jxbout);

    output_quantities.mercier =
        ComputeMercierStability(fc, output_quantities.vmec_internal_results,
                                output_quantities.mercier_intermediate);

    if (checkpoint == VmecCheckpoint::MERCIER) {
      return output_quantities;  // output_quantities partially uninitialized.
    }

    output_quantities.threed1_first_table_intermediate =
        ComputeIntermediateThreed1FirstTableQuantities(
            s, fc, output_quantities.vmec_internal_results);

    output_quantities.threed1_first_table = ComputeThreed1FirstTable(
        fc, output_quantities.vmec_internal_results, output_quantities.jxbout,
        output_quantities.threed1_first_table_intermediate);

    if (checkpoint == VmecCheckpoint::THREED1_FIRST_TABLE) {
      return output_quantities;  // output_quantities partially uninitialized.
    }

    output_quantities.threed1_geometric_magnetic_intermediate =
        ComputeIntermediateThreed1GeometricMagneticQuantities(
            s, fc, h, output_quantities.vmec_internal_results,
            output_quantities.jxbout,
            output_quantities.threed1_first_table_intermediate,
            vacuum_pressure_state);

    output_quantities.threed1_geometric_magnetic =
        ComputeThreed1GeometricMagneticQuantities(
            s, fc, h, output_quantities.vmec_internal_results,
            output_quantities.jxbout,
            output_quantities.threed1_first_table_intermediate,
            output_quantities.threed1_geometric_magnetic_intermediate);

    if (checkpoint == VmecCheckpoint::THREED1_GEOMAG) {
      return output_quantities;  // output_quantities partially uninitialized.
    }

    output_quantities.threed1_volumetrics = ComputeThreed1Volumetrics(
        output_quantities.threed1_geometric_magnetic_intermediate,
        output_quantities.threed1_geometric_magnetic);

    if (checkpoint == VmecCheckpoint::THREED1_VOLUMETRICS) {
      return output_quantities;  // output_quantities partially uninitialized.
    }

    output_quantities.threed1_axis = ComputeThreed1AxisGeometry(
        s, t, output_quantities.vmec_internal_results);

    if (checkpoint == VmecCheckpoint::THREED1_AXIS) {
      return output_quantities;  // output_quantities partially uninitialized.
    }

    output_quantities.threed1_betas = ComputeThreed1Betas(
        h, output_quantities.threed1_first_table_intermediate,
        output_quantities.threed1_geometric_magnetic_intermediate,
        output_quantities.threed1_geometric_magnetic);

    if (checkpoint == VmecCheckpoint::THREED1_BETAS) {
      return output_quantities;  // output_quantities partially uninitialized.
    }

    output_quantities.threed1_shafranov_integrals =
        ComputeThreed1ShafranovIntegrals(
            s, fc, h, output_quantities.vmec_internal_results,
            output_quantities.threed1_geometric_magnetic_intermediate,
            output_quantities.threed1_geometric_magnetic,
            vacuum_pressure_state);

    if (checkpoint == VmecCheckpoint::THREED1_SHAFRANOV_INTEGRALS) {
      return output_quantities;  // output_quantities partially uninitialized.
    }

    // NOTE: We slightly deviate from Fortran VMEC here,
    // in that we only compute the `wout` file if the run converged
    // or simply ran out of iterations.
    // The `wout` file is NOT computed in case the run crashed,
    // since the `threed1` file data would not get computed in that case
    // and the `wout` file would mainly contain garbage data.
    // For debugging, better get the input file in case VMEC++ crashed
    // and setup a stand-alone test case to figure out what went wrong
    // and how to prevent that crash in the future.
    output_quantities.wout = ComputeWOutFileContents(
        indata, s, t, fc, constants, h, mgrid_mode,
        /*m_vmec_internal_results=*/output_quantities.vmec_internal_results,
        output_quantities.bsubs_half, output_quantities.mercier,
        output_quantities.jxbout,
        output_quantities.threed1_first_table_intermediate,
        output_quantities.threed1_first_table,
        output_quantities.threed1_geometric_magnetic,
        output_quantities.threed1_axis, output_quantities.threed1_betas,
        vmec_status, iter2);

    // TODO(jons): freeb_data output to be implemented when free-boundary test
    // case is set up
  }

  output_quantities.indata = indata;

  return output_quantities;
}  // ComputeOutputQuantities

vmecpp::VmecInternalResults vmecpp::GatherDataFromThreads(
    const int sign_of_jacobian, const Sizes& s, const FlowControl& fc,
    const VmecConstants& constants,
    const std::vector<std::unique_ptr<RadialPartitioning>>& radial_partitioning,
    const std::vector<std::unique_ptr<FourierGeometry>>& decomposed_x,
    const std::vector<std::unique_ptr<IdealMhdModel>>& models_from_threads,
    const std::vector<std::unique_ptr<RadialProfiles>>& radial_profiles) {
  VmecInternalResults results;

  results.sign_of_jacobian = sign_of_jacobian;

  results.num_half = fc.ns - 1;
  results.num_full = fc.ns;

  results.nZnT_reduced = s.nThetaReduced * s.nZeta;

  results.sqrtSH = VectorXd::Zero(results.num_half);
  results.sqrtSF = VectorXd::Zero(results.num_full);

  results.sm = VectorXd::Zero(results.num_half);
  results.sp = VectorXd::Zero(results.num_half);

  results.phipF = VectorXd::Zero(results.num_full);
  results.chipF = VectorXd::Zero(results.num_full);

  // computed in RecomputeToroidalFlux here!
  results.phiF = VectorXd::Zero(results.num_full);

  results.iotaF = VectorXd::Zero(results.num_full);
  results.spectral_width = VectorXd::Zero(results.num_full);

  results.phipH = VectorXd::Zero(results.num_half);
  results.chipH = VectorXd::Zero(results.num_half);
  results.bvcoH = VectorXd::Zero(results.num_half);
  results.dVdsH = VectorXd::Zero(results.num_half);
  results.massH = VectorXd::Zero(results.num_half);
  results.presH = VectorXd::Zero(results.num_half);
  results.iotaH = VectorXd::Zero(results.num_half);
  results.currH = VectorXd::Zero(results.num_half);

  // state vector
  results.rmncc = RowMatrixXd::Zero(results.num_full, s.mnsize);
  results.zmnsc = RowMatrixXd::Zero(results.num_full, s.mnsize);
  results.lmnsc = RowMatrixXd::Zero(results.num_full, s.mnsize);
  if (s.lthreed) {
    results.rmnss = RowMatrixXd::Zero(results.num_full, s.mnsize);
    results.zmncs = RowMatrixXd::Zero(results.num_full, s.mnsize);
    results.lmncs = RowMatrixXd::Zero(results.num_full, s.mnsize);
  }
  if (s.lasym) {
    results.rmnsc = RowMatrixXd::Zero(results.num_full, s.mnsize);
    results.zmncc = RowMatrixXd::Zero(results.num_full, s.mnsize);
    results.lmncc = RowMatrixXd::Zero(results.num_full, s.mnsize);
    if (s.lthreed) {
      results.rmncs = RowMatrixXd::Zero(results.num_full, s.mnsize);
      results.zmnss = RowMatrixXd::Zero(results.num_full, s.mnsize);
      results.lmnss = RowMatrixXd::Zero(results.num_full, s.mnsize);
    }
  }

  // from inv-DFTs
  results.r_e = RowMatrixXd::Zero(results.num_full, s.nZnT);
  results.r_o = RowMatrixXd::Zero(results.num_full, s.nZnT);
  results.z_e = RowMatrixXd::Zero(results.num_full, s.nZnT);
  results.z_o = RowMatrixXd::Zero(results.num_full, s.nZnT);

  results.ru_e = RowMatrixXd::Zero(results.num_full, s.nZnT);
  results.ru_o = RowMatrixXd::Zero(results.num_full, s.nZnT);
  results.zu_e = RowMatrixXd::Zero(results.num_full, s.nZnT);
  results.zu_o = RowMatrixXd::Zero(results.num_full, s.nZnT);
  results.rv_e = RowMatrixXd::Zero(results.num_full, s.nZnT);
  results.rv_o = RowMatrixXd::Zero(results.num_full, s.nZnT);
  results.zv_e = RowMatrixXd::Zero(results.num_full, s.nZnT);
  results.zv_o = RowMatrixXd::Zero(results.num_full, s.nZnT);

  // from even-m and odd-m contributions
  results.ruFull = RowMatrixXd::Zero(results.num_full, s.nZnT);
  results.zuFull = RowMatrixXd::Zero(results.num_full, s.nZnT);

  // from Jacobian
  results.r12 = RowMatrixXd::Zero(results.num_half, s.nZnT);
  results.ru12 = RowMatrixXd::Zero(results.num_half, s.nZnT);
  results.zu12 = RowMatrixXd::Zero(results.num_half, s.nZnT);
  results.rs = RowMatrixXd::Zero(results.num_half, s.nZnT);
  results.zs = RowMatrixXd::Zero(results.num_half, s.nZnT);
  results.gsqrt = RowMatrixXd::Zero(results.num_half, s.nZnT);

  // metric elements
  results.guu = RowMatrixXd::Zero(results.num_half, s.nZnT);
  results.guv = RowMatrixXd::Zero(results.num_half, s.nZnT);
  results.gvv = RowMatrixXd::Zero(results.num_half, s.nZnT);

  // magnetic field
  results.bsupu = RowMatrixXd::Zero(results.num_half, s.nZnT);
  results.bsupv = RowMatrixXd::Zero(results.num_half, s.nZnT);
  results.bsubu = RowMatrixXd::Zero(results.num_half, s.nZnT);
  results.bsubv = RowMatrixXd::Zero(results.num_half, s.nZnT);
  results.bsubvF = RowMatrixXd::Zero(results.num_full, s.nZnT);
  results.total_pressure = RowMatrixXd::Zero(results.num_half, s.nZnT);

  const std::size_t num_threads = radial_partitioning.size();
  for (std::size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
    const RadialPartitioning& r = *radial_partitioning[thread_id];
    const IdealMhdModel& m = *models_from_threads[thread_id];
    const RadialProfiles& p = *radial_profiles[thread_id];

    const int nsMinH = r.nsMinH;
    const int nsMaxH = r.nsMaxH;

    const int nsMinF = r.nsMinF;
    const int nsMaxFIncludingLcfs = r.nsMaxFIncludingLcfs;

    const int nsMinF1 = r.nsMinF1;

    for (int jH = nsMinH; jH < nsMaxH; ++jH) {
      // half-grid points are overlapping --> only take unique ones !
      if (jH < nsMaxH - 1 || jH == fc.ns - 2) {
        results.sqrtSH[jH] = p.sqrtSH[jH - nsMinH];

        results.sm[jH] = p.sm[jH - nsMinH];
        results.sp[jH] = p.sp[jH - nsMinH];

        results.phipH[jH] = p.phipH[jH - nsMinH];
        results.chipH[jH] = p.chipH[jH - nsMinH];
        results.bvcoH[jH] = p.bvcoH[jH - nsMinH];
        results.dVdsH[jH] = p.dVdsH[jH - nsMinH];
        results.massH[jH] = p.massH[jH - nsMinH];
        results.presH[jH] = p.presH[jH - nsMinH];
        results.iotaH[jH] = p.iotaH[jH - nsMinH];
        results.currH[jH] = p.currH[jH - nsMinH];

        for (int kl = 0; kl < s.nZnT; ++kl) {
          int idx_global = jH * s.nZnT + kl;
          int idx_local = (jH - nsMinH) * s.nZnT + kl;

          // from Jacobian
          results.r12(idx_global) = m.r12[idx_local];
          results.ru12(idx_global) = m.ru12[idx_local];
          results.zu12(idx_global) = m.zu12[idx_local];
          results.rs(idx_global) = m.rs[idx_local];
          results.zs(idx_global) = m.zs[idx_local];
          results.gsqrt(idx_global) = m.gsqrt[idx_local];

          // metric elements
          results.guu(idx_global) = m.guu[idx_local];
          if (s.lthreed) {
            results.guv(idx_global) = m.guv[idx_local];
          }
          results.gvv(idx_global) = m.gvv[idx_local];

          // magnetic field
          results.bsupu(idx_global) = m.bsupu[idx_local];
          results.bsupv(idx_global) = m.bsupv[idx_local];
          results.bsubu(idx_global) = m.bsubu[idx_local];
          results.bsubv(idx_global) = m.bsubv[idx_local];
          results.total_pressure(idx_global) = m.totalPressure[idx_local];
        }  // kl
      }
    }  // jH

    for (int jF = nsMinF; jF < nsMaxFIncludingLcfs; ++jF) {
      results.sqrtSF[jF] = p.sqrtSF[jF - nsMinF1];

      // phipF is computed in RecomputeToroidalFlux here!
      results.phipF[jF] = p.phipF[jF - r.nsMinF1];

      results.chipF[jF] = p.chipF[jF - r.nsMinF1];

      results.iotaF[jF] = p.iotaF[jF - r.nsMinF1];
      results.spectral_width[jF] = p.spectral_width[jF - r.nsMinF1];

      // state vector
      for (int n = 0; n < s.ntor + 1; ++n) {
        for (int m = 0; m < s.mpol; ++m) {
          // FIXME(eguiraud) slow loop
          const int source_index =
              ((jF - nsMinF1) * s.mpol + m) * (s.ntor + 1) + n;
          const int target_index = (jF * (s.ntor + 1) + n) * s.mpol + m;

          results.rmncc(target_index) =
              decomposed_x[thread_id]->rmncc[source_index];
          results.zmnsc(target_index) =
              decomposed_x[thread_id]->zmnsc[source_index];
          results.lmnsc(target_index) =
              decomposed_x[thread_id]->lmnsc[source_index];
          if (s.lthreed) {
            results.rmnss(target_index) =
                decomposed_x[thread_id]->rmnss[source_index];
            results.zmncs(target_index) =
                decomposed_x[thread_id]->zmncs[source_index];
            results.lmncs(target_index) =
                decomposed_x[thread_id]->lmncs[source_index];
          }
          if (s.lasym) {
            results.rmnsc(target_index) =
                decomposed_x[thread_id]->rmnsc[source_index];
            results.zmncc(target_index) =
                decomposed_x[thread_id]->zmncc[source_index];
            results.lmncc(target_index) =
                decomposed_x[thread_id]->lmncc[source_index];
            if (s.lthreed) {
              results.rmncs(target_index) =
                  decomposed_x[thread_id]->rmncs[source_index];
              results.zmnss(target_index) =
                  decomposed_x[thread_id]->zmnss[source_index];
              results.lmnss(target_index) =
                  decomposed_x[thread_id]->lmnss[source_index];
            }
          }
        }  // m
      }  // n

      double unlamscale = 1.0;
      if (jF > 0) {
        unlamscale = -1.0 / constants.lamscale;
      }

      for (int kl = 0; kl < s.nZnT; ++kl) {
        int idx_global = jF * s.nZnT + kl;
        int idx_local = (jF - nsMinF) * s.nZnT + kl;
        int idx_local1 = (jF - nsMinF1) * s.nZnT + kl;

        results.bsubvF(idx_global) = m.blmn_e[idx_local] * unlamscale;

        // from inv-DFT
        results.r_e(idx_global) = m.r1_e[idx_local1];
        results.r_o(idx_global) = m.r1_o[idx_local1];
        results.z_e(idx_global) = m.z1_e[idx_local1];
        results.z_o(idx_global) = m.z1_o[idx_local1];
        results.ru_e(idx_global) = m.ru_e[idx_local1];
        results.ru_o(idx_global) = m.ru_o[idx_local1];
        results.zu_e(idx_global) = m.zu_e[idx_local1];
        results.zu_o(idx_global) = m.zu_o[idx_local1];
        if (s.lthreed) {
          results.rv_e(idx_global) = m.rv_e[idx_local1];
          results.rv_o(idx_global) = m.rv_o[idx_local1];
          results.zv_e(idx_global) = m.zv_e[idx_local1];
          results.zv_o(idx_global) = m.zv_o[idx_local1];
        } else {
          results.rv_e(idx_global) = 0.0;
          results.rv_o(idx_global) = 0.0;
          results.zv_e(idx_global) = 0.0;
          results.zv_o(idx_global) = 0.0;
        }

        // from even-m and odd-m contributions
        results.ruFull(idx_global) = m.ruFull[idx_local];
        results.zuFull(idx_global) = m.zuFull[idx_local];
      }  // kl
    }  // jF

    if (thread_id == 0) {
      // a single thread is enough; this should be consistent among threads
      results.currv = radial_profiles[thread_id]->currv;
    }
  }  // thread_id

  return results;
}  // GatherDataFromThreads

void vmecpp::MeshBledingBSubZeta(const Sizes& s, const FlowControl& fc,
                                 VmecInternalResults& m_vmec_internal_results) {
  // COMPUTE COVARIANT BSUBU,V (EVEN, ODD) ON HALF RADIAL MESH FOR FORCE BALANCE
  for (int jH = fc.ns - 3; jH >= 0; jH--) {
    // jF is the full-grid point outside jH-th half-grid point
    int jF = jH + 1;
    for (int kl = 0; kl < s.nZnT; ++kl) {
      int idx_sourceH = (jH + 1) * s.nZnT + kl;
      int idx_sourceF = jF * s.nZnT + kl;
      int idx_targetH = jH * s.nZnT + kl;

      m_vmec_internal_results.bsubv(idx_targetH) =
          2.0 * m_vmec_internal_results.bsubvF(idx_sourceF) -
          m_vmec_internal_results.bsubv(idx_sourceH);
    }  // kl
  }  // jH
}  // MeshBledingBSubZeta

vmecpp::PoloidalCurrentToFixBSubV vmecpp::ComputePoloidalCurrentToFixBSubV(
    const Sizes& s, const VmecInternalResults& vmec_internal_results) {
  PoloidalCurrentToFixBSubV poloidal_current_to_fix_bsubv;

  poloidal_current_to_fix_bsubv.poloidal_current_deviation =
      VectorXd::Zero(vmec_internal_results.num_half);

  for (int jH = 0; jH < vmec_internal_results.num_half; ++jH) {
    double poloidal_current_from_bsubv = 0.0;
    for (int kl = 0; kl < s.nZnT; ++kl) {
      int iHalf = jH * s.nZnT + kl;
      int l = kl % s.nThetaEff;
      poloidal_current_from_bsubv +=
          vmec_internal_results.bsubv(iHalf) * s.wInt[l];
    }  // kl

    // deviation = actual value - expected value
    poloidal_current_to_fix_bsubv.poloidal_current_deviation[jH] =
        poloidal_current_from_bsubv - vmec_internal_results.bvcoH[jH];
  }  // jH

  return poloidal_current_to_fix_bsubv;
}  // ComputePoloidalCurrentToFixBSubV

void vmecpp::FixupPoloidalCurrent(
    const Sizes& s,
    const PoloidalCurrentToFixBSubV& poloidal_current_to_fix_bsubv,
    VmecInternalResults& m_vmec_internal_results) {
  for (int jH = 0; jH < m_vmec_internal_results.num_half; ++jH) {
    for (int kl = 0; kl < s.nZnT; ++kl) {
      int iHalf = jH * s.nZnT + kl;

      // fix bsubv by subtracting the deviation in poloidal current
      m_vmec_internal_results.bsubv(iHalf) -=
          poloidal_current_to_fix_bsubv.poloidal_current_deviation[jH];
    }  // kl
  }  // jH
}  // FixupPoloidalCurrent

void vmecpp::RecomputeToroidalFlux(
    const FlowControl& fc, VmecInternalResults& m_vmec_internal_results) {
  // quadrature in radial direction
  m_vmec_internal_results.phiF[0] = 0.0;
  for (int jF = 1; jF < fc.ns; ++jF) {
    m_vmec_internal_results.phiF[jF] = m_vmec_internal_results.phiF[jF - 1] +
                                       m_vmec_internal_results.phipF[jF - 1];
  }  // jF

  // now apply scaling
  const double scaling_factor =
      m_vmec_internal_results.sign_of_jacobian * 2.0 * M_PI * fc.deltaS;
  for (int jF = 0; jF < fc.ns; ++jF) {
    m_vmec_internal_results.phiF[jF] *= scaling_factor;
  }  // jF
}  // RecomputeToroidalFlux

vmecpp::RemainingMetric vmecpp::ComputeRemainingMetric(
    const Sizes& s, const VmecInternalResults& vmec_internal_results) {
  RemainingMetric remaining_metric;

  // for bss
  remaining_metric.rv12 =
      RowMatrixXd::Zero(vmec_internal_results.num_half, s.nZnT);
  remaining_metric.zv12 =
      RowMatrixXd::Zero(vmec_internal_results.num_full, s.nZnT);
  remaining_metric.rs12 =
      RowMatrixXd::Zero(vmec_internal_results.num_half, s.nZnT);
  remaining_metric.zs12 =
      RowMatrixXd::Zero(vmec_internal_results.num_full, s.nZnT);
  remaining_metric.gsu =
      RowMatrixXd::Zero(vmec_internal_results.num_half, s.nZnT);
  remaining_metric.gsv =
      RowMatrixXd::Zero(vmec_internal_results.num_full, s.nZnT);

  for (int jH = 0; jH < vmec_internal_results.num_half; ++jH) {
    for (int kl = 0; kl < s.nZnT; ++kl) {
      int idxH = jH * s.nZnT + kl;
      int idxFi = idxH;
      int idxFo = (jH + 1) * s.nZnT + kl;

      double rv_e12 = (vmec_internal_results.rv_e(idxFo) +
                       vmec_internal_results.rv_e(idxFi)) /
                      2.0;
      double rv_o12 = (vmec_internal_results.rv_o(idxFo) +
                       vmec_internal_results.rv_o(idxFi)) /
                      2.0;
      remaining_metric.rv12(idxH) =
          rv_e12 + vmec_internal_results.sqrtSH[jH] * rv_o12;

      double zv_e12 = (vmec_internal_results.zv_e(idxFo) +
                       vmec_internal_results.zv_e(idxFi)) /
                      2.0;
      double zv_o12 = (vmec_internal_results.zv_o(idxFo) +
                       vmec_internal_results.zv_o(idxFi)) /
                      2.0;
      remaining_metric.zv12(idxH) =
          zv_e12 + vmec_internal_results.sqrtSH[jH] * zv_o12;

      double r_o12 = (vmec_internal_results.r_o(idxFo) +
                      vmec_internal_results.r_o(idxFi)) /
                     2.0;
      double z_o12 = (vmec_internal_results.z_o(idxFo) +
                      vmec_internal_results.z_o(idxFi)) /
                     2.0;
      remaining_metric.rs12(idxH) =
          vmec_internal_results.rs(idxH) +
          r_o12 / (2.0 * vmec_internal_results.sqrtSH[jH]);
      remaining_metric.zs12(idxH) =
          vmec_internal_results.zs(idxH) +
          z_o12 / (2.0 * vmec_internal_results.sqrtSH[jH]);

      remaining_metric.gsu(idxH) =
          remaining_metric.rs12(idxH) * vmec_internal_results.ru12(idxH) +
          remaining_metric.zs12(idxH) * vmec_internal_results.zu12(idxH);
      remaining_metric.gsv(idxH) =
          remaining_metric.rs12(idxH) * remaining_metric.rv12(idxH) +
          remaining_metric.zs12(idxH) * remaining_metric.zv12(idxH);
    }  // kl
  }  // jH

  return remaining_metric;
}  // ComputeRemainingMetric

vmecpp::CylindricalComponentsOfB vmecpp::BCylindricalComponents(
    const Sizes& s, const VmecInternalResults& vmec_internal_results,
    const RemainingMetric& remaining_metric) {
  CylindricalComponentsOfB b_cylindrical;

  b_cylindrical.b_r = RowMatrixXd::Zero(vmec_internal_results.num_half, s.nZnT);
  b_cylindrical.b_phi =
      RowMatrixXd::Zero(vmec_internal_results.num_half, s.nZnT);
  b_cylindrical.b_z = RowMatrixXd::Zero(vmec_internal_results.num_half, s.nZnT);

  for (int jH = 0; jH < vmec_internal_results.num_half; ++jH) {
    for (int kl = 0; kl < s.nZnT; ++kl) {
      int idxH = jH * s.nZnT + kl;

      b_cylindrical.b_r(idxH) =
          vmec_internal_results.bsupu(idxH) * vmec_internal_results.ru12(idxH) +
          vmec_internal_results.bsupv(idxH) * remaining_metric.rv12(idxH);
      b_cylindrical.b_phi(idxH) =
          vmec_internal_results.bsupv(idxH) * vmec_internal_results.r12(idxH);
      b_cylindrical.b_z(idxH) =
          vmec_internal_results.bsupu(idxH) * vmec_internal_results.zu12(idxH) +
          vmec_internal_results.bsupv(idxH) * remaining_metric.zv12(idxH);
    }  // kl
  }  // jH

  return b_cylindrical;
}  // BCylindricalComponents

vmecpp::BSubSHalf vmecpp::ComputeBSubSOnHalfGrid(
    const Sizes& s, const VmecInternalResults& vmec_internal_results,
    const RemainingMetric& remaining_metric) {
  BSubSHalf bsubs_half;
  bsubs_half.bsubs_half =
      RowMatrixXd::Zero(vmec_internal_results.num_half, s.nZnT);

  for (int jH = 0; jH < vmec_internal_results.num_half; ++jH) {
    for (int kl = 0; kl < s.nZnT; ++kl) {
      int idxH = jH * s.nZnT + kl;
      bsubs_half.bsubs_half(idxH) =
          vmec_internal_results.bsupu(idxH) * remaining_metric.gsu(idxH) +
          vmec_internal_results.bsupv(idxH) * remaining_metric.gsv(idxH);
    }  // kl
  }  // jH

  return bsubs_half;
}  // ComputeBSubS

vmecpp::BSubSFull vmecpp::PutBSubSOnFullGrid(
    const Sizes& s, const VmecInternalResults& vmec_internal_results,
    const BSubSHalf& bsubs_half) {
  BSubSFull bsubs_full;
  bsubs_full.bsubs_full =
      RowMatrixXd::Zero(vmec_internal_results.num_full, s.nZnT);

  // ignore axis and boundary for now
  for (int jF = 1; jF < vmec_internal_results.num_full - 1; ++jF) {
    const int jHi = jF - 1;
    const int jHo = jF;
    for (int kl = 0; kl < s.nZnT; ++kl) {
      bsubs_full.bsubs_full(jF * s.nZnT + kl) =
          (bsubs_half.bsubs_half(jHo * s.nZnT + kl) +
           bsubs_half.bsubs_half(jHi * s.nZnT + kl)) /
          2.0;
    }  // kl
  }  // jF

  return bsubs_full;
}  // PutBSubSOnFullGrid

vmecpp::SymmetryDecomposedCovariantB vmecpp::DecomposeCovariantBBySymmetry(
    const Sizes& s, const VmecInternalResults& vmec_internal_results,
    const BSubSFull& bsubs_full) {
  SymmetryDecomposedCovariantB decomposed_bcov;

  decomposed_bcov.bsubs_s = RowMatrixXd::Zero(
      vmec_internal_results.num_full, vmec_internal_results.nZnT_reduced);
  decomposed_bcov.bsubu_s = RowMatrixXd::Zero(
      vmec_internal_results.num_half, vmec_internal_results.nZnT_reduced);
  decomposed_bcov.bsubv_s = RowMatrixXd::Zero(
      vmec_internal_results.num_half, vmec_internal_results.nZnT_reduced);

  if (s.lasym) {
    decomposed_bcov.bsubs_a = VectorXd::Zero(
        vmec_internal_results.num_full, vmec_internal_results.nZnT_reduced);
    decomposed_bcov.bsubu_a = VectorXd::Zero(
        vmec_internal_results.num_half, vmec_internal_results.nZnT_reduced);
    decomposed_bcov.bsubv_a = VectorXd::Zero(
        vmec_internal_results.num_half, vmec_internal_results.nZnT_reduced);

    // fsym_fft:
    // CONTRACTS bs,bu,bv FROM FULL nu INTERVAL TO HALF-U INTERVAL
    // SO COS,SIN INTEGRALS CAN BE PERFORMED ON HALF-U INTERVAL
    // bs_s(v,u) = .5*( bs(v,u) - bs(-v,-u) )     ! * SIN(mu - nv)
    // bs_a(v,u) = .5*( bs(v,u) + bs(-v,-u) )     ! * COS(mu - nv)
    for (int jF = 0; jF < vmec_internal_results.num_full; ++jF) {
      for (int kl = 0; kl < vmec_internal_results.nZnT_reduced; ++kl) {
        const int source_index = jF * s.nZnT + kl;

        const int k = kl / s.nThetaReduced;
        const int l = kl % s.nThetaReduced;

        const int l_reversed = (s.nThetaEven - l) % s.nThetaEven;
        const int k_reversed = (s.nZeta - k) % s.nZeta;
        const int kl_reversed = k_reversed * s.nThetaEven + l_reversed;

        const int source_index_reversed = jF * s.nZnT + kl_reversed;

        const int target_index = jF * vmec_internal_results.nZnT_reduced + kl;

        decomposed_bcov.bsubs_s(target_index) =
            bsubs_full.bsubs_full(source_index) -
            bsubs_full.bsubs_full(source_index_reversed);
        decomposed_bcov.bsubs_a(target_index) =
            bsubs_full.bsubs_full(source_index) +
            bsubs_full.bsubs_full(source_index_reversed);
      }  // kl
    }  // jF
    for (int jH = 0; jH < vmec_internal_results.num_half; ++jH) {
      for (int kl = 0; kl < s.nZnT; ++kl) {
        const int source_index = jH * s.nZnT + kl;

        const int k = kl / s.nThetaEff;
        const int l = kl % s.nThetaEff;

        const int l_reversed = (s.nThetaEven - l) % s.nThetaEven;
        const int k_reversed = (s.nZeta - k) % s.nZeta;
        const int kl_reversed = k_reversed * s.nThetaEven + l_reversed;

        const int source_index_reversed = jH * s.nZnT + kl_reversed;

        const int target_index = jH * vmec_internal_results.nZnT_reduced + kl;

        decomposed_bcov.bsubu_s(target_index) =
            vmec_internal_results.bsubu(source_index) +
            vmec_internal_results.bsubu(source_index_reversed);
        decomposed_bcov.bsubu_a(target_index) =
            vmec_internal_results.bsubu(source_index) -
            vmec_internal_results.bsubu(source_index_reversed);
        decomposed_bcov.bsubv_s(target_index) =
            vmec_internal_results.bsubv(source_index) +
            vmec_internal_results.bsubv(source_index_reversed);
        decomposed_bcov.bsubv_a(target_index) =
            vmec_internal_results.bsubv(source_index) -
            vmec_internal_results.bsubv(source_index_reversed);
      }  // kl
    }  // jH
  } else {
    // stellarator-symmetric case: simply copy over data
    for (int jF = 0; jF < vmec_internal_results.num_full; ++jF) {
      for (int kl = 0; kl < vmec_internal_results.nZnT_reduced; ++kl) {
        const int source_index = jF * s.nZnT + kl;
        const int target_index = jF * vmec_internal_results.nZnT_reduced + kl;
        decomposed_bcov.bsubs_s(target_index) =
            bsubs_full.bsubs_full(source_index);
      }  // kl
    }  // jF
    for (int jH = 0; jH < vmec_internal_results.num_half; ++jH) {
      for (int kl = 0; kl < s.nZnT; ++kl) {
        const int source_index = jH * s.nZnT + kl;
        const int target_index = jH * vmec_internal_results.nZnT_reduced + kl;
        decomposed_bcov.bsubu_s(target_index) =
            vmec_internal_results.bsubu(source_index);
        decomposed_bcov.bsubv_s(target_index) =
            vmec_internal_results.bsubv(source_index);
      }  // kl
    }  // jH
  }

  return decomposed_bcov;
}  // DecomposeCovariantBBySymmetry

vmecpp::CovariantBDerivatives vmecpp::LowPassFilterCovariantB(
    const Sizes& s, const FourierBasisFastPoloidal& t,
    const SymmetryDecomposedCovariantB& decomposed_bcov,
    VmecInternalResults& m_vmec_internal_results) {
  CovariantBDerivatives covariant_b_derivatives;

  covariant_b_derivatives.bsubsu =
      RowMatrixXd::Zero(m_vmec_internal_results.num_full, s.nZnT);
  covariant_b_derivatives.bsubsv =
      RowMatrixXd::Zero(m_vmec_internal_results.num_full, s.nZnT);

  // TODO(jons): no split into (non-)stellarator-symmetric contributions?
  covariant_b_derivatives.bsubuv =
      RowMatrixXd::Zero(m_vmec_internal_results.num_half, s.nZnT);
  covariant_b_derivatives.bsubvu =
      RowMatrixXd::Zero(m_vmec_internal_results.num_half, s.nZnT);

  std::vector<double> bsubsu_s(m_vmec_internal_results.num_full * s.nZnT, 0.0);
  std::vector<double> bsubsv_s(m_vmec_internal_results.num_full * s.nZnT, 0.0);
  std::vector<double> bsubsu_a;
  std::vector<double> bsubsv_a;
  if (s.lasym) {
    bsubsu_a.resize(m_vmec_internal_results.num_full * s.nZnT);
    bsubsv_a.resize(m_vmec_internal_results.num_full * s.nZnT);
  }

  std::vector<double> bsubu_filtered_s(
      m_vmec_internal_results.num_half * s.nZnT, 0.0);
  std::vector<double> bsubv_filtered_s(
      m_vmec_internal_results.num_half * s.nZnT, 0.0);
  std::vector<double> bsubu_filtered_a;
  std::vector<double> bsubv_filtered_a;
  if (s.lasym) {
    bsubu_filtered_a.resize(m_vmec_internal_results.num_half * s.nZnT);
    bsubv_filtered_a.resize(m_vmec_internal_results.num_half * s.nZnT);
  }

  // FOURIER LOW-PASS FILTER bsubs
  for (int jF = 0; jF < m_vmec_internal_results.num_full; ++jF) {
    for (int m = 0; m < s.mpol; ++m) {
      for (int n = 0; n <= s.ntor; ++n) {
        // FOURIER TRANSFORM
        double dnorm1 = 1.0;
        if (m == s.mnyq) {
          dnorm1 /= 2.0;
        }
        if (n == s.nnyq && n != 0) {
          dnorm1 /= 2.0;
        }

        double bsubsmn1 = 0.0;
        double bsubsmn2 = 0.0;

        // only needed for non-stellarator-symmetric case
        double bsubsmn3 = 0.0;
        double bsubsmn4 = 0.0;

        for (int k = 0; k < s.nZeta; ++k) {
          // FIXME(eguiraud) slow loop
          const int idx_kn = k * (s.nnyq2 + 1) + n;
          for (int l = 0; l < s.nThetaReduced; ++l) {
            const int idx_ml = m * s.nThetaReduced + l;
            const int kl = k * s.nThetaReduced + l;

            const int source_index =
                jF * m_vmec_internal_results.nZnT_reduced + kl;

            // sin-cos
            const double tsini1 = t.sinmui[idx_ml] * t.cosnv[idx_kn] * dnorm1;
            // cos-sin
            const double tsini2 = t.cosmui[idx_ml] * t.sinnv[idx_kn] * dnorm1;

            // sin-cos
            bsubsmn1 += tsini1 * decomposed_bcov.bsubs_s(source_index);
            // cos-sin
            bsubsmn2 += tsini2 * decomposed_bcov.bsubs_s(source_index);

            if (s.lasym) {
              // cos-cos
              const double tcosi1 = t.cosmui[idx_ml] * t.cosnv[idx_kn] * dnorm1;
              // sin-sin
              const double tcosi2 = t.sinmui[idx_ml] * t.sinnv[idx_kn] * dnorm1;

              // cos-cos
              bsubsmn3 += tcosi1 * decomposed_bcov.bsubs_a(source_index);
              // sin-sin
              bsubsmn4 += tcosi2 * decomposed_bcov.bsubs_a(source_index);
            }  // lasym
          }  // l
        }  // k

        // FOURIER INVERSE TRANSFORM
        // Compute on u-v grid (must add symmetric, antisymmetric parts for
        // lasym=T)
        for (int k = 0; k < s.nZeta; ++k) {
          // FIXME(eguiraud) slow loop
          const int idx_kn = k * (s.nnyq2 + 1) + n;
          for (int l = 0; l < s.nThetaReduced; ++l) {
            const int idx_ml = m * s.nThetaReduced + l;

            const int kl = k * s.nThetaReduced + l;
            const int target_index =
                jF * m_vmec_internal_results.nZnT_reduced + kl;

            const double tcosm1 = t.cosmum[idx_ml] * t.cosnv[idx_kn];
            const double tcosm2 = t.sinmum[idx_ml] * t.sinnv[idx_kn];
            bsubsu_s[target_index] += tcosm1 * bsubsmn1 + tcosm2 * bsubsmn2;

            const double tcosn1 = t.sinmu[idx_ml] * t.sinnvn[idx_kn];
            const double tcosn2 = t.cosmu[idx_ml] * t.cosnvn[idx_kn];
            bsubsv_s[target_index] += tcosn1 * bsubsmn1 + tcosn2 * bsubsmn2;

            if (s.lasym) {
              const double tsinm1 = t.sinmum[idx_ml] * t.cosnv[idx_kn];
              const double tsinm2 = t.cosmum[idx_ml] * t.sinnv[idx_kn];
              bsubsu_a[target_index] += tsinm1 * bsubsmn3 + tsinm2 * bsubsmn4;

              const double tsinn1 = t.cosmu[idx_ml] * t.sinnvn[idx_kn];
              const double tsinn2 = t.sinmu[idx_ml] * t.cosnvn[idx_kn];
              bsubsv_a[target_index] += tsinn1 * bsubsmn3 + tsinn2 * bsubsmn4;
            }  // lasym
          }  // l
        }  // k
      }  // n
    }  // m
  }  // jF

  // FOURIER LOW-PASS FILTER bsubu AND bsubv
  for (int jH = 0; jH < m_vmec_internal_results.num_half; ++jH) {
    for (int m = 0; m < s.mpol; ++m) {
      for (int n = 0; n <= s.ntor; ++n) {
        // FOURIER TRANSFORM
        double dnorm1 = 1.0;
        if (m == s.mnyq) {
          dnorm1 /= 2.0;
        }
        if (n == s.nnyq && n != 0) {
          dnorm1 /= 2.0;
        }

        double bsubumn1 = 0.0;
        double bsubumn2 = 0.0;
        double bsubvmn1 = 0.0;
        double bsubvmn2 = 0.0;

        // only needed for non-stellarator-symmetric case
        double bsubumn3 = 0.0;
        double bsubumn4 = 0.0;
        double bsubvmn3 = 0.0;
        double bsubvmn4 = 0.0;

        for (int k = 0; k < s.nZeta; ++k) {
          // FIXME(eguiraud) slow loop
          const int idx_kn = k * (s.nnyq2 + 1) + n;
          for (int l = 0; l < s.nThetaReduced; ++l) {
            const int idx_ml = m * s.nThetaReduced + l;

            const int kl = k * s.nThetaReduced + l;
            const int source_index =
                jH * m_vmec_internal_results.nZnT_reduced + kl;

            // cos-cos
            const double tcosi1 = t.cosmui[idx_ml] * t.cosnv[idx_kn] * dnorm1;
            // sin-sin
            const double tcosi2 = t.sinmui[idx_ml] * t.sinnv[idx_kn] * dnorm1;

            // cos-cos
            bsubvmn1 += tcosi1 * decomposed_bcov.bsubv_s(source_index);
            // sin-sin
            bsubvmn2 += tcosi2 * decomposed_bcov.bsubv_s(source_index);
            // cos-cos
            bsubumn1 += tcosi1 * decomposed_bcov.bsubu_s(source_index);
            // sin-sin
            bsubumn2 += tcosi2 * decomposed_bcov.bsubu_s(source_index);

            if (s.lasym) {
              // sin-cos
              const double tsini1 = t.sinmui[idx_ml] * t.cosnv[idx_kn] * dnorm1;
              // cos-sin
              const double tsini2 = t.cosmui[idx_ml] * t.sinnv[idx_kn] * dnorm1;

              // sin-cos
              bsubvmn3 += tsini1 * decomposed_bcov.bsubv_a(source_index);
              // cos-sin
              bsubvmn4 += tsini2 * decomposed_bcov.bsubv_a(source_index);
              // sin-cos
              bsubumn3 += tsini1 * decomposed_bcov.bsubu_a(source_index);
              // cos-sin
              bsubumn4 += tsini2 * decomposed_bcov.bsubu_a(source_index);
            }  // lasym
          }  // l
        }  // k

        // FOURIER INVERSE TRANSFORM
        // Compute on u-v grid (must add symmetric, antisymmetric parts for
        // lasym=T)
        for (int k = 0; k < s.nZeta; ++k) {
          const int idx_kn = k * (s.nnyq2 + 1) + n;
          for (int l = 0; l < s.nThetaReduced; ++l) {
            const int idx_ml = m * s.nThetaReduced + l;

            const int kl = k * s.nThetaReduced + l;
            const int target_index =
                jH * m_vmec_internal_results.nZnT_reduced + kl;

            const double tcos1 = t.cosmu[idx_ml] * t.cosnv[idx_kn];
            const double tcos2 = t.sinmu[idx_ml] * t.sinnv[idx_kn];
            bsubu_filtered_s[target_index] +=
                tcos1 * bsubumn1 + tcos2 * bsubumn2;
            bsubv_filtered_s[target_index] +=
                tcos1 * bsubvmn1 + tcos2 * bsubvmn2;

            const double tsinm1 = t.sinmum[idx_ml] * t.cosnv[idx_kn];
            const double tsinm2 = t.cosmum[idx_ml] * t.sinnv[idx_kn];
            covariant_b_derivatives.bsubvu(target_index) +=
                tsinm1 * bsubvmn1 + tsinm2 * bsubvmn2;

            const double tsinn1 = t.cosmu[idx_ml] * t.sinnvn[idx_kn];
            const double tsinn2 = t.sinmu[idx_ml] * t.cosnvn[idx_kn];
            covariant_b_derivatives.bsubuv(target_index) +=
                tsinn1 * bsubumn1 + tsinn2 * bsubumn2;

            if (s.lasym) {
              const double tsin1 = t.sinmu[idx_ml] * t.cosnv[idx_kn];
              const double tsin2 = t.cosmu[idx_ml] * t.sinnv[idx_kn];
              bsubu_filtered_a[target_index] +=
                  tsin1 * bsubumn3 + tsin2 * bsubumn4;
              bsubv_filtered_a[target_index] +=
                  tsin1 * bsubvmn3 + tsin2 * bsubvmn4;

              const double tcosm1 = t.cosmum[idx_ml] * t.cosnv[idx_kn];
              const double tcosm2 = t.sinmum[idx_ml] * t.sinnv[idx_kn];
              covariant_b_derivatives.bsubvu(target_index) +=
                  tcosm1 * bsubvmn3 + tcosm2 * bsubvmn4;

              const double tcosn1 = t.sinmu[idx_ml] * t.sinnvn[idx_kn];
              const double tcosn2 = t.cosmu[idx_ml] * t.cosnvn[idx_kn];
              covariant_b_derivatives.bsubuv(target_index) +=
                  tcosn1 * bsubumn3 + tcosn2 * bsubumn4;
            }  // lasym
          }  // l
        }  // k
      }  // n
    }  // m
  }  // jH

  if (s.lasym) {
    // EXTEND FILTERED bsubu, bsubv TO NTHETA3 MESH
    // fext_fft
    for (int jH = 0; jH < m_vmec_internal_results.num_half; ++jH) {
      for (int k = 0; k < s.nZeta; ++k) {
        for (int l = 0; l < s.nThetaReduced; ++l) {
          const int kl = k * s.nThetaReduced + l;
          const int source_index =
              jH * m_vmec_internal_results.nZnT_reduced + kl;
          const int target_index = jH * s.nZnT + kl;

          m_vmec_internal_results.bsubu(target_index) =
              bsubu_filtered_s[source_index] + bsubu_filtered_a[source_index];
          m_vmec_internal_results.bsubv(target_index) =
              bsubv_filtered_s[source_index] + bsubv_filtered_a[source_index];
        }  // l
        for (int l = s.nThetaReduced; l < s.nThetaEven; ++l) {
          const int kl = k * s.nThetaEven + l;

          const int l_reversed = (s.nThetaEven - l) % s.nThetaEven;
          const int k_reversed = (s.nZeta - k) % s.nZeta;
          const int kl_reversed = k_reversed * s.nThetaEven + l_reversed;

          const int source_index_reversed = jH * s.nZnT + kl_reversed;

          const int target_index = jH * s.nZnT + kl;

          m_vmec_internal_results.bsubu(target_index) =
              bsubu_filtered_s[source_index_reversed] -
              bsubu_filtered_a[source_index_reversed];
          m_vmec_internal_results.bsubv(target_index) =
              bsubv_filtered_s[source_index_reversed] -
              bsubv_filtered_a[source_index_reversed];
        }  // l
      }  // k
    }  // jH
  } else {
    // simply overwrite non-filtered data in-place
    for (int jH = 0; jH < m_vmec_internal_results.num_half; ++jH) {
      for (int kl = 0; kl < s.nZnT; ++kl) {
        const int idx_kl = jH * s.nZnT + kl;
        m_vmec_internal_results.bsubu(idx_kl) = bsubu_filtered_s[idx_kl];
        m_vmec_internal_results.bsubv(idx_kl) = bsubv_filtered_s[idx_kl];
      }  // kl
    }  // jH
  }

  // EXTEND bsubsu, bsubsv TO NTHETA3 MESH
  if (s.lasym) {
    // fsym_invfft
    for (int jF = 0; jF < m_vmec_internal_results.num_full; ++jF) {
      for (int k = 0; k < s.nZeta; ++k) {
        for (int l = s.nThetaReduced; l < s.nThetaEven; ++l) {
          const int kl = k * s.nThetaEven + l;

          const int l_reversed = (s.nThetaEven - l) % s.nThetaEven;
          const int k_reversed = (s.nZeta - k) % s.nZeta;
          const int kl_reversed = k_reversed * s.nThetaEven + l_reversed;

          const int source_index_reversed = jF * s.nZnT + kl_reversed;

          const int target_index = jF * s.nZnT + kl;

          covariant_b_derivatives.bsubsu(target_index) =
              bsubsu_s[source_index_reversed] - bsubsu_a[source_index_reversed];
          covariant_b_derivatives.bsubsv(target_index) =
              bsubsv_s[source_index_reversed] - bsubsv_a[source_index_reversed];
        }  // l
        for (int l = 0; l < s.nThetaReduced; ++l) {
          const int kl = k * s.nThetaReduced + l;
          const int source_index =
              jF * m_vmec_internal_results.nZnT_reduced + kl;
          const int target_index = jF * s.nZnT + kl;

          covariant_b_derivatives.bsubsu(target_index) =
              bsubsu_s[source_index] + bsubsu_a[source_index];
          covariant_b_derivatives.bsubsv(target_index) =
              bsubsv_s[source_index] + bsubsv_a[source_index];
        }  // l
      }  // k
    }  // jH
  } else {
    // simply copy over data
    for (int jF = 0; jF < m_vmec_internal_results.num_full; ++jF) {
      for (int kl = 0; kl < s.nZnT; ++kl) {
        const int idx_kl = jF * s.nZnT + kl;
        covariant_b_derivatives.bsubsu(idx_kl) = bsubsu_s[idx_kl];
        covariant_b_derivatives.bsubsv(idx_kl) = bsubsv_s[idx_kl];
      }  // kl
    }  // jH
  }

  return covariant_b_derivatives;
}  // NOLINT(readability/fn_size)

void vmecpp::ExtrapolateBSubS(const Sizes& s, const FlowControl& fc,
                              BSubSFull& m_bsubs_full) {
  for (int kl = 0; kl < s.nZnT; ++kl) {
    // extrapolate towards axis from first two interior full-grid points
    const int index_0 = 0 * s.nZnT + kl;
    const int index_1 = 1 * s.nZnT + kl;
    const int index_2 = 2 * s.nZnT + kl;
    m_bsubs_full.bsubs_full(index_0) = 2.0 * m_bsubs_full.bsubs_full(index_1) -
                                       m_bsubs_full.bsubs_full(index_2);

    // extrapolate towards boundary from last two interior full-grid points
    const int index_ns_1 = (fc.ns - 1) * s.nZnT + kl;
    const int index_ns_2 = (fc.ns - 2) * s.nZnT + kl;
    const int index_ns_3 = (fc.ns - 3) * s.nZnT + kl;
    m_bsubs_full.bsubs_full(index_ns_1) =
        2.0 * m_bsubs_full.bsubs_full(index_ns_2) -
        m_bsubs_full.bsubs_full(index_ns_3);
  }  // kl
}  // ExtrapolateBSubS

vmecpp::JxBOutFileContents vmecpp::ComputeJxBOutputFileContents(
    const Sizes& s, const FlowControl& fc,
    const VmecInternalResults& vmec_internal_results,
    const BSubSFull& bsubs_full,
    const CovariantBDerivatives& covariant_b_derivatives,
    const bool return_outputs_even_if_not_converged, VmecStatus vmec_status) {
  JxBOutFileContents jxbout;

  jxbout.itheta = RowMatrixXd::Zero(vmec_internal_results.num_full, s.nZnT);
  jxbout.izeta = RowMatrixXd::Zero(vmec_internal_results.num_full, s.nZnT);
  jxbout.bdotk = RowMatrixXd::Zero(vmec_internal_results.num_full, s.nZnT);

  jxbout.amaxfor = VectorXd::Zero(vmec_internal_results.num_full);
  jxbout.aminfor = VectorXd::Zero(vmec_internal_results.num_full);
  jxbout.avforce = VectorXd::Zero(vmec_internal_results.num_full);
  jxbout.pprim = VectorXd::Zero(vmec_internal_results.num_full);
  jxbout.jdotb = VectorXd::Zero(vmec_internal_results.num_full);
  jxbout.bdotb = VectorXd::Zero(vmec_internal_results.num_full);
  jxbout.bdotgradv = VectorXd::Zero(vmec_internal_results.num_full);
  jxbout.jpar2 = VectorXd::Zero(vmec_internal_results.num_full);
  jxbout.jperp2 = VectorXd::Zero(vmec_internal_results.num_full);
  jxbout.phin = VectorXd::Zero(vmec_internal_results.num_full);

  jxbout.phin[fc.ns - 1] = 1.0;

  jxbout.jsupu3 = RowMatrixXd::Zero(vmec_internal_results.num_full, s.nZnT);
  jxbout.jsupv3 = RowMatrixXd::Zero(vmec_internal_results.num_full, s.nZnT);
  jxbout.jsups3 = RowMatrixXd::Zero(vmec_internal_results.num_half, s.nZnT);

  jxbout.bsupu3 = RowMatrixXd::Zero(vmec_internal_results.num_full, s.nZnT);
  jxbout.bsupv3 = RowMatrixXd::Zero(vmec_internal_results.num_full, s.nZnT);

  jxbout.jcrossb = RowMatrixXd::Zero(vmec_internal_results.num_full, s.nZnT);
  jxbout.jxb_gradp = RowMatrixXd::Zero(vmec_internal_results.num_full, s.nZnT);
  jxbout.jdotb_sqrtg =
      RowMatrixXd::Zero(vmec_internal_results.num_full, s.nZnT);

  jxbout.sqrtg3 = RowMatrixXd::Zero(vmec_internal_results.num_full, s.nZnT);

  jxbout.bsubu3 = RowMatrixXd::Zero(vmec_internal_results.num_half, s.nZnT);
  jxbout.bsubv3 = RowMatrixXd::Zero(vmec_internal_results.num_half, s.nZnT);
  jxbout.bsubs3 = RowMatrixXd::Zero(vmec_internal_results.num_full, s.nZnT);

  std::vector<double> pprim(fc.ns, 0.0);

  std::vector<double> sqgb2(s.nZnT, 0.0);

  std::vector<double> kperpu(s.nZnT, 0.0);
  std::vector<double> kperpv(s.nZnT, 0.0);

  std::vector<double> kp2(s.nZnT, 0.0);

  std::vector<double> sqrtg(s.nZnT, 0.0);

  std::vector<double> bsupu1(s.nZnT, 0.0);
  std::vector<double> bsupv1(s.nZnT, 0.0);

  std::vector<double> bsubu1(s.nZnT, 0.0);
  std::vector<double> bsubv1(s.nZnT, 0.0);

  std::vector<double> jxb(s.nZnT, 0.0);

  static constexpr double dnorm1 = 4.0 * M_PI * M_PI;

  for (int jF = 1; jF < fc.ns - 1; ++jF) {
    const int jHi = jF - 1;
    const int jHo = jF;

    // "over-vp"
    // 1/V' on full grid
    // and 4 pi^2 divided out
    const double ovp =
        2.0 /
        (vmec_internal_results.dVdsH[jHo] + vmec_internal_results.dVdsH[jHi]) /
        dnorm1;

    const double tjnorm = ovp * vmec_internal_results.sign_of_jacobian;

    // dp/ds here
    double pprime =
        1.0 / MU_0 *
        (vmec_internal_results.presH[jHo] - vmec_internal_results.presH[jHi]) /
        fc.deltaS;

    const double pprime_ovp = pprime * ovp;
    const double pnorm = 1.0 / (std::abs(pprime_ovp) + DBL_EPSILON);

    double force_residual_max = -DBL_MAX;
    double force_residual_min = DBL_MAX;

    double average_force_residual = 0.0;
    double average_pressure_gradient = 0.0;

    double average_jdotb = 0.0;
    double average_bdotb = 0.0;

    double average_jpar2 = 0.0;
    double average_jperp2 = 0.0;

    for (int kl = 0; kl < s.nZnT; ++kl) {
      const int l = kl % s.nThetaEff;
      const int target_index = jF * s.nZnT + kl;

      // --------

      const double gsqrt_outside =
          vmec_internal_results.gsqrt(jHo * s.nZnT + kl);
      const double gsqrt_inside =
          vmec_internal_results.gsqrt(jHi * s.nZnT + kl);

      const double bsq_outside =
          vmec_internal_results.total_pressure(jHo * s.nZnT + kl) -
          vmec_internal_results.presH[jHo];
      const double bsq_inside =
          vmec_internal_results.total_pressure(jHi * s.nZnT + kl) -
          vmec_internal_results.presH[jHi];

      sqgb2[kl] = gsqrt_outside * bsq_outside + gsqrt_inside * bsq_inside;

      // --------

      const double bsubu_outside =
          vmec_internal_results.bsubu(jHo * s.nZnT + kl);
      const double bsubu_inside =
          vmec_internal_results.bsubu(jHi * s.nZnT + kl);

      const double bsubv_outside =
          vmec_internal_results.bsubv(jHo * s.nZnT + kl);
      const double bsubv_inside =
          vmec_internal_results.bsubv(jHi * s.nZnT + kl);

      kperpu[kl] = 0.5 * (bsubv_outside + bsubv_inside) * pprime / sqgb2[kl];
      kperpv[kl] = 0.5 * (bsubu_outside + bsubu_inside) * pprime / sqgb2[kl];

      // --------

      const double guu_o = vmec_internal_results.guu(jHo * s.nZnT + kl);
      const double guu_i = vmec_internal_results.guu(jHi * s.nZnT + kl);

      const double guv_o = vmec_internal_results.guv(jHo * s.nZnT + kl);
      const double guv_i = vmec_internal_results.guv(jHi * s.nZnT + kl);

      const double gvv_o = vmec_internal_results.gvv(jHo * s.nZnT + kl);
      const double gvv_i = vmec_internal_results.gvv(jHi * s.nZnT + kl);

      const double term_uu = kperpu[kl] * kperpu[kl] * (guu_o + guu_i) / 2.0;
      const double term_uv = kperpu[kl] * kperpv[kl] * (guv_o + guv_i) / 2.0;
      const double term_vv = kperpv[kl] * kperpv[kl] * (gvv_o + gvv_i) / 2.0;

      kp2[kl] = term_uu + 2.0 * term_uv + term_vv;

      // --------

      const double bsubsu = covariant_b_derivatives.bsubsu(jF * s.nZnT + kl);
      const double bsubsv = covariant_b_derivatives.bsubsv(jF * s.nZnT + kl);

      const double bsubus = (bsubu_outside - bsubu_inside) / fc.deltaS;
      const double bsubvs = (bsubv_outside - bsubv_inside) / fc.deltaS;

      jxbout.itheta(target_index) = (bsubsv - bsubvs) / MU_0;
      jxbout.izeta(target_index) = (-bsubsu + bsubus) / MU_0;

      // --------

      sqrtg[kl] = (gsqrt_outside + gsqrt_inside) / 2.0;

      // --------

      const double bsupu_outside =
          vmec_internal_results.bsupu(jHo * s.nZnT + kl);
      const double bsupu_inside =
          vmec_internal_results.bsupu(jHi * s.nZnT + kl);

      const double bsupv_outside =
          vmec_internal_results.bsupv(jHo * s.nZnT + kl);
      const double bsupv_inside =
          vmec_internal_results.bsupv(jHi * s.nZnT + kl);

      bsupu1[kl] =
          (bsupu_outside * gsqrt_outside + bsupu_inside * gsqrt_inside) /
          (2.0 * sqrtg[kl]);
      bsupv1[kl] =
          (bsupv_outside * gsqrt_outside + bsupv_inside * gsqrt_inside) /
          (2.0 * sqrtg[kl]);

      // --------

      bsubu1[kl] = (bsubu_outside + bsubu_inside) / 2.0;
      bsubv1[kl] = (bsubv_outside + bsubv_inside) / 2.0;

      // --------

      jxb[kl] = ovp * (jxbout.itheta(target_index) * bsupv1[kl] -
                       jxbout.izeta(target_index) * bsupu1[kl]);

      // --------

      jxbout.bdotk(target_index) = jxbout.itheta(target_index) * bsubu1[kl] +
                                   jxbout.izeta(target_index) * bsubv1[kl];

      // --------

      const double force_residual = jxb[kl] - pprime_ovp;

      force_residual_max = std::max(force_residual_max, force_residual * pnorm);
      force_residual_min = std::min(force_residual_min, force_residual * pnorm);

      average_force_residual += force_residual * s.wInt[l];
      average_pressure_gradient += pprime_ovp * s.wInt[l];

      // --------

      // Compute <K dot B>, <B sup v> = signgs*phip
      // jpar2 = <j||**2>, jperp2 = <j-perp**2>,
      // with <...> = flux surface average

      average_jdotb += jxbout.bdotk(target_index) * s.wInt[l];
      average_bdotb += sqgb2[kl] * s.wInt[l];

      average_jpar2 += jxbout.bdotk(target_index) * jxbout.bdotk(target_index) *
                       s.wInt[l] / sqgb2[kl];
      average_jperp2 += kp2[kl] * sqrtg[kl] * s.wInt[l];
    }  // kl

    jxbout.amaxfor[jF] = 100.0 * std::min(force_residual_max, 9.999);
    jxbout.aminfor[jF] = 100.0 * std::max(force_residual_min, -9.999);

    jxbout.avforce[jF] = average_force_residual;
    jxbout.pprim[jF] = average_pressure_gradient;

    jxbout.jdotb[jF] = dnorm1 * tjnorm * average_jdotb;
    jxbout.bdotb[jF] = dnorm1 * tjnorm * average_bdotb;

    const double phipH_outside = vmec_internal_results.phipH[jHo];
    const double phipH_inside = vmec_internal_results.phipH[jHi];

    jxbout.bdotgradv[jF] =
        dnorm1 * tjnorm * (phipH_outside + phipH_inside) / 2.0;

    jxbout.jpar2[jF] = dnorm1 * tjnorm * average_jpar2;
    jxbout.jperp2[jF] = dnorm1 * tjnorm * average_jperp2;

    // Some quantities are only computed if VMEC++ actually converged.
    if (vmec_status == VmecStatus::SUCCESSFUL_TERMINATION ||
        return_outputs_even_if_not_converged) {
      // normalized toroidal magnetic flux
      jxbout.phin[jF] = vmec_internal_results.phiF[jF] /
                        vmec_internal_results.phiF[fc.ns - 1];

      for (int kl = 0; kl < s.nZnT; ++kl) {
        const int target_index = jF * s.nZnT + kl;

        // --------

        jxbout.jsupu3(target_index) = ovp * jxbout.itheta(target_index);
        jxbout.jsupv3(target_index) = ovp * jxbout.izeta(target_index);

        // --------

        jxbout.bsupu3(target_index) = bsupu1[kl];
        jxbout.bsupv3(target_index) = bsupv1[kl];

        // --------

        jxbout.jcrossb(target_index) = jxb[kl];
        jxbout.jxb_gradp(target_index) = jxb[kl] - pprime_ovp;
        jxbout.jdotb_sqrtg(target_index) = ovp * jxbout.bdotk(target_index);

        // --------

        jxbout.sqrtg3(target_index) = sqrtg[kl] * ovp;

        // --------

        // bsubu and bsubv remain on the half-grid -> separate loop below
        jxbout.bsubs3(target_index) = bsubs_full.bsubs_full(target_index);
      }  // kl
    }
  }  // jF

  // Some quantities are only computed if VMEC++ actually converged.
  if (vmec_status == VmecStatus::SUCCESSFUL_TERMINATION ||
      return_outputs_even_if_not_converged) {
    // The loop in jxbforce.f90:594 goes over js=2,ns1,
    // which means that the last half-grid point is not touched.
    for (int jH = 0; jH < vmec_internal_results.num_half - 1; ++jH) {
      const double ovp = 1.0 / vmec_internal_results.dVdsH[jH] / dnorm1;

      for (int kl = 0; kl < s.nZnT; ++kl) {
        const int target_index = jH * s.nZnT + kl;

        const double bsubuv = covariant_b_derivatives.bsubuv(target_index);
        const double bsubvu = covariant_b_derivatives.bsubvu(target_index);

        jxbout.jsups3(target_index) = ovp * (bsubuv - bsubvu) / MU_0;

        jxbout.bsubu3(target_index) = vmec_internal_results.bsubu(target_index);
        jxbout.bsubv3(target_index) = vmec_internal_results.bsubv(target_index);
      }  // kl
    }  // jH
  }

  // extrapolate stuff to axis and boundary
  for (int kl = 0; kl < s.nZnT; ++kl) {
    // used to extrapolate towards axis from first two interior full-grid points
    const int index_0 = 0 * s.nZnT + kl;
    const int index_1 = 1 * s.nZnT + kl;
    const int index_2 = 2 * s.nZnT + kl;

    // used to extrapolate towards boundary from last two interior full-grid
    // points
    const int index_ns_1 = (fc.ns - 1) * s.nZnT + kl;
    const int index_ns_2 = (fc.ns - 2) * s.nZnT + kl;
    const int index_ns_3 = (fc.ns - 3) * s.nZnT + kl;

    jxbout.izeta(index_0) = 2.0 * jxbout.izeta(index_1) - jxbout.izeta(index_2);
    jxbout.izeta(index_ns_1) =
        2.0 * jxbout.izeta(index_ns_2) - jxbout.izeta(index_ns_3);
  }  // kl

  jxbout.jdotb[0] = 2.0 * jxbout.jdotb[1] - jxbout.jdotb[2];
  jxbout.jdotb[fc.ns - 1] =
      2.0 * jxbout.jdotb[fc.ns - 2] - jxbout.jdotb[fc.ns - 3];

  jxbout.bdotb[0] = 2.0 * jxbout.bdotb[1] - jxbout.bdotb[2];
  jxbout.bdotb[fc.ns - 1] =
      2.0 * jxbout.bdotb[fc.ns - 2] - jxbout.bdotb[fc.ns - 3];

  jxbout.bdotgradv[0] = 2.0 * jxbout.bdotgradv[1] - jxbout.bdotgradv[2];
  jxbout.bdotgradv[fc.ns - 1] =
      2.0 * jxbout.bdotgradv[fc.ns - 2] - jxbout.bdotgradv[fc.ns - 3];

  // Note that jpar2, jperp2 have been initialized to all-0 in the beginning,
  // so there is not need to set the axis and boundary entries to zero here.

  jxbout.pprim[0] = 2.0 * jxbout.pprim[1] - jxbout.pprim[2];
  jxbout.pprim[fc.ns - 1] =
      2.0 * jxbout.pprim[fc.ns - 2] - jxbout.pprim[fc.ns - 3];

  return jxbout;
}  // ComputeJxBOutputFileContents

vmecpp::MercierStabilityIntermediateQuantities
vmecpp::ComputeIntermediateMercierQuantities(
    const Sizes& s, const FlowControl& fc,
    const VmecInternalResults& vmec_internal_results,
    const JxBOutFileContents& jxbout) {
  // SCALE VP, PHIPS TO REAL UNITS (VOLUME, TOROIDAL FLUX DERIVATIVES)
  // AND PUT GSQRT IN ABS UNITS (SIGNGS MAY BE NEGATIVE)
  // NOTE: VP has (coming into this routine) the sign of the jacobian multiplied
  // out
  //      i.e., vp = signgs*<gsqrt>
  // THE SHEAR TERM MUST BE MULTIPLIED BY THE SIGN OF THE JACOBIAN
  // (OR A BETTER SOLUTION IS TO RETAIN THE JACOBIAN SIGN IN ALL TERMS,
  // INCLUDING
  //  VP, THAT DEPEND EXPLICITLY ON THE JACOBIAN. WE CHOOSE THIS LATTER
  //  METHOD...)
  //
  // COMING INTO THIS ROUTINE, THE JACOBIAN(gsqrt) = 1./(grad-s . grad-theta X
  // grad-zeta) WE CONVERT THIS FROM grad-s to grad-phi DEPENDENCE BY DIVIDING
  // gsqrt by PHIP_real
  //
  // NOTE: WE ARE USING 0 < s < 1 AS THE FLUX VARIABLE, BEING CAREFUL
  // TO KEEP d(phi)/ds == PHIP_real FACTORS WHERE REQUIRED
  // THE V'' TERM IS d2V/d(PHI)**2, PHI IS REAL TOROIDAL FLUX
  //
  // SHEAR = d(iota)/d(phi)   :  FULL MESH
  // VPP   = d(vp)/d(phi)     :  FULL MESH
  // PRESP = d(pres)/d(phi)   :  FULL MESH  (PRES IS REAL PRES*mu0)
  // IP    = d(Itor)/d(phi)   :  FULL MESH
  //
  // ON ENTRY, BDOTJ = Jacobian * J*B  ON THE FULL RADIAL GRID
  //           BSQ = 0.5*|B**2| + p IS ON THE HALF RADIAL GRID

  // REFERENCE: BAUER, BETANCOURT, GARABEDIAN, MHD Equilibrium and Stability of
  // Stellarators We break up the Omega-subs into a positive shear term (Dshear)
  // and a net current term, Dcurr Omega_subw == Dwell Omega-subd == Dgeod
  // (geodesic curvature, Pfirsch-Schlueter term)
  //
  // Include (eventually) Suydam for reference (cylindrical limit)

  MercierStabilityIntermediateQuantities mercier_intermediate;

  mercier_intermediate.s = VectorXd::Zero(fc.ns);
  mercier_intermediate.shear = VectorXd::Zero(fc.ns);
  mercier_intermediate.vpp = VectorXd::Zero(fc.ns);
  mercier_intermediate.d_pressure_d_s = VectorXd::Zero(fc.ns);
  mercier_intermediate.d_toroidal_current_d_s = VectorXd::Zero(fc.ns);
  mercier_intermediate.phip_realH = VectorXd::Zero(fc.ns - 1);
  mercier_intermediate.phip_realF = VectorXd::Zero(fc.ns);
  mercier_intermediate.vp_real = VectorXd::Zero(fc.ns - 1);
  mercier_intermediate.torcur = VectorXd::Zero(fc.ns - 1);

  mercier_intermediate.gsqrt_full = RowMatrixXd::Zero(fc.ns, s.nZnT);
  mercier_intermediate.bdotj = RowMatrixXd::Zero(fc.ns, s.nZnT);
  mercier_intermediate.gpp = RowMatrixXd::Zero(fc.ns, s.nZnT);
  mercier_intermediate.b2 = RowMatrixXd::Zero(fc.ns - 1, s.nZnT);

  mercier_intermediate.tpp = VectorXd::Zero(fc.ns);
  mercier_intermediate.tbb = VectorXd::Zero(fc.ns);
  mercier_intermediate.tjb = VectorXd::Zero(fc.ns);
  mercier_intermediate.tjj = VectorXd::Zero(fc.ns);

  for (int jH = 0; jH < fc.ns - 1; ++jH) {
    // NOTE: phip_real should be > 0 to get the correct physical sign of
    // REAL-space gradients, for example, grad-p, grad-Ip, etc. However, with
    // phip_real defined this way, Mercier will be correct.
    mercier_intermediate.phip_realH[jH] =
        2.0 * M_PI * vmec_internal_results.phipH[jH] *
        vmec_internal_results.sign_of_jacobian;

    // dV/d(PHI) on half mesh
    mercier_intermediate.vp_real[jH] =
        vmec_internal_results.sign_of_jacobian * (4.0 * M_PI * M_PI) *
        vmec_internal_results.dVdsH[jH] / mercier_intermediate.phip_realH[jH];

    // COMPUTE INTEGRATED TOROIDAL CURRENT
    for (int kl = 0; kl < s.nZnT; ++kl) {
      const int idx_kl = jH * s.nZnT + kl;
      const int l = kl % s.nThetaEff;
      mercier_intermediate.torcur[jH] +=
          vmec_internal_results.bsubu(idx_kl) * s.wInt[l];
    }  // kl
    mercier_intermediate.torcur[jH] *=
        vmec_internal_results.sign_of_jacobian * 2.0 * M_PI;
  }  // jH

  // COMPUTE SURFACE AVERAGE VARIABLES ON FULL RADIAL MESH
  for (int jF = 1; jF < fc.ns - 1; ++jF) {
    const int jHi = jF - 1;
    const int jHo = jF;

    mercier_intermediate.phip_realF[jF] =
        (mercier_intermediate.phip_realH[jHo] +
         mercier_intermediate.phip_realH[jHi]) /
        2.0;
    const double denom = mercier_intermediate.phip_realF[jF] * fc.deltaS;

    // d(iota)/d(PHI)
    mercier_intermediate.shear[jF] =
        (vmec_internal_results.iotaH[jHo] - vmec_internal_results.iotaH[jHi]) /
        denom;

    // d(VP)/d(PHI)
    mercier_intermediate.vpp[jF] = (mercier_intermediate.vp_real[jHo] -
                                    mercier_intermediate.vp_real[jHi]) /
                                   denom;

    // d(p)/d(PHI)
    mercier_intermediate.d_pressure_d_s[jF] =
        (vmec_internal_results.presH[jHo] - vmec_internal_results.presH[jHi]) /
        denom;

    // d(Itor)/d(PHI)
    mercier_intermediate.d_toroidal_current_d_s[jF] =
        (mercier_intermediate.torcur[jHo] - mercier_intermediate.torcur[jHi]) /
        denom;

    // -------------------

    // COMPUTE:
    // GPP == |grad-phi|**2 = PHIP**2*|grad-s|**2           (on full mesh)
    // Gsqrt_FULL = JACOBIAN/PHIP == jacobian based on flux (on full mesh)
    for (int kl = 0; kl < s.nZnT; ++kl) {
      const int index_full = jF * s.nZnT + kl;

      const double gsqrt_outside =
          vmec_internal_results.gsqrt(jHo * s.nZnT + kl);
      const double gsqrt_inside =
          vmec_internal_results.gsqrt(jHi * s.nZnT + kl);
      mercier_intermediate.gsqrt_full(index_full) =
          (gsqrt_outside + gsqrt_inside) / 2.0;

      // In educational_VMEC, bdotk is scaled by mu0 in jxbforce() before
      // entering mercier(); here, we do the scaling by MU_0 at this place in
      // order to leave bdotk at the value it has in the jxbout file.
      mercier_intermediate.bdotj(index_full) =
          jxbout.bdotk(index_full) * MU_0 /
          mercier_intermediate.gsqrt_full(index_full);

      mercier_intermediate.gsqrt_full(index_full) /=
          mercier_intermediate.phip_realF[jF];
    }  // kl

    mercier_intermediate.s[jF] = jF * fc.deltaS;
    const double sqrtSF = std::sqrt(mercier_intermediate.s[jF]);

    for (int kl = 0; kl < s.nZnT; ++kl) {
      const int index_full = jF * s.nZnT + kl;

      // dR/dTheta
      const double rtf = vmec_internal_results.ru_e(index_full) +
                         sqrtSF * vmec_internal_results.ru_o(index_full);

      // dZ/dTheta
      const double ztf = vmec_internal_results.zu_e(index_full) +
                         sqrtSF * vmec_internal_results.zu_o(index_full);

      // dR/dZeta
      const double rzf = vmec_internal_results.rv_e(index_full) +
                         sqrtSF * vmec_internal_results.rv_o(index_full);

      // dZ/dZeta
      const double zzf = vmec_internal_results.zv_e(index_full) +
                         sqrtSF * vmec_internal_results.zv_o(index_full);

      // R
      const double r1f = vmec_internal_results.r_e(index_full) +
                         sqrtSF * vmec_internal_results.r_o(index_full);

      // g_uu
      const double gtt = rtf * rtf + ztf * ztf;

      const double gpp_numerator = mercier_intermediate.gsqrt_full(index_full) *
                                   mercier_intermediate.gsqrt_full(index_full);

      // TODO(jons): figure out what this really is
      const double gpp_denominator_ingredient = rtf * zzf - rzf * ztf;
      const double gpp_denominator =
          gtt * r1f * r1f +
          gpp_denominator_ingredient * gpp_denominator_ingredient;
      mercier_intermediate.gpp(index_full) = gpp_numerator / gpp_denominator;
    }  // kl
  }  // jF

  // COMPUTE SURFACE AVERAGES OVER dS/|grad-PHI|**3 => |Jac| du dv /
  // |grad-PHI|**2 WHERE Jac = gsqrt/phip_real
  for (int jH = 0; jH < fc.ns - 1; ++jH) {
    for (int kl = 0; kl < s.nZnT; ++kl) {
      const int index_half = jH * s.nZnT + kl;

      // total pressure = (|B|^2/2 + mu_0 * p)
      // presH = mu_0 * p
      // --> |B|^2 = 2 * (|B|^2/2 + mu_0 * p - mu_0 * p)
      mercier_intermediate.b2(index_half) =
          2.0 * (vmec_internal_results.total_pressure(index_half) -
                 vmec_internal_results.presH[jH]);
    }  // kl
  }  // jH

  const double four_pi_squared = 4.0 * M_PI * M_PI;
  for (int jF = 1; jF < fc.ns - 1; ++jF) {
    const int jHi = jF - 1;
    const int jHo = jF;

    for (int kl = 0; kl < s.nZnT; ++kl) {
      const int l = kl % s.nThetaEff;
      const int index_full = jF * s.nZnT + kl;
      const int index_half_o = jHo * s.nZnT + kl;
      const int index_half_i = jHi * s.nZnT + kl;

      const double b2i = (mercier_intermediate.b2(index_half_o) +
                          mercier_intermediate.b2(index_half_i)) /
                         2.0;

      // <1/B**2>
      const double ob2 = mercier_intermediate.gsqrt_full(index_full) / b2i;
      mercier_intermediate.tpp[jF] += ob2 * s.wInt[l];

      // <b*b/|grad-phi|**3>
      const double ob2_reused = b2i *
                                mercier_intermediate.gsqrt_full(index_full) *
                                mercier_intermediate.gpp(index_full);
      mercier_intermediate.tbb[jF] += ob2_reused * s.wInt[l];

      // <j*b/|grad-phi|**3>
      const double jdotb = mercier_intermediate.bdotj(index_full) *
                           mercier_intermediate.gpp(index_full) *
                           mercier_intermediate.gsqrt_full(index_full);
      mercier_intermediate.tjb[jF] += jdotb * s.wInt[l];

      // <(j*b)2/b**2*|grad-phi|**3>
      const double jdotb_reused =
          jdotb * mercier_intermediate.bdotj(index_full) / b2i;
      mercier_intermediate.tjj[jF] += jdotb_reused * s.wInt[l];
    }  // kl

    mercier_intermediate.tpp[jF] *= four_pi_squared;
    mercier_intermediate.tbb[jF] *= four_pi_squared;
    mercier_intermediate.tjb[jF] *= four_pi_squared;
    mercier_intermediate.tjj[jF] *= four_pi_squared;
  }  // jF

  return mercier_intermediate;
}  // ComputeIntermediateMercierQuantities

vmecpp::MercierFileContents vmecpp::ComputeMercierStability(
    const FlowControl& fc, const VmecInternalResults& vmec_internal_results,
    const MercierStabilityIntermediateQuantities& mercier_intermediate) {
  MercierFileContents mercier;

  mercier.s = VectorXd::Zero(fc.ns);
  mercier.toroidal_flux = VectorXd::Zero(fc.ns);
  mercier.iota = VectorXd::Zero(fc.ns);
  mercier.shear = VectorXd::Zero(fc.ns);
  mercier.d_volume_d_s = VectorXd::Zero(fc.ns);
  mercier.well = VectorXd::Zero(fc.ns);
  mercier.toroidal_current = VectorXd::Zero(fc.ns);
  mercier.d_toroidal_current_d_s = VectorXd::Zero(fc.ns);
  mercier.pressure = VectorXd::Zero(fc.ns);
  mercier.d_pressure_d_s = VectorXd::Zero(fc.ns);

  // -------------------

  mercier.DMerc = VectorXd::Zero(fc.ns);
  mercier.Dshear = VectorXd::Zero(fc.ns);
  mercier.Dwell = VectorXd::Zero(fc.ns);
  mercier.Dcurr = VectorXd::Zero(fc.ns);
  mercier.Dgeod = VectorXd::Zero(fc.ns);

  // first table in Mercier output file
  for (int jF = 1; jF < fc.ns - 1; ++jF) {
    const int jHi = jF - 1;
    const int jHo = jF;

    // S
    mercier.s[jF] = mercier_intermediate.s[jF];

    const double vp_full = (mercier_intermediate.vp_real[jHo] +
                            mercier_intermediate.vp_real[jHi]) /
                           2.0;
    if (vp_full == 0.0) {
      // skip this surface
      continue;
    }

    // PHI
    mercier.toroidal_flux[jF] = 0.0;
    for (int jF_prime = 1; jF_prime <= jF; ++jF_prime) {
      mercier.toroidal_flux[jF] += mercier_intermediate.phip_realF[jF_prime];
    }  // jF_prime
    mercier.toroidal_flux[jF] *= fc.deltaS;

    // IOTA
    mercier.iota[jF] =
        (vmec_internal_results.iotaH[jHo] + vmec_internal_results.iotaH[jHi]) /
        2.0;

    // SHEAR
    mercier.shear[jF] = mercier_intermediate.shear[jF] / vp_full;

    // VP
    mercier.d_volume_d_s[jF] = vp_full;

    // WELL
    mercier.well[jF] =
        -mercier_intermediate.vpp[jF] * vmec_internal_results.sign_of_jacobian;

    // ITOR
    mercier.toroidal_current[jF] =
        (mercier_intermediate.torcur[jHo] + mercier_intermediate.torcur[jHi]) /
        2.0;

    // ITOR'
    mercier.d_toroidal_current_d_s[jF] =
        mercier_intermediate.d_toroidal_current_d_s[jF] / vp_full;

    // PRES
    mercier.pressure[jF] =
        (vmec_internal_results.presH[jHo] + vmec_internal_results.presH[jHi]) /
        2.0;

    // PRES'
    mercier.d_pressure_d_s[jF] =
        mercier_intermediate.d_pressure_d_s[jF] / vp_full;
  }  // jF

  // second table in Mercier output file
  for (int jF = 1; jF < fc.ns - 1; ++jF) {
    // scaling by (2 pi)^2 already done above where those are computed
    const double tpp = mercier_intermediate.tpp[jF];
    const double tjb = mercier_intermediate.tjb[jF];
    const double tbb = mercier_intermediate.tbb[jF];
    const double tjj = mercier_intermediate.tjj[jF];

    mercier.Dshear[jF] =
        mercier_intermediate.shear[jF] * mercier_intermediate.shear[jF] / 4.0;

    mercier.Dcurr[jF] =
        -mercier_intermediate.shear[jF] *
        (tjb - mercier_intermediate.d_toroidal_current_d_s[jF] * tbb);

    mercier.Dwell[jF] = mercier_intermediate.d_pressure_d_s[jF] *
                        (mercier_intermediate.vpp[jF] -
                         mercier_intermediate.d_pressure_d_s[jF] * tpp) *
                        tbb;

    mercier.Dgeod[jF] = tjb * tjb - tbb * tjj;

    mercier.DMerc[jF] = mercier.Dshear[jF] + mercier.Dcurr[jF] +
                        mercier.Dwell[jF] + mercier.Dgeod[jF];
  }  // jF

  // fixup for comparison
  mercier.s[0] = 0.0;
  mercier.s[fc.ns - 1] = 1.0;

  return mercier;
}  // ComputeMercierStability

vmecpp::Threed1FirstTableIntermediate
vmecpp::ComputeIntermediateThreed1FirstTableQuantities(
    const Sizes& s, const FlowControl& fc,
    const VmecInternalResults& vmec_internal_results) {
  Threed1FirstTableIntermediate threed1_first_table_intermediate;

  threed1_first_table_intermediate.tau = RowMatrixXd::Zero(fc.ns - 1, s.nZnT);

  // [ns - 1]
  threed1_first_table_intermediate.beta_vol = VectorXd::Zero(fc.ns - 1);
  threed1_first_table_intermediate.overr = VectorXd::Zero(fc.ns - 1);
  threed1_first_table_intermediate.bvcoH = VectorXd::Zero(fc.ns - 1);
  threed1_first_table_intermediate.bucoH = VectorXd::Zero(fc.ns - 1);

  // [ns]
  threed1_first_table_intermediate.presf = VectorXd::Zero(fc.ns);
  threed1_first_table_intermediate.phipf_loc = VectorXd::Zero(fc.ns);
  threed1_first_table_intermediate.phi1 = VectorXd::Zero(fc.ns);
  threed1_first_table_intermediate.chi1 = VectorXd::Zero(fc.ns);
  threed1_first_table_intermediate.chi = VectorXd::Zero(fc.ns);
  threed1_first_table_intermediate.jcurv = VectorXd::Zero(fc.ns);
  threed1_first_table_intermediate.jcuru = VectorXd::Zero(fc.ns);
  threed1_first_table_intermediate.presgrad = VectorXd::Zero(fc.ns);
  threed1_first_table_intermediate.vpphi = VectorXd::Zero(fc.ns);
  threed1_first_table_intermediate.equif = VectorXd::Zero(fc.ns);
  threed1_first_table_intermediate.bucof = VectorXd::Zero(fc.ns);
  threed1_first_table_intermediate.bvcof = VectorXd::Zero(fc.ns);

  // NOTE:  S=normalized toroidal flux (0 - 1)
  //        U=poloidal angle (0 - 2*pi)
  //        V=geometric toroidal angle (0 - 2*pi)
  //        <RADIAL FORCE> = d(Ipol)/dPHI - IOTA*d(Itor)/dPHI - dp/dPHI *
  //        d(VOL)/dPHI
  //                       = d(VOL)/dPHI*[<JSUPU> - IOTA*<JSUPV> -
  //                       SIGN(JAC)*dp/dPHI] (NORMED TO SUM OF INDIVIDUAL
  //                       TERMS)

  // HALF-MESH VOLUME-AVERAGED BETA
  for (int jH = 0; jH < fc.ns - 1; ++jH) {
    double s2 = 0.0;
    double avg_tau_over_r12 = 0.0;
    for (int kl = 0; kl < s.nZnT; ++kl) {
      const int l = kl % s.nThetaEff;
      const int index_half = jH * s.nZnT + kl;

      threed1_first_table_intermediate.tau(index_half) =
          vmec_internal_results.sign_of_jacobian * s.wInt[l] *
          vmec_internal_results.gsqrt(index_half);

      s2 += vmec_internal_results.total_pressure(index_half) *
            threed1_first_table_intermediate.tau(index_half);

      avg_tau_over_r12 += threed1_first_table_intermediate.tau(index_half) /
                          vmec_internal_results.r12(index_half);
    }  // kl
    s2 /= vmec_internal_results.dVdsH[jH];
    s2 -= vmec_internal_results.presH[jH];

    threed1_first_table_intermediate.beta_vol[jH] =
        vmec_internal_results.presH[jH] / s2;

    threed1_first_table_intermediate.overr[jH] =
        avg_tau_over_r12 / vmec_internal_results.dVdsH[jH];
  }  // jH

  // extrapolate surface-averaged beta profile to magnetic axis
  threed1_first_table_intermediate.beta_axis =
      1.5 * threed1_first_table_intermediate.beta_vol[0] -
      0.5 * threed1_first_table_intermediate.beta_vol[1];

  // interpolate pressure and phip onto full grid
  threed1_first_table_intermediate.presf[0] =
      1.5 * vmec_internal_results.presH[0] -
      0.5 * vmec_internal_results.presH[1];
  threed1_first_table_intermediate.phipf_loc[0] =
      2.0 * M_PI * vmec_internal_results.sign_of_jacobian *
      (1.5 * vmec_internal_results.phipH[0] -
       0.5 * vmec_internal_results.phipH[1]);
  for (int jF = 1; jF < fc.ns - 1; ++jF) {
    const int jHi = jF - 1;
    const int jHo = jF;

    threed1_first_table_intermediate.presf[jF] =
        (vmec_internal_results.presH[jHo] + vmec_internal_results.presH[jHi]) /
        2.0;
    threed1_first_table_intermediate.phipf_loc[jF] =
        2.0 * M_PI * vmec_internal_results.sign_of_jacobian *
        (vmec_internal_results.phipH[jHo] + vmec_internal_results.phipH[jHi]) /
        2.0;
  }  // jF
  threed1_first_table_intermediate.presf[fc.ns - 1] =
      1.5 * vmec_internal_results.presH[fc.ns - 2] -
      0.5 * vmec_internal_results.presH[fc.ns - 3];
  threed1_first_table_intermediate.phipf_loc[fc.ns - 1] =
      2.0 * M_PI * vmec_internal_results.sign_of_jacobian *
      (1.5 * vmec_internal_results.phipH[fc.ns - 2] -
       0.5 * vmec_internal_results.phipH[fc.ns - 3]);

  // integrate flux differentials to get flux profiles
  threed1_first_table_intermediate.phi1[0] = 0.0;
  threed1_first_table_intermediate.chi1[0] = 0.0;
  for (int jF = 1; jF < fc.ns; ++jF) {
    const int jHi = jF - 1;

    threed1_first_table_intermediate.phi1[jF] =
        threed1_first_table_intermediate.phi1[jF - 1] +
        vmec_internal_results.phipH[jHi] * fc.deltaS;
    threed1_first_table_intermediate.chi1[jF] =
        threed1_first_table_intermediate.chi1[jF - 1] +
        (vmec_internal_results.phipH[jHi] * vmec_internal_results.iotaH[jHi]) *
            fc.deltaS;

    threed1_first_table_intermediate.chi[jF] =
        2.0 * M_PI * threed1_first_table_intermediate.chi1[jF];
  }  // jF

  // calc_fbal

  // Compute profiles of enclosed toroidal current and enclosed poloidal current
  // on half-grid.
  threed1_first_table_intermediate.bucoH.resize(fc.ns - 1);
  threed1_first_table_intermediate.bvcoH.resize(fc.ns - 1);
  for (int jH = 0; jH < fc.ns - 1; ++jH) {
    for (int kl = 0; kl < s.nZnT; ++kl) {
      int iHalf = jH * s.nZnT + kl;
      int l = kl % s.nThetaEff;
      threed1_first_table_intermediate.bucoH[jH] +=
          vmec_internal_results.bsubu(iHalf) * s.wInt[l];
      threed1_first_table_intermediate.bvcoH[jH] +=
          vmec_internal_results.bsubv(iHalf) * s.wInt[l];
    }  // kl
  }  // jH

  double signByDeltaS = vmec_internal_results.sign_of_jacobian / fc.deltaS;

  // Compute derivatives on interior full-grid knots
  // and from them, evaluate radial force balance residual.
  for (int jF = 1; jF < fc.ns - 1; ++jF) {
    const int jHi = jF - 1;
    const int jHo = jF;

    // radial derivatives from half-grid to full-grid
    threed1_first_table_intermediate.jcurv[jF] =
        signByDeltaS * (threed1_first_table_intermediate.bucoH[jHo] -
                        threed1_first_table_intermediate.bucoH[jHi]);
    threed1_first_table_intermediate.jcuru[jF] =
        -signByDeltaS * (threed1_first_table_intermediate.bvcoH[jHo] -
                         threed1_first_table_intermediate.bvcoH[jHi]);

    // prescribed pressure gradient from user input
    threed1_first_table_intermediate.presgrad[jF] =
        (vmec_internal_results.presH[jHo] - vmec_internal_results.presH[jHi]) /
        fc.deltaS;

    // interpolate dVds onto full grid
    threed1_first_table_intermediate.vpphi[jF] =
        0.5 *
        (vmec_internal_results.dVdsH[jHo] + vmec_internal_results.dVdsH[jHi]);

    // total resulting radial force-imbalance:
    // <F> = <-j x B + grad(p)>/V'
    threed1_first_table_intermediate.equif[jF] =
        (vmec_internal_results.chipF[jF] *
             threed1_first_table_intermediate.jcurv[jF] -
         vmec_internal_results.phipF[jF] *
             threed1_first_table_intermediate.jcuru[jF]) /
            threed1_first_table_intermediate.vpphi[jF] +
        threed1_first_table_intermediate.presgrad[jF];
  }  // jF

  // NOTE:  jcuru, jcurv on FULL radial mesh coming out of calc_fbal
  //        They are local (surface-averaged) current densities (NOT integrated
  //        in s) jcurX = (dV/ds)/twopi**2 <JsupX>   for X=u,v
  threed1_first_table_intermediate.bucof[0] = 0.0;
  threed1_first_table_intermediate.bvcof[0] =
      1.5 * threed1_first_table_intermediate.bvcoH[0] -
      0.5 * threed1_first_table_intermediate.bvcoH[1];
  for (int jF = 1; jF < fc.ns - 1; ++jF) {
    const int jHi = jF - 1;
    const int jHo = jF;

    threed1_first_table_intermediate.equif[jF] *=
        threed1_first_table_intermediate.vpphi[jF] /
        (std::abs(threed1_first_table_intermediate.jcurv[jF] *
                  vmec_internal_results.chipF[jF]) +
         std::abs(threed1_first_table_intermediate.jcuru[jF] *
                  vmec_internal_results.phipF[jF]) +
         std::abs(threed1_first_table_intermediate.presgrad[jF] *
                  threed1_first_table_intermediate.vpphi[jF]));

    threed1_first_table_intermediate.bucof[jF] =
        (threed1_first_table_intermediate.bucoH[jHo] +
         threed1_first_table_intermediate.bucoH[jHi]) /
        2.0;
    threed1_first_table_intermediate.bvcof[jF] =
        (threed1_first_table_intermediate.bvcoH[jHo] +
         threed1_first_table_intermediate.bvcoH[jHi]) /
        2.0;
  }
  threed1_first_table_intermediate.bucof[fc.ns - 1] =
      1.5 * threed1_first_table_intermediate.bucoH[fc.ns - 2] -
      0.5 * threed1_first_table_intermediate.bucoH[fc.ns - 3];
  threed1_first_table_intermediate.bvcof[fc.ns - 1] =
      1.5 * threed1_first_table_intermediate.bvcoH[fc.ns - 2] -
      0.5 * threed1_first_table_intermediate.bvcoH[fc.ns - 3];

  // extrapolate full-grid quantites to axis and LCFS
  threed1_first_table_intermediate.equif[0] =
      2.0 * threed1_first_table_intermediate.equif[1] -
      threed1_first_table_intermediate.equif[2];
  threed1_first_table_intermediate.equif[fc.ns - 1] =
      2.0 * threed1_first_table_intermediate.equif[fc.ns - 2] -
      threed1_first_table_intermediate.equif[fc.ns - 3];

  threed1_first_table_intermediate.jcurv[0] =
      2.0 * threed1_first_table_intermediate.jcurv[1] -
      threed1_first_table_intermediate.jcurv[2];
  threed1_first_table_intermediate.jcurv[fc.ns - 1] =
      2.0 * threed1_first_table_intermediate.jcurv[fc.ns - 2] -
      threed1_first_table_intermediate.jcurv[fc.ns - 3];

  threed1_first_table_intermediate.jcuru[0] =
      2.0 * threed1_first_table_intermediate.jcuru[1] -
      threed1_first_table_intermediate.jcuru[2];
  threed1_first_table_intermediate.jcuru[fc.ns - 1] =
      2.0 * threed1_first_table_intermediate.jcuru[fc.ns - 2] -
      threed1_first_table_intermediate.jcuru[fc.ns - 3];

  threed1_first_table_intermediate.presgrad[0] =
      2.0 * threed1_first_table_intermediate.presgrad[1] -
      threed1_first_table_intermediate.presgrad[2];
  threed1_first_table_intermediate.presgrad[fc.ns - 1] =
      2.0 * threed1_first_table_intermediate.presgrad[fc.ns - 2] -
      threed1_first_table_intermediate.presgrad[fc.ns - 3];

  threed1_first_table_intermediate.vpphi[0] =
      2.0 * threed1_first_table_intermediate.vpphi[1] -
      threed1_first_table_intermediate.vpphi[2];
  threed1_first_table_intermediate.vpphi[fc.ns - 1] =
      2.0 * threed1_first_table_intermediate.vpphi[fc.ns - 2] -
      threed1_first_table_intermediate.vpphi[fc.ns - 3];

  return threed1_first_table_intermediate;
}  // ComputeIntermediateThreed1FirstTableQuantities

vmecpp::Threed1FirstTable vmecpp::ComputeThreed1FirstTable(
    const FlowControl& fc, const VmecInternalResults& vmec_internal_results,
    const JxBOutFileContents& jxbout,
    const Threed1FirstTableIntermediate& threed1_first_table_intermediate) {
  const double fac = 2.0 * M_PI * vmec_internal_results.sign_of_jacobian;

  Threed1FirstTable threed1_first_table;

  threed1_first_table.s = VectorXd::Zero(fc.ns);
  threed1_first_table.radial_force = VectorXd::Zero(fc.ns);
  threed1_first_table.toroidal_flux = VectorXd::Zero(fc.ns);
  threed1_first_table.iota = VectorXd::Zero(fc.ns);
  threed1_first_table.avg_jsupu = VectorXd::Zero(fc.ns);
  threed1_first_table.avg_jsupv = VectorXd::Zero(fc.ns);
  threed1_first_table.d_volume_d_phi = VectorXd::Zero(fc.ns);
  threed1_first_table.d_pressure_d_phi = VectorXd::Zero(fc.ns);
  threed1_first_table.spectral_width = VectorXd::Zero(fc.ns);
  threed1_first_table.pressure = VectorXd::Zero(fc.ns);
  threed1_first_table.buco_full = VectorXd::Zero(fc.ns);
  threed1_first_table.bvco_full = VectorXd::Zero(fc.ns);
  threed1_first_table.j_dot_b = VectorXd::Zero(fc.ns);
  threed1_first_table.b_dot_b = VectorXd::Zero(fc.ns);

  // NOTE: phipf = phipf_loc/(twopi), phipf_loc ACTUAL (twopi factor) Toroidal
  // flux derivative SPH/JDH (060211): remove twopi factors from <JSUPU,V>
  // (agree with output in JXBOUT file)

  for (int jF = 0; jF < fc.ns; ++jF) {
    // normalized toroidal flux on full-grid
    const double s = jF * fc.deltaS;

    const double vpphi = threed1_first_table_intermediate.vpphi[jF];
    const double phipf_loc = threed1_first_table_intermediate.phipf_loc[jF];

    // dV/ds = dV/dPHI * d(PHI/ds)  (V=actual volume)
    const double cur0 = fac * vpphi * 2.0 * M_PI;

    // S
    threed1_first_table.s[jF] = s;

    // <RADIAL FORCE>
    threed1_first_table.radial_force[jF] =
        threed1_first_table_intermediate.equif[jF];

    // TOROIDAL FLUX
    threed1_first_table.toroidal_flux[jF] =
        fac * threed1_first_table_intermediate.phi1[jF];

    // IOTA
    threed1_first_table.iota[jF] = vmec_internal_results.iotaF[jF];

    // <JSUPU>
    threed1_first_table.avg_jsupu[jF] =
        threed1_first_table_intermediate.jcuru[jF] / vpphi / MU_0;

    // <JSUPV>
    threed1_first_table.avg_jsupv[jF] =
        threed1_first_table_intermediate.jcurv[jF] / vpphi / MU_0;

    // d(VOL)/d(PHI)
    threed1_first_table.d_volume_d_phi[jF] = cur0 / phipf_loc;

    // d(PRES)/d(PHI)
    threed1_first_table.d_pressure_d_phi[jF] =
        threed1_first_table_intermediate.presgrad[jF] / phipf_loc / MU_0;

    // <M>
    threed1_first_table.spectral_width[jF] =
        vmec_internal_results.spectral_width[jF];

    // PRESF
    threed1_first_table.pressure[jF] =
        threed1_first_table_intermediate.presf[jF] / MU_0;

    // <BSUBU>
    threed1_first_table.buco_full[jF] =
        threed1_first_table_intermediate.bucof[jF];

    // <BSUBV>
    threed1_first_table.bvco_full[jF] =
        threed1_first_table_intermediate.bvcof[jF];

    // <J.B>
    threed1_first_table.j_dot_b[jF] = jxbout.jdotb[jF];

    // <B.B>
    threed1_first_table.b_dot_b[jF] = jxbout.bdotb[jF];
  }  // jF

  return threed1_first_table;
}  // ComputeThreed1FirstTable

vmecpp::Threed1GeometricAndMagneticQuantitiesIntermediate
vmecpp::ComputeIntermediateThreed1GeometricMagneticQuantities(
    const Sizes& s, const FlowControl& fc,
    const HandoverStorage& handover_storage,
    const VmecInternalResults& vmec_internal_results,
    const JxBOutFileContents& jxbout,
    const Threed1FirstTableIntermediate& threed1_first_table_intermediate,
    VacuumPressureState vacuum_pressure_state) {
  Threed1GeometricAndMagneticQuantitiesIntermediate intermediate;

  // Calculate mean (toroidally averaged) poloidal cross section area & toroidal
  // flux.
  intermediate.anorm = 2.0 * M_PI * fc.deltaS;
  intermediate.vnorm = 2.0 * M_PI * intermediate.anorm;

  // Calculate poloidal circumference and normal surface area and aspect ratio
  // Normal is | dr/du X dr/dv | = SQRT [R**2 guu + (RuZv - RvZu)**2]
  intermediate.surf_area = VectorXd::Zero(s.nZnT);
  intermediate.circumference_sum = 0.0;
  for (int kl = 0; kl < s.nZnT; ++kl) {
    const int l = kl % s.nThetaEff;
    const int lcfs_kl = (fc.ns - 1) * s.nZnT + kl;

    const double ru0 = vmec_internal_results.ruFull(lcfs_kl);
    const double zu0 = vmec_internal_results.zuFull(lcfs_kl);
    const double guu_1u = ru0 * ru0 + zu0 * zu0;

    // compute circumference
    intermediate.circumference_sum += std::sqrt(guu_1u) * s.wInt[l];

    const double r =
        vmec_internal_results.r_e(lcfs_kl) + vmec_internal_results.r_o(lcfs_kl);
    const double rv = vmec_internal_results.rv_e(lcfs_kl) +
                      vmec_internal_results.rv_o(lcfs_kl);
    const double zv = vmec_internal_results.zv_e(lcfs_kl) +
                      vmec_internal_results.zv_o(lcfs_kl);

    // TODO(jons): figure out what this really is
    const double rv_zu_minus_zv_ru = rv * zu0 - zv * ru0;

    intermediate.surf_area[kl] =
        s.wInt[l] *
        std::sqrt(r * r * guu_1u + rv_zu_minus_zv_ru * rv_zu_minus_zv_ru);
  }  // kl

  // OUTPUT BETAS, INDUCTANCES, SAFETY FACTORS, ETC.
  // (EXTRACTED FROM FQ-CODE, 9-10-92)
  //
  // b poloidals (cylindrical estimates)

  // TODO(jons): rcenin (not used anywhere?)
  // TODO(jons): aminr2in (not used anywhere?)
  // TODO(jons): bminz2in (not used anywhere?)
  // TODO(jons): bminz2 (not used anywhere?)

  // cylindrical estimates for beta poloidal
  double sump_sum = 0.0;
  for (int jH = 0; jH < fc.ns - 1; ++jH) {
    sump_sum +=
        vmec_internal_results.dVdsH[jH] * vmec_internal_results.presH[jH];
  }  // jH
  intermediate.sump = intermediate.vnorm * sump_sum;

  // delphid_exact = Integral[ (Bvac - B) * dSphi ]
  // rshaf [= RT in Eq.(12), Phys Fluids B 5 (1993) 3119]
  //
  // Note: tau = |gsqrt|*wint
  intermediate.btor_vac = VectorXd::Zero(s.nZnT);
  intermediate.btor1 = VectorXd::Zero(s.nZnT);
  intermediate.dbtor = VectorXd::Zero(s.nZnT);
  intermediate.phat = VectorXd::Zero(s.nZnT);

  // Eq. 20 in Shafranov
  intermediate.delphid_exact = 0.0;
  intermediate.musubi = 0.0;
  intermediate.rshaf1 = 0.0;
  intermediate.rshaf2 = 0.0;
  for (int jH = 0; jH < fc.ns - 1; ++jH) {
    for (int kl = 0; kl < s.nZnT; ++kl) {
      const int index_half = jH * s.nZnT + kl;

      // TODO(jons): assumes B_tor ~ 1/R ???
      intermediate.btor_vac[kl] =
          handover_storage.rBtor / vmec_internal_results.r12(index_half);

      intermediate.delphid_exact +=
          (intermediate.btor_vac[kl] / vmec_internal_results.r12(index_half) -
           vmec_internal_results.bsupv(index_half)) *
          threed1_first_table_intermediate.tau(index_half);

      intermediate.btor1[kl] = vmec_internal_results.r12(index_half) *
                               vmec_internal_results.bsupv(index_half);

      intermediate.dbtor[kl] =
          intermediate.btor1[kl] * intermediate.btor1[kl] -
          intermediate.btor_vac[kl] * intermediate.btor_vac[kl];

      intermediate.musubi -= intermediate.dbtor[kl] *
                             threed1_first_table_intermediate.tau(index_half);

      // phat is temporarily re-used for something else...
      intermediate.phat[kl] =
          vmec_internal_results.total_pressure(index_half) -
          0.5 * intermediate.btor_vac[kl] * intermediate.btor_vac[kl];
      intermediate.phat[kl] -= intermediate.dbtor[kl];
      intermediate.phat[kl] *= threed1_first_table_intermediate.tau(index_half);

      intermediate.rshaf1 += intermediate.phat[kl];
      intermediate.rshaf2 +=
          intermediate.phat[kl] / vmec_internal_results.r12(index_half);
    }  // kl
  }  // jH
  intermediate.delphid_exact *= intermediate.anorm;
  intermediate.rshaf = intermediate.rshaf1 / intermediate.rshaf2;

  // TODO(jons): could also use threed1_first_table_intermediate.bvcof[0]
  // directly...
  intermediate.fpsi0 = 1.5 * threed1_first_table_intermediate.bvcoH[0] -
                       0.5 * threed1_first_table_intermediate.bvcoH[1];

  intermediate.redge = VectorXd::Zero(s.nZnT);
  for (int kl = 0; kl < s.nZnT; ++kl) {
    const int lcfs_kl = (fc.ns - 1) * s.nZnT + kl;
    intermediate.redge[kl] =
        vmec_internal_results.r_e(lcfs_kl) + vmec_internal_results.r_o(lcfs_kl);
  }  // kl
  if (fc.lfreeb && vacuum_pressure_state == VacuumPressureState::kActive) {
    for (int k = 0; k < s.nZeta; ++k) {
      for (int l = 0; l < s.nThetaEff; ++l) {
        // FIXME(eguiraud) slow loop for nestor
        int idx_kl = k * s.nThetaEff + l;
        int idx_lk = l * s.nZeta + k;

        const double bsubvvac_r =
            handover_storage.bSubVVac / intermediate.redge[idx_kl];
        intermediate.phat[idx_kl] =
            handover_storage.vacuum_magnetic_pressure[idx_lk] -
            0.5 * bsubvvac_r * bsubvvac_r;
      }
    }
  } else {
    for (int kl = 0; kl < s.nZnT; ++kl) {
      const int last_half = (fc.ns - 2) * s.nZnT + kl;
      const int last1_half = (fc.ns - 3) * s.nZnT + kl;
      const double bsq_at_lcfs =
          1.5 * vmec_internal_results.total_pressure(last_half) -
          0.5 * vmec_internal_results.total_pressure(last1_half);
      const double btor_edge = handover_storage.rBtor / intermediate.redge[kl];
      intermediate.phat[kl] = bsq_at_lcfs - 0.5 * btor_edge * btor_edge;
    }  // kl
  }

  double bsq_tau_sum = 0.0;
  double btor_sum = 0.0;
  double p2_sum = 0.0;
  for (int jH = 0; jH < fc.ns - 1; ++jH) {
    for (int kl = 0; kl < s.nZnT; ++kl) {
      const int index_half = jH * s.nZnT + kl;

      const double tau = threed1_first_table_intermediate.tau(index_half);
      bsq_tau_sum += vmec_internal_results.total_pressure(index_half) * tau;

      const double r_bsupv = vmec_internal_results.r12(index_half) *
                             vmec_internal_results.bsupv(index_half);
      btor_sum += tau * r_bsupv * r_bsupv;
    }  // kl

    const double pressure = vmec_internal_results.presH[jH];
    p2_sum += pressure * pressure * vmec_internal_results.dVdsH[jH];
  }  // jH

  intermediate.sumbtot =
      2.0 * (intermediate.vnorm * bsq_tau_sum - intermediate.sump);
  intermediate.sumbtor = intermediate.vnorm * btor_sum;
  intermediate.sumbpol = intermediate.sumbtot - intermediate.sumbtor;

  intermediate.sump20 = 2.0 * intermediate.sump;
  intermediate.sump2 = intermediate.vnorm * p2_sum;

  intermediate.jPS2.resize(fc.ns);
  intermediate.jpar_perp_sum = 0.0;
  intermediate.jparPS_perp_sum = 0.0;
  intermediate.s2 = 0.0;
  for (int jF = 1; jF < fc.ns - 1; ++jF) {
    const int jHi = jF - 1;
    const int jHo = jF;

    intermediate.jPS2[jF] = jxbout.jpar2[jF] - jxbout.jdotb[jF] *
                                                   jxbout.jdotb[jF] /
                                                   jxbout.bdotb[jF];

    // The factor of 1/2 from radial interpolation/averaging
    // is left out, since it cancels in numerator and denominator.
    // Premature optimization in Fortran VMEC, but leave it as-is for comparison
    // (for now).
    const double two_dVds_full =
        vmec_internal_results.dVdsH[jHo] + vmec_internal_results.dVdsH[jHi];
    intermediate.jpar_perp_sum += jxbout.jpar2[jF] * two_dVds_full;
    intermediate.jparPS_perp_sum += intermediate.jPS2[jF] * two_dVds_full;
    intermediate.s2 += jxbout.jperp2[jF] * two_dVds_full;
  }  // jH

  // TODO(jons): figure out what fac is and assign a better name
  intermediate.fac =
      2.0 * M_PI * fc.deltaS * vmec_internal_results.sign_of_jacobian;
  intermediate.r3v = VectorXd::Zero(fc.ns - 1);
  for (int jH = 0; jH < fc.ns - 1; ++jH) {
    intermediate.r3v[jH] = intermediate.fac * vmec_internal_results.phipH[jH] *
                           vmec_internal_results.iotaH[jH];
  }  // jH

  return intermediate;
}  // ComputeIntermediateThreed1GeometricMagneticQuantities

vmecpp::Threed1GeometricAndMagneticQuantities
vmecpp::ComputeThreed1GeometricMagneticQuantities(
    const Sizes& s, const FlowControl& fc,
    const HandoverStorage& handover_storage,
    const VmecInternalResults& vmec_internal_results,
    const JxBOutFileContents& jxbout,
    const Threed1FirstTableIntermediate& threed1_first_table_intermediate,
    const Threed1GeometricAndMagneticQuantitiesIntermediate& intermediate) {
  Threed1GeometricAndMagneticQuantities result;

  double toroidal_flux_sum = 0.0;
  for (int jH = 0; jH < fc.ns - 1; ++jH) {
    for (int kl = 0; kl < s.nZnT; ++kl) {
      const int index_half = jH * s.nZnT + kl;

      toroidal_flux_sum += vmec_internal_results.bsupv(index_half) *
                           threed1_first_table_intermediate.tau(index_half);
    }  // kl
  }  // jH
  result.toroidal_flux = intermediate.anorm * toroidal_flux_sum;

  result.circum_p = 2.0 * M_PI * intermediate.circumference_sum;

  double surface_area_sum = 0.0;
  for (int kl = 0; kl < s.nZnT; ++kl) {
    surface_area_sum += intermediate.surf_area[kl];
  }  // kl
  result.surf_area_p = 4.0 * M_PI * M_PI * surface_area_sum;

  double cross_section_area_sum = 0.0;
  double volume_sum = 0.0;
  for (int kl = 0; kl < s.nZnT; ++kl) {
    const int l = kl % s.nThetaEff;
    const int lcfs_kl = (fc.ns - 1) * s.nZnT + kl;

    const double rb =
        vmec_internal_results.r_e(lcfs_kl) + vmec_internal_results.r_o(lcfs_kl);
    const double zub = vmec_internal_results.zu_e(lcfs_kl) +
                       vmec_internal_results.zu_o(lcfs_kl);
    const double t1 = rb * zub * s.wInt[l];

    cross_section_area_sum += t1;
    volume_sum += rb * t1;
  }  // kl
  result.cross_area_p = 2.0 * M_PI * std::abs(cross_section_area_sum);
  result.volume_p = 2.0 * M_PI * M_PI * std::abs(volume_sum);

  result.Rmajor_p = result.volume_p / (2.0 * M_PI * result.cross_area_p);
  result.Aminor_p = std::sqrt(result.cross_area_p / M_PI);

  result.aspect = result.Rmajor_p / result.Aminor_p;

  // Also, estimate mean elongation of plasma from the following relations
  // for an axisymmetric torus with elliptical cross section and semi-axes
  // a and a * kappa (kappa >= 1)
  //
  // surf_area _p = 2*pi*R * 2*pi*a ctwiddle(kappa_p)
  // volume_p    = 2*pi*R * pi*a ** 2 * kappa_p
  // cross_area _p =   pi*a ** 2 * kappa_p
  //
  // The cirumference of an ellipse of semi-axes a and a * kappa_p is
  //    2 * pi * a ctwiddle(kappa_p)
  // The exact form for ctwiddle is 4 E(1 - kappa_p^2) / (2 pi), where
  //  E is the complete elliptic integral of the second kind
  // (with parameter argument m, not modulus argument k)
  //
  // The coding below implements an approximate inverse of the function
  // d(kappa) = ctwiddle(kappa) / sqrt(kappa)
  // The approximate inverse is
  //    kappa = 1 + (pi^2/8) * (d^2+sqrt(d^4-1)-1)
  // Note that the variable aminor_p, for an elliptic cross section,
  // would be a * sqrt(kappa)
  const double d_of_kappa =
      result.surf_area_p * result.Aminor_p / (2 * result.volume_p);
  const double d_of_kappa_sqared = d_of_kappa * d_of_kappa;
  result.kappa_p =
      1.0 +
      (M_PI * M_PI / 8.0) *
          (d_of_kappa_sqared +
           std::sqrt(std::abs(d_of_kappa_sqared * d_of_kappa_sqared - 1.0)) -
           1.0);

  RadialExtent radial_extent = handover_storage.GetRadialExtent();
  result.rcen = (radial_extent.r_outer + radial_extent.r_inner) / 2.0;

  // volume-averaged minor radius
  const GeometricOffset& geometric_offset =
      handover_storage.GetGeometricOffset();
  result.aminr1 = std::sqrt(2.0 * result.volume_p /
                            (4.0 * M_PI * M_PI * geometric_offset.r_00));

  result.pavg = intermediate.sump / result.volume_p;
  result.factor = result.pavg * 2;

  result.b0 = intermediate.fpsi0 / geometric_offset.r_00;

  result.rmax_surf = 0.0;
  result.rmin_surf = DBL_MAX;
  result.zmax_surf = 0.0;
  for (int kl = 0; kl < s.nZnT; ++kl) {
    const int lcfs_kl = (fc.ns - 1) * s.nZnT + kl;
    const double r =
        vmec_internal_results.r_e(lcfs_kl) + vmec_internal_results.r_o(lcfs_kl);
    const double z =
        vmec_internal_results.z_e(lcfs_kl) + vmec_internal_results.z_o(lcfs_kl);
    result.rmax_surf = std::max(result.rmax_surf, r);
    result.rmin_surf = std::min(result.rmin_surf, r);
    result.zmax_surf = std::max(result.zmax_surf, z);
  }  // kl

  result.bmin = RowMatrixXd::Ones(fc.ns - 1, s.nThetaReduced) * DBL_MAX;
  result.bmax = RowMatrixXd::Zero(fc.ns - 1, s.nThetaReduced);

  for (int jH = 0; jH < fc.ns - 1; ++jH) {
    for (int k = 0; k < s.nZeta; ++k) {
      for (int l = 0; l < s.nThetaReduced; ++l) {
        const int kl = k * s.nThetaReduced + l;
        const int index_half = jH * s.nZnT + kl;

        const double mod_b =
            std::sqrt(2.0 * (vmec_internal_results.total_pressure(index_half) -
                             vmec_internal_results.presH[jH]));
        result.bmax(jH * s.nThetaReduced + l) =
            std::max(result.bmax(jH * s.nThetaEff + l), mod_b);
        result.bmin(jH * s.nThetaReduced + l) =
            std::min(result.bmin(jH * s.nThetaEff + l), mod_b);
      }  // k
    }  // l
  }  // jH

  // Compute Waist thickness and height in \f$\varphi = 0, \pi\f$ symmetry
  // planes.
  int symmetry_planes_count = 1;
  if (s.ntor > 0) {
    symmetry_planes_count = 2;
  }
  result.waist = VectorXd::Zero(symmetry_planes_count);
  result.height = VectorXd::Zero(symmetry_planes_count);

  int symmetry_plane_index = 0;
  for (int k = 0; k < s.nZeta / 2 + 1; ++k) {
    if (k != 0 && k != s.nZeta / 2) {
      continue;
    }

    const int index_outboard = ((fc.ns - 1) * s.nZeta + k) * s.nThetaEff + 0;
    const double r_outboard = vmec_internal_results.r_e(index_outboard) +
                              vmec_internal_results.r_o(index_outboard);

    const int index_inboard =
        ((fc.ns - 1) * s.nZeta + k) * s.nThetaReduced + (s.nThetaReduced - 1);
    const double r_inboard = vmec_internal_results.r_e(index_inboard) +
                             vmec_internal_results.r_o(index_inboard);

    result.waist[symmetry_plane_index] = r_outboard - r_inboard;

    result.height[symmetry_plane_index] = 0.0;
    for (int l = 0; l < s.nThetaEff; ++l) {
      const int index_zeta = ((fc.ns - 1) * s.nZeta + k) * s.nThetaEff + l;
      const double z = vmec_internal_results.z_e(index_zeta) +
                       vmec_internal_results.z_o(index_zeta);
      result.height[symmetry_plane_index] =
          std::max(result.height[symmetry_plane_index], z);
    }  // l
    result.height[symmetry_plane_index] *= 2.0;

    symmetry_plane_index++;
  }  // k

  result.betapol = 2.0 * intermediate.sump / intermediate.sumbpol;
  result.betatot = intermediate.sump20 / intermediate.sumbtot;
  result.betator = intermediate.sump20 / intermediate.sumbtor;
  result.VolAvgB = std::sqrt(std::abs(intermediate.sumbtot / result.volume_p));

  // TODO(jons): which ion is assumed here ?
  result.IonLarmor = 3.2e-3 / result.VolAvgB;

  if (intermediate.s2 != 0.0) {
    result.jpar_perp = intermediate.jpar_perp_sum / intermediate.s2;
    result.jparPS_perp = intermediate.jparPS_perp_sum / intermediate.s2;
  } else {
    result.jpar_perp = intermediate.jpar_perp_sum;
    result.jparPS_perp = intermediate.jparPS_perp_sum;
  }

  result.toroidal_current = handover_storage.cTor / MU_0;

  result.rbtor = handover_storage.rBtor;
  result.rbtor0 = handover_storage.rBtor0;

  result.psi = VectorXd::Zero(fc.ns);
  for (int jF = 1; jF < fc.ns; ++jF) {
    const int jFi = jF - 1;
    const int jHi = jF - 1;
    result.psi[jF] = result.psi[jFi] + intermediate.r3v[jHi];
  }  // jF

  result.loc_jpar_perp = VectorXd::Zero(fc.ns);
  result.loc_jparPS_perp = VectorXd::Zero(fc.ns);
  for (int jF = 1; jF < fc.ns; ++jF) {
    double jperp2 = DBL_EPSILON;
    if (jxbout.jperp2[jF] != 0.0) {
      // TODO(jons): Actually, in-place overwrite within Fortran VMEC.
      // -> need to do in-place overwrite for follow-up quanties?
      jperp2 = jxbout.jperp2[jF];
    }

    result.loc_jpar_perp[jF] = jxbout.jpar2[jF] / jperp2;
    if (jF < fc.ns - 1) {
      result.loc_jparPS_perp[jF] = intermediate.jPS2[jF] / jperp2;
    } else {
      result.loc_jparPS_perp[jF] = 0.0;
    }
  }  // jF

  result.ygeo = VectorXd::Zero(2 * fc.ns);
  result.yinden = VectorXd::Zero(2 * fc.ns);
  result.yellip = VectorXd::Zero(2 * fc.ns);
  result.ytrian = VectorXd::Zero(2 * fc.ns);
  result.yshift = VectorXd::Zero(2 * fc.ns);
  for (int nplanes = 0; nplanes < 2; ++nplanes) {
    // phi = 0 deg -> first symmetry plane per toroidal module
    int k = 0;
    if (nplanes > 0) {
      if (s.nZeta == 1) {
        break;
      }
      // phi = 180 deg / nfp -> second symmetry plane per toroidal module
      k = s.nZeta / 2;
    }

    // omit magnetic axis at jF = 0
    for (int jF = 1; jF < fc.ns; ++jF) {
      double minimum_z = DBL_MAX;
      double maximum_z = -DBL_MAX;
      // TODO(jons): rename to ..._r
      double minimum_x = DBL_MAX;
      // TODO(jons): rename to ..._r
      double maximum_x = -DBL_MAX;

      double rzmax = 0.0;

      // TODO(jons): these seem to not be used anywhere in Fortran VMEC?
      // double rzmin = 0.0;
      // double zxmax = 0.0;
      // double zxmin = 0.0;

      // Theta = 0 to pi in upper half of X-Z plane
      // TODO(jons): why second loop over toroidal offset ?
      for (int icount = 0; icount < 2; ++icount) {
        int k1 = k;
        int t1 = 1;
        if (icount == 1) {
          // (twopi-v), reflected plane
          k1 = (s.nZeta - k) % s.nZeta;
          t1 = -1;
        }

        for (int l = 0; l < s.nThetaReduced; ++l) {
          const int l_off = (jF * s.nZeta + k1) * s.nThetaEff + l;

          const double yr1u = vmec_internal_results.r_e(l_off) +
                              vmec_internal_results.sqrtSF[jF] *
                                  vmec_internal_results.r_o(l_off);
          const double yz1u = t1 * (vmec_internal_results.z_e(l_off) +
                                    vmec_internal_results.sqrtSF[jF] *
                                        vmec_internal_results.z_o(l_off));

          if (yz1u >= maximum_z) {
            maximum_z = std::abs(yz1u);
            rzmax = yr1u;
          } else if (yz1u <= minimum_z) {
            minimum_z = yz1u;

            // TODO(jons): these seem to not be used anywhere in Fortran VMEC?
            // rzmin = yr1u;
          }

          if (yr1u >= maximum_x) {
            maximum_x = yr1u;

            // TODO(jons): these seem to not be used anywhere  in Fortran VMEC?
            // zxmax = yz1u;
          } else if (yr1u <= minimum_x) {
            minimum_x = yr1u;

            // TODO(jons): these seem to not be used anywhere in Fortran VMEC?
            // zxmin = yz1u;
          }
        }  // l
      }  // icount

      // theta=180 deg
      const int l_pi = (jF * s.nZeta + k) * s.nThetaEff + (s.nThetaReduced - 1);
      double xmida =
          vmec_internal_results.r_e(l_pi) +
          vmec_internal_results.sqrtSF[jF] * vmec_internal_results.r_o(l_pi);

      // theta=0
      const int l_t = (jF * s.nZeta + k) * s.nThetaEff + 0;
      double xmidb =
          vmec_internal_results.r_e(l_t) +
          vmec_internal_results.sqrtSF[jF] * vmec_internal_results.r_o(l_t);

      // Geometric major radius
      const double rgeo = (xmidb + xmida) / 2.0;

      result.ygeo[nplanes * fc.ns + jF] = (xmidb - xmida) / 2.0;

      result.yinden[nplanes * fc.ns + jF] =
          (xmida - minimum_x) / (maximum_x - minimum_x);

      result.yellip[nplanes * fc.ns + jF] =
          (maximum_z - minimum_z) / (maximum_x - minimum_x);

      result.ytrian[nplanes * fc.ns + jF] =
          (rgeo - rzmax) / (maximum_x - minimum_x);

      const double r_axis = vmec_internal_results.r_e(k * s.nThetaEff + 0);
      result.yshift[nplanes * fc.ns + jF] =
          (r_axis - rgeo) / (maximum_x - minimum_x);
    }  // jF
  }  // nplanes

  return result;
}  // ComputeThreed1GeometricMagneticQuantities

vmecpp::Threed1Volumetrics vmecpp::ComputeThreed1Volumetrics(
    const Threed1GeometricAndMagneticQuantitiesIntermediate&
        threed1_geometric_magnetic_intermediate,
    const Threed1GeometricAndMagneticQuantities& threed1_geomag) {
  const double fac = 0.5 / MU_0;

  Threed1Volumetrics result;

  // pressure
  result.int_p = threed1_geometric_magnetic_intermediate.sump / MU_0;
  result.avg_p = threed1_geomag.pavg / MU_0;

  // bpol**2 /(2 mu0)
  result.int_bpol = fac * threed1_geometric_magnetic_intermediate.sumbpol;
  result.avg_bpol = fac * threed1_geometric_magnetic_intermediate.sumbpol /
                    threed1_geomag.volume_p;

  // btor**2 /(2 mu0)
  result.int_btor = fac * threed1_geometric_magnetic_intermediate.sumbtor;
  result.avg_btor = fac * threed1_geometric_magnetic_intermediate.sumbtor /
                    threed1_geomag.volume_p;

  // b**2 /(2 mu0)
  result.int_modb = fac * threed1_geometric_magnetic_intermediate.sumbtot;
  result.avg_modb = fac * threed1_geometric_magnetic_intermediate.sumbtot /
                    threed1_geomag.volume_p;

  // EKIN (3/2p)
  result.int_ekin = 1.5 * threed1_geometric_magnetic_intermediate.sump / MU_0;
  result.avg_ekin = 1.5 * threed1_geomag.pavg / MU_0;

  return result;
}  // ComputeThreed1Volumetrics

vmecpp::Threed1AxisGeometry vmecpp::ComputeThreed1AxisGeometry(
    const Sizes& s, const FourierBasisFastPoloidal& fourier_basis,
    const VmecInternalResults& vmec_internal_results) {
  Threed1AxisGeometry result;

  result.raxis_symm = VectorXd::Zero(s.ntor + 1);
  result.zaxis_symm = VectorXd::Zero(s.ntor + 1);
  if (s.lasym) {
    result.raxis_asym = VectorXd::Zero(s.ntor + 1);
    result.zaxis_asym = VectorXd::Zero(s.ntor + 1);
  }

  // magnetic axis is at radial index 0
  const int jF = 0;
  const int m = 0;
  for (int n = 0; n <= s.ntor; ++n) {
    const double basis_scaling =
        fourier_basis.mscale[0] * fourier_basis.nscale[n];

    double tz = basis_scaling;
    if (!s.lthreed) {
      tz = 0.0;
    }

    const int index = (jF * (s.ntor + 1) + n) * s.mpol + m;

    result.raxis_symm[n] = basis_scaling * vmec_internal_results.rmncc(index);
    if (s.lthreed) {
      result.zaxis_symm[n] = -tz * vmec_internal_results.zmncs(index);
    }
    if (s.lasym) {
      result.zaxis_asym[n] = basis_scaling * vmec_internal_results.zmncc(index);
      if (s.lthreed) {
        result.raxis_asym[n] = -tz * vmec_internal_results.rmncs(index);
      }
    }
  }  // n

  return result;
}  // ComputeThreed1AxisGeometry

vmecpp::Threed1Betas vmecpp::ComputeThreed1Betas(
    const HandoverStorage& handover_storage,
    const Threed1FirstTableIntermediate& threed1_first_table_intermediate,
    const Threed1GeometricAndMagneticQuantitiesIntermediate&
        threed1_geomag_intermediate,
    const Threed1GeometricAndMagneticQuantities& threed1_geomag) {
  Threed1Betas result;

  result.betatot = threed1_geomag.betatot;
  result.betapol = threed1_geomag.betapol;
  result.betator = threed1_geomag.betator;

  // TODO(jons): should this maybe be bsubvvac ?
  result.rbtor = handover_storage.rBtor;
  result.betaxis = threed1_first_table_intermediate.beta_axis;
  result.betstr =
      2.0 *
      std::sqrt(threed1_geomag_intermediate.sump2 / threed1_geomag.volume_p) /
      (threed1_geomag_intermediate.sumbtot / threed1_geomag.volume_p);

  return result;
}  // ComputeThreed1Betas

vmecpp::Threed1ShafranovIntegrals vmecpp::ComputeThreed1ShafranovIntegrals(
    const Sizes& s, const FlowControl& fc,
    const HandoverStorage& handover_storage,
    const VmecInternalResults& vmec_internal_results,
    const Threed1GeometricAndMagneticQuantitiesIntermediate&
        threed1_geometric_magnetic_intermediate,
    const Threed1GeometricAndMagneticQuantities& threed1_geomag,
    VacuumPressureState vacuum_pressure_state) {
  Threed1ShafranovIntegrals result;

  // Shafranov surface integrals s1,s2
  // Plasma Physics vol 13, pp 757-762 (1971)
  // Also, s3 = .5*S3, defined in Lao, Nucl. Fusion 25, p.1421 (1985)
  // Note: if ctor = 0, use Int(Bsupu*Bsubu dV) for ctor*ctor/R
  // Phys. Fluids B, Vol 5 (1993) p 3121, Eq. 9a-9d
  std::vector<double> bpol2vac(s.nZnT, 0.0);
  if (fc.lfreeb &&
      vacuum_pressure_state == vmecpp::VacuumPressureState::kActive) {
    for (int l = 0; l < s.nThetaEff; ++l) {
      for (int k = 0; k < s.nZeta; ++k) {
        // FIXME(eguiraud) slow loop for nestor
        int idx_lk = l * s.nZeta + k;
        int idx_kl = k * s.nThetaEff + l;
        const double bphiv = handover_storage.vacuum_b_phi[idx_lk];
        bpol2vac[idx_kl] =
            2.0 * handover_storage.vacuum_magnetic_pressure[idx_lk] -
            bphiv * bphiv;
      }
    }
  } else {
    for (int kl = 0; kl < s.nZnT; ++kl) {
      const int jH_ns2 = (fc.ns - 2) * s.nZnT + kl;
      const int jH_ns3 = (fc.ns - 3) * s.nZnT + kl;

      const double bsq_lcfs =
          1.5 * vmec_internal_results.total_pressure(jH_ns2) -
          0.5 * vmec_internal_results.total_pressure(jH_ns3);

      const double bsupv_lcfs = 1.5 * vmec_internal_results.bsupv(jH_ns2) -
                                0.5 * vmec_internal_results.bsupv(jH_ns3);
      const double btor_lfcs =
          bsupv_lcfs * threed1_geometric_magnetic_intermediate.redge[kl];

      bpol2vac[kl] = 2.0 * bsq_lcfs - btor_lfcs * btor_lfcs;
    }  // kl
  }

  // Compute current-like norm (factor) in Eq.(8), <a> * int(Bpol**2 * dA)
  // where <a> == 2*pi*Rs in Eq. 8 is the effective minor radius = Vol/Asurf
  // (corrects wrong description of Rs in paper, which is NOT the major radius).
  // This aminr1 = 1/2 the "correct" aminr1.
  const double aminr1 = threed1_geomag.volume_p / threed1_geomag.surf_area_p;

  double bpol2vac_surf_area_sum = 0.0;
  for (int kl = 0; kl < s.nZnT; ++kl) {
    bpol2vac_surf_area_sum +=
        bpol2vac[kl] * threed1_geometric_magnetic_intermediate.surf_area[kl];
  }  // kl
  const double factor =
      1.0 / (4.0 * M_PI * M_PI * aminr1 * bpol2vac_surf_area_sum);
  const double facnorm = factor * 4.0 * M_PI * M_PI;

  // Lao's definition of normalization factor
  const double toroidal_current_per_circumference =
      vmec_internal_results.currv / threed1_geomag.circum_p;
  result.scaling_ratio = factor * toroidal_current_per_circumference *
                         toroidal_current_per_circumference *
                         threed1_geomag.volume_p;

  double sigr0 = 0.0;
  double sigr1 = 0.0;
  double sigz1 = 0.0;
  for (int kl = 0; kl < s.nZnT; ++kl) {
    const int l = kl % s.nThetaEff;
    const int lcfs_kl = (fc.ns - 1) * s.nZnT + kl;

    const double r_edge = threed1_geometric_magnetic_intermediate.redge[kl];
    const double z_edge =
        vmec_internal_results.z_e(lcfs_kl) + vmec_internal_results.z_o(lcfs_kl);

    const double p_hat = threed1_geometric_magnetic_intermediate.phat[kl];
    const double rbps1u = facnorm * r_edge * p_hat * s.wInt[l];

    sigr0 += rbps1u * vmec_internal_results.zuFull(lcfs_kl);
    sigr1 += rbps1u * vmec_internal_results.zuFull(lcfs_kl) * r_edge;
    sigz1 -= rbps1u * vmec_internal_results.ruFull(lcfs_kl) * z_edge;
  }  // kl

  const double er = sigr1 + sigz1;

  // LAO, NUCL.FUS. 25 (1985) 1421
  const double rshaf = threed1_geometric_magnetic_intermediate.rshaf;
  result.r_lao =
      threed1_geomag.volume_p / (2.0 * M_PI * threed1_geomag.cross_area_p);
  result.f_lao = rshaf / result.r_lao;
  result.f_geo = rshaf / threed1_geomag.rcen;

  result.smaleli = factor * threed1_geometric_magnetic_intermediate.sumbpol;
  result.betai = 2.0 * factor * threed1_geometric_magnetic_intermediate.sump;
  result.musubi = threed1_geometric_magnetic_intermediate.vnorm * factor *
                  threed1_geometric_magnetic_intermediate.musubi;
  result.lambda = 0.5 * result.smaleli + result.betai;

  // Shafranov def. based on RT, Eq.(12)
  result.s11 = er - rshaf * sigr0;

  // R = Rgeometric
  result.s12 = er - threed1_geomag.rcen * sigr0;

  // R = RLao
  result.s13 = er - result.r_lao * sigr0;
  result.s2 = sigr0 * rshaf;

  // 1/2 S3 in Eq.(14c)
  result.s3 = sigz1;

  result.delta1 = 0.0;
  result.delta2 = 1.0 - result.f_geo;
  result.delta3 = 1.0 - result.f_lao;

  return result;
}  // ComputeThreed1ShafranovIntegrals

vmecpp::WOutFileContents vmecpp::ComputeWOutFileContents(
    const VmecINDATA& indata, const Sizes& s, const FourierBasisFastPoloidal& t,
    const FlowControl& fc, const VmecConstants& constants,
    const HandoverStorage& handover_storage, const std::string& mgrid_mode,
    VmecInternalResults& m_vmec_internal_results, const BSubSHalf& bsubs_half,
    const MercierFileContents& mercier, const JxBOutFileContents& jxbout,
    const Threed1FirstTableIntermediate& threed1_first_table_intermediate,
    const Threed1FirstTable& threed1_first_table,
    const Threed1GeometricAndMagneticQuantities& threed1_geomag,
    const Threed1AxisGeometry& threed1_axis, const Threed1Betas& threed1_betas,
    VmecStatus vmec_status, int iter2) {
  // THIS SUBROUTINE CREATES THE FILE WOUT.
  // IT CONTAINS THE CYLINDRICAL COORDINATE SPECTRAL COEFFICIENTS
  // RMN,ZMN (full), LMN (half_mesh - CONVERTED FROM INTERNAL full
  // REPRESENTATION), AS WELL AS COEFFICIENTS (ON NYQ MESH) FOR COMPUTED
  // QUANTITIES: BSQ, BSUPU,V, BSUBU,V, GSQRT (HALF); BSUBS (FULL-CONVERTED IN
  // JXBFORCE)

  WOutFileContents wout;

  // take version from educational_VMEC for now
  // TODO(jons): Upgrade VMEC++ to match PARVMEC and then change version to
  // "9.0".
  wout.version = "8.52";

  // We cannot provide a meaningful value for input_extension here, as we run
  // from a json input, but the input_extension indicates the existence of a
  // corresponding Fortran namelist with naming convention input.extension
  // To avoid confusion we leave it blank.
  wout.input_extension = "";

  wout.sign_of_jacobian = m_vmec_internal_results.sign_of_jacobian;

  // TODO(jons): Extend data set such that all input file contents are available
  // in the output file.
  wout.gamma = indata.gamma;

  wout.pcurr_type = indata.pcurr_type;
  wout.pmass_type = indata.pmass_type;
  wout.piota_type = indata.piota_type;

  // mass profile: am, am_aux_s, am_aux_f
  wout.am = NonEmptyVectorOr(indata.am, 0.0);
  wout.am_aux_s = NonEmptyVectorOr(indata.am_aux_s, -1.0);
  wout.am_aux_f = NonEmptyVectorOr(indata.am_aux_f, 0.0);
  wout.ac = NonEmptyVectorOr(indata.ac, 0.0);
  wout.ac_aux_s = NonEmptyVectorOr(indata.ac_aux_s, -1.0);
  wout.ac_aux_f = NonEmptyVectorOr(indata.ac_aux_f, 0.0);
  wout.ai = NonEmptyVectorOr(indata.ai, 0.0);
  wout.ai_aux_s = NonEmptyVectorOr(indata.ai_aux_s, -1.0);
  wout.ai_aux_f = NonEmptyVectorOr(indata.ai_aux_f, 0.0);

  wout.nfp = indata.nfp;
  wout.mpol = indata.mpol;
  wout.ntor = indata.ntor;
  wout.lasym = indata.lasym;

  wout.ns = fc.ns;
  wout.ftolv = fc.ftolv;

  // TODO(jons): Technically, this is not an input but an output (should go into
  // output data section).
  wout.maximum_iterations = iter2;

  wout.lfreeb = indata.lfreeb;
  wout.mgrid_file = indata.mgrid_file;
  // copy STL vector into Eigen vector
  wout.nextcur = static_cast<int>(indata.extcur.size());
  wout.extcur = ToEigenVector(indata.extcur);
  wout.mgrid_mode = mgrid_mode;

  // -------------------
  // scalar quantities

  wout.wb = handover_storage.magneticEnergy;
  wout.wp = handover_storage.thermalEnergy;

  wout.rmax_surf = threed1_geomag.rmax_surf;
  wout.rmin_surf = threed1_geomag.rmin_surf;
  wout.zmax_surf = threed1_geomag.zmax_surf;

  wout.mnmax = s.mnmax;
  wout.mnmax_nyq = s.mnmax_nyq;

  // map SUCCESSFUL_TERMINATION(11) to NORMAL_TERMINATION(0) for wout file
  // (fileout.f90:45 in educational_VMEC)
  if (vmec_status == VmecStatus::SUCCESSFUL_TERMINATION) {
    wout.ier_flag = VmecStatusCode(VmecStatus::NORMAL_TERMINATION);
  } else {
    wout.ier_flag = VmecStatusCode(vmec_status);
  }

  wout.aspect = threed1_geomag.aspect;

  wout.betatot = threed1_betas.betatot;
  wout.betapol = threed1_betas.betapol;
  wout.betator = threed1_betas.betator;
  wout.betaxis = threed1_betas.betaxis;

  wout.b0 = threed1_geomag.b0;

  wout.rbtor0 = handover_storage.rBtor0;
  wout.rbtor = handover_storage.rBtor;

  wout.IonLarmor = threed1_geomag.IonLarmor;
  wout.VolAvgB = threed1_geomag.VolAvgB;

  wout.ctor = handover_storage.cTor / MU_0;

  wout.Aminor_p = threed1_geomag.Aminor_p;
  wout.Rmajor_p = threed1_geomag.Rmajor_p;
  wout.volume_p = threed1_geomag.volume_p;

  wout.fsqr = fc.fsqr;
  wout.fsqz = fc.fsqz;
  wout.fsql = fc.fsql;

  wout.force_residual_r = ToEigenVector(fc.force_residual_r);
  wout.force_residual_z = ToEigenVector(fc.force_residual_z);
  wout.force_residual_lambda = ToEigenVector(fc.force_residual_lambda);
  wout.fsqt = wout.force_residual_r + wout.force_residual_z +
              wout.force_residual_lambda;
  wout.delbsq = ToEigenVector(fc.delbsq);
  // Convert status codes to integer values
  wout.restart_reasons = Eigen::VectorXi::Zero(fc.restart_reasons.size());
  for (Eigen::Index i = 0;
       i < static_cast<Eigen::Index>(fc.restart_reasons.size()); ++i) {
    // VmecStatusCode
    wout.restart_reasons(i) = static_cast<int>(fc.restart_reasons[i]);
  }
  wout.itfsq = wout.fsqt.size();
  // First entry is the staring energy W
  wout.wdot = ToEigenVector(fc.mhd_energy);
  if (wout.wdot.size() > 1) {
    // Compute decay rate wdot = -W/dt = (W[1:] - W[0:]) / W[1:]
    wout.wdot.tail(wout.wdot.size() - 1) =
        (wout.wdot.tail(wout.wdot.size() - 1) -
         wout.wdot.head(wout.wdot.size() - 1))
            .cwiseQuotient(wout.wdot.tail(wout.wdot.size() - 1));
  }
  // -------------------
  // one-dimensional array quantities

  wout.iota_full = m_vmec_internal_results.iotaF;

  wout.safety_factor = VectorXd::Ones(fc.ns) * DBL_MAX;

  wout.pressure_full = VectorXd::Zero(fc.ns);
  wout.phipf = VectorXd::Zero(fc.ns);
  wout.chipf = VectorXd::Zero(fc.ns);
  wout.jcuru = VectorXd::Zero(fc.ns);
  wout.jcurv = VectorXd::Zero(fc.ns);

  for (int jF = 0; jF < fc.ns; ++jF) {
    if (wout.iota_full[jF] != 0.0) {
      wout.safety_factor[jF] = 1.0 / wout.iota_full[jF];
    }
    wout.pressure_full[jF] = threed1_first_table_intermediate.presf[jF] / MU_0;
    wout.phipf[jF] = m_vmec_internal_results.sign_of_jacobian * 2.0 * M_PI *
                     m_vmec_internal_results.phipF[jF];
    wout.chipf[jF] = m_vmec_internal_results.sign_of_jacobian * 2.0 * M_PI *
                     m_vmec_internal_results.chipF[jF];
    wout.jcuru[jF] = threed1_first_table_intermediate.jcuru[jF] / MU_0;
    wout.jcurv[jF] = threed1_first_table_intermediate.jcurv[jF] / MU_0;
  }  // jF
  wout.toroidal_flux = m_vmec_internal_results.phiF;
  wout.poloidal_flux = threed1_first_table_intermediate.chi;
  wout.spectral_width = m_vmec_internal_results.spectral_width;

  wout.mass.resize(fc.ns - 1);
  wout.pressure_half.resize(fc.ns - 1);
  for (int jH = 0; jH < fc.ns - 1; ++jH) {
    wout.mass[jH] = m_vmec_internal_results.massH[jH] / MU_0;
    wout.pressure_half[jH] = m_vmec_internal_results.presH[jH] / MU_0;
  }  // jH
  wout.iota_half = m_vmec_internal_results.iotaH;
  wout.beta = threed1_first_table_intermediate.beta_vol;
  wout.buco = threed1_first_table_intermediate.bucoH;
  wout.bvco = threed1_first_table_intermediate.bvcoH;
  wout.dVds = m_vmec_internal_results.dVdsH;
  wout.phips = m_vmec_internal_results.phipH;
  wout.overr = threed1_first_table_intermediate.overr;

  wout.jdotb = jxbout.jdotb;
  wout.bdotb = jxbout.bdotb;
  wout.bdotgradv = jxbout.bdotgradv;

  wout.DMerc = mercier.DMerc;
  wout.Dshear = mercier.Dshear;
  wout.Dwell = mercier.Dwell;
  wout.Dcurr = mercier.Dcurr;
  wout.Dgeod = mercier.Dgeod;

  wout.equif = threed1_first_table.radial_force;

  // TODO(jons): curlabel, potvac: once free-boundary works

  // -------------------
  // mode numbers for Fourier coefficient arrays below

  // copy STL vectors into Eigen vectors
  wout.xm = ToEigenVector(t.xm);
  wout.xn = ToEigenVector(t.xn);
  wout.xm_nyq = ToEigenVector(t.xm_nyq);
  wout.xn_nyq = ToEigenVector(t.xn_nyq);

  // -------------------
  // stellarator-symmetric Fourier coefficients

  wout.raxis_c = threed1_axis.raxis_symm;
  wout.zaxis_s = threed1_axis.zaxis_symm;

  // NYQUIST FREQUENCY REQUIRES FACTOR OF 1/2
  std::vector<double> cosmui(s.nThetaReduced * (s.mnyq2 + 1));
  for (int ml = 0; ml < s.nThetaReduced * (s.mnyq2 + 1); ++ml) {
    cosmui[ml] = t.cosmui[ml];
  }
  if (s.mnyq != 0) {
    for (int l = 0; l < s.nThetaReduced; ++l) {
      const int ml = s.mnyq * s.nThetaReduced + l;
      cosmui[ml] /= 2.0;
    }
  }

  std::vector<double> cosnv((s.nnyq2 + 1) * s.nZeta);
  for (int kn = 0; kn < (s.nnyq2 + 1) * s.nZeta; ++kn) {
    cosnv[kn] = t.cosnv[kn];
  }
  if (s.nnyq != 0) {
    for (int k = 0; k < s.nZeta; ++k) {
      // FIXME(eguiraud) slow loop
      const int kn = k * (s.nnyq2 + 1) + s.nnyq;
      cosnv[kn] /= 2.0;
    }
  }

  // MUST CONVERT m=1 MODES... FROM INTERNAL TO PHYSICAL FORM
  // Extrapolation of m=0 Lambda (cs) modes, which are not evolved at j=1, done
  // in CONVERT
  if (s.lthreed) {
    for (int jF = 0; jF < fc.ns; ++jF) {
      for (int n = 0; n < s.ntor + 1; ++n) {
        const int m = 1;
        const int idx_fc = (jF * (s.ntor + 1) + n) * s.mpol + m;

        const double old_rss = m_vmec_internal_results.rmnss(idx_fc);
        m_vmec_internal_results.rmnss(idx_fc) =
            (old_rss + m_vmec_internal_results.zmncs(idx_fc));
        m_vmec_internal_results.zmncs(idx_fc) =
            (old_rss - m_vmec_internal_results.zmncs(idx_fc));
      }  // n
    }  // jF
  }

  // CONVERT TO rmnc, zmns, lmns, etc EXTERNAL representation (without internal
  // mscale, nscale) IF B^v ~ phip + lamu, MUST DIVIDE BY phipf(js) below to
  // maintain old-style format
  wout.rmnc = RowMatrixXd::Zero(fc.ns, s.mnmax);
  wout.zmns = RowMatrixXd::Zero(fc.ns, s.mnmax);
  wout.lmns_full = RowMatrixXd::Zero(fc.ns, s.mnmax);
  for (int jF = 0; jF < fc.ns; ++jF) {
    std::vector<double> rmnc1(s.mnmax, 0.0);
    std::vector<double> zmns1(s.mnmax, 0.0);
    std::vector<double> lmns1(s.mnmax, 0.0);

    // DO M = 0 MODES SEPARATELY (ONLY KEEP N >= 0 HERE: COS(-NV), SIN(-NV))
    int mn = -1;
    int m_0 = 0;
    for (int n = 0; n <= s.ntor; ++n) {
      mn++;
      const int idx_fc = (jF * (s.ntor + 1) + n) * s.mpol + m_0;
      const double t1 = t.mscale[m_0] * t.nscale[n];
      rmnc1[mn] = t1 * m_vmec_internal_results.rmncc(idx_fc);
      if (s.lthreed) {
        zmns1[mn] = -t1 * m_vmec_internal_results.zmncs(idx_fc);
        lmns1[mn] = -t1 * m_vmec_internal_results.lmncs(idx_fc);
      }
      // NOTE: Z and lambda do not have m=0 contributions in 2D,
      // since sin(m * theta) == 0 for m = 0
    }  // n

    // extrapolate to axis if 3D
    if (s.lthreed && jF == 0) {
      int mn = -1;
      for (int n = 0; n <= s.ntor; ++n) {
        mn++;
        const int idx_ns_1 = (1 * (s.ntor + 1) + n) * s.mpol + m_0;
        const int idx_ns_2 = (2 * (s.ntor + 1) + n) * s.mpol + m_0;
        const double t1 = t.mscale[m_0] * t.nscale[n];
        lmns1[mn] = -t1 * (2.0 * m_vmec_internal_results.lmncs(idx_ns_1) -
                           m_vmec_internal_results.lmncs(idx_ns_2));
      }  // n
    }

    // now come the m>0, n=-ntor, ..., ntor entries
    for (int m = 1; m < s.mpol; ++m) {
      for (int n = -s.ntor; n <= s.ntor; ++n) {
        mn++;
        const int abs_n = std::abs(n);
        const int idx_fc = (jF * (s.ntor + 1) + abs_n) * s.mpol + m;
        const double t1 = t.mscale[m] * t.nscale[abs_n];
        if (n == 0) {
          rmnc1[mn] = t1 * m_vmec_internal_results.rmncc(idx_fc);
          zmns1[mn] = t1 * m_vmec_internal_results.zmnsc(idx_fc);
          lmns1[mn] = t1 * m_vmec_internal_results.lmnsc(idx_fc);
        } else if (jF > 0) {
          rmnc1[mn] = t1 * m_vmec_internal_results.rmncc(idx_fc) / 2.0;
          zmns1[mn] = t1 * m_vmec_internal_results.zmnsc(idx_fc) / 2.0;
          lmns1[mn] = t1 * m_vmec_internal_results.lmnsc(idx_fc) / 2.0;
          if (s.lthreed) {
            const int sign_n = signum(n);
            rmnc1[mn] +=
                t1 * sign_n * m_vmec_internal_results.rmnss(idx_fc) / 2.0;
            zmns1[mn] -=
                t1 * sign_n * m_vmec_internal_results.zmncs(idx_fc) / 2.0;
            lmns1[mn] -=
                t1 * sign_n * m_vmec_internal_results.lmncs(idx_fc) / 2.0;
          }
        }
        // NOTE: can omit assigning jF=0 entries to 0, since rmnc1, ..., lmns1
        // are initialized to 0.0 already
      }  // n
    }  // m

    // pre-incrementing means that we are off by one at the end
    CHECK_EQ(mn + 1, s.mnmax) << "counting error: (mn + 1)=" << (mn + 1)
                              << " should be mnmax=" << s.mnmax;

    for (int mn = 0; mn < s.mnmax; ++mn) {
      wout.rmnc(jF * s.mnmax + mn) = rmnc1[mn];
      wout.zmns(jF * s.mnmax + mn) = zmns1[mn];
      wout.lmns_full(jF * s.mnmax + mn) =
          lmns1[mn] / m_vmec_internal_results.phipF[jF] * constants.lamscale;
    }  // mn
  }  // jF

  // INTERPOLATE LAMBDA ONTO HALF-MESH FOR BACKWARDS CONSISTENCY WITH EARLIER
  // VERSIONS OF VMEC AND SMOOTHS POSSIBLE UNPHYSICAL "WIGGLE" ON RADIAL MESH
  wout.lmns = RowMatrixXd::Zero(fc.ns - 1, s.mnmax);
  for (int jH = 0; jH < fc.ns - 1; ++jH) {
    const int jFi = jH;
    const int jFo = jH + 1;

    for (int mn = 0; mn < s.mnmax; ++mn) {
      const double lmns_outside = wout.lmns_full(jFo * s.mnmax + mn);

      double lmns_inside = wout.lmns_full(jFi * s.mnmax + mn);
      if (jFi == 0 && wout.xm[mn] <= 1) {
        lmns_inside = lmns_outside;
      }

      if (wout.xm[mn] % 2 == 0) {
        // m is even
        wout.lmns(jH * s.mnmax + mn) = (lmns_outside + lmns_inside) / 2.0;
      } else {
        // m is odd
        const double sm = m_vmec_internal_results.sm[jH];
        const double sp = m_vmec_internal_results.sp[jH];
        wout.lmns(jH * s.mnmax + mn) =
            (sm * lmns_outside + sp * lmns_inside) / 2.0;
      }
    }  // mn
  }  // jH

  // COMPUTE |B| = SQRT(|B|**2) and store in bsq, bsqa
  std::vector<double> magnetic_pressure((fc.ns - 1) * s.nZnT, 0.0);
  for (int jH = 0; jH < fc.ns - 1; ++jH) {
    for (int kl = 0; kl < s.nZnT; ++kl) {
      const int idx_kl = jH * s.nZnT + kl;
      const double total_pressure =
          m_vmec_internal_results.total_pressure(idx_kl);
      magnetic_pressure[idx_kl] = std::sqrt(
          2.0 * std::abs(total_pressure - m_vmec_internal_results.presH[jH]));
    }  // kl
  }  // jH

  double tmult = 0.5;
  if (s.lasym) {
    // Changed integration norm in fixaray, SPH012314
    // TODO(jons): figure out how this works with running a symmetric case in
    // lasym=true mode
    // -> should agree, but I suspect that
    // https://github.com/ORNL-Fusion/PARVMEC/issues/21
    //    has not been fixed yet for educational_VMEC.
    tmult *= 2.0;

    // TODO(jons): implement symoutput() once lasym=true test case is set up
  }

  // -------------------
  // Fourier-transform derived quantities for each surface individually

  // half-grid
  wout.gmnc = RowMatrixXd::Zero(fc.ns - 1, s.mnmax_nyq);
  wout.bmnc = RowMatrixXd::Zero(fc.ns - 1, s.mnmax_nyq);
  wout.bsubumnc = RowMatrixXd::Zero(fc.ns - 1, s.mnmax_nyq);
  wout.bsubvmnc = RowMatrixXd::Zero(fc.ns - 1, s.mnmax_nyq);

  // Note: bsubsmns is a half-grid quantity,
  // but stored in Fortran VMEC fashion offset by 1 index to the right,
  // in order to also have the (wrong) extrapolation
  // beyond the axis on the j=0 grid point
  // for backwards compatibility.
  wout.bsubsmns = RowMatrixXd::Zero(fc.ns, s.mnmax_nyq);

  wout.bsupumnc = RowMatrixXd::Zero(fc.ns - 1, s.mnmax_nyq);
  wout.bsupvmnc = RowMatrixXd::Zero(fc.ns - 1, s.mnmax_nyq);
  for (int jH = 0; jH < fc.ns - 1; ++jH) {
    for (int mn_nyq = 0; mn_nyq < s.mnmax_nyq; ++mn_nyq) {
      const int idx_mn_nyq = jH * s.mnmax_nyq + mn_nyq;
      const int idx_mn_nyq1 = (jH + 1) * s.mnmax_nyq + mn_nyq;

      const int m = wout.xm_nyq[mn_nyq];
      const int n = wout.xn_nyq[mn_nyq] / wout.nfp;
      const int abs_n = std::abs(n);
      const int sign_n = signum(n);

      double dmult = t.mscale[m] * t.nscale[abs_n] * tmult;
      if (m == 0 || n == 0) {
        dmult *= 2.0;
      }

      // perform Fourier integrals
      for (int l = 0; l < s.nThetaReduced; ++l) {
        const int ml = m * s.nThetaReduced + l;
        for (int k = 0; k < s.nZeta; ++k) {
          // FIXME(eguiraud) slow loop
          const int kn = k * (s.nnyq2 + 1) + abs_n;

          // cos(mu - nv)
          const double tcosi = dmult * (cosmui[ml] * cosnv[kn] +
                                        sign_n * t.sinmui[ml] * t.sinnv[kn]);

          // sin(mu - nv)
          const double tsini = dmult * (t.sinmui[ml] * cosnv[kn] -
                                        sign_n * cosmui[ml] * t.sinnv[kn]);

          const int idx_kl = (jH * s.nZeta + k) * s.nThetaEff + l;
          wout.gmnc(idx_mn_nyq) +=
              tcosi * m_vmec_internal_results.gsqrt(idx_kl);
          wout.bmnc(idx_mn_nyq) += tcosi * magnetic_pressure[idx_kl];
          wout.bsubumnc(idx_mn_nyq) +=
              tcosi * m_vmec_internal_results.bsubu(idx_kl);
          wout.bsubvmnc(idx_mn_nyq) +=
              tcosi * m_vmec_internal_results.bsubv(idx_kl);
          wout.bsubsmns(idx_mn_nyq1) += tsini * bsubs_half.bsubs_half(idx_kl);
          wout.bsupumnc(idx_mn_nyq) +=
              tcosi * m_vmec_internal_results.bsupu(idx_kl);
          wout.bsupvmnc(idx_mn_nyq) +=
              tcosi * m_vmec_internal_results.bsupv(idx_kl);
        }  // k
      }  // l
    }  // mn_nyq
  }  // jH

  // Note that bsubs in wrout.f in Fortran VMEC is on the half-grid,
  // as it is computed from bsup(u,v) (both of which are on the half-grid)
  // and the metric elements g_su and g_sv, which are also on the half-grid.
  // The wout file attributes document bsubsmns to be on the full-grid though,
  // and an extrapolation towards the axis as if bsubsmns was on the full grid
  // is done, in Fortran VMEC. For now, we replicate the axis extrapolation
  // (which is wrong, because it extrapolates one full grid step further than
  // the innermost half-grid point, i.e., from s=0.5 and s=1.5 to s=-0.5, i.e.,
  // beyond the magnetic axis) to be consistent with Fortran VMEC, but will have
  // to revisit this later. Also note that a full-grid version of bsubs is
  // computed in jxbforce, and is available in the jxbout file contents in
  // realspace as bsubs3.
  for (int mn_nyq = 0; mn_nyq < s.mnmax_nyq; ++mn_nyq) {
    wout.bsubsmns(0, mn_nyq) =
        2.0 * wout.bsubsmns(1, mn_nyq) - wout.bsubsmns(2, mn_nyq);
  }  // mn_nyq

  // -------------------
  // non-stellarator-symmetric Fourier coefficients

  if (s.lasym) {
    wout.raxis_s = threed1_axis.raxis_asym;
    wout.zaxis_c = threed1_axis.zaxis_asym;

    // MUST CONVERT m=1 MODES... FROM INTERNAL TO PHYSICAL FORM
    // Extrapolation of m=0 Lambda (cs) modes, which are not evolved at j=1,
    // done in CONVERT
    for (int jF = 0; jF < fc.ns; ++jF) {
      for (int n = 0; n < s.ntor + 1; ++n) {
        const int m = 1;
        const int idx_fc = (jF * (s.ntor + 1) + n) * s.mpol + m;

        const double old_rsc = m_vmec_internal_results.rmnsc(idx_fc);
        m_vmec_internal_results.rmnsc(idx_fc) =
            (old_rsc + m_vmec_internal_results.zmncc(idx_fc));
        m_vmec_internal_results.zmncc(idx_fc) =
            (old_rsc - m_vmec_internal_results.zmncc(idx_fc));
      }  // n
    }  // jF

    // CONVERT TO rmnc, zmns, lmns, etc EXTERNAL representation (without
    // internal mscale, nscale) IF B^v ~ phip + lamu, MUST DIVIDE BY phipf(js)
    // below to maintain old-style format

    wout.rmns = RowMatrixXd::Zero(fc.ns, s.mnmax);
    wout.zmnc = RowMatrixXd::Zero(fc.ns, s.mnmax);
    wout.lmnc_full = RowMatrixXd::Zero(fc.ns, s.mnmax);

    // TODO(jons): implement when first non-stellarator-symmetric test case is
    // ready

    // INTERPOLATE LAMBDA ONTO HALF-MESH FOR BACKWARDS CONSISTENCY WITH EARLIER
    // VERSIONS OF VMEC AND SMOOTHS POSSIBLE UNPHYSICAL "WIGGLE" ON RADIAL MESH

    wout.lmnc = RowMatrixXd::Zero(fc.ns - 1, s.mnmax);

    // TODO(jons): implement when first non-stellarator-symmetric test case is
    // ready
  }  // lasym

  // RESTORE nyq ENDPOINT VALUES
  // --> not needed here, since cosmui and cosnv were duplicated in local scope

  return wout;
}  // NOLINT(readability/fn_size)

void vmecpp::CompareWOut(const WOutFileContents& test_wout,
                         const WOutFileContents& expected_wout,
                         double tolerance,
                         bool check_equal_maximum_iterations) {
  CHECK_EQ(test_wout.sign_of_jacobian, expected_wout.sign_of_jacobian);
  CHECK_EQ(test_wout.gamma, expected_wout.gamma);
  CHECK_EQ(test_wout.pcurr_type, expected_wout.pcurr_type);
  CHECK_EQ(test_wout.pmass_type, expected_wout.pmass_type);
  CHECK_EQ(test_wout.piota_type, expected_wout.piota_type);

  CHECK(IsVectorCloseRelAbs(expected_wout.am, test_wout.am, tolerance));
  CHECK(IsVectorCloseRelAbs(expected_wout.ac, test_wout.ac, tolerance));
  CHECK(IsVectorCloseRelAbs(expected_wout.ai, test_wout.ai, tolerance));

  CHECK_EQ(test_wout.nfp, expected_wout.nfp);
  CHECK_EQ(test_wout.mpol, expected_wout.mpol);
  CHECK_EQ(test_wout.ntor, expected_wout.ntor);
  CHECK_EQ(test_wout.lasym, expected_wout.lasym);

  CHECK_EQ(test_wout.ns, expected_wout.ns);
  CHECK_EQ(test_wout.ftolv, expected_wout.ftolv);
  if (check_equal_maximum_iterations) {
    CHECK_EQ(test_wout.maximum_iterations, expected_wout.maximum_iterations);
  }

  CHECK_EQ(test_wout.lfreeb, expected_wout.lfreeb);
  CHECK_EQ(test_wout.mgrid_mode, expected_wout.mgrid_mode);

  // -------------------
  // scalar quantities

  CHECK(IsCloseRelAbs(expected_wout.wb, test_wout.wb, tolerance));
  CHECK(IsCloseRelAbs(expected_wout.wp, test_wout.wp, tolerance));

  CHECK(IsCloseRelAbs(expected_wout.rmax_surf, test_wout.rmax_surf, tolerance));
  CHECK(IsCloseRelAbs(expected_wout.rmin_surf, test_wout.rmin_surf, tolerance));
  CHECK(IsCloseRelAbs(expected_wout.zmax_surf, test_wout.zmax_surf, tolerance));

  CHECK_EQ(test_wout.mnmax, expected_wout.mnmax);
  CHECK_EQ(test_wout.mnmax_nyq, expected_wout.mnmax_nyq);

  CHECK_EQ(test_wout.ier_flag, expected_wout.ier_flag);

  CHECK(IsCloseRelAbs(expected_wout.aspect, test_wout.aspect, tolerance));

  CHECK(IsCloseRelAbs(expected_wout.betatot, test_wout.betatot, tolerance));
  CHECK(IsCloseRelAbs(expected_wout.betapol, test_wout.betapol, tolerance));
  CHECK(IsCloseRelAbs(expected_wout.betator, test_wout.betator, tolerance));
  CHECK(IsCloseRelAbs(expected_wout.betaxis, test_wout.betaxis, tolerance));

  CHECK(IsCloseRelAbs(expected_wout.b0, test_wout.b0, tolerance));

  CHECK(IsCloseRelAbs(expected_wout.rbtor0, test_wout.rbtor0, tolerance));
  CHECK(IsCloseRelAbs(expected_wout.rbtor, test_wout.rbtor, tolerance));

  CHECK(IsCloseRelAbs(expected_wout.IonLarmor, test_wout.IonLarmor, tolerance));
  CHECK(IsCloseRelAbs(expected_wout.VolAvgB, test_wout.VolAvgB, tolerance));

  CHECK(IsCloseRelAbs(expected_wout.ctor, test_wout.ctor, tolerance));

  CHECK(IsCloseRelAbs(expected_wout.Aminor_p, test_wout.Aminor_p, tolerance));
  CHECK(IsCloseRelAbs(expected_wout.Rmajor_p, test_wout.Rmajor_p, tolerance));
  CHECK(IsCloseRelAbs(expected_wout.volume_p, test_wout.volume_p, tolerance));

  CHECK(IsCloseRelAbs(expected_wout.fsqr, test_wout.fsqr, tolerance));
  CHECK(IsCloseRelAbs(expected_wout.fsqz, test_wout.fsqz, tolerance));
  CHECK(IsCloseRelAbs(expected_wout.fsql, test_wout.fsql, tolerance));

  // -------------------
  // one-dimensional array quantities

  const int ns = static_cast<int>(expected_wout.iota_full.size());
  for (int jF = 0; jF < ns; ++jF) {
    CHECK(IsCloseRelAbs(expected_wout.iota_full[jF], test_wout.iota_full[jF],
                        tolerance));
    CHECK(IsCloseRelAbs(expected_wout.safety_factor[jF],
                        test_wout.safety_factor[jF], tolerance));
    CHECK(IsCloseRelAbs(expected_wout.pressure_full[jF],
                        test_wout.pressure_full[jF], tolerance));
    CHECK(IsCloseRelAbs(expected_wout.toroidal_flux[jF],
                        test_wout.toroidal_flux[jF], tolerance));
    CHECK(IsCloseRelAbs(expected_wout.poloidal_flux[jF],
                        test_wout.poloidal_flux[jF], tolerance));
    CHECK(
        IsCloseRelAbs(expected_wout.phipf[jF], test_wout.phipf[jF], tolerance));
    CHECK(
        IsCloseRelAbs(expected_wout.chipf[jF], test_wout.chipf[jF], tolerance));
    CHECK(
        IsCloseRelAbs(expected_wout.jcuru[jF], test_wout.jcuru[jF], tolerance))
        << "jF = " << jF;
    CHECK(
        IsCloseRelAbs(expected_wout.jcurv[jF], test_wout.jcurv[jF], tolerance))
        << "jF = " << jF;
    CHECK(IsCloseRelAbs(expected_wout.spectral_width[jF],
                        test_wout.spectral_width[jF], tolerance));
  }  // jF

  for (int jH = 0; jH < ns - 1; ++jH) {
    CHECK(IsCloseRelAbs(expected_wout.iota_half[jH], test_wout.iota_half[jH],
                        tolerance));
    CHECK(IsCloseRelAbs(expected_wout.mass[jH], test_wout.mass[jH], tolerance));
    CHECK(IsCloseRelAbs(expected_wout.pressure_half[jH],
                        test_wout.pressure_half[jH], tolerance));
    CHECK(IsCloseRelAbs(expected_wout.beta[jH], test_wout.beta[jH], tolerance));
    CHECK(IsCloseRelAbs(expected_wout.buco[jH], test_wout.buco[jH], tolerance));
    CHECK(IsCloseRelAbs(expected_wout.bvco[jH], test_wout.bvco[jH], tolerance));
    CHECK(IsCloseRelAbs(expected_wout.dVds[jH], test_wout.dVds[jH], tolerance));
    CHECK(
        IsCloseRelAbs(expected_wout.phips[jH], test_wout.phips[jH], tolerance));
    CHECK(
        IsCloseRelAbs(expected_wout.overr[jH], test_wout.overr[jH], tolerance));
  }  // jH

  for (int jF = 0; jF < ns; ++jF) {
    CHECK(
        IsCloseRelAbs(expected_wout.jdotb[jF], test_wout.jdotb[jF], tolerance));
    CHECK(
        IsCloseRelAbs(expected_wout.bdotb[jF], test_wout.bdotb[jF], tolerance));
    CHECK(IsCloseRelAbs(expected_wout.bdotgradv[jF], test_wout.bdotgradv[jF],
                        tolerance));
  }  // jF

  for (int jF = 0; jF < ns; ++jF) {
    CHECK(
        IsCloseRelAbs(expected_wout.DMerc[jF], test_wout.DMerc[jF], tolerance))
        << "jF = " << jF;
    CHECK(IsCloseRelAbs(expected_wout.Dshear[jF], test_wout.Dshear[jF],
                        tolerance))
        << "jF = " << jF;
    CHECK(
        IsCloseRelAbs(expected_wout.Dwell[jF], test_wout.Dwell[jF], tolerance))
        << "jF = " << jF;
    CHECK(
        IsCloseRelAbs(expected_wout.Dcurr[jF], test_wout.Dcurr[jF], tolerance))
        << "jF = " << jF;
    CHECK(
        IsCloseRelAbs(expected_wout.Dgeod[jF], test_wout.Dgeod[jF], tolerance))
        << "jF = " << jF;
  }  // jF

  for (int jF = 0; jF < ns; ++jF) {
    CHECK(
        IsCloseRelAbs(expected_wout.equif[jF], test_wout.equif[jF], tolerance))
        << "jF = " << jF;
  }

  // -------------------
  // mode numbers for Fourier coefficient arrays below

  for (int mn = 0; mn < test_wout.mnmax; ++mn) {
    CHECK_EQ(test_wout.xm[mn], expected_wout.xm[mn]);
    CHECK_EQ(test_wout.xn[mn], expected_wout.xn[mn]);
  }  // mn

  for (int mn_nyq = 0; mn_nyq < test_wout.mnmax_nyq; ++mn_nyq) {
    CHECK_EQ(test_wout.xm_nyq[mn_nyq], expected_wout.xm_nyq[mn_nyq]);
    CHECK_EQ(test_wout.xn_nyq[mn_nyq], expected_wout.xn_nyq[mn_nyq]);
  }  // mn_nyq

  // -------------------
  // stellarator-symmetric Fourier coefficients

  for (int n = 0; n <= test_wout.ntor; ++n) {
    CHECK(IsCloseRelAbs(expected_wout.raxis_c[n], test_wout.raxis_c[n],
                        tolerance));
    CHECK(IsCloseRelAbs(expected_wout.zaxis_s[n], test_wout.zaxis_s[n],
                        tolerance));
  }  // n

  for (int jF = 0; jF < ns; ++jF) {
    for (int mn = 0; mn < test_wout.mnmax; ++mn) {
      CHECK(IsCloseRelAbs(expected_wout.rmnc(jF * test_wout.mnmax + mn),
                          test_wout.rmnc(jF * test_wout.mnmax + mn), tolerance))
          << "jF = " << jF << " mn = " << mn;
      CHECK(IsCloseRelAbs(expected_wout.zmns(jF * test_wout.mnmax + mn),
                          test_wout.zmns(jF * test_wout.mnmax + mn), tolerance))
          << "jF = " << jF << " mn = " << mn;
    }  // mn
  }  // jF

  for (int jH = 0; jH < ns - 1; ++jH) {
    for (int mn = 0; mn < test_wout.mnmax; ++mn) {
      CHECK(IsCloseRelAbs(expected_wout.lmns(jH * test_wout.mnmax + mn),
                          test_wout.lmns(jH * test_wout.mnmax + mn), tolerance))
          << "jH = " << jH << " mn = " << mn;
    }  // mn
  }  // jH

  for (int jH = 0; jH < ns - 1; ++jH) {
    for (int mn_nyq = 0; mn_nyq < test_wout.mnmax_nyq; ++mn_nyq) {
      CHECK(IsCloseRelAbs(expected_wout.gmnc(jH * test_wout.mnmax_nyq + mn_nyq),
                          test_wout.gmnc(jH * test_wout.mnmax_nyq + mn_nyq),
                          tolerance))
          << "jH = " << jH << " mn_nyq = " << mn_nyq;
      CHECK(IsCloseRelAbs(expected_wout.bmnc(jH * test_wout.mnmax_nyq + mn_nyq),
                          test_wout.bmnc(jH * test_wout.mnmax_nyq + mn_nyq),
                          tolerance))
          << "jH = " << jH << " mn_nyq = " << mn_nyq;
      CHECK(IsCloseRelAbs(
          expected_wout.bsubumnc(jH * test_wout.mnmax_nyq + mn_nyq),
          test_wout.bsubumnc(jH * test_wout.mnmax_nyq + mn_nyq), tolerance))
          << "jH = " << jH << " mn_nyq = " << mn_nyq;
      CHECK(IsCloseRelAbs(
          expected_wout.bsubvmnc(jH * test_wout.mnmax_nyq + mn_nyq),
          test_wout.bsubvmnc(jH * test_wout.mnmax_nyq + mn_nyq), tolerance))
          << "jH = " << jH << " mn_nyq = " << mn_nyq;
      CHECK(IsCloseRelAbs(
          expected_wout.bsubsmns(jH * test_wout.mnmax_nyq + mn_nyq),
          test_wout.bsubsmns(jH * test_wout.mnmax_nyq + mn_nyq), tolerance))
          << "jH = " << jH << " mn_nyq = " << mn_nyq;
      CHECK(IsCloseRelAbs(
          expected_wout.bsupumnc(jH * test_wout.mnmax_nyq + mn_nyq),
          test_wout.bsupumnc(jH * test_wout.mnmax_nyq + mn_nyq), tolerance))
          << "jH = " << jH << " mn_nyq = " << mn_nyq;
      CHECK(IsCloseRelAbs(
          expected_wout.bsupvmnc(jH * test_wout.mnmax_nyq + mn_nyq),
          test_wout.bsupvmnc(jH * test_wout.mnmax_nyq + mn_nyq), tolerance))
          << "jH = " << jH << " mn_nyq = " << mn_nyq;
    }  // mn_nyq
  }  // jH

  // also test the wrong extrapolation of bsubsmns
  // beyond the magnetic axis for backward compatibility
  for (int mn_nyq = 0; mn_nyq < test_wout.mnmax_nyq; ++mn_nyq) {
    CHECK(IsCloseRelAbs(
        expected_wout.bsubsmns(0 * test_wout.mnmax_nyq + mn_nyq),
        test_wout.bsubsmns(0 * test_wout.mnmax_nyq + mn_nyq), tolerance));
  }  // mn_nyq

  // -------------------
  // non-stellarator-symmetric Fourier coefficients

  if (test_wout.lasym) {
    for (int n = 0; n <= test_wout.ntor; ++n) {
      CHECK(IsCloseRelAbs(expected_wout.raxis_s[n], test_wout.raxis_s[n],
                          tolerance));
      CHECK(IsCloseRelAbs(expected_wout.zaxis_c[n], test_wout.zaxis_c[n],
                          tolerance));
    }  // n
  }
}
