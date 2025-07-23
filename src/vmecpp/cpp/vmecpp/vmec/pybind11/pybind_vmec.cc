// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include <pybind11/eigen.h>     // to wrap Eigen matrices
#include <pybind11/iostream.h>  // py::add_ostream_redirect
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // to wrap std::vector
#include <pybind11/stl/filesystem.h>

#include <Eigen/Dense>
#include <filesystem>
#include <optional>
#include <string>
#include <type_traits>  // std::is_same_v
#include <utility>      // std::move
#include <vector>

#include "vmecpp/common/magnetic_configuration_lib/magnetic_configuration_lib.h"
#include "vmecpp/common/makegrid_lib/makegrid_lib.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/output_quantities/output_quantities.h"
#include "vmecpp/vmec/pybind11/vmec_indata_pywrapper.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace py = pybind11;
using Eigen::VectorXd;
using Eigen::VectorXi;
using vmecpp::VmecINDATAPyWrapper;
using pybind11::literals::operator""_a;

namespace {

// Add a property that gets/sets an Eigen data members to a Pybind11 wrapper.
// Simply using e.g. def_readwrite("mat", &WOutFileContents::mat) does
// not work because, under the hood, def_readwrite casts the data member to
// const before returning it from the getter (so the "write" part of
// "readwrite" refers to the data member itself, but not its contents). The
// getter function added here instead allows modification of the matrix or
// vector elements themselves.
//
// Use as: def_eigen_property(pywrapperclass, "rmnc", &WOutFileContents::rmnc);
template <typename PywrapperClass, typename EigenMatrix, typename Class>
void DefEigenProperty(PywrapperClass &pywrapper, const std::string &name,
                      EigenMatrix Class::*member_ptr) {
  static_assert(std::is_same_v<EigenMatrix, vmecpp::RowMatrixXd> ||
                std::is_same_v<EigenMatrix, Eigen::VectorXd> ||
                std::is_same_v<EigenMatrix, Eigen::VectorXi>);
  // similar to what pybind11's def_readwrite does, but returning a non-const
  // value from the getter
  auto getter = [member_ptr](Class &obj) -> EigenMatrix & {
    return obj.*member_ptr;
  };
  auto setter = [member_ptr](Class &obj, const EigenMatrix &val) {
    obj.*member_ptr = val;
  };
  pywrapper.def_property(name.c_str(), getter, setter);
}

template <typename T>
T &GetValueOrThrow(absl::StatusOr<T> &s) {
  if (!s.ok()) {
    // Could handle more exceptions, but only some are translated to meaningful
    // python exception types.
    // https://pybind11.readthedocs.io/en/stable/advanced/exceptions.html
    if (absl::IsInvalidArgument(s.status())) {
      throw pybind11::attribute_error(std::string(s.status().message()));
    } else {
      throw std::runtime_error(std::string(s.status().message()));
    }
  }
  return s.value();
}

vmecpp::HotRestartState MakeHotRestartState(
    vmecpp::WOutFileContents wout, const vmecpp::VmecINDATAPyWrapper &indata) {
  return vmecpp::HotRestartState(std::move(wout), vmecpp::VmecINDATA(indata));
}

}  // anonymous namespace

// IMPORTANT: The first argument must be the name of the module, else
// compilation will succeed but import will fail with:
//     ImportError: dynamic module does not define module export function
//     (PyInit_example)
PYBIND11_MODULE(_vmecpp, m) {
  m.doc() = "pybind11 VMEC++ plugin";

  // C++ stdout and stderr cannot easily be captured or redirected from Python.
  // This adds a Python context manager that can be used to redirect them like
  // this:
  //
  // with _vmecpp.ostream_redirect(stdout=True, stderr=True):
  //   _vmecpp.run(indata, max_thread=1) # or some other noisy function
  //
  // WARNING: Pybind11's C++ iostream redirection is thread-unsafe and does not
  // play well with OpenMP: only use it with max_thread=1 or OMP_NUM_THREADS=1!
  py::add_ostream_redirect(m, "ostream_redirect");

  auto pyindata =
      py::class_<VmecINDATAPyWrapper>(m, "VmecINDATAPyWrapper")
          .def(py::init<>())
          .def("_set_mpol_ntor", &VmecINDATAPyWrapper::SetMpolNtor,
               py::arg("new_mpol"), py::arg("new_ntor"))
          .def("from_file", &VmecINDATAPyWrapper::FromFile)
          .def("from_json", &VmecINDATAPyWrapper::FromJson)
          .def("to_json", &VmecINDATAPyWrapper::ToJson)
          .def("copy", &VmecINDATAPyWrapper::Copy)

          // numerical resolution, symmetry assumption
          .def_readwrite("lasym", &VmecINDATAPyWrapper::lasym)
          .def_readwrite("nfp", &VmecINDATAPyWrapper::nfp)
          .def_readonly("mpol", &VmecINDATAPyWrapper::mpol)  // readonly!
          .def_readonly("ntor", &VmecINDATAPyWrapper::ntor)  // readonly!
          .def_readwrite("ntheta", &VmecINDATAPyWrapper::ntheta)
          .def_readwrite("nzeta", &VmecINDATAPyWrapper::nzeta);

  // multi-grid steps
  DefEigenProperty(pyindata, "ns_array", &VmecINDATAPyWrapper::ns_array);
  DefEigenProperty(pyindata, "ftol_array", &VmecINDATAPyWrapper::ftol_array);
  DefEigenProperty(pyindata, "niter_array", &VmecINDATAPyWrapper::niter_array);

  // global physics parameters
  pyindata.def_readwrite("phiedge", &VmecINDATAPyWrapper::phiedge)
      .def_readwrite("ncurr", &VmecINDATAPyWrapper::ncurr)

      // mass / pressure profile
      .def_readwrite("pmass_type", &VmecINDATAPyWrapper::pmass_type);
  // fully read-write
  DefEigenProperty(pyindata, "am", &VmecINDATAPyWrapper::am);
  DefEigenProperty(pyindata, "am_aux_s", &VmecINDATAPyWrapper::am_aux_s);
  DefEigenProperty(pyindata, "am_aux_f", &VmecINDATAPyWrapper::am_aux_f);
  pyindata.def_readwrite("pres_scale", &VmecINDATAPyWrapper::pres_scale)
      .def_readwrite("gamma", &VmecINDATAPyWrapper::gamma)
      .def_readwrite("spres_ped", &VmecINDATAPyWrapper::spres_ped)

      // (initial guess for) iota profile
      .def_readwrite("piota_type", &VmecINDATAPyWrapper::piota_type);
  DefEigenProperty(pyindata, "ai", &VmecINDATAPyWrapper::ai);
  DefEigenProperty(pyindata, "ai_aux_s", &VmecINDATAPyWrapper::ai_aux_s);
  DefEigenProperty(pyindata, "ai_aux_f", &VmecINDATAPyWrapper::ai_aux_f);

  // enclosed toroidal current profile
  pyindata.def_readwrite("pcurr_type", &VmecINDATAPyWrapper::pcurr_type);
  DefEigenProperty(pyindata, "ac", &VmecINDATAPyWrapper::ac);
  DefEigenProperty(pyindata, "ac_aux_s", &VmecINDATAPyWrapper::ac_aux_s);
  DefEigenProperty(pyindata, "ac_aux_f", &VmecINDATAPyWrapper::ac_aux_f);
  pyindata.def_readwrite("curtor", &VmecINDATAPyWrapper::curtor)
      .def_readwrite("bloat", &VmecINDATAPyWrapper::bloat)

      // free-boundary parameters
      .def_readwrite("lfreeb", &VmecINDATAPyWrapper::lfreeb)
      .def_readwrite("mgrid_file", &VmecINDATAPyWrapper::mgrid_file);
  DefEigenProperty(pyindata, "extcur", &VmecINDATAPyWrapper::extcur);
  pyindata.def_readwrite("nvacskip", &VmecINDATAPyWrapper::nvacskip)
      .def_readwrite("free_boundary_method",
                     &VmecINDATAPyWrapper::free_boundary_method)

      // tweaking parameters
      .def_readwrite("nstep", &VmecINDATAPyWrapper::nstep);
  DefEigenProperty(pyindata, "aphi", &VmecINDATAPyWrapper::aphi);
  pyindata.def_readwrite("delt", &VmecINDATAPyWrapper::delt)
      .def_readwrite("tcon0", &VmecINDATAPyWrapper::tcon0)
      .def_readwrite("lforbal", &VmecINDATAPyWrapper::lforbal)
      .def_readwrite("return_outputs_even_if_not_converged",
                     &VmecINDATAPyWrapper::return_outputs_even_if_not_converged)

      // initial guess for magnetic axis
      // disallow re-assignment of the whole vector (to preserve sizes
      // consistent with mpol/ntor) but allow changing the individual elements
      .def_property_readonly(
          "raxis_c",
          [](VmecINDATAPyWrapper &w) -> VectorXd & { return w.raxis_c; })
      .def_property_readonly(
          "zaxis_s",
          [](VmecINDATAPyWrapper &w) -> VectorXd & { return w.zaxis_s; })
      .def_property_readonly(
          "raxis_s",
          [](VmecINDATAPyWrapper &w) -> std::optional<VectorXd> & {
            return w.raxis_s;
          })
      .def_property_readonly(
          "zaxis_c",
          [](VmecINDATAPyWrapper &w) -> std::optional<VectorXd> & {
            return w.zaxis_c;
          })

      // (initial guess for) boundary shape
      // disallow re-assignment of the whole matrix (to preserve shapes
      // consistent with mpol/ntor) but allow changing the individual elements
      .def_property_readonly(
          "rbc",
          [](VmecINDATAPyWrapper &w) -> vmecpp::RowMatrixXd & { return w.rbc; })
      .def_property_readonly(
          "zbs",
          [](VmecINDATAPyWrapper &w) -> vmecpp::RowMatrixXd & { return w.zbs; })
      .def_property_readonly(
          "rbs",
          [](VmecINDATAPyWrapper &w) -> std::optional<vmecpp::RowMatrixXd> & {
            return w.rbs;
          })
      .def_property_readonly(
          "zbc",
          [](VmecINDATAPyWrapper &w) -> std::optional<vmecpp::RowMatrixXd> & {
            return w.zbc;
          });

  py::enum_<vmecpp::FreeBoundaryMethod>(m, "FreeBoundaryMethod")
      .value("NESTOR", vmecpp::FreeBoundaryMethod::NESTOR)
      .value("BIEST", vmecpp::FreeBoundaryMethod::BIEST);

  py::class_<vmecpp::VmecCheckpoint>(m, "VmecCheckpoint");

  py::class_<vmecpp::JxBOutFileContents>(m, "JxBOutFileContents")
      .def_readonly("itheta", &vmecpp::JxBOutFileContents::itheta)
      .def_readonly("izeta", &vmecpp::JxBOutFileContents::izeta)
      .def_readonly("bdotk", &vmecpp::JxBOutFileContents::bdotk)
      //
      .def_readonly("amaxfor", &vmecpp::JxBOutFileContents::amaxfor)
      .def_readonly("aminfor", &vmecpp::JxBOutFileContents::aminfor)
      .def_readonly("avforce", &vmecpp::JxBOutFileContents::avforce)
      .def_readonly("pprim", &vmecpp::JxBOutFileContents::pprim)
      .def_readonly("jdotb", &vmecpp::JxBOutFileContents::jdotb)
      .def_readonly("bdotb", &vmecpp::JxBOutFileContents::bdotb)
      .def_readonly("bdotgradv", &vmecpp::JxBOutFileContents::bdotgradv)
      .def_readonly("jpar2", &vmecpp::JxBOutFileContents::jpar2)
      .def_readonly("jperp2", &vmecpp::JxBOutFileContents::jperp2)
      .def_readonly("phin", &vmecpp::JxBOutFileContents::phin)
      //
      .def_readonly("jsupu3", &vmecpp::JxBOutFileContents::jsupu3)
      .def_readonly("jsupv3", &vmecpp::JxBOutFileContents::jsupv3)
      .def_readonly("jsups3", &vmecpp::JxBOutFileContents::jsups3)
      .def_readonly("bsupu3", &vmecpp::JxBOutFileContents::bsupu3)
      .def_readonly("bsupv3", &vmecpp::JxBOutFileContents::bsupv3)
      .def_readonly("jcrossb", &vmecpp::JxBOutFileContents::jcrossb)
      .def_readonly("jxb_gradp", &vmecpp::JxBOutFileContents::jxb_gradp)
      .def_readonly("jdotb_sqrtg", &vmecpp::JxBOutFileContents::jdotb_sqrtg)
      .def_readonly("sqrtg3", &vmecpp::JxBOutFileContents::sqrtg3)
      .def_readonly("bsubu3", &vmecpp::JxBOutFileContents::bsubu3)
      .def_readonly("bsubv3", &vmecpp::JxBOutFileContents::bsubv3)
      .def_readonly("bsubs3", &vmecpp::JxBOutFileContents::bsubs3);

  py::class_<vmecpp::MercierFileContents>(m, "MercierFileContents")
      .def_readonly("s", &vmecpp::MercierFileContents::s)
      //
      .def_readonly("toroidal_flux",
                    &vmecpp::MercierFileContents::toroidal_flux)
      .def_readonly("iota", &vmecpp::MercierFileContents::iota)
      .def_readonly("shear", &vmecpp::MercierFileContents::shear)
      .def_readonly("d_volume_d_s", &vmecpp::MercierFileContents::d_volume_d_s)
      .def_readonly("well", &vmecpp::MercierFileContents::well)
      .def_readonly("toroidal_current",
                    &vmecpp::MercierFileContents::toroidal_current)
      .def_readonly("d_toroidal_current_d_s",
                    &vmecpp::MercierFileContents::d_toroidal_current_d_s)
      .def_readonly("pressure", &vmecpp::MercierFileContents::pressure)
      .def_readonly("d_pressure_d_s",
                    &vmecpp::MercierFileContents::d_pressure_d_s)
      //
      .def_readonly("DMerc", &vmecpp::MercierFileContents::DMerc)
      .def_readonly("Dshear", &vmecpp::MercierFileContents::Dshear)
      .def_readonly("Dwell", &vmecpp::MercierFileContents::Dwell)
      .def_readonly("Dcurr", &vmecpp::MercierFileContents::Dcurr)
      .def_readonly("Dgeod", &vmecpp::MercierFileContents::Dgeod);

  py::class_<vmecpp::Threed1FirstTable>(m, "Threed1FirstTable")
      .def_readonly("s", &vmecpp::Threed1FirstTable::s)
      .def_readonly("radial_force", &vmecpp::Threed1FirstTable::radial_force)
      .def_readonly("toroidal_flux", &vmecpp::Threed1FirstTable::toroidal_flux)
      .def_readonly("iota", &vmecpp::Threed1FirstTable::iota)
      .def_readonly("avg_jsupu", &vmecpp::Threed1FirstTable::avg_jsupu)
      .def_readonly("avg_jsupv", &vmecpp::Threed1FirstTable::avg_jsupv)
      .def_readonly("d_volume_d_phi",
                    &vmecpp::Threed1FirstTable::d_volume_d_phi)
      .def_readonly("d_pressure_d_phi",
                    &vmecpp::Threed1FirstTable::d_pressure_d_phi)
      .def_readonly("spectral_width",
                    &vmecpp::Threed1FirstTable::spectral_width)
      .def_readonly("pressure", &vmecpp::Threed1FirstTable::pressure)
      .def_readonly("buco_full", &vmecpp::Threed1FirstTable::buco_full)
      .def_readonly("bvco_full", &vmecpp::Threed1FirstTable::bvco_full)
      .def_readonly("j_dot_b", &vmecpp::Threed1FirstTable::j_dot_b)
      .def_readonly("b_dot_b", &vmecpp::Threed1FirstTable::b_dot_b);

  py::class_<vmecpp::Threed1GeometricAndMagneticQuantities>(
      m, "Threed1GeometricAndMagneticQuantities")
      .def_readonly(
          "toroidal_flux",
          &vmecpp::Threed1GeometricAndMagneticQuantities::toroidal_flux)
      //
      .def_readonly("circum_p",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::circum_p)
      .def_readonly("surf_area_p",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::surf_area_p)
      //
      .def_readonly(
          "cross_area_p",
          &vmecpp::Threed1GeometricAndMagneticQuantities::cross_area_p)
      .def_readonly("volume_p",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::volume_p)
      //
      .def_readonly("Rmajor_p",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::Rmajor_p)
      .def_readonly("Aminor_p",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::Aminor_p)
      .def_readonly("aspect",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::aspect)
      //
      .def_readonly("kappa_p",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::kappa_p)
      .def_readonly("rcen",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::rcen)
      //
      .def_readonly("aminr1",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::aminr1)
      //
      .def_readonly("pavg",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::pavg)
      .def_readonly("factor",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::factor)
      //
      .def_readonly("b0", &vmecpp::Threed1GeometricAndMagneticQuantities::b0)
      //
      .def_readonly("rmax_surf",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::rmax_surf)
      .def_readonly("rmin_surf",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::rmin_surf)
      .def_readonly("zmax_surf",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::zmax_surf)
      //
      .def_readonly("bmin",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::bmin)
      .def_readonly("bmax",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::bmax)
      //
      .def_readonly("waist",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::waist)
      .def_readonly("height",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::height)
      //
      .def_readonly("betapol",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::betapol)
      .def_readonly("betatot",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::betatot)
      .def_readonly("betator",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::betator)
      .def_readonly("VolAvgB",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::VolAvgB)
      .def_readonly("IonLarmor",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::IonLarmor)
      //
      .def_readonly("jpar_perp",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::jpar_perp)
      .def_readonly("jparPS_perp",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::jparPS_perp)
      //
      .def_readonly(
          "toroidal_current",
          &vmecpp::Threed1GeometricAndMagneticQuantities::toroidal_current)
      //
      .def_readonly("rbtor",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::rbtor)
      .def_readonly("rbtor0",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::rbtor0)
      //
      .def_readonly("psi", &vmecpp::Threed1GeometricAndMagneticQuantities::psi)
      .def_readonly("ygeo",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::ygeo)
      .def_readonly("yinden",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::yinden)
      .def_readonly("yellip",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::yellip)
      .def_readonly("ytrian",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::ytrian)
      .def_readonly("yshift",
                    &vmecpp::Threed1GeometricAndMagneticQuantities::yshift)
      //
      .def_readonly(
          "loc_jpar_perp",
          &vmecpp::Threed1GeometricAndMagneticQuantities::loc_jpar_perp)
      .def_readonly(
          "loc_jparPS_perp",
          &vmecpp::Threed1GeometricAndMagneticQuantities::loc_jparPS_perp);

  py::class_<vmecpp::Threed1Volumetrics>(m, "Threed1Volumetrics")
      .def_readonly("int_p", &vmecpp::Threed1Volumetrics::int_p)
      .def_readonly("avg_p", &vmecpp::Threed1Volumetrics::avg_p)
      //
      .def_readonly("int_bpol", &vmecpp::Threed1Volumetrics::int_bpol)
      .def_readonly("avg_bpol", &vmecpp::Threed1Volumetrics::avg_bpol)
      //
      .def_readonly("int_btor", &vmecpp::Threed1Volumetrics::int_btor)
      .def_readonly("avg_btor", &vmecpp::Threed1Volumetrics::avg_btor)
      //
      .def_readonly("int_modb", &vmecpp::Threed1Volumetrics::int_modb)
      .def_readonly("avg_modb", &vmecpp::Threed1Volumetrics::avg_modb)
      //
      .def_readonly("int_ekin", &vmecpp::Threed1Volumetrics::int_ekin)
      .def_readonly("avg_ekin", &vmecpp::Threed1Volumetrics::avg_ekin);

  py::class_<vmecpp::Threed1AxisGeometry>(m, "Threed1AxisGeometry")
      .def_readonly("raxis_symm", &vmecpp::Threed1AxisGeometry::raxis_symm)
      .def_readonly("zaxis_symm", &vmecpp::Threed1AxisGeometry::zaxis_symm)
      .def_readonly("raxis_asym", &vmecpp::Threed1AxisGeometry::raxis_asym)
      .def_readonly("zaxis_asym", &vmecpp::Threed1AxisGeometry::zaxis_asym);

  py::class_<vmecpp::Threed1Betas>(m, "Threed1Betas")
      .def_readonly("betatot", &vmecpp::Threed1Betas::betatot)
      .def_readonly("betapol", &vmecpp::Threed1Betas::betapol)
      .def_readonly("betator", &vmecpp::Threed1Betas::betator)
      .def_readonly("rbtor", &vmecpp::Threed1Betas::rbtor)
      .def_readonly("betaxis", &vmecpp::Threed1Betas::betaxis)
      .def_readonly("betstr", &vmecpp::Threed1Betas::betstr);

  py::class_<vmecpp::Threed1ShafranovIntegrals>(m, "Threed1ShafranovIntegrals")
      .def_readonly("scaling_ratio",
                    &vmecpp::Threed1ShafranovIntegrals::scaling_ratio)
      //
      .def_readonly("r_lao", &vmecpp::Threed1ShafranovIntegrals::r_lao)
      .def_readonly("f_lao", &vmecpp::Threed1ShafranovIntegrals::f_lao)
      .def_readonly("f_geo", &vmecpp::Threed1ShafranovIntegrals::f_geo)
      //
      .def_readonly("smaleli", &vmecpp::Threed1ShafranovIntegrals::smaleli)
      .def_readonly("betai", &vmecpp::Threed1ShafranovIntegrals::betai)
      .def_readonly("musubi", &vmecpp::Threed1ShafranovIntegrals::musubi)
      .def_readonly("lambda", &vmecpp::Threed1ShafranovIntegrals::lambda)
      //
      .def_readonly("s11", &vmecpp::Threed1ShafranovIntegrals::s11)
      .def_readonly("s12", &vmecpp::Threed1ShafranovIntegrals::s12)
      .def_readonly("s13", &vmecpp::Threed1ShafranovIntegrals::s13)
      .def_readonly("s2", &vmecpp::Threed1ShafranovIntegrals::s2)
      .def_readonly("s3", &vmecpp::Threed1ShafranovIntegrals::s3)
      //
      .def_readonly("delta1", &vmecpp::Threed1ShafranovIntegrals::delta1)
      .def_readonly("delta2", &vmecpp::Threed1ShafranovIntegrals::delta2)
      .def_readonly("delta3", &vmecpp::Threed1ShafranovIntegrals::delta3);

  py::class_<vmecpp::WOutFileContents>(m, "WOutFileContents")
      .def(py::init<const vmecpp::WOutFileContents &>(), py::arg("wout"))
      .def(py::init())
      .def_readwrite("version", &vmecpp::WOutFileContents::version)
      .def_readwrite("input_extension",
                     &vmecpp::WOutFileContents::input_extension)
      //
      .def_readwrite("sign_of_jacobian",
                     &vmecpp::WOutFileContents::sign_of_jacobian)
      //
      .def_readwrite("gamma", &vmecpp::WOutFileContents::gamma)
      //
      .def_readwrite("pcurr_type", &vmecpp::WOutFileContents::pcurr_type)
      .def_readwrite("pmass_type", &vmecpp::WOutFileContents::pmass_type)
      .def_readwrite("piota_type", &vmecpp::WOutFileContents::piota_type)
      //
      .def_readwrite("am", &vmecpp::WOutFileContents::am)
      .def_readwrite("ac", &vmecpp::WOutFileContents::ac)
      .def_readwrite("ai", &vmecpp::WOutFileContents::ai)
      //
      .def_readwrite("am_aux_s", &vmecpp::WOutFileContents::am_aux_s)
      .def_readwrite("am_aux_f", &vmecpp::WOutFileContents::am_aux_f)
      //
      .def_readwrite("ac_aux_s", &vmecpp::WOutFileContents::ac_aux_s)
      .def_readwrite("ac_aux_f", &vmecpp::WOutFileContents::ac_aux_f)
      //
      .def_readwrite("ai_aux_s", &vmecpp::WOutFileContents::ai_aux_s)
      .def_readwrite("ai_aux_f", &vmecpp::WOutFileContents::ai_aux_f)
      //
      .def_readwrite("nfp", &vmecpp::WOutFileContents::nfp)
      .def_readwrite("mpol", &vmecpp::WOutFileContents::mpol)
      .def_readwrite("ntor", &vmecpp::WOutFileContents::ntor)
      .def_readwrite("lasym", &vmecpp::WOutFileContents::lasym)
      //
      .def_readwrite("ns", &vmecpp::WOutFileContents::ns)
      .def_readwrite("ftolv", &vmecpp::WOutFileContents::ftolv)
      .def_readwrite("maximum_iterations",
                     &vmecpp::WOutFileContents::maximum_iterations)
      //
      .def_readwrite("lfreeb", &vmecpp::WOutFileContents::lfreeb)
      .def_readwrite("mgrid_file", &vmecpp::WOutFileContents::mgrid_file)
      .def_readwrite("nextcur", &vmecpp::WOutFileContents::nextcur)
      .def_readwrite("extcur", &vmecpp::WOutFileContents::extcur)
      .def_readwrite("mgrid_mode", &vmecpp::WOutFileContents::mgrid_mode)
      //
      .def_readwrite("wb", &vmecpp::WOutFileContents::wb)
      .def_readwrite("wp", &vmecpp::WOutFileContents::wp)
      //
      .def_readwrite("rmax_surf", &vmecpp::WOutFileContents::rmax_surf)
      .def_readwrite("rmin_surf", &vmecpp::WOutFileContents::rmin_surf)
      .def_readwrite("zmax_surf", &vmecpp::WOutFileContents::zmax_surf)
      //
      .def_readwrite("mnmax", &vmecpp::WOutFileContents::mnmax)
      .def_readwrite("mnmax_nyq", &vmecpp::WOutFileContents::mnmax_nyq)
      //
      .def_readwrite("ier_flag", &vmecpp::WOutFileContents::ier_flag)
      //
      .def_readwrite("aspect", &vmecpp::WOutFileContents::aspect)
      //
      .def_readwrite("betatot", &vmecpp::WOutFileContents::betatot)
      .def_readwrite("betapol", &vmecpp::WOutFileContents::betapol)
      .def_readwrite("betator", &vmecpp::WOutFileContents::betator)
      .def_readwrite("betaxis", &vmecpp::WOutFileContents::betaxis)
      //
      .def_readwrite("b0", &vmecpp::WOutFileContents::b0)
      //
      .def_readwrite("rbtor0", &vmecpp::WOutFileContents::rbtor0)
      .def_readwrite("rbtor", &vmecpp::WOutFileContents::rbtor)
      //
      .def_readwrite("IonLarmor", &vmecpp::WOutFileContents::IonLarmor)
      .def_readwrite("VolAvgB", &vmecpp::WOutFileContents::VolAvgB)
      //
      .def_readwrite("ctor", &vmecpp::WOutFileContents::ctor)
      //
      .def_readwrite("Aminor_p", &vmecpp::WOutFileContents::Aminor_p)
      .def_readwrite("Rmajor_p", &vmecpp::WOutFileContents::Rmajor_p)
      .def_readwrite("volume_p", &vmecpp::WOutFileContents::volume_p)
      //
      .def_readwrite("fsqr", &vmecpp::WOutFileContents::fsqr)
      .def_readwrite("fsqz", &vmecpp::WOutFileContents::fsqz)
      .def_readwrite("fsql", &vmecpp::WOutFileContents::fsql)
      .def_readwrite("itfsq", &vmecpp::WOutFileContents::itfsq)
      //
      .def_readwrite("iota_full", &vmecpp::WOutFileContents::iota_full)
      .def_readwrite("safety_factor", &vmecpp::WOutFileContents::safety_factor)
      .def_readwrite("pressure_full", &vmecpp::WOutFileContents::pressure_full)
      .def_readwrite("toroidal_flux", &vmecpp::WOutFileContents::toroidal_flux)
      .def_readwrite("phipf", &vmecpp::WOutFileContents::phipf)
      .def_readwrite("poloidal_flux", &vmecpp::WOutFileContents::poloidal_flux)
      .def_readwrite("chipf", &vmecpp::WOutFileContents::chipf)
      .def_readwrite("jcuru", &vmecpp::WOutFileContents::jcuru)
      .def_readwrite("jcurv", &vmecpp::WOutFileContents::jcurv)
      //
      .def_readwrite("force_residual_r",
                     &vmecpp::WOutFileContents::force_residual_r)
      .def_readwrite("force_residual_z",
                     &vmecpp::WOutFileContents::force_residual_z)
      .def_readwrite("force_residual_lambda",
                     &vmecpp::WOutFileContents::force_residual_lambda)
      .def_readwrite("fsqt", &vmecpp::WOutFileContents::fsqt)
      .def_readwrite("delbsq", &vmecpp::WOutFileContents::delbsq)
      .def_readwrite("wdot", &vmecpp::WOutFileContents::wdot)
      .def_readwrite("restart_reasons",
                     &vmecpp::WOutFileContents::restart_reasons)
      //
      .def_readwrite("iota_half", &vmecpp::WOutFileContents::iota_half)
      .def_readwrite("mass", &vmecpp::WOutFileContents::mass)
      .def_readwrite("pressure_half", &vmecpp::WOutFileContents::pressure_half)
      .def_readwrite("beta", &vmecpp::WOutFileContents::beta)
      .def_readwrite("buco", &vmecpp::WOutFileContents::buco)
      .def_readwrite("bvco", &vmecpp::WOutFileContents::bvco)
      .def_readwrite("dVds", &vmecpp::WOutFileContents::dVds)
      .def_readwrite("spectral_width",
                     &vmecpp::WOutFileContents::spectral_width)
      .def_readwrite("phips", &vmecpp::WOutFileContents::phips)
      .def_readwrite("overr", &vmecpp::WOutFileContents::overr)
      //
      .def_readwrite("jdotb", &vmecpp::WOutFileContents::jdotb)
      .def_readwrite("bdotb", &vmecpp::WOutFileContents::bdotb)
      .def_readwrite("bdotgradv", &vmecpp::WOutFileContents::bdotgradv)
      //
      .def_readwrite("DMerc", &vmecpp::WOutFileContents::DMerc)
      .def_readwrite("Dshear", &vmecpp::WOutFileContents::Dshear)
      .def_readwrite("Dwell", &vmecpp::WOutFileContents::Dwell)
      .def_readwrite("Dcurr", &vmecpp::WOutFileContents::Dcurr)
      .def_readwrite("Dgeod", &vmecpp::WOutFileContents::Dgeod)
      //
      .def_readwrite("equif", &vmecpp::WOutFileContents::equif)
      //
      .def_readwrite("curlabel", &vmecpp::WOutFileContents::curlabel)
      //
      .def_readwrite("potvac", &vmecpp::WOutFileContents::potvac)
      //
      .def_readwrite("xm", &vmecpp::WOutFileContents::xm)
      .def_readwrite("xn", &vmecpp::WOutFileContents::xn)
      .def_readwrite("xm_nyq", &vmecpp::WOutFileContents::xm_nyq)
      .def_readwrite("xn_nyq", &vmecpp::WOutFileContents::xn_nyq)
      //
      .def_readwrite("raxis_c", &vmecpp::WOutFileContents::raxis_c)
      .def_readwrite("zaxis_s", &vmecpp::WOutFileContents::zaxis_s)
      //
      .def_readwrite("rmnc", &vmecpp::WOutFileContents::rmnc)
      .def_readwrite("zmns", &vmecpp::WOutFileContents::zmns)
      .def_readwrite("lmns_full", &vmecpp::WOutFileContents::lmns_full)
      .def_readwrite("lmns", &vmecpp::WOutFileContents::lmns)
      .def_readwrite("gmnc", &vmecpp::WOutFileContents::gmnc)
      .def_readwrite("bmnc", &vmecpp::WOutFileContents::bmnc)
      .def_readwrite("bsubumnc", &vmecpp::WOutFileContents::bsubumnc)
      .def_readwrite("bsubvmnc", &vmecpp::WOutFileContents::bsubvmnc)
      .def_readwrite("bsubsmns", &vmecpp::WOutFileContents::bsubsmns)
      .def_readwrite("bsubsmns_full", &vmecpp::WOutFileContents::bsubsmns_full)
      .def_readwrite("bsupumnc", &vmecpp::WOutFileContents::bsupumnc)
      .def_readwrite("bsupvmnc", &vmecpp::WOutFileContents::bsupvmnc)
      //
      .def_readwrite("raxis_s", &vmecpp::WOutFileContents::raxis_s)
      .def_readwrite("zaxis_c", &vmecpp::WOutFileContents::zaxis_c)
      //
      .def_readwrite("rmns", &vmecpp::WOutFileContents::rmns)
      .def_readwrite("zmnc", &vmecpp::WOutFileContents::zmnc)
      .def_readwrite("lmnc_full", &vmecpp::WOutFileContents::lmnc_full)
      .def_readwrite("lmnc", &vmecpp::WOutFileContents::lmnc)
      .def_readwrite("gmns", &vmecpp::WOutFileContents::gmns)
      .def_readwrite("bmns", &vmecpp::WOutFileContents::bmns)
      .def_readwrite("bsubumns", &vmecpp::WOutFileContents::bsubumns)
      .def_readwrite("bsubvmns", &vmecpp::WOutFileContents::bsubvmns)
      .def_readwrite("bsubsmnc", &vmecpp::WOutFileContents::bsubsmnc)
      .def_readwrite("bsubsmnc_full", &vmecpp::WOutFileContents::bsubsmnc_full)
      .def_readwrite("bsupumns", &vmecpp::WOutFileContents::bsupumns)
      .def_readwrite("bsupvmns", &vmecpp::WOutFileContents::bsupvmns);

  py::class_<vmecpp::OutputQuantities>(m, "OutputQuantities")
      .def_readonly("jxbout", &vmecpp::OutputQuantities::jxbout)
      .def_readonly("mercier", &vmecpp::OutputQuantities::mercier)
      .def_readonly("threed1_first_table",
                    &vmecpp::OutputQuantities::threed1_first_table)
      .def_readonly("threed1_geometric_magnetic",
                    &vmecpp::OutputQuantities::threed1_geometric_magnetic)
      .def_readonly("threed1_volumetrics",
                    &vmecpp::OutputQuantities::threed1_volumetrics)
      .def_readonly("threed1_axis", &vmecpp::OutputQuantities::threed1_axis)
      .def_readonly("threed1_betas", &vmecpp::OutputQuantities::threed1_betas)
      .def_readonly("threed1_shafranov_integrals",
                    &vmecpp::OutputQuantities::threed1_shafranov_integrals)
      .def_readonly("wout", &vmecpp::OutputQuantities::wout)
      .def_property_readonly(
          "indata",
          [](const vmecpp::OutputQuantities &oq) -> VmecINDATAPyWrapper {
            // TODO(eguiraud): this conversion requires a copy as long as
            // https://github.com/proximafusion/repo/issues/2593 is not solved
            return VmecINDATAPyWrapper(oq.indata);
          })
      .def(
          "save",
          [](const vmecpp::OutputQuantities &oq,
             const std::filesystem::path &path) {
            absl::Status s = oq.Save(path);

            if (!s.ok()) {
              const std::string msg =
                  "There was an error saving OutputQuantities to file '" +
                  std::string(path) + "':\n" + std::string(s.message());
              throw std::runtime_error(msg);
            }
          },
          py::arg("path"))
      .def_static("load", [](const std::filesystem::path &path) {
        auto maybe_oq = vmecpp::OutputQuantities::Load(path);
        if (!maybe_oq.ok()) {
          const std::string msg =
              "There was an error loading OutputQuantities from file '" +
              std::string(path) + "':\n" +
              std::string(maybe_oq.status().message());
          throw std::runtime_error(msg);
        }
        return maybe_oq.value();
      });

  py::class_<vmecpp::HotRestartState>(m, "HotRestartState")
      .def(py::init(&MakeHotRestartState), "wout"_a, "indata"_a)
      .def_readwrite("wout", &vmecpp::HotRestartState::wout)
      .def_readwrite("indata", &vmecpp::HotRestartState::indata);

  m.def(
      "run",
      [](const VmecINDATAPyWrapper &indata,
         std::optional<vmecpp::HotRestartState> initial_state,
         std::optional<int> max_threads,
         bool verbose = true) -> vmecpp::OutputQuantities {
        auto ret = vmecpp::run(vmecpp::VmecINDATA(indata),
                               std::move(initial_state), max_threads, verbose);
        return GetValueOrThrow(ret);
      },
      py::arg("indata"), py::arg("initial_state") = std::nullopt,
      py::arg("max_threads") = std::nullopt, py::arg("verbose") = true);

  py::class_<makegrid::MakegridParameters>(m, "MakegridParameters")
      .def(py::init<bool, bool, int, double, double, int, double, double, int,
                    int>(),
           "normalize_by_currents"_a, "assume_stellarator_symmetry"_a,
           "number_of_field_periods"_a, "r_grid_minimum"_a, "r_grid_maximum"_a,
           "number_of_r_grid_points"_a, "z_grid_minimum"_a, "z_grid_maximum"_a,
           "number_of_z_grid_points"_a, "number_of_phi_grid_points"_a)
      .def_static(
          "from_file",
          [](const std::filesystem::path &file) {
            auto maybe_params =
                makegrid::ImportMakegridParametersFromFile(file);
            return GetValueOrThrow(maybe_params);
          },
          py::arg("file"))
      .def_readonly("normalize_by_currents",
                    &makegrid::MakegridParameters::normalize_by_currents)
      .def_readonly("assume_stellarator_symmetry",
                    &makegrid::MakegridParameters::assume_stellarator_symmetry)
      .def_readonly("number_of_field_periods",
                    &makegrid::MakegridParameters::number_of_field_periods)
      .def_readonly("r_grid_minimum",
                    &makegrid::MakegridParameters::r_grid_minimum)
      .def_readonly("r_grid_maximum",
                    &makegrid::MakegridParameters::r_grid_maximum)
      .def_readonly("number_of_r_grid_points",
                    &makegrid::MakegridParameters::number_of_r_grid_points)
      .def_readonly("z_grid_minimum",
                    &makegrid::MakegridParameters::z_grid_minimum)
      .def_readonly("z_grid_maximum",
                    &makegrid::MakegridParameters::z_grid_maximum)
      .def_readonly("number_of_z_grid_points",
                    &makegrid::MakegridParameters::number_of_z_grid_points)
      .def_readonly("number_of_phi_grid_points",
                    &makegrid::MakegridParameters::number_of_phi_grid_points);

  py::class_<magnetics::MagneticConfiguration>(m, "MagneticConfiguration")
      .def_static(
          "from_file",
          [](const std::filesystem::path &file) {
            auto maybe_config =
                magnetics::ImportMagneticConfigurationFromCoilsFile(file);
            return GetValueOrThrow(maybe_config);
          },
          py::arg("file"));
  auto response_table =
      py::class_<makegrid::MagneticFieldResponseTable>(
          m, "MagneticFieldResponseTable")
          .def(py::init<const makegrid::MakegridParameters &,
                        const makegrid::RowMatrixXd &,
                        const makegrid::RowMatrixXd &,
                        const makegrid::RowMatrixXd &>(),
               py::arg("parameters"), py::arg("b_r"), py::arg("b_p"),
               py::arg("b_z"))
          .def_readonly("parameters",
                        &makegrid::MagneticFieldResponseTable::parameters);
  DefEigenProperty(response_table, "b_r",
                   &makegrid::MagneticFieldResponseTable::b_r);
  DefEigenProperty(response_table, "b_p",
                   &makegrid::MagneticFieldResponseTable::b_p);
  DefEigenProperty(response_table, "b_z",
                   &makegrid::MagneticFieldResponseTable::b_z);

  m.def(
      "compute_magnetic_field_response_table",
      [](const makegrid::MakegridParameters &mgrid_params,
         const magnetics::MagneticConfiguration &magnetic_configuration) {
        auto ret = makegrid::ComputeMagneticFieldResponseTable(
            mgrid_params, magnetic_configuration);
        return GetValueOrThrow(ret);
      },
      py::arg("makegrid_parameters"), py::arg("magnetic_configuration"));

  m.def(
      "run",
      [](const VmecINDATAPyWrapper &indata,
         const makegrid::MagneticFieldResponseTable &magnetic_response_table,
         std::optional<vmecpp::HotRestartState> initial_state,
         std::optional<int> max_threads, bool verbose = true) {
        auto ret =
            vmecpp::run(vmecpp::VmecINDATA(indata), magnetic_response_table,
                        std::move(initial_state), max_threads, verbose);
        return GetValueOrThrow(ret);
      },
      py::arg("indata"), py::arg("magnetic_response_table"),
      py::arg("initial_state") = std::nullopt,
      py::arg("max_threads") = std::nullopt, py::arg("verbose") = true);
}  // NOLINT(readability/fn_size)
