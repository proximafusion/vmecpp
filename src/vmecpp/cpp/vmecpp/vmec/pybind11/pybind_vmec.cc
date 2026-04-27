// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include <pybind11/eigen.h>     // to wrap Eigen matrices
#include <pybind11/iostream.h>  // py::add_ostream_redirect
#include <pybind11/native_enum.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // to wrap std::vector
#include <pybind11/stl/filesystem.h>

#include <Eigen/Dense>
#include <filesystem>
#include <optional>
#include <string>
#include <type_traits>  // std::is_same_v
#include <utility>      // std::move

#include "vmecpp/common/magnetic_configuration_lib/magnetic_configuration_lib.h"
#include "vmecpp/common/makegrid_lib/makegrid_lib.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/output_quantities/output_quantities.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace py = pybind11;
using Eigen::VectorXd;
using Eigen::VectorXi;
using pybind11::literals::operator""_a;
using vmecpp::VmecINDATA;

namespace {

// Add a property that gets/sets an Eigen data member to a Pybind11 wrapper,
// converting between the internal real_t (long double) storage and Python's
// float64 (double) numpy arrays at the I/O boundary.
//
// For VmecINDATA members that are already double (RowMatrixXd / VectorXd),
// the getter returns a non-const reference so Python can mutate elements.
//
// For output struct members that use RowMatrixXr / Eigen::Matrix<real_t,...>,
// the getter copies and casts to double for Python; the setter casts back.
//
// Use as: DefEigenProperty(pywrapperclass, "rmnc", &WOutFileContents::rmnc);
template <typename PywrapperClass, typename EigenMatrix, typename Class>
void DefEigenProperty(PywrapperClass &pywrapper, const std::string &name,
                      EigenMatrix Class::*member_ptr) {
  static_assert(
      std::is_same_v<EigenMatrix, vmecpp::RowMatrixXd> ||
      std::is_same_v<EigenMatrix, Eigen::VectorXd> ||
      std::is_same_v<EigenMatrix, Eigen::VectorXi> ||
      std::is_same_v<EigenMatrix, vmecpp::RowMatrixXr> ||
      std::is_same_v<EigenMatrix,
                     Eigen::Matrix<vmecpp::real_t, Eigen::Dynamic, 1>>);

  if constexpr (std::is_same_v<EigenMatrix, vmecpp::RowMatrixXd> ||
                std::is_same_v<EigenMatrix, Eigen::VectorXd> ||
                std::is_same_v<EigenMatrix, Eigen::VectorXi>) {
    // double storage: return mutable reference so Python can modify elements
    auto getter = [member_ptr](Class &obj) -> EigenMatrix & {
      return obj.*member_ptr;
    };
    auto setter = [member_ptr](Class &obj, const EigenMatrix &val) {
      obj.*member_ptr = val;
    };
    pywrapper.def_property(name.c_str(), getter, setter);
  } else if constexpr (std::is_same_v<
                           EigenMatrix,
                           Eigen::Matrix<vmecpp::real_t, Eigen::Dynamic, 1>>) {
    // real_t column vector: cast to double and expose as 1-D numpy array
    auto getter = [member_ptr](const Class &obj) -> Eigen::VectorXd {
      return (obj.*member_ptr).template cast<double>();
    };
    auto setter = [member_ptr](Class &obj, const Eigen::VectorXd &val) {
      obj.*member_ptr = val.cast<vmecpp::real_t>();
    };
    pywrapper.def_property(name.c_str(), getter, setter);
  } else {
    // real_t row-major matrix: cast to double row-major matrix
    auto getter = [member_ptr](const Class &obj) -> vmecpp::RowMatrixXd {
      return (obj.*member_ptr).template cast<double>();
    };
    auto setter = [member_ptr](Class &obj, const vmecpp::RowMatrixXd &val) {
      obj.*member_ptr = val.cast<vmecpp::real_t>();
    };
    pywrapper.def_property(name.c_str(), getter, setter);
  }
}

// Add a readonly property for an Eigen real_t member, returning a double copy.
// Use instead of def_readonly for RowMatrixXr / Eigen::Matrix<real_t,...>
// members since pybind11 has no built-in caster for long double.
template <typename PywrapperClass, typename EigenMatrix, typename Class>
void DefEigenReadonly(PywrapperClass &pywrapper, const std::string &name,
                      EigenMatrix Class::*member_ptr) {
  if constexpr (std::is_same_v<EigenMatrix, Eigen::Matrix<vmecpp::real_t,
                                                          Eigen::Dynamic, 1>>) {
    pywrapper.def_property_readonly(
        name.c_str(), [member_ptr](const Class &obj) -> Eigen::VectorXd {
          return (obj.*member_ptr).template cast<double>();
        });
  } else {
    pywrapper.def_property_readonly(
        name.c_str(), [member_ptr](const Class &obj) -> vmecpp::RowMatrixXd {
          return (obj.*member_ptr).template cast<double>();
        });
  }
}

// Add a readonly property for a real_t scalar member, returning a double.
template <typename PywrapperClass, typename Class>
void DefRealReadonly(PywrapperClass &pywrapper, const std::string &name,
                     vmecpp::real_t Class::*member_ptr) {
  pywrapper.def_property_readonly(name.c_str(),
                                  [member_ptr](const Class &obj) -> double {
                                    return static_cast<double>(obj.*member_ptr);
                                  });
}

// Add a property that gets/sets a real_t scalar member as a Python float
// (double). Needed because pybind11 has no built-in caster for long double.
template <typename PywrapperClass, typename Class>
void DefRealProperty(PywrapperClass &pywrapper, const std::string &name,
                     vmecpp::real_t Class::*member_ptr) {
  pywrapper.def_property(
      name.c_str(),
      [member_ptr](const Class &obj) -> double {
        return static_cast<double>(obj.*member_ptr);
      },
      [member_ptr](Class &obj, double val) {
        obj.*member_ptr = static_cast<vmecpp::real_t>(val);
      });
}

// Add a property that gets/sets a std::vector<real_t> member as a Python list
// of floats (doubles).
template <typename PywrapperClass, typename Class>
void DefRealVecProperty(PywrapperClass &pywrapper, const std::string &name,
                        std::vector<vmecpp::real_t> Class::*member_ptr) {
  pywrapper.def_property(
      name.c_str(),
      [member_ptr](const Class &obj) -> std::vector<double> {
        const auto &v = obj.*member_ptr;
        std::vector<double> out;
        out.reserve(v.size());
        for (auto x : v) {
          out.push_back(static_cast<double>(x));
        }
        return out;
      },
      [member_ptr](Class &obj, const std::vector<double> &val) {
        auto &v = obj.*member_ptr;
        v.resize(val.size());
        for (std::size_t i = 0; i < val.size(); ++i) {
          v[i] = static_cast<vmecpp::real_t>(val[i]);
        }
      });
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

vmecpp::HotRestartState MakeHotRestartState(vmecpp::WOutFileContents wout,
                                            const vmecpp::VmecINDATA &indata) {
  return vmecpp::HotRestartState(std::move(wout), indata);
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

  auto pyindata = py::class_<VmecINDATA>(m, "VmecINDATA")
                      .def(py::init<>())
                      .def("_set_mpol_ntor", &VmecINDATA::SetMpolNtor,
                           py::arg("new_mpol"), py::arg("new_ntor"))
                      .def("from_file", &VmecINDATA::FromFile)
                      .def("from_json", &VmecINDATA::FromJson)
                      .def("to_json", &VmecINDATA::ToJsonOrException)
                      .def("copy", &VmecINDATA::Copy)

                      // numerical resolution, symmetry assumption
                      .def_readwrite("lasym", &VmecINDATA::lasym)
                      .def_readwrite("nfp", &VmecINDATA::nfp)
                      .def_readonly("mpol", &VmecINDATA::mpol)  // readonly!
                      .def_readonly("ntor", &VmecINDATA::ntor)  // readonly!
                      .def_readwrite("ntheta", &VmecINDATA::ntheta)
                      .def_readwrite("nzeta", &VmecINDATA::nzeta);

  // multi-grid steps
  DefEigenProperty(pyindata, "ns_array", &VmecINDATA::ns_array);
  DefEigenProperty(pyindata, "ftol_array", &VmecINDATA::ftol_array);
  DefEigenProperty(pyindata, "niter_array", &VmecINDATA::niter_array);

  // global physics parameters
  pyindata.def_readwrite("phiedge", &VmecINDATA::phiedge)
      .def_readwrite("ncurr", &VmecINDATA::ncurr)

      // mass / pressure profile
      .def_readwrite("pmass_type", &VmecINDATA::pmass_type);
  // fully read-write
  DefEigenProperty(pyindata, "am", &VmecINDATA::am);
  DefEigenProperty(pyindata, "am_aux_s", &VmecINDATA::am_aux_s);
  DefEigenProperty(pyindata, "am_aux_f", &VmecINDATA::am_aux_f);
  pyindata.def_readwrite("pres_scale", &VmecINDATA::pres_scale)
      .def_readwrite("gamma", &VmecINDATA::gamma)
      .def_readwrite("spres_ped", &VmecINDATA::spres_ped)

      // (initial guess for) iota profile
      .def_readwrite("piota_type", &VmecINDATA::piota_type);
  DefEigenProperty(pyindata, "ai", &VmecINDATA::ai);
  DefEigenProperty(pyindata, "ai_aux_s", &VmecINDATA::ai_aux_s);
  DefEigenProperty(pyindata, "ai_aux_f", &VmecINDATA::ai_aux_f);

  // enclosed toroidal current profile
  pyindata.def_readwrite("pcurr_type", &VmecINDATA::pcurr_type);
  DefEigenProperty(pyindata, "ac", &VmecINDATA::ac);
  DefEigenProperty(pyindata, "ac_aux_s", &VmecINDATA::ac_aux_s);
  DefEigenProperty(pyindata, "ac_aux_f", &VmecINDATA::ac_aux_f);
  pyindata.def_readwrite("curtor", &VmecINDATA::curtor)
      .def_readwrite("bloat", &VmecINDATA::bloat)

      // free-boundary parameters
      .def_readwrite("lfreeb", &VmecINDATA::lfreeb)
      .def_readwrite("mgrid_file", &VmecINDATA::mgrid_file);
  DefEigenProperty(pyindata, "extcur", &VmecINDATA::extcur);
  pyindata.def_readwrite("nvacskip", &VmecINDATA::nvacskip)
      .def_readwrite("free_boundary_method", &VmecINDATA::free_boundary_method)

      // tweaking parameters
      .def_readwrite("nstep", &VmecINDATA::nstep);
  DefEigenProperty(pyindata, "aphi", &VmecINDATA::aphi);
  pyindata.def_readwrite("delt", &VmecINDATA::delt)
      .def_readwrite("tcon0", &VmecINDATA::tcon0)
      .def_readwrite("lforbal", &VmecINDATA::lforbal)
      .def_readwrite("iteration_style", &VmecINDATA::iteration_style)
      .def_readwrite("return_outputs_even_if_not_converged",
                     &VmecINDATA::return_outputs_even_if_not_converged)

      // initial guess for magnetic axis
      // disallow re-assignment of the whole vector (to preserve sizes
      // consistent with mpol/ntor) but allow changing the individual elements
      .def_property_readonly(
          "raxis_c", [](VmecINDATA &w) -> VectorXd & { return w.raxis_c; })
      .def_property_readonly(
          "zaxis_s", [](VmecINDATA &w) -> VectorXd & { return w.zaxis_s; })
      .def_property_readonly(
          "raxis_s",
          [](VmecINDATA &w) -> std::optional<VectorXd> & { return w.raxis_s; })
      .def_property_readonly(
          "zaxis_c",
          [](VmecINDATA &w) -> std::optional<VectorXd> & { return w.zaxis_c; })

      // (initial guess for) boundary shape
      // disallow re-assignment of the whole matrix (to preserve shapes
      // consistent with mpol/ntor) but allow changing the individual elements
      .def_property_readonly(
          "rbc", [](VmecINDATA &w) -> vmecpp::RowMatrixXd & { return w.rbc; })
      .def_property_readonly(
          "zbs", [](VmecINDATA &w) -> vmecpp::RowMatrixXd & { return w.zbs; })
      .def_property_readonly(
          "rbs",
          [](VmecINDATA &w) -> std::optional<vmecpp::RowMatrixXd> & {
            return w.rbs;
          })
      .def_property_readonly(
          "zbc", [](VmecINDATA &w) -> std::optional<vmecpp::RowMatrixXd> & {
            return w.zbc;
          });

  py::native_enum<vmecpp::FreeBoundaryMethod>(m, "FreeBoundaryMethod",
                                              "enum.Enum")
      .value("NESTOR", vmecpp::FreeBoundaryMethod::NESTOR)
      .value("ONLY_COILS", vmecpp::FreeBoundaryMethod::ONLY_COILS)
      .value("BIEST", vmecpp::FreeBoundaryMethod::BIEST)
      .export_values()
      .finalize();

  py::native_enum<vmecpp::OutputMode>(m, "OutputMode", "enum.IntEnum")
      .value("SILENT", vmecpp::OutputMode::kSilent)
      .value("LEGACY", vmecpp::OutputMode::kLegacy)
      .value("PROGRESS", vmecpp::OutputMode::kProgress)
      .value("PROGRESS_NON_TTY", vmecpp::OutputMode::kProgressNonTTY)
      .export_values()
      .finalize();

  py::native_enum<vmecpp::IterationStyle>(m, "IterationStyle", "enum.Enum")
      .value("VMEC_8_52", vmecpp::IterationStyle::VMEC_8_52)
      .export_values()
      .finalize();

  py::class_<vmecpp::VmecCheckpoint>(m, "VmecCheckpoint");

  {
    auto c = py::class_<vmecpp::JxBOutFileContents>(m, "JxBOutFileContents");
    DefEigenReadonly(c, "itheta", &vmecpp::JxBOutFileContents::itheta);
    DefEigenReadonly(c, "izeta", &vmecpp::JxBOutFileContents::izeta);
    DefEigenReadonly(c, "bdotk", &vmecpp::JxBOutFileContents::bdotk);
    DefEigenReadonly(c, "amaxfor", &vmecpp::JxBOutFileContents::amaxfor);
    DefEigenReadonly(c, "aminfor", &vmecpp::JxBOutFileContents::aminfor);
    DefEigenReadonly(c, "avforce", &vmecpp::JxBOutFileContents::avforce);
    DefEigenReadonly(c, "pprim", &vmecpp::JxBOutFileContents::pprim);
    DefEigenReadonly(c, "jdotb", &vmecpp::JxBOutFileContents::jdotb);
    DefEigenReadonly(c, "bdotb", &vmecpp::JxBOutFileContents::bdotb);
    DefEigenReadonly(c, "bdotgradv", &vmecpp::JxBOutFileContents::bdotgradv);
    DefEigenReadonly(c, "jpar2", &vmecpp::JxBOutFileContents::jpar2);
    DefEigenReadonly(c, "jperp2", &vmecpp::JxBOutFileContents::jperp2);
    DefEigenReadonly(c, "phin", &vmecpp::JxBOutFileContents::phin);
    DefEigenReadonly(c, "jsupu3", &vmecpp::JxBOutFileContents::jsupu3);
    DefEigenReadonly(c, "jsupv3", &vmecpp::JxBOutFileContents::jsupv3);
    DefEigenReadonly(c, "jsups3", &vmecpp::JxBOutFileContents::jsups3);
    DefEigenReadonly(c, "bsupu3", &vmecpp::JxBOutFileContents::bsupu3);
    DefEigenReadonly(c, "bsupv3", &vmecpp::JxBOutFileContents::bsupv3);
    DefEigenReadonly(c, "jcrossb", &vmecpp::JxBOutFileContents::jcrossb);
    DefEigenReadonly(c, "jxb_gradp", &vmecpp::JxBOutFileContents::jxb_gradp);
    DefEigenReadonly(c, "jdotb_sqrtg",
                     &vmecpp::JxBOutFileContents::jdotb_sqrtg);
    DefEigenReadonly(c, "sqrtg3", &vmecpp::JxBOutFileContents::sqrtg3);
    DefEigenReadonly(c, "bsubu3", &vmecpp::JxBOutFileContents::bsubu3);
    DefEigenReadonly(c, "bsubv3", &vmecpp::JxBOutFileContents::bsubv3);
    DefEigenReadonly(c, "bsubs3", &vmecpp::JxBOutFileContents::bsubs3);
  }

  {
    auto c = py::class_<vmecpp::MercierFileContents>(m, "MercierFileContents");
    DefEigenReadonly(c, "s", &vmecpp::MercierFileContents::s);
    DefEigenReadonly(c, "toroidal_flux",
                     &vmecpp::MercierFileContents::toroidal_flux);
    DefEigenReadonly(c, "iota", &vmecpp::MercierFileContents::iota);
    DefEigenReadonly(c, "shear", &vmecpp::MercierFileContents::shear);
    DefEigenReadonly(c, "d_volume_d_s",
                     &vmecpp::MercierFileContents::d_volume_d_s);
    DefEigenReadonly(c, "well", &vmecpp::MercierFileContents::well);
    DefEigenReadonly(c, "toroidal_current",
                     &vmecpp::MercierFileContents::toroidal_current);
    DefEigenReadonly(c, "d_toroidal_current_d_s",
                     &vmecpp::MercierFileContents::d_toroidal_current_d_s);
    DefEigenReadonly(c, "pressure", &vmecpp::MercierFileContents::pressure);
    DefEigenReadonly(c, "d_pressure_d_s",
                     &vmecpp::MercierFileContents::d_pressure_d_s);
    DefEigenReadonly(c, "DMerc", &vmecpp::MercierFileContents::DMerc);
    DefEigenReadonly(c, "Dshear", &vmecpp::MercierFileContents::Dshear);
    DefEigenReadonly(c, "Dwell", &vmecpp::MercierFileContents::Dwell);
    DefEigenReadonly(c, "Dcurr", &vmecpp::MercierFileContents::Dcurr);
    DefEigenReadonly(c, "Dgeod", &vmecpp::MercierFileContents::Dgeod);
  }

  {
    auto c = py::class_<vmecpp::Threed1FirstTable>(m, "Threed1FirstTable");
    DefEigenReadonly(c, "s", &vmecpp::Threed1FirstTable::s);
    DefEigenReadonly(c, "radial_force",
                     &vmecpp::Threed1FirstTable::radial_force);
    DefEigenReadonly(c, "toroidal_flux",
                     &vmecpp::Threed1FirstTable::toroidal_flux);
    DefEigenReadonly(c, "iota", &vmecpp::Threed1FirstTable::iota);
    DefEigenReadonly(c, "avg_jsupu", &vmecpp::Threed1FirstTable::avg_jsupu);
    DefEigenReadonly(c, "avg_jsupv", &vmecpp::Threed1FirstTable::avg_jsupv);
    DefEigenReadonly(c, "d_volume_d_phi",
                     &vmecpp::Threed1FirstTable::d_volume_d_phi);
    DefEigenReadonly(c, "d_pressure_d_phi",
                     &vmecpp::Threed1FirstTable::d_pressure_d_phi);
    DefEigenReadonly(c, "spectral_width",
                     &vmecpp::Threed1FirstTable::spectral_width);
    DefEigenReadonly(c, "pressure", &vmecpp::Threed1FirstTable::pressure);
    DefEigenReadonly(c, "buco_full", &vmecpp::Threed1FirstTable::buco_full);
    DefEigenReadonly(c, "bvco_full", &vmecpp::Threed1FirstTable::bvco_full);
    DefEigenReadonly(c, "j_dot_b", &vmecpp::Threed1FirstTable::j_dot_b);
    DefEigenReadonly(c, "b_dot_b", &vmecpp::Threed1FirstTable::b_dot_b);
  }

  {
    using T = vmecpp::Threed1GeometricAndMagneticQuantities;
    auto c = py::class_<T>(m, "Threed1GeometricAndMagneticQuantities");
    DefRealReadonly(c, "toroidal_flux", &T::toroidal_flux);
    DefRealReadonly(c, "circum_p", &T::circum_p);
    DefRealReadonly(c, "surf_area_p", &T::surf_area_p);
    DefRealReadonly(c, "cross_area_p", &T::cross_area_p);
    DefRealReadonly(c, "volume_p", &T::volume_p);
    DefRealReadonly(c, "Rmajor_p", &T::Rmajor_p);
    DefRealReadonly(c, "Aminor_p", &T::Aminor_p);
    DefRealReadonly(c, "aspect", &T::aspect);
    DefRealReadonly(c, "kappa_p", &T::kappa_p);
    DefRealReadonly(c, "rcen", &T::rcen);
    DefRealReadonly(c, "aminr1", &T::aminr1);
    DefRealReadonly(c, "pavg", &T::pavg);
    DefRealReadonly(c, "factor", &T::factor);
    DefRealReadonly(c, "b0", &T::b0);
    DefRealReadonly(c, "rmax_surf", &T::rmax_surf);
    DefRealReadonly(c, "rmin_surf", &T::rmin_surf);
    DefRealReadonly(c, "zmax_surf", &T::zmax_surf);
    DefEigenReadonly(c, "bmin", &T::bmin);
    DefEigenReadonly(c, "bmax", &T::bmax);
    DefEigenReadonly(c, "waist", &T::waist);
    DefEigenReadonly(c, "height", &T::height);
    DefRealReadonly(c, "betapol", &T::betapol);
    DefRealReadonly(c, "betatot", &T::betatot);
    DefRealReadonly(c, "betator", &T::betator);
    DefRealReadonly(c, "VolAvgB", &T::VolAvgB);
    DefRealReadonly(c, "IonLarmor", &T::IonLarmor);
    DefRealReadonly(c, "jpar_perp", &T::jpar_perp);
    DefRealReadonly(c, "jparPS_perp", &T::jparPS_perp);
    DefRealReadonly(c, "toroidal_current", &T::toroidal_current);
    DefRealReadonly(c, "rbtor", &T::rbtor);
    DefRealReadonly(c, "rbtor0", &T::rbtor0);
    DefEigenReadonly(c, "psi", &T::psi);
    DefEigenReadonly(c, "ygeo", &T::ygeo);
    DefEigenReadonly(c, "yinden", &T::yinden);
    DefEigenReadonly(c, "yellip", &T::yellip);
    DefEigenReadonly(c, "ytrian", &T::ytrian);
    DefEigenReadonly(c, "yshift", &T::yshift);
    DefEigenReadonly(c, "loc_jpar_perp", &T::loc_jpar_perp);
    DefEigenReadonly(c, "loc_jparPS_perp", &T::loc_jparPS_perp);
  }

  {
    auto c = py::class_<vmecpp::Threed1Volumetrics>(m, "Threed1Volumetrics");
    DefRealReadonly(c, "int_p", &vmecpp::Threed1Volumetrics::int_p);
    DefRealReadonly(c, "avg_p", &vmecpp::Threed1Volumetrics::avg_p);
    DefRealReadonly(c, "int_bpol", &vmecpp::Threed1Volumetrics::int_bpol);
    DefRealReadonly(c, "avg_bpol", &vmecpp::Threed1Volumetrics::avg_bpol);
    DefRealReadonly(c, "int_btor", &vmecpp::Threed1Volumetrics::int_btor);
    DefRealReadonly(c, "avg_btor", &vmecpp::Threed1Volumetrics::avg_btor);
    DefRealReadonly(c, "int_modb", &vmecpp::Threed1Volumetrics::int_modb);
    DefRealReadonly(c, "avg_modb", &vmecpp::Threed1Volumetrics::avg_modb);
    DefRealReadonly(c, "int_ekin", &vmecpp::Threed1Volumetrics::int_ekin);
    DefRealReadonly(c, "avg_ekin", &vmecpp::Threed1Volumetrics::avg_ekin);
  }

  {
    auto c = py::class_<vmecpp::Threed1AxisGeometry>(m, "Threed1AxisGeometry");
    DefEigenReadonly(c, "raxis_symm", &vmecpp::Threed1AxisGeometry::raxis_symm);
    DefEigenReadonly(c, "zaxis_symm", &vmecpp::Threed1AxisGeometry::zaxis_symm);
    DefEigenReadonly(c, "raxis_asym", &vmecpp::Threed1AxisGeometry::raxis_asym);
    DefEigenReadonly(c, "zaxis_asym", &vmecpp::Threed1AxisGeometry::zaxis_asym);
  }

  {
    auto c = py::class_<vmecpp::Threed1Betas>(m, "Threed1Betas");
    DefRealReadonly(c, "betatot", &vmecpp::Threed1Betas::betatot);
    DefRealReadonly(c, "betapol", &vmecpp::Threed1Betas::betapol);
    DefRealReadonly(c, "betator", &vmecpp::Threed1Betas::betator);
    DefRealReadonly(c, "rbtor", &vmecpp::Threed1Betas::rbtor);
    DefRealReadonly(c, "betaxis", &vmecpp::Threed1Betas::betaxis);
    DefRealReadonly(c, "betstr", &vmecpp::Threed1Betas::betstr);
  }

  {
    using T = vmecpp::Threed1ShafranovIntegrals;
    auto c = py::class_<T>(m, "Threed1ShafranovIntegrals");
    DefRealReadonly(c, "scaling_ratio", &T::scaling_ratio);
    DefRealReadonly(c, "r_lao", &T::r_lao);
    DefRealReadonly(c, "f_lao", &T::f_lao);
    DefRealReadonly(c, "f_geo", &T::f_geo);
    DefRealReadonly(c, "smaleli", &T::smaleli);
    DefRealReadonly(c, "betai", &T::betai);
    DefRealReadonly(c, "musubi", &T::musubi);
    DefRealReadonly(c, "lambda", &T::lambda);
    DefRealReadonly(c, "s11", &T::s11);
    DefRealReadonly(c, "s12", &T::s12);
    DefRealReadonly(c, "s13", &T::s13);
    DefRealReadonly(c, "s2", &T::s2);
    DefRealReadonly(c, "s3", &T::s3);
    DefRealReadonly(c, "delta1", &T::delta1);
    DefRealReadonly(c, "delta2", &T::delta2);
    DefRealReadonly(c, "delta3", &T::delta3);
  }

  auto pywout =
      py::class_<vmecpp::WOutFileContents>(m, "WOutFileContents")
          .def(py::init<const vmecpp::WOutFileContents &>(), py::arg("wout"))
          .def(py::init())
          //
          .def_readwrite("input_extension",
                         &vmecpp::WOutFileContents::input_extension)
          //
          .def_readwrite("signgs", &vmecpp::WOutFileContents::signgs)
          //
          .def_readwrite("pcurr_type", &vmecpp::WOutFileContents::pcurr_type)
          .def_readwrite("pmass_type", &vmecpp::WOutFileContents::pmass_type)
          .def_readwrite("piota_type", &vmecpp::WOutFileContents::piota_type)
          //
          .def_readwrite("nfp", &vmecpp::WOutFileContents::nfp)
          .def_readwrite("mpol", &vmecpp::WOutFileContents::mpol)
          .def_readwrite("ntor", &vmecpp::WOutFileContents::ntor)
          .def_readwrite("lasym", &vmecpp::WOutFileContents::lasym)
          .def_readwrite("lrfp", &vmecpp::WOutFileContents::lrfp)
          //
          .def_readwrite("ns", &vmecpp::WOutFileContents::ns)
          .def_readwrite("ftolv", &vmecpp::WOutFileContents::ftolv)
          .def_readwrite("niter", &vmecpp::WOutFileContents::niter)
          //
          .def_readwrite("lfreeb", &vmecpp::WOutFileContents::lfreeb)
          .def_readwrite("mgrid_file", &vmecpp::WOutFileContents::mgrid_file)
          .def_readwrite("nextcur", &vmecpp::WOutFileContents::nextcur)
          .def_readwrite("extcur", &vmecpp::WOutFileContents::extcur)
          .def_readwrite("mgrid_mode", &vmecpp::WOutFileContents::mgrid_mode)
          //
          .def_readwrite("mnmax", &vmecpp::WOutFileContents::mnmax)
          .def_readwrite("mnmax_nyq", &vmecpp::WOutFileContents::mnmax_nyq)
          //
          .def_readwrite("ier_flag", &vmecpp::WOutFileContents::ier_flag)
          //
          .def_readwrite("itfsq", &vmecpp::WOutFileContents::itfsq)
          //
          .def_readwrite("restart_reason_timetrace",
                         &vmecpp::WOutFileContents::restart_reason_timetrace)
          //
          .def_readwrite("curlabel", &vmecpp::WOutFileContents::curlabel)
          //
          .def_readwrite("potvac", &vmecpp::WOutFileContents::potvac)
          //
          .def_readwrite("xm", &vmecpp::WOutFileContents::xm)
          .def_readwrite("xn", &vmecpp::WOutFileContents::xn)
          .def_readwrite("xm_nyq", &vmecpp::WOutFileContents::xm_nyq)
          .def_readwrite("xn_nyq", &vmecpp::WOutFileContents::xn_nyq);

  // real_t scalars: exposed as Python float (double) with explicit cast
  DefRealProperty(pywout, "version_", &vmecpp::WOutFileContents::version_);
  DefRealProperty(pywout, "gamma", &vmecpp::WOutFileContents::gamma);
  DefRealProperty(pywout, "wb", &vmecpp::WOutFileContents::wb);
  DefRealProperty(pywout, "wp", &vmecpp::WOutFileContents::wp);
  DefRealProperty(pywout, "rmax_surf", &vmecpp::WOutFileContents::rmax_surf);
  DefRealProperty(pywout, "rmin_surf", &vmecpp::WOutFileContents::rmin_surf);
  DefRealProperty(pywout, "zmax_surf", &vmecpp::WOutFileContents::zmax_surf);
  DefRealProperty(pywout, "aspect", &vmecpp::WOutFileContents::aspect);
  DefRealProperty(pywout, "betatotal", &vmecpp::WOutFileContents::betatotal);
  DefRealProperty(pywout, "betapol", &vmecpp::WOutFileContents::betapol);
  DefRealProperty(pywout, "betator", &vmecpp::WOutFileContents::betator);
  DefRealProperty(pywout, "betaxis", &vmecpp::WOutFileContents::betaxis);
  DefRealProperty(pywout, "b0", &vmecpp::WOutFileContents::b0);
  DefRealProperty(pywout, "rbtor0", &vmecpp::WOutFileContents::rbtor0);
  DefRealProperty(pywout, "rbtor", &vmecpp::WOutFileContents::rbtor);
  DefRealProperty(pywout, "IonLarmor", &vmecpp::WOutFileContents::IonLarmor);
  DefRealProperty(pywout, "volavgB", &vmecpp::WOutFileContents::volavgB);
  DefRealProperty(pywout, "ctor", &vmecpp::WOutFileContents::ctor);
  DefRealProperty(pywout, "Aminor_p", &vmecpp::WOutFileContents::Aminor_p);
  DefRealProperty(pywout, "Rmajor_p", &vmecpp::WOutFileContents::Rmajor_p);
  DefRealProperty(pywout, "volume", &vmecpp::WOutFileContents::volume);
  DefRealProperty(pywout, "fsqr", &vmecpp::WOutFileContents::fsqr);
  DefRealProperty(pywout, "fsqz", &vmecpp::WOutFileContents::fsqz);
  DefRealProperty(pywout, "fsql", &vmecpp::WOutFileContents::fsql);

  // Eigen::Matrix<real_t,...> 1D and 2D arrays: cast to double at boundary
  DefEigenProperty(pywout, "am", &vmecpp::WOutFileContents::am);
  DefEigenProperty(pywout, "ac", &vmecpp::WOutFileContents::ac);
  DefEigenProperty(pywout, "ai", &vmecpp::WOutFileContents::ai);
  DefEigenProperty(pywout, "am_aux_s", &vmecpp::WOutFileContents::am_aux_s);
  DefEigenProperty(pywout, "am_aux_f", &vmecpp::WOutFileContents::am_aux_f);
  DefEigenProperty(pywout, "ac_aux_s", &vmecpp::WOutFileContents::ac_aux_s);
  DefEigenProperty(pywout, "ac_aux_f", &vmecpp::WOutFileContents::ac_aux_f);
  DefEigenProperty(pywout, "ai_aux_s", &vmecpp::WOutFileContents::ai_aux_s);
  DefEigenProperty(pywout, "ai_aux_f", &vmecpp::WOutFileContents::ai_aux_f);
  DefEigenProperty(pywout, "iotaf", &vmecpp::WOutFileContents::iotaf);
  DefEigenProperty(pywout, "q_factor", &vmecpp::WOutFileContents::q_factor);
  DefEigenProperty(pywout, "presf", &vmecpp::WOutFileContents::presf);
  DefEigenProperty(pywout, "phi", &vmecpp::WOutFileContents::phi);
  DefEigenProperty(pywout, "phipf", &vmecpp::WOutFileContents::phipf);
  DefEigenProperty(pywout, "chi", &vmecpp::WOutFileContents::chi);
  DefEigenProperty(pywout, "chipf", &vmecpp::WOutFileContents::chipf);
  DefEigenProperty(pywout, "jcuru", &vmecpp::WOutFileContents::jcuru);
  DefEigenProperty(pywout, "jcurv", &vmecpp::WOutFileContents::jcurv);
  DefEigenProperty(pywout, "force_residual_r",
                   &vmecpp::WOutFileContents::force_residual_r);
  DefEigenProperty(pywout, "force_residual_z",
                   &vmecpp::WOutFileContents::force_residual_z);
  DefEigenProperty(pywout, "force_residual_lambda",
                   &vmecpp::WOutFileContents::force_residual_lambda);
  DefEigenProperty(pywout, "fsqt", &vmecpp::WOutFileContents::fsqt);
  DefEigenProperty(pywout, "delbsq", &vmecpp::WOutFileContents::delbsq);
  DefEigenProperty(pywout, "wdot", &vmecpp::WOutFileContents::wdot);
  DefEigenProperty(pywout, "iotas", &vmecpp::WOutFileContents::iotas);
  DefEigenProperty(pywout, "mass", &vmecpp::WOutFileContents::mass);
  DefEigenProperty(pywout, "pres", &vmecpp::WOutFileContents::pres);
  DefEigenProperty(pywout, "beta_vol", &vmecpp::WOutFileContents::beta_vol);
  DefEigenProperty(pywout, "buco", &vmecpp::WOutFileContents::buco);
  DefEigenProperty(pywout, "bvco", &vmecpp::WOutFileContents::bvco);
  DefEigenProperty(pywout, "vp", &vmecpp::WOutFileContents::vp);
  DefEigenProperty(pywout, "specw", &vmecpp::WOutFileContents::specw);
  DefEigenProperty(pywout, "phips", &vmecpp::WOutFileContents::phips);
  DefEigenProperty(pywout, "over_r", &vmecpp::WOutFileContents::over_r);
  DefEigenProperty(pywout, "jdotb", &vmecpp::WOutFileContents::jdotb);
  DefEigenProperty(pywout, "bdotb", &vmecpp::WOutFileContents::bdotb);
  DefEigenProperty(pywout, "bdotgradv", &vmecpp::WOutFileContents::bdotgradv);
  DefEigenProperty(pywout, "DMerc", &vmecpp::WOutFileContents::DMerc);
  DefEigenProperty(pywout, "DShear", &vmecpp::WOutFileContents::DShear);
  DefEigenProperty(pywout, "DWell", &vmecpp::WOutFileContents::DWell);
  DefEigenProperty(pywout, "DCurr", &vmecpp::WOutFileContents::DCurr);
  DefEigenProperty(pywout, "DGeod", &vmecpp::WOutFileContents::DGeod);
  DefEigenProperty(pywout, "equif", &vmecpp::WOutFileContents::equif);
  DefEigenProperty(pywout, "raxis_cc", &vmecpp::WOutFileContents::raxis_cc);
  DefEigenProperty(pywout, "zaxis_cs", &vmecpp::WOutFileContents::zaxis_cs);
  DefEigenProperty(pywout, "rmnc", &vmecpp::WOutFileContents::rmnc);
  DefEigenProperty(pywout, "zmns", &vmecpp::WOutFileContents::zmns);
  DefEigenProperty(pywout, "lmns_full", &vmecpp::WOutFileContents::lmns_full);
  DefEigenProperty(pywout, "lmns", &vmecpp::WOutFileContents::lmns);
  DefEigenProperty(pywout, "gmnc", &vmecpp::WOutFileContents::gmnc);
  DefEigenProperty(pywout, "bmnc", &vmecpp::WOutFileContents::bmnc);
  DefEigenProperty(pywout, "bsubumnc", &vmecpp::WOutFileContents::bsubumnc);
  DefEigenProperty(pywout, "bsubvmnc", &vmecpp::WOutFileContents::bsubvmnc);
  DefEigenProperty(pywout, "bsubsmns", &vmecpp::WOutFileContents::bsubsmns);
  DefEigenProperty(pywout, "bsubsmns_full",
                   &vmecpp::WOutFileContents::bsubsmns_full);
  DefEigenProperty(pywout, "bsupumnc", &vmecpp::WOutFileContents::bsupumnc);
  DefEigenProperty(pywout, "bsupvmnc", &vmecpp::WOutFileContents::bsupvmnc);
  DefEigenProperty(pywout, "currumnc", &vmecpp::WOutFileContents::currumnc);
  DefEigenProperty(pywout, "currvmnc", &vmecpp::WOutFileContents::currvmnc);
  DefEigenProperty(pywout, "raxis_cs", &vmecpp::WOutFileContents::raxis_cs);
  DefEigenProperty(pywout, "zaxis_cc", &vmecpp::WOutFileContents::zaxis_cc);
  // non-stellarator symmetric
  DefEigenProperty(pywout, "rmns", &vmecpp::WOutFileContents::rmns);
  DefEigenProperty(pywout, "zmnc", &vmecpp::WOutFileContents::zmnc);
  DefEigenProperty(pywout, "lmnc_full", &vmecpp::WOutFileContents::lmnc_full);
  DefEigenProperty(pywout, "lmnc", &vmecpp::WOutFileContents::lmnc);
  DefEigenProperty(pywout, "gmns", &vmecpp::WOutFileContents::gmns);
  DefEigenProperty(pywout, "bmns", &vmecpp::WOutFileContents::bmns);
  DefEigenProperty(pywout, "bsubumns", &vmecpp::WOutFileContents::bsubumns);
  DefEigenProperty(pywout, "bsubvmns", &vmecpp::WOutFileContents::bsubvmns);
  DefEigenProperty(pywout, "bsubsmnc", &vmecpp::WOutFileContents::bsubsmnc);
  DefEigenProperty(pywout, "bsubsmnc_full",
                   &vmecpp::WOutFileContents::bsubsmnc_full);
  DefEigenProperty(pywout, "bsupumns", &vmecpp::WOutFileContents::bsupumns);
  DefEigenProperty(pywout, "bsupvmns", &vmecpp::WOutFileContents::bsupvmns);
  DefEigenProperty(pywout, "currumns", &vmecpp::WOutFileContents::currumns);
  DefEigenProperty(pywout, "currvmns", &vmecpp::WOutFileContents::currvmns);

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
      .def_readonly("indata", &vmecpp::OutputQuantities::indata)
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
      [](const VmecINDATA &indata,
         std::optional<vmecpp::HotRestartState> initial_state,
         std::optional<int> max_threads,
         vmecpp::OutputMode verbose) -> vmecpp::OutputQuantities {
        bool was_interrupted = false;
        auto interrupt_check = [&was_interrupted]() -> bool {
          if (was_interrupted) {
            return true;
          }
          py::gil_scoped_acquire acquire;
          if (PyErr_CheckSignals() != 0) {
            was_interrupted = true;
            return true;
          }
          return false;
        };
        absl::StatusOr<vmecpp::OutputQuantities> ret;
        {
          py::gil_scoped_release release;
          ret = vmecpp::run(indata, std::move(initial_state), max_threads,
                            verbose, interrupt_check);
        }
        if (was_interrupted) {
          throw py::error_already_set();
        }
        return GetValueOrThrow(ret);
      },
      py::arg("indata"), py::arg("initial_state") = std::nullopt,
      py::arg("max_threads") = std::nullopt,
      py::arg("verbose") = vmecpp::OutputMode::kProgress);

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
      [](const VmecINDATA &indata,
         const makegrid::MagneticFieldResponseTable &magnetic_response_table,
         std::optional<vmecpp::HotRestartState> initial_state,
         std::optional<int> max_threads, vmecpp::OutputMode verbose) {
        bool was_interrupted = false;
        auto interrupt_check = [&was_interrupted]() -> bool {
          if (was_interrupted) return true;
          py::gil_scoped_acquire acquire;
          if (PyErr_CheckSignals() != 0) {
            was_interrupted = true;
            return true;
          }
          return false;
        };
        absl::StatusOr<vmecpp::OutputQuantities> ret;
        {
          py::gil_scoped_release release;
          ret = vmecpp::run(indata, magnetic_response_table,
                            std::move(initial_state), max_threads, verbose,
                            interrupt_check);
        }
        if (was_interrupted) {
          throw py::error_already_set();
        }
        return GetValueOrThrow(ret);
      },
      py::arg("indata"), py::arg("magnetic_response_table"),
      py::arg("initial_state") = std::nullopt,
      py::arg("max_threads") = std::nullopt,
      py::arg("verbose") = vmecpp::OutputMode::kProgress);
}  // NOLINT(readability/fn_size)
