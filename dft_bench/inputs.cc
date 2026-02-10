#include "vmecpp_tools/dft_bench/inputs.h"

#include <filesystem>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "util/file_io/file_io.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/fourier_geometry/fourier_geometry.h"
#include "vmecpp/vmec/handover_storage/handover_storage.h"
#include "vmecpp/vmec/radial_profiles/radial_profiles.h"

namespace fs = std::filesystem;

dft_bench::Inputs dft_bench::Inputs::FromConfig(const fs::path &config) {
  CHECK(fs::exists(config));

  // benchmarks are single-thread
  constexpr int num_threads = 1;
  constexpr int thread_id = 0;
  // FIXME(eguiraud) is this ok?
  constexpr int ns = 99;

  absl::StatusOr<std::string> indata_json = file_io::ReadFile(config);
  CHECK(indata_json.ok());

  absl::StatusOr<vmecpp::VmecINDATA> vmec_indata =
      vmecpp::VmecINDATA::FromJson(*indata_json);
  CHECK(vmec_indata.ok());

  vmecpp::RadialPartitioning radial_partitioning;
  radial_partitioning.adjustRadialPartitioning(num_threads, thread_id, ns,
                                               vmec_indata->lfreeb,
                                               /*printout*/ false);

  vmecpp::Sizes s(vmec_indata->lasym, vmec_indata->nfp, vmec_indata->mpol,
                  vmec_indata->ntor, vmec_indata->ntheta, vmec_indata->nzeta);

  vmecpp::FourierBasisFastPoloidal fb(&s);

  vmecpp::FlowControl fc(vmec_indata->lfreeb, vmec_indata->delt,
                         static_cast<int>(vmec_indata->ns_array.size()) + 1);
  // ForcesToFourier3D accesses fc.ns
  fc.ns = ns;

  vmecpp::HandoverStorage hs(&s);
  vmecpp::RadialProfiles radial_profiles(
      &radial_partitioning, &hs, &(*vmec_indata), &fc, 1 /*FIXME is this ok?*/,
      0. /*FIXME is this ok?*/);
  vmecpp::FourierGeometry fg(&s, &radial_partitioning, ns);

  return {
      radial_partitioning,        std::move(s), std::move(fb), std::move(fc),
      std::move(radial_profiles), std::move(fg)};
}
