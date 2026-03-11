#include <Eigen/Dense>
#include <filesystem>
#include <iostream>
#include <string>

#include "absl/log/check.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "util/file_io/file_io.h"
#include "vmecpp/common/makegrid_lib/makegrid_lib.h"

namespace {
using file_io::ReadFile;

using magnetics::ImportMagneticConfigurationFromCoilsFile;
using magnetics::MagneticConfiguration;

using makegrid::ImportMakegridParametersFromFile;
using makegrid::MagneticFieldResponseTable;
using makegrid::MakegridCachedVectorPotential;
using makegrid::MakegridParameters;
using makegrid::WriteMakegridNetCDFFile;
}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cout << "usage: " << argv[0]
              << " makegrid_parameters.json coils.makegrid\n";
    return -1;
  }

  // read file for MakegridParameters specified on command line
  const std::filesystem::path makegrid_parameters_filename(argv[1]);
  CHECK_EQ(makegrid_parameters_filename.extension(), ".json")
      << "first command line argument has to be a '<makegrid_parameters>.json' "
         "file.";

  absl::StatusOr<MakegridParameters> maybe_makegrid_parameters =
      ImportMakegridParametersFromFile(makegrid_parameters_filename);
  CHECK_OK(maybe_makegrid_parameters);
  const MakegridParameters& makegrid_parameters = *maybe_makegrid_parameters;

  // read file containing the coil geometry specified on command line
  const std::filesystem::path makegrid_coils_filename(argv[2]);
  CHECK(absl::StartsWith(makegrid_coils_filename.filename().string(), "coils."))
      << "second command line argument has to be a "
         "'coils.<configuration_name>' file.";

  absl::StatusOr<MagneticConfiguration> maybe_magnetic_configuration =
      magnetics::ImportMagneticConfigurationFromCoilsFile(
          makegrid_coils_filename);
  CHECK_OK(maybe_magnetic_configuration);
  const MagneticConfiguration& magnetic_configuration =
      *maybe_magnetic_configuration;

  // make first internal copy to be able to
  // migrate num_windings into circuit currents
  MagneticConfiguration m_magnetic_configuration = magnetic_configuration;

  // Normalized mode actually means in MAKEGRID-speak
  // that the data from the 4-th column in the `coils.` file is divided out,
  // and that is parsed here into the number of windings in each Coil.
  // Hence, need to:
  // a) make sure that num_windings is the same
  //    for all Coils in a given SerialCircuit
  // b) "migrate" the num_windings factor into
  //    the circuit current, so that
  //    num_windings can be 1
  // c) set the number of windings to 1.0
  //    if normalize_by_currents is true
  if (makegrid_parameters.normalize_by_currents) {
    CHECK_OK(NumWindingsToCircuitCurrents(m_magnetic_configuration));
  }

  absl::StatusOr<Eigen::VectorXd> maybe_circuit_currents =
      GetCircuitCurrents(magnetic_configuration);
  CHECK_OK(maybe_circuit_currents);
  const Eigen::VectorXd& circuit_currents = *maybe_circuit_currents;

  // compute the magnetic field cache and the vector potential cache
  absl::StatusOr<MagneticFieldResponseTable> maybe_magnetic_response_table =
      ComputeMagneticFieldResponseTable(makegrid_parameters,
                                        m_magnetic_configuration);
  CHECK_OK(maybe_magnetic_response_table);
  const MagneticFieldResponseTable& magnetic_response_table =
      *maybe_magnetic_response_table;

  absl::StatusOr<MakegridCachedVectorPotential> maybe_vector_potential_cache =
      ComputeVectorPotentialCache(makegrid_parameters,
                                  m_magnetic_configuration);
  CHECK_OK(maybe_vector_potential_cache);
  const MakegridCachedVectorPotential& vector_potential_cache =
      *maybe_vector_potential_cache;

  // construct output filename based on part after `coils.` of filename
  std::vector<std::string> filename_parts =
      absl::StrSplit(makegrid_coils_filename.filename().string(), "coils.");
  CHECK_EQ(filename_parts.size(), (size_t)2);
  std::string makegrid_filename =
      absl::StrCat("mgrid_", filename_parts[1], ".nc");

  // write into NetCDF file for use by Fortran VMEC etc.
  absl::Status status = WriteMakegridNetCDFFile(
      makegrid_filename, makegrid_parameters, circuit_currents,
      magnetic_response_table, vector_potential_cache);
  CHECK_OK(status);

  return 0;
}
