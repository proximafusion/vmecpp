// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/common/magnetic_configuration_lib/magnetic_configuration_lib.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iostream>
#include <locale>
#include <sstream>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "util/file_io/file_io.h"
#include "vmecpp/common/composed_types_definition/composed_types.h"
#include "vmecpp/common/composed_types_lib/composed_types_lib.h"

namespace magnetics {

using composed_types::FourierCoefficient1D;
using composed_types::OrthonormalFrameAroundAxis;
using composed_types::Vector3d;

// Read all current carriers, starting from the line below "begin filament"
// until "end" is found or the stream ends.
absl::Status ParseCurrentCarriers(
    std::stringstream& m_makegrid_coils_ss,
    MagneticConfiguration& m_magnetic_configuration) {
  std::vector<int> coil_ids;

  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;
  std::vector<double> w;

  for (std::string raw_line; std::getline(m_makegrid_coils_ss, raw_line);) {
    absl::string_view stripped_line = absl::StripAsciiWhitespace(raw_line);

    if (absl::StartsWith(stripped_line, "mirror")) {
      // coils can be mirrored within MAKEGRID --> ignore this line for now
      if (!(absl::EndsWithIgnoreCase(stripped_line, "NIL") ||
            absl::EndsWithIgnoreCase(stripped_line, "NUL"))) {
        return absl::InvalidArgumentError(absl::StrCat(
            "The magnetic_configuration_lib only supports coilsets for which "
            "the mirror option is deactivated. Please check your "
            "coils file, should be 'mirror NUL' but found ",
            stripped_line));
      }
      continue;
    } else if (absl::StartsWith(stripped_line, "end")) {
      // current carrier geometry ended on this line
      return absl::OkStatus();
    }

    std::vector<std::string> line_parts = absl::StrSplit(
        stripped_line, absl::ByAnyChar(" \t"), absl::SkipWhitespace());

    const std::size_t num_line_parts = line_parts.size();

    if (num_line_parts == 4 || num_line_parts == 6) {
      // handle first four columns: x, y, z, w
      x.push_back(std::stod(line_parts[0]));  // x or r
      y.push_back(std::stod(line_parts[1]));  // y or 0.0
      z.push_back(std::stod(line_parts[2]));  // z
      w.push_back(std::stod(line_parts[3]));  // num_windings or 0.0 or current
                                              // or any product of those

      if (num_line_parts == 6) {
        // handle six columns: x, y, z, 0.0, serial_circuit_id,
        // current_carrier_name
        int serial_circuit_id = std::stoi(line_parts[4]);
        std::string current_carrier_name = line_parts[5];

        // find or create target serial circuit based on circuit ID
        std::vector<int>::iterator index_of_circuit_id =
            std::find(coil_ids.begin(), coil_ids.end(), serial_circuit_id);
        SerialCircuit* serial_circuit;
        if (index_of_circuit_id != coil_ids.end()) {
          const int serial_circuit_index =
              static_cast<int>(index_of_circuit_id - coil_ids.begin());
          serial_circuit = m_magnetic_configuration.mutable_serial_circuits(
              serial_circuit_index);
        } else {
          serial_circuit = m_magnetic_configuration.add_serial_circuits();
          serial_circuit->set_current(1.0);
          coil_ids.push_back(serial_circuit_id);
        }

        Coil* coil = serial_circuit->add_coils();
        coil->set_num_windings(w.at(0));

        CurrentCarrier* current_carrier = coil->add_current_carriers();

        if (x.size() == 1) {
          // circular filament has only one line with 6 columns

          CircularFilament* circular_filament =
              current_carrier->mutable_circular_filament();
          circular_filament->set_name(current_carrier_name);
          circular_filament->set_radius(x.at(0));

          Vector3d* center = circular_filament->mutable_center();
          center->set_x(0.0);
          center->set_y(0.0);
          center->set_z(z.at(0));

          Vector3d* normal = circular_filament->mutable_normal();
          normal->set_x(0.0);
          normal->set_y(0.0);
          normal->set_z(1.0);

        } else {
          // polygon filament has some lines with 4 columns
          // and ends at a line with 6 columns

          // check if all number of windings are equal -> error if they are not
          // all equal ignore last entry, which is always 0.0
          for (size_t i = 1; i < w.size() - 1; ++i) {
            if (w.at(i) != coil->num_windings()) {
              // abort parsing
              std::stringstream error_message;
              error_message << "number of windings different at point " << i;
              return absl::InvalidArgumentError(error_message.str());
            }
          }

          PolygonFilament* polygon_filament =
              current_carrier->mutable_polygon_filament();
          polygon_filament->set_name(current_carrier_name);
          for (size_t i = 0; i < x.size(); ++i) {
            Vector3d* vertex = polygon_filament->add_vertices();
            vertex->set_x(x.at(i));
            vertex->set_y(y.at(i));
            vertex->set_z(z.at(i));
          }
        }

        x.clear();
        y.clear();
        z.clear();
        w.clear();
      }
    } else {
      std::stringstream error_message;
      error_message << "cannot parse line: '" << stripped_line << "': has "
                    << num_line_parts << " parts";
      return absl::InvalidArgumentError(error_message.str());
    }
  }

  return absl::ResourceExhaustedError(
      "did not find 'end' line in makegrid_coils file");
}  // ParseCurrentCarriers

absl::StatusOr<MagneticConfiguration> ImportMagneticConfigurationFromMakegrid(
    const std::string& makegrid_coils) {
  MagneticConfiguration magnetic_configuration;

  std::stringstream makegrid_coils_ss(makegrid_coils);
  for (std::string raw_line; std::getline(makegrid_coils_ss, raw_line);
       /* no-op */) {
    absl::string_view stripped_line = absl::StripAsciiWhitespace(raw_line);

    if (absl::StartsWith(stripped_line, "periods")) {
      // number of field periods: "periods 5" --> 5
      std::vector<std::string> line_parts = absl::StrSplit(
          stripped_line, absl::ByAnyChar(" \t"), absl::SkipWhitespace());
      if (line_parts.size() != 2) {
        std::stringstream error_message;
        error_message << "expected number of field periods after 'periods', "
                         "but no second part was found on line '"
                      << stripped_line << "'";
        return absl::NotFoundError(error_message.str());
      } else {
        magnetic_configuration.set_num_field_periods(std::stoi(line_parts[1]));
      }
    } else if (absl::StartsWith(stripped_line, "begin filament")) {
      // expect current carrier geometry starting on next line until line "end"
      absl::Status status =
          ParseCurrentCarriers(makegrid_coils_ss, magnetic_configuration);
      if (status != absl::OkStatus()) {
        // If something did not work out during parsing, return an empty
        // MagneticConfiguration, since the data is likely messed up anyway.
        magnetic_configuration.Clear();
        return status;
      }
    }
  }

  return magnetic_configuration;
}  // ImportMagneticConfigurationFromMakegrid

absl::StatusOr<MagneticConfiguration> ImportMagneticConfigurationFromCoilsFile(
    const std::filesystem::path& mgrid_coils_file) {
  const auto maybe_coils_file_content = file_io::ReadFile(mgrid_coils_file);
  if (!maybe_coils_file_content.ok()) {
    return maybe_coils_file_content.status();
  }
  return ImportMagneticConfigurationFromMakegrid(*maybe_coils_file_content);
}

absl::StatusOr<std::vector<double> > GetCircuitCurrents(
    const MagneticConfiguration& magnetic_configuration) {
  absl::Status status =
      IsMagneticConfigurationFullyPopulated(magnetic_configuration);
  if (!status.ok()) {
    return status;
  }

  const int number_of_serial_circuits =
      magnetic_configuration.serial_circuits_size();
  std::vector<double> circuit_currents(number_of_serial_circuits, 0.0);
  for (int i = 0; i < number_of_serial_circuits; ++i) {
    circuit_currents[i] = magnetic_configuration.serial_circuits(i).current();
  }

  return circuit_currents;
}  // GetCircuitCurrents

absl::Status SetCircuitCurrents(
    const std::vector<double>& circuit_currents,
    MagneticConfiguration& m_magnetic_configuration) {
  const int number_of_serial_circuits =
      m_magnetic_configuration.serial_circuits_size();
  const int number_of_circuit_currents =
      static_cast<int>(circuit_currents.size());
  if (number_of_serial_circuits != number_of_circuit_currents) {
    std::stringstream error_message;
    error_message << "The number of circuit currents ("
                  << number_of_circuit_currents << ") ";
    error_message << "has to equal number of SerialCircuits ("
                  << number_of_serial_circuits << ") ";
    error_message << "in the given MagneticConfiguration.";
    return absl::InvalidArgumentError(error_message.str());
  }

  for (int i = 0; i < number_of_serial_circuits; ++i) {
    m_magnetic_configuration.mutable_serial_circuits(i)->set_current(
        circuit_currents[i]);
  }

  return absl::OkStatus();
}  // SetCircuitCurrents

absl::Status NumWindingsToCircuitCurrents(
    MagneticConfiguration& m_magnetic_configuration) {
  const int num_serial_circuits =
      m_magnetic_configuration.serial_circuits_size();
  for (int idx_circuit = 0; idx_circuit < num_serial_circuits; ++idx_circuit) {
    SerialCircuit* m_serial_circuit =
        m_magnetic_configuration.mutable_serial_circuits(idx_circuit);
    const int num_coils = m_serial_circuit->coils_size();

    // step 1: determine unique number of windings in all coils (and error out
    // if not all num_windings are the same)
    double unique_num_windings = 0.0;
    // This contains the sign of the number of windings of each circuit
    // with respect to the first circuit.
    // The first element is thus always expected to be 1.
    // It is used in stellarator-symmetrically-flipped coils,
    // where the order of the points along the coil stayed the same
    // and the stellarator-symmetric reversal of the poloidal coordinate
    // was incorporated by reversing the sign of the number of windings instead.
    std::vector<int> num_windings_signs(num_coils);
    for (int idx_coil = 0; idx_coil < num_coils; ++idx_coil) {
      const Coil& coil = m_serial_circuit->coils(idx_coil);
      if (idx_coil == 0) {
        unique_num_windings = coil.num_windings();
      } else if (std::abs(coil.num_windings()) !=
                 std::abs(unique_num_windings)) {
        return absl::InvalidArgumentError(absl::StrCat(
            "not all num_windings are |equal| in coil: |", coil.num_windings(),
            "| =!= |", unique_num_windings, "|"));
      }
      num_windings_signs[idx_coil] =
          coil.num_windings() * unique_num_windings < 0 ? -1 : 1;
    }

    // step 2: migrate num_windings into circuit current
    const double current_times_num_windings =
        m_serial_circuit->current() * unique_num_windings;
    m_serial_circuit->set_current(current_times_num_windings);

    // step 3: set num_windings to +1 or -1 in all coils
    for (int idx_coil = 0; idx_coil < num_coils; ++idx_coil) {
      Coil* m_coil = m_serial_circuit->mutable_coils(idx_coil);
      m_coil->set_num_windings(num_windings_signs[idx_coil]);
    }
  }

  return absl::OkStatus();
}  // NumWindingsToCircuitCurrents

// ------------------

absl::Status MoveRadially(double radial_step,
                          CircularFilament& m_circular_filament) {
  // check that center is on origin in x and y
  const Vector3d& center = m_circular_filament.center();
  if (center.x() != 0.0 || center.y() != 0.0) {
    return absl::InvalidArgumentError(
        "center has to be on origin in x and y to perform radial movement");
  }

  const Vector3d& normal = m_circular_filament.normal();
  if (normal.x() != 0.0 || normal.y() != 0.0 || normal.z() == 0.0) {
    return absl::InvalidArgumentError(
        "normal has to be along z axis to perform radial movement");
  }

  m_circular_filament.set_radius(m_circular_filament.radius() + radial_step);

  return absl::OkStatus();
}  // MoveRadially

absl::Status MoveRadially(double radial_step,
                          PolygonFilament& m_polygon_filament) {
  int num_vertices = m_polygon_filament.vertices_size();
  for (int i = 0; i < num_vertices; ++i) {
    Vector3d* vertex = m_polygon_filament.mutable_vertices(i);
    const double r =
        std::sqrt(vertex->x() * vertex->x() + vertex->y() * vertex->y());
    const double phi = std::atan2(vertex->y(), vertex->x());
    vertex->set_x((r + radial_step) * std::cos(phi));
    vertex->set_y((r + radial_step) * std::sin(phi));
    // z is unchanged, since radial movement happens only in the x-y plane
  }
  return absl::OkStatus();
}  // MoveRadially

absl::Status MoveRadially(double radial_step,
                          MagneticConfiguration& m_magnetic_configuration) {
  const int num_serial_circuits =
      m_magnetic_configuration.serial_circuits_size();
  for (int idx_circuit = 0; idx_circuit < num_serial_circuits; ++idx_circuit) {
    SerialCircuit* m_serial_circuit =
        m_magnetic_configuration.mutable_serial_circuits(idx_circuit);
    const int num_coils = m_serial_circuit->coils_size();
    for (int idx_coil = 0; idx_coil < num_coils; ++idx_coil) {
      Coil* m_coil = m_serial_circuit->mutable_coils(idx_coil);
      const int num_current_carriers = m_coil->current_carriers_size();
      for (int idx_current_carrier = 0;
           idx_current_carrier < num_current_carriers; ++idx_current_carrier) {
        CurrentCarrier* m_current_carrier =
            m_coil->mutable_current_carriers(idx_current_carrier);
        switch (m_current_carrier->type_case()) {
          case CurrentCarrier::TypeCase::kInfiniteStraightFilament:
            return absl::InvalidArgumentError(
                "Cannot perform radial movement if an InfiniteStraightFilament "
                "is present in the MagneticConfiguration");
          case CurrentCarrier::TypeCase::kCircularFilament:
            CHECK_OK(MoveRadially(
                radial_step,
                *(m_current_carrier->mutable_circular_filament())));
            break;
          case CurrentCarrier::TypeCase::kPolygonFilament:
            CHECK_OK(MoveRadially(
                radial_step, *(m_current_carrier->mutable_polygon_filament())));
            break;
          case CurrentCarrier::TypeCase::kFourierFilament:
            return absl::InvalidArgumentError(
                "Cannot perform radial movement if an FourierFilament is "
                "present in the MagneticConfiguration");
          case CurrentCarrier::TypeCase::kTypeNotSet:
            // consider as empty CurrentCarrier -> ignore
            break;
          default:
            std::stringstream error_message;
            error_message << "current carrier type ";
            error_message << m_current_carrier->type_case();
            error_message << " not implemented yet.";
            LOG(FATAL) << error_message.str();
        }
      }  // CurrentCarrier
    }  // Coil
  }  // SerialCircuit

  return absl::OkStatus();
}  // MoveRadially

// ------------------

std::string CurrentCarrierIdentifier(
    const InfiniteStraightFilament& infinite_straight_filament) {
  std::stringstream current_carrier_identifier;
  current_carrier_identifier << "InfiniteStraightFilament";
  if (infinite_straight_filament.has_name()) {
    current_carrier_identifier << " " << infinite_straight_filament.name();
  }
  return current_carrier_identifier.str();
}

std::string CurrentCarrierIdentifier(
    const CircularFilament& circular_filament) {
  std::stringstream current_carrier_identifier;
  current_carrier_identifier << "CircularFilament";
  if (circular_filament.has_name()) {
    current_carrier_identifier << " " << circular_filament.name();
  }
  return current_carrier_identifier.str();
}

std::string CurrentCarrierIdentifier(const PolygonFilament& polygon_filament) {
  std::stringstream current_carrier_identifier;
  current_carrier_identifier << "PolygonFilament";
  if (polygon_filament.has_name()) {
    current_carrier_identifier << " " << polygon_filament.name();
  }
  return current_carrier_identifier.str();
}

absl::Status IsInfiniteStraightFilamentFullyPopulated(
    const InfiniteStraightFilament& infinite_straight_filament) {
  if (!infinite_straight_filament.has_origin()) {
    std::stringstream error_message;
    error_message << CurrentCarrierIdentifier(infinite_straight_filament);
    error_message << " has no origin.";
    return absl::NotFoundError(error_message.str());
  } else {
    // has origin, now check that all components are set
    const Vector3d& origin = infinite_straight_filament.origin();
    absl::Status status = IsVector3dFullyPopulated(
        origin, absl::StrCat("origin of ", CurrentCarrierIdentifier(
                                               infinite_straight_filament)));
    if (!status.ok()) {
      return status;
    }
  }

  if (!infinite_straight_filament.has_direction()) {
    std::stringstream error_message;
    error_message << CurrentCarrierIdentifier(infinite_straight_filament);
    error_message << " has no direction.";
    return absl::NotFoundError(error_message.str());
  } else {
    // has direction, now check that all components are set
    const Vector3d& direction = infinite_straight_filament.direction();
    absl::Status status = IsVector3dFullyPopulated(
        direction,
        absl::StrCat("direction of ",
                     CurrentCarrierIdentifier(infinite_straight_filament)));
    if (!status.ok()) {
      return status;
    }
  }

  return absl::OkStatus();
}  // IsInfiniteStraightFilamentFullyPopulated

absl::Status IsCircularFilamentFullyPopulated(
    const CircularFilament& circular_filament) {
  if (!circular_filament.has_center()) {
    std::stringstream error_message;
    error_message << CurrentCarrierIdentifier(circular_filament);
    error_message << " has no center.";
    return absl::NotFoundError(error_message.str());
  } else {
    // has center, now check that all components are set
    const Vector3d& center = circular_filament.center();
    absl::Status status = IsVector3dFullyPopulated(
        center, absl::StrCat("center of ",
                             CurrentCarrierIdentifier(circular_filament)));
    if (!status.ok()) {
      return status;
    }
  }

  if (!circular_filament.has_normal()) {
    std::stringstream error_message;
    error_message << CurrentCarrierIdentifier(circular_filament);
    error_message << " has no normal.";
    return absl::NotFoundError(error_message.str());
  } else {
    // has normal, now check that all components are set
    const Vector3d& normal = circular_filament.normal();
    absl::Status status = IsVector3dFullyPopulated(
        normal, absl::StrCat("normal of ",
                             CurrentCarrierIdentifier(circular_filament)));
    if (!status.ok()) {
      return status;
    }
  }

  if (!circular_filament.has_radius()) {
    std::stringstream error_message;
    error_message << CurrentCarrierIdentifier(circular_filament);
    error_message << " has no radius.";
    return absl::NotFoundError(error_message.str());
  }

  return absl::OkStatus();
}  // IsCircularFilamentFullyPopulated

absl::Status IsPolygonFilamentFullyPopulated(
    const PolygonFilament& polygon_filament) {
  if (polygon_filament.vertices_size() < 2) {
    std::stringstream error_message;
    error_message << CurrentCarrierIdentifier(polygon_filament);
    error_message << " has too few vertices ("
                  << polygon_filament.vertices_size() << "); need at least 2.";
    return absl::NotFoundError(error_message.str());
  }

  for (int i = 0; i < polygon_filament.vertices_size(); ++i) {
    const Vector3d& vertex = polygon_filament.vertices(i);
    std::stringstream vertex_identifier;
    vertex_identifier << "vertex[" << i << "]";
    absl::Status status = IsVector3dFullyPopulated(
        vertex, absl::StrCat(vertex_identifier.str(), " of ",
                             CurrentCarrierIdentifier(polygon_filament)));
    if (!status.ok()) {
      return status;
    }
  }

  return absl::OkStatus();
}  // IsPolygonFilamentFullyPopulated

absl::Status IsMagneticConfigurationFullyPopulated(
    const MagneticConfiguration& magnetic_configuration) {
  for (const SerialCircuit& serial_circuit :
       magnetic_configuration.serial_circuits()) {
    for (const Coil& coil : serial_circuit.coils()) {
      for (const CurrentCarrier& current_carrier : coil.current_carriers()) {
        absl::Status status;
        switch (current_carrier.type_case()) {
          case CurrentCarrier::TypeCase::kInfiniteStraightFilament:
            status = IsInfiniteStraightFilamentFullyPopulated(
                current_carrier.infinite_straight_filament());
            break;
          case CurrentCarrier::TypeCase::kCircularFilament:
            status = IsCircularFilamentFullyPopulated(
                current_carrier.circular_filament());
            break;
          case CurrentCarrier::TypeCase::kPolygonFilament:
            status = IsPolygonFilamentFullyPopulated(
                current_carrier.polygon_filament());
            break;
          case CurrentCarrier::TypeCase::kTypeNotSet:
            // consider as empty CurrentCarrier -> ignore
            break;
          default:
            std::stringstream error_message;
            error_message << "current carrier type ";
            error_message << current_carrier.type_case();
            error_message << " not implemented yet.";
            status = absl::UnimplementedError(error_message.str());
        }

        if (!status.ok()) {
          return status;
        }
      }  // CurrentCarrier
    }  // Coil
  }  // SerialCircuit

  return absl::OkStatus();
}  // IsMagneticConfigurationFullyPopulated

void PrintInfiniteStraightFilament(
    const InfiniteStraightFilament& infinite_straight_filament,
    int indentation) {
  std::string prefix;
  for (int i = 0; i < indentation; ++i) {
    prefix += " ";
  }

  std::cout << prefix << "InfiniteStraightFilament {" << '\n';

  if (infinite_straight_filament.has_name()) {
    std::cout << prefix << "  name: '" << infinite_straight_filament.name()
              << "'" << '\n';
  } else {
    std::cout << prefix << "  name: none" << '\n';
  }

  if (infinite_straight_filament.has_origin()) {
    const Vector3d& origin = infinite_straight_filament.origin();
    std::cout << prefix << "  origin: [" << origin.x() << ", " << origin.y()
              << ", " << origin.z() << "]" << '\n';
  } else {
    std::cout << prefix << "  origin: none" << '\n';
  }

  if (infinite_straight_filament.has_direction()) {
    const Vector3d& direction = infinite_straight_filament.direction();
    std::cout << prefix << "  direction: [" << direction.x() << ", "
              << direction.y() << ", " << direction.z() << "]" << '\n';
  } else {
    std::cout << prefix << "  direction: none" << '\n';
  }

  std::cout << prefix << "}" << '\n';
}  // PrintInfiniteStraightFilament

void PrintCircularFilament(const CircularFilament& circular_filament,
                           int indentation) {
  std::string prefix;
  for (int i = 0; i < indentation; ++i) {
    prefix += " ";
  }

  std::cout << prefix << "CircularFilament {" << '\n';

  if (circular_filament.has_name()) {
    std::cout << prefix << "  name: '" << circular_filament.name() << "'"
              << '\n';
  } else {
    std::cout << prefix << "  name: none" << '\n';
  }

  if (circular_filament.has_center()) {
    const Vector3d& center = circular_filament.center();
    std::cout << prefix << "  center: [" << center.x() << ", " << center.y()
              << ", " << center.z() << "]" << '\n';
  } else {
    std::cout << prefix << "  center: none" << '\n';
  }

  if (circular_filament.has_normal()) {
    const Vector3d& normal = circular_filament.normal();
    std::cout << prefix << "  normal: [" << normal.x() << ", " << normal.y()
              << ", " << normal.z() << "]" << '\n';
  } else {
    std::cout << prefix << "  normal: none" << '\n';
  }

  if (circular_filament.has_radius()) {
    const double radius = circular_filament.radius();
    std::cout << prefix << "  radius: " << radius << '\n';
  } else {
    std::cout << prefix << "  radius: none" << '\n';
  }

  std::cout << prefix << "}" << '\n';
}  // PrintCircularFilament

void PrintPolygonFilament(const PolygonFilament& polygon_filament,
                          int indentation) {
  std::string prefix;
  for (int i = 0; i < indentation; ++i) {
    prefix += " ";
  }

  std::cout << prefix << "PolygonFilament {" << '\n';

  if (polygon_filament.has_name()) {
    std::cout << prefix << "  name: '" << polygon_filament.name() << "'"
              << '\n';
  } else {
    std::cout << prefix << "  name: none" << '\n';
  }

  if (polygon_filament.vertices_size() > 0) {
    std::cout << prefix << "  vertices: [" << polygon_filament.vertices_size()
              << "]" << '\n';
  } else {
    std::cout << prefix << "  vertices: none" << '\n';
  }

  std::cout << prefix << "}" << '\n';
}  // PrintPolygonFilament

void PrintCurrentCarrier(const CurrentCarrier& current_carrier,
                         int indentation) {
  std::string prefix;
  for (int i = 0; i < indentation; ++i) {
    prefix += " ";
  }

  std::cout << prefix << "CurrentCarrier {" << '\n';

  switch (current_carrier.type_case()) {
    case CurrentCarrier::TypeCase::kInfiniteStraightFilament:
      PrintInfiniteStraightFilament(
          current_carrier.infinite_straight_filament(), indentation + 2);
      break;
    case CurrentCarrier::TypeCase::kCircularFilament:
      PrintCircularFilament(current_carrier.circular_filament(),
                            indentation + 2);
      break;
    case CurrentCarrier::TypeCase::kPolygonFilament:
      PrintPolygonFilament(current_carrier.polygon_filament(), indentation + 2);
      break;
    case CurrentCarrier::TypeCase::kTypeNotSet:
      // consider as empty CurrentCarrier -> ignore
      break;
    default:
      std::stringstream error_message;
      error_message << "current carrier type ";
      error_message << current_carrier.type_case();
      error_message << " not implemented yet.";
      LOG(FATAL) << error_message.str();
  }

  std::cout << prefix << "}" << '\n';
}  // PrintCurrentCarrier

void PrintCoil(const Coil& coil, int indentation) {
  std::string prefix;
  for (int i = 0; i < indentation; ++i) {
    prefix += " ";
  }

  std::cout << prefix << "Coil {" << '\n';

  if (coil.has_name()) {
    std::cout << prefix << "  name: '" << coil.name() << "'" << '\n';
  } else {
    std::cout << prefix << "  name: none" << '\n';
  }

  if (coil.has_num_windings()) {
    std::cout << prefix << "  num_windings: " << coil.num_windings() << '\n';
  } else {
    std::cout << prefix << "  num_windings: none" << '\n';
  }

  for (const CurrentCarrier& current_carrier : coil.current_carriers()) {
    PrintCurrentCarrier(current_carrier, indentation + 2);
  }

  std::cout << prefix << "}" << '\n';
}  // PrintCoil

void PrintSerialCircuit(const SerialCircuit& serial_circuit, int indentation) {
  std::string prefix;
  for (int i = 0; i < indentation; ++i) {
    prefix += " ";
  }

  std::cout << prefix << "SerialCircuit {" << '\n';

  if (serial_circuit.has_name()) {
    std::cout << prefix << "  name: '" << serial_circuit.name() << "'" << '\n';
  } else {
    std::cout << prefix << "  name: none" << '\n';
  }

  if (serial_circuit.has_current()) {
    std::cout << prefix << "  current: " << serial_circuit.current() << '\n';
  } else {
    std::cout << prefix << "  current: none" << '\n';
  }

  for (const Coil& coil : serial_circuit.coils()) {
    PrintCoil(coil, indentation + 2);
  }

  std::cout << prefix << "}" << '\n';
}  // PrintSerialCircuit

void PrintMagneticConfiguration(
    const MagneticConfiguration& magnetic_configuration, int indentation) {
  std::string prefix;
  for (int i = 0; i < indentation; ++i) {
    prefix += " ";
  }

  std::cout << prefix << "MagneticConfiguration {" << '\n';

  if (magnetic_configuration.has_name()) {
    std::cout << prefix << "  name: '" << magnetic_configuration.name() << "'"
              << '\n';
  } else {
    std::cout << prefix << "  name: none" << '\n';
  }

  if (magnetic_configuration.has_num_field_periods()) {
    std::cout << prefix << "  num_field_periods: "
              << magnetic_configuration.num_field_periods() << '\n';
  } else {
    std::cout << prefix << "  num_field_periods: none" << '\n';
  }

  for (const SerialCircuit& serial_circuit :
       magnetic_configuration.serial_circuits()) {
    PrintSerialCircuit(serial_circuit, indentation + 2);
  }

  std::cout << prefix << "}" << '\n';
}  // PrintMagneticConfiguration

}  // namespace magnetics
