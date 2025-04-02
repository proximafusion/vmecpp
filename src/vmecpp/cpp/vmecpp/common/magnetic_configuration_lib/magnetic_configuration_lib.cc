// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/common/magnetic_configuration_lib/magnetic_configuration_lib.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iostream>
#include <locale>
#include <optional> // Ensure optional is included where used
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

  template<typename T>
  std::ostream& operator<<(std::ostream& os, std::optional<T> const& opt)
  {
      return opt ? os << opt.value() : os;
  }

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
        SerialCircuit* serial_circuit{};
        if (index_of_circuit_id != coil_ids.end()) {
          const int serial_circuit_index =
              static_cast<int>(index_of_circuit_id - coil_ids.begin());
          serial_circuit = m_magnetic_configuration.mutable_serial_circuits(
              serial_circuit_index);
        } else {
          serial_circuit = m_magnetic_configuration.add_serial_circuits();
          // Set default current if creating a new circuit
          serial_circuit->current_ = 1.0;
          coil_ids.push_back(serial_circuit_id);
        }

        Coil* coil = serial_circuit->add_coils();
        coil->num_windings_ = w.at(0);

        // Use CurrentCarrier methods to manage the union
        CurrentCarrier* current_carrier = coil->add_current_carriers();

        if (x.size() == 1) {
          // circular filament: use mutable accessor to get/create and then set fields directly
          CircularFilament* circular_filament = current_carrier->mutable_circular_filament();
          circular_filament->name_ = current_carrier_name;
          circular_filament->radius_ = x.at(0);

          // Ensure center and normal exist before setting components
          circular_filament->center_.emplace(); // Creates Vector3d if std::nullopt
          circular_filament->center_->set_x(0.0);
          circular_filament->center_->set_y(0.0);
          circular_filament->center_->set_z(z.at(0));

          circular_filament->normal_.emplace(); // Creates Vector3d if std::nullopt
          circular_filament->normal_->set_x(0.0);
          circular_filament->normal_->set_y(0.0);
          circular_filament->normal_->set_z(1.0);

        } else {
          // polygon filament
          // Check num_windings consistency
          if (!coil->num_windings_) {
             return absl::InternalError("Coil num_windings should have been set but wasn't.");
          }
          double coil_num_windings = *coil->num_windings_; // Dereference optional

          for (size_t i = 1; i < w.size() - 1; ++i) {
            if (w.at(i) != coil_num_windings) {
              std::stringstream error_message;
              error_message << "Number of windings mismatch at point " << i
                            << " (" << w.at(i) << " vs " << coil_num_windings << ") in coil ending with name '"
                            << current_carrier_name << "'";
              return absl::InvalidArgumentError(error_message.str());
            }
          }

          // Use mutable accessor to get/create, then set fields directly
          PolygonFilament* polygon_filament = current_carrier->mutable_polygon_filament();
          polygon_filament->name_ = current_carrier_name;

          // Add vertices using the helper method
          for (size_t i = 0; i < x.size(); ++i) {
            Vector3d* vertex = polygon_filament->add_vertices(); // Use helper
            vertex->set_x(x.at(i));
            vertex->set_y(y.at(i));
            vertex->set_z(z.at(i));
          }
        }

        // Clear temporary vectors for the next coil/filament
        x.clear();
        y.clear();
        z.clear();
        w.clear();
      }
    } else {
      std::stringstream error_message;
      error_message << "Cannot parse line: '" << stripped_line << "': has "
                    << num_line_parts << " parts, expected 4 or 6";
      return absl::InvalidArgumentError(error_message.str());
    }
  }
    // If the loop finishes without finding "end", it's an error
   return absl::ResourceExhaustedError(
      "Reached end of stream without finding 'end' marker in filament data.");
}  // ParseCurrentCarriers

absl::StatusOr<MagneticConfiguration> ImportMagneticConfigurationFromMakegrid(
    const std::string& makegrid_coils) {
  MagneticConfiguration magnetic_configuration;
  std::stringstream makegrid_coils_ss(makegrid_coils);
  bool in_filament_section = false; // Flag to track if we are parsing filaments

  for (std::string raw_line; std::getline(makegrid_coils_ss, raw_line);) {
    absl::string_view stripped_line = absl::StripAsciiWhitespace(raw_line);

    if (stripped_line.empty() || absl::StartsWith(stripped_line, "#")) {
        continue; // Skip empty/comment lines
    }

    if (!in_filament_section) {
        if (absl::StartsWith(stripped_line, "periods")) {
          std::vector<std::string> line_parts = absl::StrSplit(
              stripped_line, absl::ByAnyChar(" \t"), absl::SkipWhitespace());
          if (line_parts.size() != 2) {
             return absl::InvalidArgumentError(absl::StrCat(
                "Expected 2 parts for 'periods' line, found ", line_parts.size(),
                " in '", stripped_line, "'"));
          }
          try {
            // Use direct access to set the optional member
            magnetic_configuration.num_field_periods_ = std::stoi(line_parts[1]);
          } catch (const std::exception& e) {
             return absl::InvalidArgumentError(absl::StrCat(
                "Invalid number format for periods '", line_parts[1], "': ", e.what()));
          }
        } else if (absl::StartsWith(stripped_line, "begin filament")) {
          in_filament_section = true;
          // Call ParseCurrentCarriers, which reads until "end"
          absl::Status status = ParseCurrentCarriers(makegrid_coils_ss, magnetic_configuration);

          in_filament_section = false; // Reset flag after parsing attempt

          if (!status.ok()) {
             magnetic_configuration.Clear(); // Clear potentially partial data
             return status; // Propagate error
          }
          // If OkStatus, ParseCurrentCarriers consumed until 'end', continue loop
        }
        // Ignore other lines outside the filament section
    }
    // If in_filament_section is true, ParseCurrentCarriers handles the lines
  }

   // Check if essential parts were found after parsing the whole string
  if (!magnetic_configuration.num_field_periods_) { // Check optional directly
      return absl::NotFoundError("MAKEGRID data did not contain 'periods' line.");
  }
   if (magnetic_configuration.serial_circuits_size() == 0) {
       // This might be valid but often indicates an issue.
       LOG(WARNING) << "MAKEGRID parsing resulted in zero serial circuits. "
                    << "Check 'begin filament' and 'end' markers.";
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

absl::StatusOr<Eigen::VectorXd> GetCircuitCurrents(
    const MagneticConfiguration& magnetic_configuration) {
  absl::Status status =
      IsMagneticConfigurationFullyPopulated(magnetic_configuration);
  if (!status.ok()) {
    return status;
  }

  const int number_of_serial_circuits =
      magnetic_configuration.serial_circuits_size();
  Eigen::VectorXd circuit_currents(number_of_serial_circuits);
  for (int i = 0; i < number_of_serial_circuits; ++i) {
    circuit_currents[i] = magnetic_configuration.serial_circuits(i).current_.value();
  }

  return circuit_currents;
}  // GetCircuitCurrents

absl::Status SetCircuitCurrents(
    const Eigen::VectorXd& circuit_currents,
    MagneticConfiguration& m_magnetic_configuration) {
  const int number_of_serial_circuits =
      m_magnetic_configuration.serial_circuits_size();
  const Eigen::VectorXd::Index number_of_circuit_currents =
      circuit_currents.size();
  if (number_of_serial_circuits != number_of_circuit_currents) {
     return absl::InvalidArgumentError(absl::StrCat(
        "Number of provided circuit currents (", number_of_circuit_currents,
        ") does not match number of SerialCircuits (", number_of_serial_circuits,
        ") in MagneticConfiguration."));
  }

  for (int i = 0; i < number_of_serial_circuits; ++i) {
    // Get mutable pointer and set optional member directly
    m_magnetic_configuration.mutable_serial_circuits(i)->current_ = circuit_currents[i];
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
        unique_num_windings = coil.num_windings_.value();
      } else if (std::abs(coil.num_windings_.value()) !=
                 std::abs(unique_num_windings)) {
            return absl::InvalidArgumentError(absl::StrCat(
            "not all num_windings are |equal| in coil: |", coil.num_windings_.value(),
            "| =!= |", unique_num_windings, "|"));
         }
      num_windings_signs[idx_coil] =
          coil.num_windings_.value() * unique_num_windings < 0 ? -1 : 1;
    }

    // step 2: migrate num_windings into circuit current
    const double current_times_num_windings =
        m_serial_circuit->current_.value() * unique_num_windings;
    m_serial_circuit->current_ = current_times_num_windings;

    // step 3: set num_windings to +1 or -1 in all coils
    for (int idx_coil = 0; idx_coil < num_coils; ++idx_coil) {
      Coil* m_coil = m_serial_circuit->mutable_coils(idx_coil);
      m_coil->num_windings_ = static_cast<double>(num_windings_signs[idx_coil]);
    }
  }

  return absl::OkStatus();
}  // NumWindingsToCircuitCurrents

// ------------------

absl::Status MoveRadially(double radial_step,
                          CircularFilament& m_circular_filament) {
  if (!m_circular_filament.center_.has_value()) {
      return absl::InvalidArgumentError("CircularFilament requires 'center' for radial movement.");
  }
   if (!m_circular_filament.normal_.has_value()) {
      return absl::InvalidArgumentError("CircularFilament requires 'normal' for radial movement.");
  }
   if (!m_circular_filament.radius_.has_value()) {
      return absl::InvalidArgumentError("CircularFilament requires 'radius' for radial movement.");
  }

  const Vector3d& center = m_circular_filament.center_.value();
  if (center.x() != 0.0 || center.y() != 0.0) {
    return absl::InvalidArgumentError(
        "Center must be on the z-axis (x=0, y=0) for this radial movement");
  }

  const Vector3d& normal = *m_circular_filament.normal_;
   // Check normal alignment with z-axis (allow positive or negative)
  if (normal.x() != 0.0 || normal.y() != 0.0 || std::abs(normal.z()) < 1e-9) {
     return absl::InvalidArgumentError(
        "Normal must be aligned with the z-axis (x=0, y=0) for this radial movement");
  }

  // Modify the radius directly (dereference, modify, assign back)
  *m_circular_filament.radius_ += radial_step;
   // Optionally, check if the new radius is non-negative
   if (*m_circular_filament.radius_ < 0.0) {
       // Revert change and return error? Or clamp? Let's error.
       *m_circular_filament.radius_ -= radial_step; // Revert
       return absl::InvalidArgumentError("Radial step results in negative radius.");
   }


  return absl::OkStatus();
}  // MoveRadially

absl::Status MoveRadially(double radial_step,
                          PolygonFilament& m_polygon_filament) {
  // Iterate through the list directly or using helpers
  for (int i = 0; i < m_polygon_filament.vertices_size(); ++i) {
    Vector3d* vertex = m_polygon_filament.mutable_vertices(i); // Use helper
    if (!vertex) continue; // Should not happen if index is valid

    const double r = std::sqrt(vertex->x() * vertex->x() + vertex->y() * vertex->y());
    double new_r = r + radial_step;

     if (new_r < 0) {
         return absl::InvalidArgumentError(absl::StrCat(
            "Radial step results in negative radius for vertex ", i));
     }

     if (r > 1e-9) { // Check if vertex is not effectively at the origin
         const double phi = std::atan2(vertex->y(), vertex->x());
         vertex->set_x(new_r * std::cos(phi));
         vertex->set_y(new_r * std::sin(phi));
     } else {
         // Vertex starts at R=0. Leave it there as direction is undefined.
         vertex->set_x(0.0);
         vertex->set_y(0.0);
     }
    // z is unchanged
  }
  return absl::OkStatus();
}  // MoveRadially

absl::Status MoveRadially(double radial_step,
                          MagneticConfiguration& m_magnetic_configuration) {
  // Iterate through the structure using helpers or direct list access
  for (int idx_circuit = 0; idx_circuit < m_magnetic_configuration.serial_circuits_size(); ++idx_circuit) {
    SerialCircuit* m_serial_circuit = m_magnetic_configuration.mutable_serial_circuits(idx_circuit);
    for (int idx_coil = 0; idx_coil < m_serial_circuit->coils_size(); ++idx_coil) {
      Coil* m_coil = m_serial_circuit->mutable_coils(idx_coil);
      for (int idx_current_carrier = 0; idx_current_carrier < m_coil->current_carriers_size(); ++idx_current_carrier) {
        CurrentCarrier* m_current_carrier = m_coil->mutable_current_carriers(idx_current_carrier);
        absl::Status status = absl::OkStatus();

        // Use CurrentCarrier methods to check type and get mutable pointer
        switch (m_current_carrier->type_case()) {
          case CurrentCarrier::TypeCase::kInfiniteStraightFilament:
             return absl::InvalidArgumentError(
                "Cannot perform radial movement on an InfiniteStraightFilament.");
          case CurrentCarrier::TypeCase::kCircularFilament:
             // Pass the dereferenced object obtained via mutable accessor
             status = MoveRadially(radial_step, *(m_current_carrier->mutable_circular_filament()));
             if (!status.ok()) return status; // Propagate error
             break;
          case CurrentCarrier::TypeCase::kPolygonFilament:
             status = MoveRadially(radial_step, *(m_current_carrier->mutable_polygon_filament()));
             if (!status.ok()) return status; // Propagate error
             break;
          case CurrentCarrier::TypeCase::kFourierFilament:
            return absl::InvalidArgumentError(
                "Cannot perform radial movement if an FourierFilament is "
                "present in the MagneticConfiguration");
          case CurrentCarrier::TypeCase::kTypeNotSet:
            // consider as empty CurrentCarrier -> ignore
            break;
          default:
            return absl::InternalError(absl::StrCat(
                "Unsupported current carrier type encountered during radial move: ",
                m_current_carrier->type_case()));
        }
      }  // CurrentCarrier
    }  // Coil
  }  // SerialCircuit

  return absl::OkStatus();
}  // MoveRadially


// ------------------
// Helper functions to create identifiers (use direct optional check and dereference)

std::string CurrentCarrierIdentifier(const InfiniteStraightFilament& inf_filament) {
  std::stringstream ss;
  ss << "InfiniteStraightFilament";
  if (inf_filament.name_) { // Check optional directly
    ss << " (name: '" << *inf_filament.name_ << "')"; // Dereference
  }
  return ss.str();
}

std::string CurrentCarrierIdentifier(const CircularFilament& circ_filament) {
  std::stringstream ss;
  ss << "CircularFilament";
  if (circ_filament.name_) {
    ss << " (name: '" << *circ_filament.name_ << "')";
  }
  return ss.str();
}

std::string CurrentCarrierIdentifier(const PolygonFilament& poly_filament) {
   std::stringstream ss;
   ss << "PolygonFilament";
  if (poly_filament.name_) {
    ss << " (name: '" << *poly_filament.name_ << "')";
  }
  return ss.str();
}


// --- Fully Populated Checks (use direct optional check and dereference) ---

absl::Status IsInfiniteStraightFilamentFullyPopulated(
    const InfiniteStraightFilament& inf_filament) {
  // Check optional directly
  if (!inf_filament.origin_) {
    return absl::FailedPreconditionError(absl::StrCat(
        CurrentCarrierIdentifier(inf_filament), " requires 'origin' to be set."));
  } else {
    // Check the Vector3d components after ensuring origin exists
    absl::Status status = IsVector3dFullyPopulated(
        *inf_filament.origin_, // Dereference
        absl::StrCat("origin of ", CurrentCarrierIdentifier(inf_filament)));
    if (!status.ok()) return status;
  }

  if (!inf_filament.direction_) {
     return absl::FailedPreconditionError(absl::StrCat(
        CurrentCarrierIdentifier(inf_filament), " requires 'direction' to be set."));
  } else {
    absl::Status status = IsVector3dFullyPopulated(
        *inf_filament.direction_, // Dereference
        absl::StrCat("direction of ", CurrentCarrierIdentifier(inf_filament)));
    if (!status.ok()) return status;
    // Optional: Check if direction is non-zero
     const Vector3d& dir = *inf_filament.direction_;
     if (std::abs(dir.x()) < 1e-9 && std::abs(dir.y()) < 1e-9 && std::abs(dir.z()) < 1e-9) {
        return absl::InvalidArgumentError(absl::StrCat(
            CurrentCarrierIdentifier(inf_filament), " has zero direction vector."));
     }
  }
  // Name is optional, not checked for geometric population

  return absl::OkStatus();
}

absl::Status IsCircularFilamentFullyPopulated(
    const CircularFilament& circ_filament) {
  if (!circ_filament.center_) {
      return absl::FailedPreconditionError(absl::StrCat(
        CurrentCarrierIdentifier(circ_filament), " requires 'center' to be set."));
  } else {
     absl::Status status = IsVector3dFullyPopulated(
        *circ_filament.center_,
        absl::StrCat("center of ", CurrentCarrierIdentifier(circ_filament)));
     if (!status.ok()) return status;
  }

   if (!circ_filament.normal_) {
        return absl::FailedPreconditionError(absl::StrCat(
            CurrentCarrierIdentifier(circ_filament), " requires 'normal' to be set."));
   } else {
       absl::Status status = IsVector3dFullyPopulated(
           *circ_filament.normal_,
           absl::StrCat("normal of ", CurrentCarrierIdentifier(circ_filament)));
       if (!status.ok()) return status;
       // Optional: Check if normal is non-zero
       const Vector3d& n = *circ_filament.normal_;
       if (std::abs(n.x()) < 1e-9 && std::abs(n.y()) < 1e-9 && std::abs(n.z()) < 1e-9) {
           return absl::InvalidArgumentError(absl::StrCat(
               CurrentCarrierIdentifier(circ_filament), " has zero normal vector."));
       }
   }

    if (!circ_filament.radius_) {
         return absl::FailedPreconditionError(absl::StrCat(
            CurrentCarrierIdentifier(circ_filament), " requires 'radius' to be set."));
    } else {
        // Check radius non-negative
        if (*circ_filament.radius_ < 0.0) {
           return absl::InvalidArgumentError(absl::StrCat(
               CurrentCarrierIdentifier(circ_filament),
               " has negative radius (", *circ_filament.radius_, ")."));
        }
    }
    // Name is optional

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
        vertex,
        absl::StrFormat("vertex[%d] of %s", i, CurrentCarrierIdentifier(polygon_filament)));
    if (!status.ok()) {
      return status;
    }
  }

  return absl::OkStatus();
}  // IsPolygonFilamentFullyPopulated

absl::Status IsMagneticConfigurationFullyPopulated(
    const MagneticConfiguration& magnetic_configuration) {

   // Check top-level fields directly
   if (!magnetic_configuration.num_field_periods_) {
       return absl::FailedPreconditionError("MagneticConfiguration requires 'num_field_periods'.");
   } else if (*magnetic_configuration.num_field_periods_ <= 0) {
       return absl::InvalidArgumentError(absl::StrCat(
           "MagneticConfiguration num_field_periods must be positive (is ",
           *magnetic_configuration.num_field_periods_, ")."));
   }

  // Iterate using helpers
  for (int idx_circuit = 0; idx_circuit < magnetic_configuration.serial_circuits_size(); ++idx_circuit) {
    const SerialCircuit& serial_circuit = magnetic_configuration.serial_circuits(idx_circuit);

     // Check circuit fields directly
     if (!serial_circuit.current_) {
         return absl::FailedPreconditionError(absl::StrCat(
            "SerialCircuit index ", idx_circuit,
            (serial_circuit.name_ ? absl::StrCat(" ('", *serial_circuit.name_, "')") : ""),
            " requires 'current'."));
     }

     for (int idx_coil = 0; idx_coil < serial_circuit.coils_size(); ++idx_coil) {
        const Coil& coil = serial_circuit.coils(idx_coil);

        // Check coil fields directly
         if (!coil.num_windings_) {
              return absl::FailedPreconditionError(absl::StrCat(
                 "Coil index ", idx_coil, " in Circuit index ", idx_circuit,
                 (coil.name_ ? absl::StrCat(" ('", *coil.name_, "')") : ""),
                 " requires 'num_windings'."));
         }


         for (int idx_carrier = 0; idx_carrier < coil.current_carriers_size(); ++idx_carrier) {
             const CurrentCarrier& current_carrier = coil.current_carriers(idx_carrier);
             absl::Status status = absl::OkStatus();

             // Use CurrentCarrier methods to check type and access data safely
             switch (current_carrier.type_case()) {
                 case CurrentCarrier::TypeCase::kInfiniteStraightFilament:
                    status = IsInfiniteStraightFilamentFullyPopulated(current_carrier.infinite_straight_filament());
                    break;
                 case CurrentCarrier::TypeCase::kCircularFilament:
                    status = IsCircularFilamentFullyPopulated(current_carrier.circular_filament());
                    break;
                 case CurrentCarrier::TypeCase::kPolygonFilament:
                    status = IsPolygonFilamentFullyPopulated(current_carrier.polygon_filament());
                    break;
                 case CurrentCarrier::TypeCase::kFourierFilament:
                    status = absl::UnimplementedError("Population check for FourierFilament not implemented.");
                    break;
                 case CurrentCarrier::TypeCase::kTypeNotSet:
                    status = absl::FailedPreconditionError(absl::StrCat(
                       "CurrentCarrier index ", idx_carrier, " in Coil index ", idx_coil,
                       " in Circuit index ", idx_circuit, " has kTypeNotSet."));
                    break;
                 default:
                    status = absl::InternalError(absl::StrCat(
                        "Unknown current carrier type ", current_carrier.type_case(),
                        " encountered during population check."));
             }

        if (!status.ok()) {
          return status;
        }
      }  // CurrentCarrier
    }  // Coil
  }  // SerialCircuit

  return absl::OkStatus();
}  // IsMagneticConfigurationFullyPopulated


// --- Print Functions (use direct optional check and dereference, list helpers) ---

void PrintInfiniteStraightFilament(
    const InfiniteStraightFilament& inf_filament, int indentation) {
  std::string prefix(indentation, ' ');
  std::cout << prefix << "InfiniteStraightFilament {" << '\n';

  // Use direct optional check and dereference
  std::cout << prefix << "  name: "
            << (inf_filament.name_ ? absl::StrCat("'", *inf_filament.name_, "'") : "[not set]")
            << '\n';
  if (inf_filament.origin_) {
      const Vector3d& origin = *inf_filament.origin_;
      std::cout << prefix << "  origin: [" << origin.x() << ", " << origin.y() << ", " << origin.z() << "]\n";
  } else {
       std::cout << prefix << "  origin: [not set]\n";
  }
  if (inf_filament.direction_) {
       const Vector3d& direction = *inf_filament.direction_;
       std::cout << prefix << "  direction: [" << direction.x() << ", " << direction.y() << ", " << direction.z() << "]\n";
  } else {
      std::cout << prefix << "  direction: [not set]\n";
  }
  std::cout << prefix << "}" << '\n';
}

void PrintCircularFilament(const CircularFilament& circ_filament, int indentation) {
    std::string prefix(indentation, ' ');
    std::cout << prefix << "CircularFilament {" << '\n';
    std::cout << prefix << "  name: "
              << (circ_filament.name_ ? absl::StrCat("'", *circ_filament.name_, "'") : "[not set]")
              << '\n';
    if (circ_filament.center_) {
        const Vector3d& center = *circ_filament.center_;
        std::cout << prefix << "  center: [" << center.x() << ", " << center.y() << ", " << center.z() << "]\n";
    } else {
         std::cout << prefix << "  center: [not set]\n";
    }
    if (circ_filament.normal_) {
         const Vector3d& normal = *circ_filament.normal_;
         std::cout << prefix << "  normal: [" << normal.x() << ", " << normal.y() << ", " << normal.z() << "]\n";
    } else {
        std::cout << prefix << "  normal: [not set]\n";
    }
     std::cout << prefix << "  radius: "
               << (circ_filament.radius_ ? std::to_string(*circ_filament.radius_) : "[not set]")
               << '\n';
    std::cout << prefix << "}" << '\n';
}

void PrintPolygonFilament(const PolygonFilament& polygon_filament,
                          int indentation) {
  std::string prefix;
  for (int i = 0; i < indentation; ++i) {
    prefix += " ";
  }

     std::cout << prefix << "PolygonFilament {" << '\n';

    std::cout << prefix << "  name: '" << polygon_filament.name_ << "'"
               << '\n';

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

    std::cout << prefix << "  name: '" << coil.name_ << "'" << '\n';
    std::cout << prefix << "  num_windings: " << coil.num_windings_<< '\n';
  for (const CurrentCarrier& current_carrier : coil.current_carriers_) {
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
     std::cout << prefix << "  name: "
               << serial_circuit.name_
               << '\n';
    std::cout << prefix << "  current: "
               << serial_circuit.current_
               << '\n';

  for (const Coil& coil : serial_circuit.coils_) {
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
    std::cout << prefix << "  name: '" << magnetic_configuration.name_ << "'"
             << '\n';

   std::cout << prefix << "  num_field_periods: "
              << magnetic_configuration.num_field_periods_ << '\n';

  for (const SerialCircuit& serial_circuit :
       magnetic_configuration.serial_circuits()) {
    PrintSerialCircuit(serial_circuit, indentation + 2);
   }

   std::cout << prefix << "}" << '\n';
}  // PrintMagneticConfiguration

}  // namespace magnetics
