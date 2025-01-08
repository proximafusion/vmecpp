#ifndef VMECPP_COMMON_MAGNETIC_CONFIGURATION_DEFINITION_MAGNETIC_CONFIGURATION_H_
#define VMECPP_COMMON_MAGNETIC_CONFIGURATION_DEFINITION_MAGNETIC_CONFIGURATION_H_

#include <string>
#include <list>

// FIXME(jons): to be replaced in the end
#include "vmecpp/common/magnetic_configuration_definition/magnetic_configuration.pb.h"

namespace magnetics {

struct Coil {
  // a human-readable name, e.g., for plotting
  bool has_name_ = false;
  std::string name_;

  // number of windings == multiplier for current along geometry;
  // num_windings == 1 is assumed if this field is not populated
  bool has_num_windings_ = false;
  double num_windings_ = 0.0;

  // objects that define the single-turn geometry of the coil
  std::list<CurrentCarrier> current_carriers_;

  // name
  bool has_name() const { return has_name_; }
  const std::string& name() const { return name_; }
  void set_name(const std::string& value) {
    name_ = value;
    has_name_ = true;
  }
  void set_name(std::string&& value) {
    name_ = std::move(value);
    has_name_ = true;
  }
  void clear_name() {
    name_.clear();
    has_name_ = false;
  }

  // num_windings
  bool has_num_windings() const { return has_num_windings_; }
  double num_windings() const { return num_windings_; }
  void set_num_windings(double value) {
    num_windings_ = value;
    has_num_windings_ = true;
  }
  void clear_num_windings() {
    num_windings_ = 0.0;
    has_num_windings_ = false;
  }

  // current_carriers
  int current_carriers_size() const {
    return static_cast<int>(current_carriers_.size());
  }
  const CurrentCarrier& current_carriers(int index) const {
    auto it = current_carriers_.cbegin();
    // No bounds checks here for brevity
    std::advance(it, index);
    return *it;
  }
  CurrentCarrier* mutable_current_carriers(int index) {
    auto it = current_carriers_.begin();
    // No bounds checks here for brevity
    std::advance(it, index);
    return &(*it);
  }
  CurrentCarrier* add_current_carriers() {
    current_carriers_.emplace_back();
    auto it = current_carriers_.end();
    // Move to newly added element
    --it;
    return &(*it);
  }
  const std::list<CurrentCarrier>& current_carriers() const {
    return current_carriers_;
  }
  std::list<CurrentCarrier>* mutable_current_carriers() {
    return &current_carriers_;
  }
  void clear_current_carriers() {
    current_carriers_.clear();
  }

  // Clear the entire structure
  void Clear() {
    clear_name();
    clear_num_windings();
    clear_current_carriers();
  }
}; // Coil

struct SerialCircuit {
  // a human-readable name, e.g., for plotting
  bool has_name_ = false;
  std::string name_;

  // current along each of the current carriers
  bool has_current_ = false;
  double current_ = 0.0;

  // objects that define the geometry of coils
  std::list<Coil> coils_;

  // name
  bool has_name() const { return has_name_; }
  const std::string& name() const { return name_; }
  void set_name(const std::string& value) {
    name_ = value;
    has_name_ = true;
  }
  void set_name(std::string&& value) {
    name_ = std::move(value);
    has_name_ = true;
  }
  void clear_name() {
    name_.clear();
    has_name_ = false;
  }

  // current
  bool has_current() const { return has_current_; }
  double current() const { return current_; }
  void set_current(double value) {
    current_ = value;
    has_current_ = true;
  }
  void clear_current() {
    current_ = 0.0;
    has_current_ = false;
  }

  // coils
  int coils_size() const {
    return static_cast<int>(coils_.size());
  }
  const Coil& coils(int index) const {
    auto it = coils_.cbegin();
    // No bounds checks here for brevity
    std::advance(it, index);
    return *it;
  }
  Coil* mutable_coils(int index) {
    auto it = coils_.begin();
    // No bounds checks here for brevity
    std::advance(it, index);
    return &(*it);
  }
  Coil* add_coils() {
    coils_.emplace_back();
    auto it = coils_.end();
    --it;
    return &(*it);
  }
  const std::list<Coil>& coils() const {
    return coils_;
  }
  std::list<Coil>* mutable_coils() {
    return &coils_;
  }
  void clear_coils() {
    coils_.clear();
  }

  // Clear the entire structure
  void Clear() {
    clear_name();
    clear_current();
    clear_coils();
  }
}; // SerialCircuit

struct MagneticConfiguration {
  // a human-readable name, e.g., for plotting
  bool has_name_ = false;
  std::string name_;

  // number of field periods of this coil set
  bool has_num_field_periods_ = false;
  int num_field_periods_ = 0;

  // objects that specify geometry and currents of coils
  std::list<SerialCircuit> serial_circuits_;

  // name
  bool has_name() const { return has_name_; }
  const std::string& name() const { return name_; }
  void set_name(const std::string& value) {
    name_ = value;
    has_name_ = true;
  }
  void set_name(std::string&& value) {
    name_ = std::move(value);
    has_name_ = true;
  }
  void clear_name() {
    name_.clear();
    has_name_ = false;
  }

  // num_field_periods
  bool has_num_field_periods() const { return has_num_field_periods_; }
  int num_field_periods() const { return num_field_periods_; }
  void set_num_field_periods(int value) {
    num_field_periods_ = value;
    has_num_field_periods_ = true;
  }
  void clear_num_field_periods() {
    num_field_periods_ = 0;
    has_num_field_periods_ = false;
  }

  // serial_circuits
  int serial_circuits_size() const {
    return static_cast<int>(serial_circuits_.size());
  }
  const SerialCircuit& serial_circuits(int index) const {
    auto it = serial_circuits_.cbegin();
    // No bounds checks here for brevity
    std::advance(it, index);
    return *it;
  }
  SerialCircuit* mutable_serial_circuits(int index) {
    auto it = serial_circuits_.begin();
    // No bounds checks here for brevity
    std::advance(it, index);
    return &(*it);
  }
  SerialCircuit* add_serial_circuits() {
    serial_circuits_.emplace_back();
    auto it = serial_circuits_.end();
    --it;
    return &(*it);
  }
  const std::list<SerialCircuit>& serial_circuits() const {
    return serial_circuits_;
  }
  std::list<SerialCircuit>* mutable_serial_circuits() {
    return &serial_circuits_;
  }
  void clear_serial_circuits() {
    serial_circuits_.clear();
  }

  // Clear the entire structure (all fields)
  void Clear() {
    clear_name();
    clear_num_field_periods();
    clear_serial_circuits();
  }
}; // MagneticConfiguration

} // namespace magnetics

#endif // VMECPP_COMMON_MAGNETIC_CONFIGURATION_DEFINITION_MAGNETIC_CONFIGURATION_H_
