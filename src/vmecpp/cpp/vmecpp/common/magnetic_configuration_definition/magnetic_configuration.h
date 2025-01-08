#ifndef VMECPP_COMMON_MAGNETIC_CONFIGURATION_DEFINITION_MAGNETIC_CONFIGURATION_H_
#define VMECPP_COMMON_MAGNETIC_CONFIGURATION_DEFINITION_MAGNETIC_CONFIGURATION_H_

#include <string>
#include <list>

// FIXME(jons): to be replaced in the end
#include "vmecpp/common/magnetic_configuration_definition/magnetic_configuration.pb.h"

namespace magnetics {

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
