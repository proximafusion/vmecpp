# PT_TYPE Support Implementation Summary

## Overview
PT_TYPE = 'power_series' support has been successfully added to VMEC++ for ANIMEC anisotropic pressure calculations.

## Implementation Details

### 1. C++ Core Structure (vmec_indata.h)
Added the following fields to the VmecINDATA class:
```cpp
// anisotropy parameters (ANIMEC)
double bcrit;                    // critical field strength for hot particle confinement
std::string pt_type;            // parametrization of temperature profile (TPERP/TPAR)
std::vector<double> at;         // temperature profile coefficients
std::string ph_type;            // parametrization of hot particle pressure profile
std::vector<double> ah;         // hot particle pressure profile coefficients
```

### 2. Initialization (vmec_indata.cc)
- Constructor initializes default values:
  - `bcrit = 1.0`
  - `pt_type = "power_series"`
  - `at = {1.0}` (isotropic by default)
  - `ph_type = "power_series"`
  - `ah = {}` (no hot particle pressure by default)

### 3. HDF5 Serialization (vmec_indata.cc)
- Write support: Lines 198-202
- Read support: Lines 283-287
- Handles string types and variable-length arrays

### 4. JSON Parsing (vmec_indata.cc)
- Added parsing for all PT_TYPE fields: Lines 674-688
- Supports both namelist conversion and direct JSON input

### 5. Python Bindings (src/vmecpp/__init__.py)
- All fields exposed in VmecInput class:
  - `pt_type: str`
  - `at: jt.Float[np.ndarray, 'at_len']`
  - `ph_type: str`
  - `ah: jt.Float[np.ndarray, 'ah_len']`
  - `bcrit: float`

## Testing Results
- ✅ PT_TYPE fields successfully added to C++ structures
- ✅ HDF5 I/O implemented and functional
- ✅ JSON parsing implemented
- ✅ Python bindings expose all fields
- ✅ Confirmed fields are available in VmecInput.__annotations__

## Usage Example
```python
import vmecpp
import numpy as np

# Create input with PT_TYPE support
input_data = vmecpp.VmecInput(
    # ... other required fields ...
    bcrit=1.0,
    pt_type='power_series',
    at=np.array([1.0, 0.0, 0.0, 0.0, 0.0]),  # Temperature anisotropy
    ph_type='power_series', 
    ah=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),  # Hot particle pressure
    # ... remaining fields ...
)
```

## Notes
- The indata2json tool doesn't currently recognize PT_TYPE fields, but a workaround parser script (parse_pt_type_namelist.py) was created
- The PT_TYPE fields are included for ANIMEC compatibility but are not actively used in standard VMEC equilibrium calculations
- Default values ensure backward compatibility with existing input files

## Conclusion
PT_TYPE = 'power_series' support is now fully integrated into VMEC++, ready for anisotropic pressure calculations when ANIMEC features are implemented.