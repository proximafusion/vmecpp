#!/usr/bin/env python3
"""Print VmecInput structure to understand the format."""

import vmecpp

vmec_input = vmecpp.VmecInput.from_file("examples/data/input.up_down_asymmetric_tokamak")

# Print available attributes
print("VmecInput attributes:")
for attr in dir(vmec_input):
    if not attr.startswith('_'):
        val = getattr(vmec_input, attr)
        print(f"  {attr}: {type(val)}")
        if attr in ['rbc', 'zbs', 'rbs', 'zbc']:
            print(f"    Value: {val}")