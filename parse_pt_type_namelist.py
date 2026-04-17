#!/usr/bin/env python3
"""
Parse PT_TYPE and PH_TYPE from namelist files that indata2json doesn't support yet.
This is a temporary solution until indata2json is updated.
"""

import re
import json
import sys
from pathlib import Path


def parse_pt_ph_from_namelist(namelist_file):
    """Extract PT_TYPE, PH_TYPE, AT, and AH from a namelist file."""
    
    with open(namelist_file, 'r') as f:
        content = f.read()
    
    # Find the INDATA section
    indata_match = re.search(r'&INDATA(.*?)/', content, re.DOTALL | re.IGNORECASE)
    if not indata_match:
        print("ERROR: No &INDATA section found")
        return None
        
    indata_content = indata_match.group(1)
    
    # Parse PT_TYPE
    pt_type_match = re.search(r'PT_TYPE\s*=\s*["\']([^"\']+)["\']', indata_content, re.IGNORECASE)
    pt_type = pt_type_match.group(1) if pt_type_match else "power_series"
    
    # Parse PH_TYPE
    ph_type_match = re.search(r'PH_TYPE\s*=\s*["\']([^"\']+)["\']', indata_content, re.IGNORECASE)
    ph_type = ph_type_match.group(1) if ph_type_match else "power_series"
    
    # Parse BCRIT
    bcrit_match = re.search(r'BCRIT\s*=\s*([\d.eE+-]+)', indata_content, re.IGNORECASE)
    bcrit = float(bcrit_match.group(1)) if bcrit_match else 1.0
    
    # Parse AT array
    at_match = re.search(r'AT\s*=\s*((?:[\d.eE+-]+\s*)+)', indata_content, re.IGNORECASE)
    if at_match:
        at_str = at_match.group(1).strip()
        at = [float(x) for x in at_str.split()]
    else:
        at = [1.0]  # Default: isotropic
    
    # Parse AH array
    ah_match = re.search(r'AH\s*=\s*((?:[\d.eE+-]+\s*)+)', indata_content, re.IGNORECASE)
    if ah_match:
        ah_str = ah_match.group(1).strip()
        ah = [float(x) for x in ah_str.split()]
    else:
        ah = []  # Default: no hot particle pressure
    
    return {
        'bcrit': bcrit,
        'pt_type': pt_type.lower(),
        'at': at,
        'ph_type': ph_type.lower(),
        'ah': ah
    }


def enhance_json_with_pt_ph(json_file, namelist_file):
    """Add PT_TYPE and PH_TYPE fields to an existing JSON file."""
    
    # Parse the namelist for PT/PH fields
    pt_ph_data = parse_pt_ph_from_namelist(namelist_file)
    if not pt_ph_data:
        return False
    
    # Load existing JSON
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    
    # Add the new fields
    json_data.update(pt_ph_data)
    
    # Write back
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Successfully added PT_TYPE/PH_TYPE fields to {json_file}")
    print(f"  bcrit = {pt_ph_data['bcrit']}")
    print(f"  pt_type = '{pt_ph_data['pt_type']}'")
    print(f"  at = {pt_ph_data['at']}")
    print(f"  ph_type = '{pt_ph_data['ph_type']}'")
    print(f"  ah = {pt_ph_data['ah']}")
    
    return True


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: parse_pt_type_namelist.py <json_file> <namelist_file>")
        sys.exit(1)
    
    json_file = Path(sys.argv[1])
    namelist_file = Path(sys.argv[2])
    
    if not json_file.exists():
        print(f"ERROR: JSON file {json_file} does not exist")
        sys.exit(1)
        
    if not namelist_file.exists():
        print(f"ERROR: Namelist file {namelist_file} does not exist")
        sys.exit(1)
    
    success = enhance_json_with_pt_ph(json_file, namelist_file)
    sys.exit(0 if success else 1)