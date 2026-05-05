#!/usr/bin/env python3
"""Debug symmetric equilibrium failure."""

import json
import subprocess
import sys
from pathlib import Path

def test_quick_symmetric():
    """Test minimal symmetric case."""
    print("Testing minimal symmetric equilibrium...")
    
    # Use the quick test case
    json_path = Path("test_symmetric_quick.json")
    if not json_path.exists():
        print(f"Error: {json_path} not found")
        return False
        
    # Run with standalone executable
    cmd = ["./build/vmec_standalone", str(json_path)]
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        
        print(f"Return code: {result.returncode}")
        print(f"\nSTDOUT:\n{result.stdout}")
        print(f"\nSTDERR:\n{result.stderr}")
        
        return result.returncode == 0
            
    except subprocess.TimeoutExpired:
        print("✗ Timed out after 5 seconds")
        return False
    except Exception as e:
        print(f"✗ Exception: {e}")
        return False

if __name__ == "__main__":
    success = test_quick_symmetric()
    sys.exit(0 if success else 1)