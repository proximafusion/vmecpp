#!/usr/bin/env python3
"""
Compare VMEC++ and VMEC2000 on local input files.
Based on benchmark_vmec patterns but simplified for our needs.
"""

import os
import sys
import subprocess
import tempfile
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class VmecResult:
    """Store VMEC computation results"""
    wb: Optional[float] = None
    betatotal: Optional[float] = None 
    aspect: Optional[float] = None
    volume_p: Optional[float] = None
    raxis_cc: Optional[float] = None
    iotaf_edge: Optional[float] = None
    success: bool = False
    error: Optional[str] = None

class VmecppRunner:
    """Run VMEC++ using Python bindings"""
    
    def __init__(self):
        self.name = "VMEC++"
        
    def run_case(self, input_file: Path) -> VmecResult:
        try:
            import vmecpp
            vmec_input = vmecpp.VmecInput.from_file(str(input_file))
            output = vmecpp.run(vmec_input, verbose=False)
            
            return VmecResult(
                wb=output.wout.wb,
                betatotal=getattr(output.wout, 'betatotal', 0.0),
                aspect=getattr(output.wout, 'aspect', 0.0),
                volume_p=getattr(output.wout, 'volume_p', 0.0),
                raxis_cc=getattr(output.wout, 'raxis_cc', [0.0])[0] if hasattr(output.wout, 'raxis_cc') else 0.0,
                iotaf_edge=getattr(output.wout, 'iotas', [0.0])[-1] if hasattr(output.wout, 'iotas') else 0.0,
                success=True
            )
        except Exception as e:
            return VmecResult(success=False, error=str(e))

class Vmec2000Runner:
    """Run VMEC2000 using Python import"""
    
    def __init__(self):
        self.name = "VMEC2000"
        self.available = self._check_available()
        
    def _check_available(self) -> bool:
        try:
            result = subprocess.run(['python', '-c', 'import vmec; print("OK")'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
            
    def run_case(self, input_file: Path) -> VmecResult:
        if not self.available:
            return VmecResult(success=False, error="VMEC2000 not available")
            
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Copy input file to temp directory
                temp_input = Path(tmpdir) / "input.test"
                temp_input.write_text(input_file.read_text())
                
                # Run VMEC2000
                cmd = f"""
import sys
sys.path.insert(0, '{tmpdir}')
import vmec
import os
os.chdir('{tmpdir}')
result = vmec.run('{temp_input}')
if result.success:
    wout = result.wout
    print(f"SUCCESS {{wout.wb}} {{getattr(wout, 'betatotal', 0.0)}} {{getattr(wout, 'aspect', 0.0)}} {{getattr(wout, 'volume_p', 0.0)}} {{getattr(wout, 'raxis_cc', [0.0])[0] if hasattr(wout, 'raxis_cc') else 0.0}} {{getattr(wout, 'iotas', [0.0])[-1] if hasattr(wout, 'iotas') else 0.0}}")
else:
    print(f"FAILED {{result.error}}")
"""
                
                result = subprocess.run(['python', '-c', cmd], 
                                      capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0 and result.stdout.startswith("SUCCESS"):
                    parts = result.stdout.strip().split()
                    return VmecResult(
                        wb=float(parts[1]),
                        betatotal=float(parts[2]),
                        aspect=float(parts[3]),
                        volume_p=float(parts[4]),
                        raxis_cc=float(parts[5]),
                        iotaf_edge=float(parts[6]),
                        success=True
                    )
                else:
                    error_msg = result.stdout if result.stdout else result.stderr
                    return VmecResult(success=False, error=error_msg.strip())
                    
        except subprocess.TimeoutExpired:
            return VmecResult(success=False, error="Timeout after 300s")
        except Exception as e:
            return VmecResult(success=False, error=str(e))

def main():
    # Input files to test (excluding input.test as requested)
    input_files = [
        "examples/data/input.solovev",
        "examples/data/input.w7x", 
        "examples/data/input.nfp4_QH_warm_start",
        "examples/data/input.cth_like_fixed_bdy",
        "examples/data/input.up_down_asymmetric_tokamak",
        "src/vmecpp/cpp/vmecpp/test_data/input.cma",
        "src/vmecpp/cpp/vmecpp/test_data/input.cth_like_fixed_bdy",
        "src/vmecpp/cpp/vmecpp/test_data/input.cth_like_fixed_bdy_nzeta_37",
        "src/vmecpp/cpp/vmecpp/test_data/input.cth_like_free_bdy",
        "src/vmecpp/cpp/vmecpp/test_data/input.li383_low_res",
        "src/vmecpp/cpp/vmecpp/test_data/input.solovev",
        "src/vmecpp/cpp/vmecpp/test_data/input.solovev_analytical",
        "src/vmecpp/cpp/vmecpp/test_data/input.solovev_no_axis",
        "src/vmecpp/cpp/vmecpp/test_data/input.test_asymmetric",
        "src/vmecpp/cpp/vmecpp/test_data/input.up_down_asymmetric_tokamak.json",
        "build/_deps/indata2json-src/demo_inputs/input.HELIOTRON",
        "build/_deps/indata2json-src/demo_inputs/input.w7x_ref_167_12_12",
        "build/_deps/indata2json-src/demo_inputs/input.vmec",
        "build/_deps/indata2json-src/demo_inputs/input.W7X_s2048_M16_N16_f12_cpu8"
    ]
    
    # Initialize runners
    vmecpp = VmecppRunner()
    vmec2000 = Vmec2000Runner()
    
    print("VMEC Implementation Comparison")
    print("=" * 60)
    print(f"VMEC++: Available")
    print(f"VMEC2000: {'Available' if vmec2000.available else 'Not Available'}")
    print()
    
    results = {}
    success_counts = {"VMEC++": 0, "VMEC2000": 0}
    total_tests = 0
    
    for input_file in input_files:
        input_path = Path(input_file)
        if not input_path.exists():
            continue
            
        total_tests += 1
        print(f"Testing: {input_file}")
        
        # Run VMEC++
        vmecpp_result = vmecpp.run_case(input_path)
        if vmecpp_result.success:
            success_counts["VMEC++"] += 1
            print(f"  VMEC++: ✅ MHD Energy = {vmecpp_result.wb:.6e}")
        else:
            error = vmecpp_result.error[:50] + "..." if len(vmecpp_result.error) > 50 else vmecpp_result.error
            print(f"  VMEC++: ❌ {error}")
        
        # Run VMEC2000
        vmec2000_result = vmec2000.run_case(input_path)
        if vmec2000_result.success:
            success_counts["VMEC2000"] += 1
            print(f"  VMEC2000: ✅ MHD Energy = {vmec2000_result.wb:.6e}")
        else:
            error = vmec2000_result.error[:50] + "..." if len(vmec2000_result.error) > 50 else vmec2000_result.error
            print(f"  VMEC2000: ❌ {error}")
        
        results[input_file] = {
            "VMEC++": vmecpp_result,
            "VMEC2000": vmec2000_result
        }
        print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total input files tested: {total_tests}")
    print(f"VMEC++ successful runs: {success_counts['VMEC++']} ({success_counts['VMEC++'] / total_tests * 100:.1f}%)")
    print(f"VMEC2000 successful runs: {success_counts['VMEC2000']} ({success_counts['VMEC2000'] / total_tests * 100:.1f}%)")
    
    # Detailed comparison for successful cases
    print("\nSuccessful Cases Comparison:")
    print("-" * 60)
    for input_file, case_results in results.items():
        vmecpp_res = case_results["VMEC++"]
        vmec2000_res = case_results["VMEC2000"]
        
        if vmecpp_res.success and vmec2000_res.success:
            rel_diff = abs(vmecpp_res.wb - vmec2000_res.wb) / abs(vmec2000_res.wb) * 100
            print(f"{Path(input_file).name}:")
            print(f"  VMEC++:   {vmecpp_res.wb:.6e}")
            print(f"  VMEC2000: {vmec2000_res.wb:.6e}")
            print(f"  Rel diff: {rel_diff:.2f}%")

if __name__ == "__main__":
    main()