#!/usr/bin/env python3
"""Compare VMEC++ and VMEC2000 on local input files.

Includes immediate validation against reference data.
"""

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import netCDF4 as nc


@dataclass
class VmecResult:
    """Store VMEC computation results."""

    wb: float | None = None
    betatotal: float | None = None
    aspect: float | None = None
    volume_p: float | None = None
    raxis_cc: float | None = None
    iotaf_edge: float | None = None
    success: bool = False
    error: str | None = None


class VmecppRunner:
    """Run VMEC++ using Python bindings."""

    def __init__(self):
        self.name = "VMEC++"

    def run_case(self, input_file: Path) -> VmecResult:
        try:
            import vmecpp

            vmec_input = vmecpp.VmecInput.from_file(str(input_file))
            output = vmecpp.run(vmec_input, verbose=False)

            return VmecResult(
                wb=output.wout.wb,
                betatotal=getattr(output.wout, "betatotal", 0.0),
                aspect=getattr(output.wout, "aspect", 0.0),
                volume_p=getattr(output.wout, "volume_p", 0.0),
                raxis_cc=getattr(output.wout, "raxis_cc", [0.0])[0]
                if hasattr(output.wout, "raxis_cc")
                else 0.0,
                iotaf_edge=getattr(output.wout, "iotas", [0.0])[-1]
                if hasattr(output.wout, "iotas")
                else 0.0,
                success=True,
            )
        except Exception as e:
            return VmecResult(success=False, error=str(e))


class Vmec2000Runner:
    """Run VMEC2000 using xvmec2000 executable."""

    def __init__(self):
        self.name = "VMEC2000"
        self.available = self._check_available()

    def _check_available(self) -> bool:
        try:
            result = subprocess.run(
                ["which", "xvmec2000"], capture_output=True, text=True, check=False
            )
            return result.returncode == 0
        except Exception:
            return False

    def run_case(self, input_file: Path) -> VmecResult:
        if not self.available:
            return VmecResult(success=False, error="xvmec2000 not available")

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Copy input file to temp directory
                temp_input = Path(tmpdir) / "input.test"
                temp_input.write_text(input_file.read_text())

                # Run VMEC2000
                result = subprocess.run(
                    ["xvmec2000", str(temp_input)],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=tmpdir,
                    check=False,
                )

                # Look for wout file
                wout_file = Path(tmpdir) / f"wout_{temp_input.stem}.nc"
                if wout_file.exists():
                    try:
                        with nc.Dataset(str(wout_file), "r") as f:
                            wb = f.variables["wb"][()]
                            betatotal = f.variables.get("betatotal", [0.0])[()]
                            aspect = f.variables.get("aspect", [0.0])[()]
                            volume_p = f.variables.get("volume_p", [0.0])[()]
                            raxis_cc = (
                                f.variables.get("raxis_cc", [0.0])[()][0]
                                if "raxis_cc" in f.variables
                                else 0.0
                            )
                            iotas = f.variables.get("iotas", [0.0])[()]
                            iotaf_edge = iotas[-1] if len(iotas) > 0 else 0.0

                            return VmecResult(
                                wb=float(wb),
                                betatotal=float(betatotal),
                                aspect=float(aspect),
                                volume_p=float(volume_p),
                                raxis_cc=float(raxis_cc),
                                iotaf_edge=float(iotaf_edge),
                                success=True,
                            )
                    except Exception as e:
                        return VmecResult(
                            success=False, error=f"Error reading wout file: {e}"
                        )
                else:
                    error_msg = result.stdout if result.stdout else result.stderr
                    return VmecResult(success=False, error=error_msg.strip())

        except subprocess.TimeoutExpired:
            return VmecResult(success=False, error="Timeout after 300s")
        except Exception as e:
            return VmecResult(success=False, error=str(e))


def validate_against_reference():
    """Validate VMEC++ against reference circular tokamak case."""
    print("=" * 60)
    print("REFERENCE VALIDATION CHECK")
    print("=" * 60)

    try:
        import vmecpp

        # Test VMEC++ with reference input
        vmec_input = vmecpp.VmecInput.from_file(
            "src/vmecpp/cpp/vmecpp/test_data/circular_tokamak.json"
        )
        output = vmecpp.run(vmec_input, verbose=False)
        vmecpp_energy = output.wout.wb

        # Read reference file
        ref_file = Path(
            "src/vmecpp/cpp/vmecpp/test_data/wout_circular_tokamak_reference.nc"
        )
        with nc.Dataset(str(ref_file), "r") as f:
            ref_energy = f.variables["wb"][()]

        # Compare
        rel_diff = abs(vmecpp_energy - ref_energy) / abs(ref_energy) * 100

        print(f"VMEC++ Energy:    {vmecpp_energy:.6e}")
        print(f"Reference Energy: {ref_energy:.6e}")
        print(f"Relative Diff:    {rel_diff:.6f}%")

        if rel_diff < 0.001:
            print("✅ VMEC++ MATCHES REFERENCE - VALIDATION PASSED")
        else:
            print("❌ VMEC++ DIFFERS FROM REFERENCE - VALIDATION FAILED")

        print()
        return rel_diff < 0.001

    except Exception as e:
        print(f"❌ VALIDATION ERROR: {e}")
        print()
        return False


def main():
    # First validate against reference
    reference_ok = validate_against_reference()
    if not reference_ok:
        print("⚠️  Reference validation failed - proceeding with comparison anyway")
        print()
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
        "build/_deps/indata2json-src/demo_inputs/input.W7X_s2048_M16_N16_f12_cpu8",
    ]

    # Initialize runners
    vmecpp = VmecppRunner()
    vmec2000 = Vmec2000Runner()

    print("VMEC Implementation Comparison")
    print("=" * 60)
    print("VMEC++: Available")
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
            error = (
                vmecpp_result.error[:50] + "..."
                if vmecpp_result.error and len(vmecpp_result.error) > 50
                else vmecpp_result.error or "Unknown error"
            )
            print(f"  VMEC++: ❌ {error}")

        # Run VMEC2000
        vmec2000_result = vmec2000.run_case(input_path)
        if vmec2000_result.success:
            success_counts["VMEC2000"] += 1
            print(f"  VMEC2000: ✅ MHD Energy = {vmec2000_result.wb:.6e}")
        else:
            error = (
                vmec2000_result.error[:50] + "..."
                if vmec2000_result.error and len(vmec2000_result.error) > 50
                else vmec2000_result.error or "Unknown error"
            )
            print(f"  VMEC2000: ❌ {error}")

        results[input_file] = {"VMEC++": vmecpp_result, "VMEC2000": vmec2000_result}
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total input files tested: {total_tests}")
    print(
        f"VMEC++ successful runs: {success_counts['VMEC++']} ({success_counts['VMEC++'] / total_tests * 100:.1f}%)"
    )
    print(
        f"VMEC2000 successful runs: {success_counts['VMEC2000']} ({success_counts['VMEC2000'] / total_tests * 100:.1f}%)"
    )

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
