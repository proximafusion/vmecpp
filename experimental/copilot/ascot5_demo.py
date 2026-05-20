"""ASCOT5 Python API demonstration script.

This script demonstrates the a5py Python API from the ASCOT5 introduction
tutorial (https://ascot4fusion.github.io/ascot5/).

Run setup_ascot5.sh first to build libascot.so and install a5py.

Steps demonstrated:
  1. Create an ASCOT5 HDF5 input file.
  2. Populate it with physics inputs using built-in templates.
  3. Initialise the magnetic field in memory via libascot.so and query it.
  4. Print a summary of the input data tree.
"""

import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from a5py import Ascot  # noqa: E402  # type: ignore[import-untyped]
from a5py.ascot5io.marker import Marker  # noqa: E402  # type: ignore[import-untyped]

OUTPUT_FILE = Path(__file__).resolve().parent / "ascot.h5"

print("=== ASCOT5 Python API demo ===")
print()

# ------------------------------------------------------------------
# 1. Create a new HDF5 input file
# ------------------------------------------------------------------
print("Step 1: Creating ASCOT5 input file:", OUTPUT_FILE)
a5 = Ascot(str(OUTPUT_FILE), create=True)
print("  OK")

# ------------------------------------------------------------------
# 2. Populate inputs using built-in templates
# ------------------------------------------------------------------
print("Step 2: Creating physics inputs from templates ...")

# Simulation options with a short mileage end condition
a5.data.create_input("options tutorial")

# Analytical ITER-like circular equilibrium magnetic field
a5.data.create_input("bfield analytical iter circular")

# A simple rectangular 2-D wall
a5.data.create_input("wall rectangular")

# Flat (constant) plasma background
a5.data.create_input("plasma flat")

# Zero electric field
a5.data.create_input("E_TC", exyz=np.array([0, 0, 0]))

# 100 alpha-particle guiding-centre markers on the midplane
rng = np.random.default_rng(seed=42)
mrk = Marker.generate("gc", n=100, species="alpha")
mrk["energy"][:] = 3.5e6  # 3.5 MeV (fusion alpha energy)
mrk["pitch"][:] = 0.99 - 1.98 * rng.random(100)
mrk["r"][:] = np.linspace(6.2, 8.2, 100)  # R in [m]
a5.data.create_input("gc", **mrk)

# Dummy inputs for unused physics modules (required for a valid input file)
a5.data.create_input("N0_3D")
a5.data.create_input("Boozer")
a5.data.create_input("MHD_STAT")
a5.data.create_input("asigma_loc")

print("  All inputs created.")

# ------------------------------------------------------------------
# 3. Print a summary of the data tree
# ------------------------------------------------------------------
print()
print("Step 3: Input data tree")
print("-" * 50)
a5.data.ls(show=True)
print("-" * 50)

# ------------------------------------------------------------------
# 4. Initialise the magnetic field and query it at a specific point
# ------------------------------------------------------------------
print()
print("Step 4: Initialising magnetic field via libascot.so ...")
try:
    a5.input_init(bfield=True)
except Exception as e:
    msg = (
        f"Failed to initialise the magnetic field: {e}\n"
        "Ensure libascot.so is compiled and a5py is installed via setup_ascot5.sh."
    )
    raise RuntimeError(msg) from e

r_query = 6.2  # metres
z_query = 0.0
phi_query = 0.0
t_query = 0.0

try:
    result = a5.input_eval(r_query, phi_query, z_query, t_query, "psi", "rho")
    psi_val = result[0]
    rho_val = result[1]

    print(
        f"  Magnetic flux at (R={r_query} m, phi={phi_query}, z={z_query} m,"
        f" t={t_query}):"
    )
    print(f"    psi = {psi_val}")
    print(f"    rho = {rho_val}")
finally:
    a5.input_free(bfield=True)
    print("  Magnetic field freed from memory.")

print()
print("=== Demo completed successfully ===")
print(f"ASCOT5 HDF5 file written to: {OUTPUT_FILE}")
