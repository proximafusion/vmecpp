#!/usr/bin/env bash
# Setup script to build and install the ASCOT5 Python package (a5py)
# from the third_party/ascot5 submodule.
#
# Prerequisites (Debian/Ubuntu):
#   sudo apt-get install libhdf5-dev
#
# Usage:
#   cd experimental/copilot
#   bash setup_ascot5.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ASCOT5_DIR="${REPO_ROOT}/third_party/ascot5"

if [ ! -d "${ASCOT5_DIR}/src" ]; then
    echo "ERROR: ascot5 submodule not initialised. Run:"
    echo "  git submodule update --init third_party/ascot5"
    exit 1
fi

# Check for h5cc (HDF5 compiler wrapper required by the Makefile)
if ! command -v h5cc &>/dev/null; then
    echo "ERROR: h5cc not found. Install HDF5 development libraries:"
    echo "  sudo apt-get install libhdf5-dev   # Debian/Ubuntu"
    echo "  brew install hdf5                  # macOS"
    exit 1
fi

echo "==> Building libascot.so in ${ASCOT5_DIR} ..."
make -C "${ASCOT5_DIR}" libascot -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
echo "==> libascot.so built successfully."

echo "==> Installing a5py Python package (editable) ..."
pip install -e "${ASCOT5_DIR}"
echo "==> a5py installed successfully."

echo ""
echo "Setup complete. You can now run the demo:"
echo "  python ${SCRIPT_DIR}/ascot5_demo.py"
