# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
#
# CUDA compiler launcher of the MSVC build: rewrites each bare /utf-8 that CMake passes
# to nvcc as -Xcompiler=/utf-8 and runs nvcc.
import subprocess
import sys

args = sys.argv[1:]
fixed = ["-Xcompiler=/utf-8" if a == "/utf-8" else a for a in args]
sys.exit(subprocess.call(fixed))
