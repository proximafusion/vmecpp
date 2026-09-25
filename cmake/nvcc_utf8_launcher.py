# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
#
# CUDA compiler launcher used only for the MSVC (native Windows) build.
#
# CMake's generators inject a bare ``/utf-8`` into the nvcc command line for
# MSVC. cl.exe accepts it, but nvcc parses ``/utf-8`` as a second input file
# (a ``/``-prefixed path on Windows) and aborts with "A single input file is
# required for a non-link phase". CMake offers no variable to suppress the
# injected flag, so this launcher rewrites each bare ``/utf-8`` argument into
# ``-Xcompiler=/utf-8`` (passing it through to the host compiler, which is
# what was intended) and then execs the real nvcc. It is wired in through
# CMAKE_CUDA_COMPILER_LAUNCHER, which the Ninja and Makefile generators honor.
import subprocess
import sys

args = sys.argv[1:]
fixed = ["-Xcompiler=/utf-8" if a == "/utf-8" else a for a in args]
sys.exit(subprocess.call(fixed))
