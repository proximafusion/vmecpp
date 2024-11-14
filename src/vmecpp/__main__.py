# SPDX-FileCopyrightText: 2024-present Proxima Fusion Gmbh <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
import sys

from vmecpp._vmecpp import VmecINDATAPyWrapper, run

indata = VmecINDATAPyWrapper.from_file(sys.argv[1])
run(indata)
