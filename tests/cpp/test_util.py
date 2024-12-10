# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
from pathlib import Path

from vmecpp import _util


def test_repo_root():
    # check we can find a file that should be there from the repo root
    assert Path(_util.package_root(), "_util.py").is_file()
