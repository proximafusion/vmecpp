from pathlib import Path

from vmecpp import _util


def test_repo_root():
    # check we can find a file that should be there from the repo root
    assert Path(_util.package_root(), "_util.py").is_file()
