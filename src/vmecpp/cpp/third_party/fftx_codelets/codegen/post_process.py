"""Post-process SPIRAL-generated FFTX kernel sources for vmecpp.

After running SPIRAL via FFTX's `gen_dftbat.py`, the emitted .cpp files have
two issues that this script fixes in-place:

1. Compound-literal pattern `(({...constants...} + offset))` is emitted by
   SPIRAL when a (small, repeating) Diag/RCDiag table appears at use sites.
   That syntax is not valid C/C++ -- a compound literal needs an explicit
   `(type){...}` cast.  We rewrite `(({` -> `(((const double[]){`.

2. Workspace arrays inside the kernel body are declared as plain `static`,
   which makes them shared global state that breaks under OpenMP.  We rewrite
   function-local `static double <name>[N]` -> `static thread_local double ...`.
   File-scope `static double D1[…]` arrays (twiddle tables populated by the
   `init_*` function and read-only thereafter) are intentionally NOT promoted
   to thread_local: a single global copy is sharable across threads after init.

Usage:
    python3 post_process.py <directory> [<directory> ...]
"""

import re
import sys
from pathlib import Path

# Pattern 1: SPIRAL compound-literal emission
RE_COMPOUND = re.compile(r"\(\(\{")

# Pattern 2: function-local static workspaces (indented `static double NAME[N];`)
# We deliberately match only INDENTED `static double` so we leave file-scope
# `static double D1[24];` (twiddle tables) alone.
RE_LOCAL_STATIC = re.compile(r"(\n\s+)static double ")


def patch_file(path: Path) -> bool:
    with open(path) as f:
        text = f.read()
    new = RE_COMPOUND.sub("(((const double[]){", text)
    new = RE_LOCAL_STATIC.sub(r"\1static thread_local double ", new)
    if new == text:
        return False
    with open(path, "w") as f:
        f.write(new)
    return True


def main(argv):
    if len(argv) < 2:
        print(__doc__)  # noqa: T201
        return 1
    n_changed = 0
    n_total = 0
    for d in argv[1:]:
        for path in sorted(Path(d).glob("fftx_*.cpp")):
            n_total += 1
            if patch_file(path):
                n_changed += 1
    print(f"post-processed {n_changed}/{n_total} files")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
