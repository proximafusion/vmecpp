"""Plot the QUASR-0065579 free-boundary multigrid counterexample.

Panel A: FSQR vs cumulative iteration for vmecpp 0.6.1 (fails), vmecpp 0.7.0
         (converges), and stock PARVMEC (fails), ns_array = [12, 50, 201].
Panel B: the "just faster" case (QUASR-0029346, ns_array = [8, 16, 31]),
         vmecpp 0.6.1 vs 0.7.0, both converge.
"""

import contextlib
import re
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

VP = re.compile(r"^\s*(\d+)\s*\|\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*\|")
VP_NS = re.compile(r"NS =\s*(\d+)\s+NO\. FOURIER")
PV = re.compile(r"^\s*(\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+")
PV_NS = re.compile(r"NS =\s*(\d+) NO\. FOURIER MODES")


def parse(path, ns_re, it_re):
    stages, cur = [], None
    with open(path) as f:
        for line in f:
            m = ns_re.search(line)
            if m:
                cur = (int(m.group(1)), [])
                stages.append(cur)
                continue
            m = it_re.match(line)
            if m and cur is not None:
                with contextlib.suppress(ValueError):
                    cur[1].append(
                        (int(m.group(1)), float(m.group(2)), float(m.group(3)))
                    )
    return stages


def flatten(stages):
    git, fsqr, bounds, off = [], [], [], 0
    for ns, pts in stages:
        bounds.append((off, ns))
        for it, r, _z in pts:
            git.append(off + it)
            fsqr.append(r)
        if pts:
            off += pts[-1][0]
    return git, fsqr, bounds


base = sys.argv[1] if len(sys.argv) > 1 else "logs"  # dir with the logs

fig, (axA, axB) = plt.subplots(2, 1, figsize=(10, 9))

# ---- Panel A: counterexample 65579 ----
series = [
    (
        f"{base}/vmecpp061_ns12-50-201.log",
        VP_NS,
        VP,
        "vmecpp 0.6.1 (bug) -- FAILS",
        "#d62728",
    ),
    (
        f"{base}/vmecpp070_ns12-50-201.log",
        VP_NS,
        VP,
        "vmecpp 0.7.0 (fix) -- converges",
        "#2ca02c",
    ),
    (
        f"{base}/parvmec_ns12-50-201.log",
        PV_NS,
        PV,
        "PARVMEC (stock) -- FAILS",
        "#1f77b4",
    ),
]
allb = None
for path, ns_re, it_re, label, color in series:
    st = parse(path, ns_re, it_re)
    git, fsqr, bounds = flatten(st)
    axA.semilogy(git, fsqr, label=label, color=color, lw=0.9)
    if "0.7.0" in label:
        allb = bounds
for off, ns in allb or []:
    axA.axvline(off, color="k", ls="--", lw=0.6, alpha=0.5)
    axA.text(off, 3e6, f"ns={ns}", rotation=90, fontsize=8, va="top")
axA.axhline(1e-9, color="gray", lw=0.6, ls=":", label="ftol = 1e-9")
axA.set_ylim(1e-10, 1e7)
axA.set_ylabel("FSQR (radial force residual)")
axA.set_title(
    "QUASR-0065579 (nfp=4), free boundary, ns_array=[12,50,201]: "
    "cold start fails, multigrid required"
)
axA.legend(fontsize=8, loc="center right")

# ---- Panel B: just-faster 29346 ----
for path, label, color in [
    (f"{base}/vmecpp061_29346_ns8-16-31.log", "vmecpp 0.6.1", "#d62728"),
    (f"{base}/vmecpp070_29346_ns8-16-31.log", "vmecpp 0.7.0", "#2ca02c"),
]:
    st = parse(path, VP_NS, VP)
    git, fsqr, bounds = flatten(st)
    axB.semilogy(git, fsqr, label=label, color=color, lw=0.9)
    if "0.7.0" in label:
        for off, ns in bounds:
            axB.axvline(off, color="k", ls="--", lw=0.6, alpha=0.5)
            axB.text(off, 3e0, f"ns={ns}", rotation=90, fontsize=8, va="top")
axB.axhline(1e-9, color="gray", lw=0.6, ls=":")
axB.set_ylim(1e-10, 1e1)
axB.set_ylabel("FSQR")
axB.set_xlabel("cumulative iteration")
axB.set_title(
    "QUASR-0029346 (nfp=2), free boundary, ns_array=[8,16,31]: both converge, "
    "0.7.0 ~1.8x fewer total iters (2.8x on the final ns=31 grid)"
)
axB.legend(fontsize=8, loc="upper right")

fig.tight_layout()
fig.savefig("quasr_counterexample.png", dpi=150)
print("wrote quasr_counterexample.png")
