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
                    cur[1].append((int(m.group(1)), float(m.group(2))))
    return stages


def flatten(stages):
    git, fsqr, bounds, off = [], [], [], 0
    for ns, pts in stages:
        bounds.append((off, ns))
        for it, r in pts:
            git.append(off + it)
            fsqr.append(r)
        if pts:
            off += pts[-1][0]
    return git, fsqr, bounds


base = sys.argv[1] if len(sys.argv) > 1 else "logs"
series = [
    (f"{base}/vmecpp061_ns8-768.log", VP_NS, VP, "vmecpp 0.6.1", "#d62728"),
    (f"{base}/vmecpp070_ns8-768.log", VP_NS, VP, "vmecpp 0.7.0", "#2ca02c"),
    (f"{base}/parvmec_ns8-768.log", PV_NS, PV, "PARVMEC", "#1f77b4"),
]

fig, ax = plt.subplots(figsize=(10, 5))
# draw survivors first, 0.6.1 on top so its trace stays visible where they overlap
order = {"vmecpp 0.6.1": 3, "vmecpp 0.7.0": 2, "PARVMEC": 1}
transition = None
fail_pt = None
for path, ns_re, it_re, label, color in series:
    git, fsqr, bounds = flatten(parse(path, ns_re, it_re))
    ax.semilogy(git, fsqr, label=label, color=color, lw=1.0, zorder=order[label])
    if label == "vmecpp 0.6.1":
        fail_pt = (git[-1], fsqr[-1])
    if label == "vmecpp 0.7.0" and len(bounds) > 1:
        transition = bounds[1][0]

if fail_pt is not None:
    ax.plot(*fail_pt, "x", color="#d62728", ms=13, mew=3, zorder=5)
    ax.annotate(
        "0.6.1: bad Jacobian\nat the transition",
        fail_pt,
        xytext=(fail_pt[0] + 900, 1e-9),
        fontsize=9,
        color="#d62728",
        arrowprops={"arrowstyle": "->", "color": "#d62728"},
    )

if transition is not None:
    ax.axvline(transition, color="k", ls="--", lw=0.7, alpha=0.5)
    ax.text(transition, 3e1, "ns 8 -> 768", rotation=90, fontsize=9, va="top")
ax.axhline(1e-11, color="gray", lw=0.6, ls=":", label="ftol = 1e-11")
ax.set_ylim(1e-12, 1e2)
ax.set_xlabel("cumulative iteration")
ax.set_ylabel("FSQR (radial force residual)")
ax.set_title(
    "W7-X free boundary, ns=[8, 768], delt=1.0\n"
    "vmecpp 0.6.1 fails at the transition (bad Jacobian); 0.7.0 and PARVMEC get through"
)
ax.legend(fontsize=9, loc="upper right")
fig.tight_layout()
fig.savefig("w7x_high_ns_jump.png", dpi=150)
print("wrote w7x_high_ns_jump.png")
