import re

import matplotlib.pyplot as plt

STAGE_RE = re.compile(r"^\s*NS\s*=\s*(\d+)")
VMECPP_ITER_RE = re.compile(
    r"^\s*(\d+)\s*\|\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*\|"
)
PARVMEC_ITER_RE = re.compile(
    r"^\s*(\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+"
)
PARVMEC_STAGE_RE = re.compile(r"^\s*NS =\s*(\d+) NO\. FOURIER MODES")


def parse_vmecpp(path):
    stages = []  # list of (ns, [(iter, fsqr, fsqz)])
    cur = None
    with open(path) as f:
        for line in f:
            m = STAGE_RE.match(line)
            if m:
                cur = (int(m.group(1)), [])
                stages.append(cur)
                continue
            m = VMECPP_ITER_RE.match(line)
            if m and cur is not None:
                cur[1].append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    return stages


def parse_parvmec(path):
    stages = []
    cur = None
    with open(path) as f:
        for line in f:
            m = PARVMEC_STAGE_RE.match(line)
            if m:
                cur = (int(m.group(1)), [])
                stages.append(cur)
                continue
            m = PARVMEC_ITER_RE.match(line)
            if m and cur is not None:
                cur[1].append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    return stages


def flatten(stages):
    """Concatenate stages into one continuous iteration axis, returning (global_iter,
    fsqr, fsqz, stage_boundaries)."""
    git = []
    fsqr = []
    fsqz = []
    boundaries = []
    offset = 0
    for ns, points in stages:
        boundaries.append((offset, ns))
        for it, r, z in points:
            git.append(offset + it)
            fsqr.append(r)
            fsqz.append(z)
        if points:
            offset += points[-1][0]
    return git, fsqr, fsqz, boundaries


vmecpp_before = parse_vmecpp("vmecpp_bug_present.log")
vmecpp_after = parse_vmecpp("vmecpp_fixed.log")
parvmec = parse_parvmec("parvmec.log")

fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=False)

for ax, (stages, title) in zip(
    axes,
    [
        (vmecpp_before, "vmecpp -- bug present (vacuum state not seeded)"),
        (vmecpp_after, "vmecpp -- fixed (vacuum state seeded across transitions)"),
        (parvmec, "PARVMEC (Fortran) -- same iter2>1 gate, same behavior"),
    ],
    strict=True,
):
    git, fsqr, fsqz, boundaries = flatten(stages)
    ax.semilogy(git, fsqr, label="FSQR", lw=1)
    ax.semilogy(git, fsqz, label="FSQZ", lw=1, alpha=0.8)
    for offset, ns in boundaries:
        ax.axvline(offset, color="k", ls="--", lw=0.7, alpha=0.6)
        ax.text(
            offset,
            ax.get_ylim()[1] if False else 1e2,
            f"ns={ns}",
            rotation=90,
            fontsize=8,
            va="top",
        )
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("force residual")
    ax.set_ylim(1e-13, 1e2)
    ax.legend(fontsize=8, loc="upper right")

axes[-1].set_xlabel("cumulative iteration")
fig.suptitle(
    "W7-X free-boundary multigrid transitions (ns = 4,9,28,99):\n"
    "force-residual spike at each stage entry"
)
fig.tight_layout()
fig.savefig("multigrid_transition_comparison.png", dpi=150)
print("wrote multigrid_transition_comparison.png")

stage_labels = []
before_vals = []
after_vals = []
parvmec_vals = []
for i in range(1, len(vmecpp_before)):
    ns_prev = vmecpp_before[i - 1][0]
    ns = vmecpp_before[i][0]
    stage_labels.append(f"{ns_prev}->{ns}")
    before_vals.append(
        vmecpp_before[i][1][0][1] if vmecpp_before[i][1] else float("nan")
    )
    after_vals.append(vmecpp_after[i][1][0][1] if vmecpp_after[i][1] else float("nan"))
    parvmec_vals.append(
        parvmec[i][1][0][1] if i < len(parvmec) and parvmec[i][1] else float("nan")
    )

fig2, ax2 = plt.subplots(figsize=(7, 4.5))
x = range(len(stage_labels))
width = 0.25
ax2.bar([i - width for i in x], before_vals, width, label="vmecpp, bug present")
ax2.bar(x, after_vals, width, label="vmecpp, fixed")
ax2.bar([i + width for i in x], parvmec_vals, width, label="PARVMEC (Fortran)")
ax2.set_yscale("log")
ax2.set_xticks(list(x))
ax2.set_xticklabels([f"ns {s}" for s in stage_labels])
ax2.set_ylabel("FSQR at iteration 1 of new stage")
ax2.set_title("Force-residual spike at multigrid stage entry, W7-X free boundary")
ax2.axhline(1e-2, color="gray", lw=0.5, ls=":")
ax2.legend(fontsize=9)
fig2.tight_layout()
fig2.savefig("stage_entry_fsqr_bars.png", dpi=150)
print("wrote stage_entry_fsqr_bars.png")

# print a compact summary table of the stage-entry (iter==1) FSQR spikes
print()
print(f"{'ns transition':<16}{'vmecpp before':>16}{'vmecpp after':>16}{'PARVMEC':>16}")
for i in range(1, len(vmecpp_before)):
    ns_prev = vmecpp_before[i - 1][0]
    ns = vmecpp_before[i][0]
    b = vmecpp_before[i][1][0][1] if vmecpp_before[i][1] else float("nan")
    a = vmecpp_after[i][1][0][1] if vmecpp_after[i][1] else float("nan")
    p = parvmec[i][1][0][1] if i < len(parvmec) and parvmec[i][1] else float("nan")
    print(f"{ns_prev:>4} -> {ns:<8}{b:>16.3e}{a:>16.3e}{p:>16.3e}")
