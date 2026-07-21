# QUASR counterexample: a case the #663 fix rescues

The sibling `../README.md` demo shows that the free-boundary multigrid
transition bug (skipping the vacuum `B^2` contribution for one iteration at
every continuation stage, gated on `iter2 > 1`) produces a force-residual
spike but that W7-X still converges either way. This directory provides the
stronger statement: a **highly shaped, large-`ns` free-boundary case that does
not converge without the fix**, in both stock PARVMEC and vmecpp 0.6.1, but
converges cleanly in vmecpp 0.7.0.

Everything needed to reproduce is checked in here -- the INDATA files and the
mgrid vacuum-field files -- so no external tooling (SIMSOPT, makegrid, the
QUASR database) is required. The scripts that *generate* these inputs from a
QUASR serial live in a separate PR.

## The counterexample: QUASR-0065579

A `nfp = 4` stellarator from the QUASR database, run free-boundary with the
coil set that produced it. Resolution `mpol = ntor = 6`, `nzeta = 24`, target
`ftol = 1e-9`, `delt = 0.9`. Files:

* `input.quasr65579_12-50-201` -- VMEC INDATA (multigrid ramp `ns = [12,50,201]`)
* `mgrid_quasr0065579.nc` -- the coil vacuum field (`EXTCUR = 1.0`, referenced
  by the INDATA); readable by both vmecpp and PARVMEC.

**Cold start is impossible.** A single `ns = 201` grid fails immediately
("solver failed during the first iterations" -- the cold interpolated guess
self-intersects on a 201-surface mesh), so the multigrid ramp is *required*.
See `logs/coldstart_ns201_fails.log`.

**On that required multigrid ramp:**

| code | outcome at `ns_array=[12,50,201]` |
|------|-----------------------------------|
| **vmecpp 0.7.0** (fix) | **converges**, FSQR = 9.9e-10, 3065 cumulative iters |
| **vmecpp 0.6.1** (bug) | **fails** -- FSQR diverges to ~9e6 at the `ns=201` stage, bad-Jacobian abort |
| **PARVMEC** (stock, unmodified) | **fails** -- FSQR spikes to ~4e3 at `ns=201` entry, oscillates with 37 Jacobian resets, hits the iteration cap |

`quasr_counterexample.png`, panel A, overlays the three force-residual
histories on a shared cumulative-iteration axis. 0.7.0 (green) descends
cleanly through every transition to `ftol`; 0.6.1 (red) and PARVMEC (blue)
never recover from the `ns=201` transition kick. The traces are in `logs/`.

## The fix, isolated to #663

vmecpp 0.7.0 differs from 0.6.1 in *two* flow-control changes (a restart-
mechanism fix that brings vmecpp in line with PARVMEC, and the #663 vacuum-
seed fix). The failure here is attributable specifically to **#663**:

* **PARVMEC already has** the restart behaviour that 0.7.0 adopted, yet it
  still fails on this case -- so 0.7.0's success is not the restart change.
* Reverting **only** the 18-line #663 patch
  (`vacuum_pressure_state_ = kInitialized` in `Vmec::InitializeRadial`) from
  the source and rebuilding flips the result from converge to fail **at the
  exact `50 -> 201` transition** ("solver failed during the first
  iterations"); see `logs/fix663_reverted_ns12-50-201.log`. Restoring the
  patch restores convergence.

The mechanism the `iter2 > 1` gate causes: at stage entry the LCFS force is
applied with the vacuum `B^2` term missing (visible as `DELBSQ = -nan` on
iteration 1 of the new stage in `logs/vmecpp061_ns12-50-201.log`), inflating
the boundary; at `ns = 201` the problem is stiff enough that the kick
self-intersects the flux surfaces and the solve cannot recover.

## A "just faster" case: QUASR-0029346

Panel B shows a `nfp = 2` case where both versions converge on
`ns_array = [8, 16, 31]`, but 0.7.0 needs fewer iterations
(`input.quasr29346_8-16-31`, `mgrid_quasr0029346.nc`):

| code        | ns=8 | ns=16 | ns=31 | cumulative |
|-------------|------|-------|-------|------------|
| vmecpp 0.6.1 | 739  | 789   | 779   | 2307 |
| vmecpp 0.7.0 | 739  | 292   | 274   | 1305 |
| PARVMEC      | 722  | 365   | 365   | 1452 |

All three converge here. The `ns=8` stage is essentially identical (no
transition has happened yet); the fix pays off at each subsequent transition,
bringing vmecpp from ~1.8x slower than 0.6.1 to faster than PARVMEC. (At
`ns=[12,50,201]` this same config is another fail->converge case for vmecpp,
while PARVMEC still converges on it -- an illustration of the "fractal" nature
of the stability boundary: the codes share the `iter2>1` bug but differ
elsewhere, so their failure sets differ.)

## Reproducing

No SIMSOPT / makegrid / QUASR access required -- just the checked-in INDATA +
mgrid and the two vmecpp releases.

```bash
# vmecpp: install the two releases in separate environments and run the CLI
pip install vmecpp==0.7.0   # -> converges
pip install vmecpp==0.6.1   # -> fails at the ns=201 transition
python -m vmecpp input.quasr65579_12-50-201        # multigrid: [12,50,201]
python -m vmecpp input.quasr65579_12-50-201 ...    # (edit NS_ARRAY to 201 for the cold-start case)

# PARVMEC (stock): reads the same INDATA + mgrid file directly
xvmec input.quasr65579_12-50-201

# regenerate the figure from the force-residual logs in logs/
python plot_counterexample.py logs
```

`plot_counterexample.py` parses the VMEC/PARVMEC iteration tables in `logs/`;
regenerate those logs by capturing the solver output of the runs above.
