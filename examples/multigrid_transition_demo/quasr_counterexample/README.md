# QUASR counterexample: a case the #663 fix rescues

The sibling `../README.md` demo shows that the free-boundary multigrid
transition bug (skipping the vacuum `B^2` contribution for one iteration at
every continuation stage, gated on `iter2 > 1`) produces a force-residual
spike but that W7-X still converges either way. This directory provides the
stronger statement requested on the mailing-list thread: a **highly shaped,
large-`ns` free-boundary case that does not converge without the fix**, in
both stock PARVMEC and vmecpp 0.6.1, but converges cleanly in vmecpp 0.7.0.

## The counterexample: QUASR-0065579

A `nfp = 4` stellarator from the QUASR database
(`tests/data/quasr/serial0065579.json`), run free-boundary with the coil set
that produced it (mgrid built from the coils; `phiedge` = enclosed vacuum
flux). Resolution `mpol = ntor = 6`, `nzeta = 24`, target `ftol = 1e-9`,
`delt = 0.9`.

**Cold start is impossible.** A single `ns = 201` grid fails immediately
("solver failed during the first iterations" -- the cold interpolated guess
self-intersects on a 201-surface mesh), so a multigrid ramp
`ns_array = [12, 50, 201]` is *required*. See
`logs/coldstart_ns201_fails.log`.

**On that required multigrid ramp:**

| code | outcome at `ns_array=[12,50,201]` |
|------|-----------------------------------|
| **vmecpp 0.7.0** (fix) | **converges**, FSQR = 9.9e-10, 3065 cumulative iters |
| **vmecpp 0.6.1** (bug) | **fails** -- FSQR diverges to ~9e6 at the `ns=201` stage, bad-Jacobian abort |
| **PARVMEC** (stock, unmodified) | **fails** -- FSQR spikes to ~4e3 at `ns=201` entry, oscillates with 37 Jacobian resets, hits the iteration cap |

`quasr_counterexample.png`, panel A, overlays the three force-residual
histories on a shared cumulative-iteration axis. 0.7.0 (green) descends
cleanly through every transition to `ftol`; 0.6.1 (red) and PARVMEC (blue)
never recover from the `ns=201` transition kick.

## The fix, isolated

vmecpp 0.7.0 differs from 0.6.1 in *two* flow-control changes (a restart-
mechanism fix that brings vmecpp in line with PARVMEC, and the #663 vacuum-
seed fix). The failure here is attributable specifically to **#663**:

* **PARVMEC already has** the restart behaviour that 0.7.0 adopted, yet it
  still fails on this case -- so 0.7.0's success is not the restart change.
* Reverting **only** the 18-line #663 patch
  (`vacuum_pressure_state_ = kInitialized` in `Vmec::InitializeRadial`) from
  the current source and rebuilding flips the result from converge to fail
  **at the exact `50 -> 201` transition** ("solver failed during the first
  iterations"); see `logs/fix663_reverted_ns12-50-201.log`. Restoring the
  patch restores convergence.

This is the mechanism the `iter2 > 1` gate causes: at stage entry the LCFS
force is applied with the vacuum `B^2` term missing (visible as `DELBSQ = -nan`
on iteration 1 of the new stage in `logs/vmecpp061_ns12-50-201.log`),
inflating the boundary; at `ns = 201` the problem is stiff enough that the
kick self-intersects the flux surfaces and the solve cannot recover.

## A "just faster" case: QUASR-0029346

Panel B shows a `nfp = 2` case (`serial0029346.json`) where both versions
converge on `ns_array = [8, 16, 31]`, but 0.7.0 needs fewer iterations:

| version | ns=8 | ns=16 | ns=31 | cumulative |
|---------|------|-------|-------|------------|
| 0.6.1   | 739  | 789   | 779   | 2307 |
| 0.7.0   | 739  | 292   | 274   | 1305 |

The `ns=8` stage is identical (no transition has happened yet); the fix pays
off at each subsequent transition, ~1.8x fewer total iterations (2.8x on the
final grid). (At `ns_array=[12,50,201]` this same config is another
fail->converge case for vmecpp, though PARVMEC happens to converge on it --
an illustration of the "fractal" nature of the stability boundary: the codes
share the `iter2>1` bug but differ elsewhere, so their failure sets differ.)

## Reproducing

Requires `vmecpp`, `simsopt`, and (for the PARVMEC/mgrid comparison) a built
`makegrid` (`mgrid`) and `xvmec`. The QUASR serials are already checked in
under `tests/data/quasr/`.

```bash
# 1. Build the cached free-boundary config (mgrid response table + boundary + phiedge)
python build_config.py ../../../tests/data/quasr/serial0065579.json cache/65579 6 6 24 101 0.4

# 2. vmecpp runs (swap the interpreter for a 0.6.1 vs 0.7.0 venv)
python run_case.py cache/65579 12,50,201 0.9 3000 1e-9 logs/vmecpp070_ns12-50-201.log   # 0.7.0 venv
python run_case.py cache/65579 201       0.9 3000 1e-9 logs/coldstart_ns201_fails.log

# 3. PARVMEC: mgrid via makegrid (R-mode, EXTCUR=1.0), then xvmec on the INDATA
#    (see input.quasr65579_12-50-201; write_indata.py regenerates it)
mgrid quasr0065579 R T <rmin> <rmax> <zmin> <zmax> 24 101 101   # in cache/65579
xvmec input.quasr65579_12-50-201

# 4. Figure
python plot_counterexample.py logs
```

`run_case.py` uses the in-memory mgrid response table; `run_case_file.py`
uses an on-disk `mgrid_*.nc` (validated to reproduce the in-memory result
bit-for-bit at `EXTCUR_OVERRIDE=1.0`) -- the exact same field PARVMEC reads.
