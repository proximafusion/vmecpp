# Free-boundary multigrid transition demo (W7-X)

This demonstrates the force-balance spike at free-boundary multigrid stage
transitions that was fixed in "Seed the vacuum state across free-boundary
multigrid transitions" (#663), and shows that the same behavior reproduces
in PARVMEC (Fortran VMEC's parallel successor), confirming it is an
inherited bug rather than something introduced by VMEC++.

## Root cause

Both PARVMEC (`Sources/General/funct3d.f`) and VMEC++
(`ideal_mhd_model.cc`) gate the whole free-boundary/vacuum force block on
`iter2 > 1`:

```fortran
IF (lfreeb .and. iter2.gt.1 .and. iequi.eq.0) THEN
   ...
   CALL vacuum(...)              ! full NESTOR solve
   gcon(l) = bsqvac(l) + presf_ns  ! force-balanced edge term
   ...
END IF
```

`iter2` is reset to 1 at the start of *every* multigrid continuation stage
(`initialize_radial.f:41` in PARVMEC; `Vmec::InitializeRadial` in VMEC++).
So the very first iteration of every new (finer) grid stage skips the
vacuum solve entirely, and the edge force is assembled with a stale or
zeroed vacuum pressure instead of the true vacuum solution -- even though
the previous stage's NESTOR solution is still exactly valid, since the
angular grid and LCFS geometry are unchanged by radial interpolation.

The fix seeds `vacuum_pressure_state_ = kInitialized` in
`InitializeRadial` on free-boundary continuation stages (mirroring the
existing hot-restart path), so the new stage's first iteration runs the
full vacuum block instead of skipping it.

## Demo case

`input.w7x_free_bdy_multigrid_demo` / `w7x_free_bdy_multigrid_demo.json`:
W7-X free-boundary equilibrium (same boundary/coils as VMEC++'s
`w7x_free_bdy_vac.json`) with a 4-stage multigrid sequence
`ns_array = 4, 9, 28, 99` -- a large jump in the last two transitions.

Three runs, same input:

- `vmecpp_bug_present.log`: VMEC++ with the #663 fix reverted (`vacuum_pressure_state_`
  not re-seeded across transitions).
- `vmecpp_fixed.log`: VMEC++ with the fix applied.
- `parvmec.log`: PARVMEC (Fortran), run directly with `xvmec` on the
  equivalent INDATA file, unmodified upstream source.

Run `python3 plot_transition_comparison.py` to regenerate the plots and the
summary table from the logs.

## Result

FSQR at iteration 1 of each new stage (force residual right after the
radial grid is refined):

| ns transition | vmecpp, bug present | vmecpp, fixed | PARVMEC (Fortran) |
|---|---|---|---|
| 4 -> 9   | 3.4e-02 | 3.4e-02 | 3.8e-02 |
| 9 -> 28  | **1.1e+01** | 9.5e-03 | **5.5e+00** |
| 28 -> 99 | **4.0e+01** | 1.5e-02 | **2.1e+01** |

(The 4->9 transition is unaffected because the vacuum field only just
turned on partway through the ns=4 stage, so there is nothing stale to
carry across yet.)

PARVMEC shows the same qualitative signature as unfixed VMEC++: a residual
spike of order 1-40 right at stage entry that grows as the grid is
refined (the problem gets stiffer at higher `ns`), followed by a slow
ring-down. VMEC++ with the fix applied stays within ~1e-2 of the
pre-transition residual level at every stage -- confirming the bug is
inherited from Fortran VMEC's flow control and is not VMEC++-specific, and
that seeding the vacuum state across transitions removes it in both
codes' shared mechanism.

See `multigrid_transition_comparison.png` for the full per-iteration
force-residual history.
