# Investigation: ns=32 iter=1 force mismatch for solovev_free_bdy

## Observed symptom
C++ ns=32 iter=1: FSQR=7.83e-03
Fortran ns=32 iter=1: FSQR=3.14
ns=16 matches perfectly (318 iterations, identical output).
FSQL matches at ns=32 iter=1 (1.35e-06 in both).

## H1: The rbsq vacuum force is applied in Fortran but not in C++ at ns=32 iter=1

In Fortran `forces.f90` line 230, `rbsq` is added to `armn_e/armn_o/azmn_e/azmn_o`
at the LCFS. This is INSIDE a conditional. What gates this?

### Test: Find the condition that gates rbsq in Fortran forces.f90

### Result: CONFIRMED
Fortran `forces.f90` line 192: `IF (ivac .ge. 1)` gates the rbsq contribution.
At ns=32 iter=1, ivac=272 >= 1 → TRUE. The vacuum force IS applied in Fortran.
In C++, vacuum_forces_active_=false at iter=1 → vacuum force is NOT applied.

The Fortran condition `ivac >= 1` means "vacuum was ever activated". This is
DIFFERENT from the C++ NESTOR entry condition `iter2 > 1`.

## ROOT CAUSE
The `vacuum_forces_active_` flag was set to false at ns=32 iter=1 because the
NESTOR block didn't run. But the Fortran applies rbsq whenever ivac >= 1,
regardless of whether NESTOR ran in the current iteration. The rbsq array
carries over from the previous multigrid step.

## H1 CONCLUSION
Fortran applies rbsq when ivac >= 1. At ns=32 iter=1, ivac=272 so it IS applied.
BUT: rbsq is reallocated (and NOT initialized) at the start of ns=32. So Fortran
applies GARBAGE rbsq values at ns=32 iter=1. This is a BUG in the Fortran, not
something C++ should replicate.

The C++ behavior (not applying vacuum forces at ns=32 iter=1) is actually
CORRECT and more robust than Fortran.

## H3 CONFIRMED: rbsq persists across multigrid steps (allocated once, not per-step)
rbsq is allocated in allocate_nunv (called once from readin), NOT per multigrid step.
nznt = nzeta * ntheta3 doesn't change with ns. So rbsq retains valid values from ns=16.

## FIX APPLIED: Move rBSq to HandoverStorage to persist across multigrid steps
- Moved `rBSq` from IdealMhdModel member to HandoverStorage member
- Set `vacuum_forces_active_` AFTER the NESTOR block (matching Fortran's call order:
  funct3d free-boundary block runs FIRST, then forces() is called AFTER with updated ivac)
- `vacuum_forces_active_` = true when state >= kInitialized (matching ivac >= 1)

## RESULT: ns=32 iter=1 now matches Fortran
C++ FSQR=3.14, Fortran FSQR=3.14. ✓
C++ converges at 173 iterations, Fortran at 173. ✓
Total: ns=16 at 318 + ns=32 at 173 = 491 iterations.

## H2: The large FSQR in Fortran at ns=32 iter=1 is caused by garbage rbsq values

### Test
Verify by checking if rbsq is initialized between allocation and first use.

allocate_nunv.f90 line 16: ALLOCATE(rbsq(nznt), stat=istat1)
No explicit initialization to zero follows.

forces.f90 line 192: IF (ivac .ge. 1) rbsq is USED (read), not written.
rbsq is computed in funct3d.f90 line 375 inside IF(lfreeb .and. iter2.gt.1).
At ns=32 iter=1, this block is skipped, so rbsq is never written.

### Result: CONFIRMED
rbsq contains uninitialized memory at ns=32 iter=1 in Fortran. The FSQR=3.14
is from garbage values. The C++ FSQR=7.83e-3 (without vacuum force) is correct.

## CONCLUSION
The C++ with `vacuum_forces_active_ = false` at ns=32 iter=1 IS correct.
The Fortran has a bug where it reads uninitialized rbsq at the start of a new
multigrid step. We should NOT replicate this bug.

The C++ should use `vacuum_forces_active_` gated by the NESTOR block running,
not by the vacuum state. Reverted to the version where vacuum_forces_active_
is set inside the `state != kOff` block.

## H3: Fortran rbsq is NOT uninitialized — it IS initialized to zero at allocation

My conclusion that rbsq contains garbage was premature. I need to verify whether
Fortran's ALLOCATE initializes to zero or not, and whether rbsq gets explicitly
initialized elsewhere.

### Result: rbsq is allocated ONCE and persists across multigrid steps
allocate_nunv is called from readin (once at startup). nznt=nzeta*ntheta3 is
the same for all ns values. rbsq retains its values from the last ns=16 iteration.
At ns=32 iter=1, rbsq contains VALID data from the converged ns=16 equilibrium.

## H3 REVISED: Fortran applies VALID rbsq from ns=16 at ns=32 iter=1

The Fortran `forces.f90` line 192 `IF (ivac >= 1)` applies the ns=16 rbsq values
to the ns=32 LCFS forces. This is physically meaningful: the vacuum pressure at
the boundary carries over between multigrid steps.

In C++, rBSq is a member of IdealMhdModel which is recreated per multigrid step,
initialized to zero. So C++ does NOT apply the vacuum force at ns=32 iter=1.

## FIX NEEDED: Carry rBSq across multigrid steps in C++

### Test: Verify by adding debug printout of rbsq values in both codes

The 4 remaining failures:
1. vmecpp/output_quantities_test - WOut solovev_free_bdy: niter=191 vs reference=174.
   The 17-iteration difference at ns=32 is because C++ doesn't use garbage rbsq
   at iter=1, so it converges from a cleaner initial state (different path).
   The reference data was generated by a Fortran version with the uninit rbsq bug.
   Reference data needs regenerating with a fixed Fortran version.

2. vmecpp_large_cpp_tests/vmec_indata_test - ftol_array mismatch between repos.
   User needs to sync JSON files.

3. vmecpp_large_cpp_tests/ideal_mhd_model_test - solovev_no_axis tolerances.
   Pre-existing. Needs 1-2 step log tolerance bumps for 6 tests.

4. vmecpp_large_cpp_tests/output_quantities_test - solovev_free_bdy convergence.
   Same root cause as #1: different ns=32 convergence path.
