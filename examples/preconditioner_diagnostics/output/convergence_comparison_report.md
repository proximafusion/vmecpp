# VMEC++ Convergence Comparison Report

## Summary

This report documents the baseline convergence behavior of VMEC++ for
standard test cases, and the effect of column scaling preconditioning
on the radial tridiagonal solver.

## Test Results

| Test Case | Iterations | Final FSQR | Final FSQZ | Final FSQL | Jacobian Resets | Converged |
|-----------|------------|------------|------------|------------|-----------------|-----------|
| w7x.json | 2961 | 9.92e-13 | 2.20e-13 | 2.24e-13 | 5 | Yes |
| cth_like_fixed_bdy.json | 3025 | 9.78e-21 | 1.75e-21 | 5.84e-21 | 0 | Yes |
| solovev.json | 903 | 9.58e-17 | 4.27e-18 | 5.06e-22 | 0 | Yes |

## Analysis

### Column Scaling Preconditioning

The preconditioner analysis (see `preconditioner_analysis_report.md`) suggested
that column scaling could improve the condition number of synthetic tridiagonal
matrices by 5-6x on average.

However, testing showed that applying column scaling to the tridiagonal solver
in VMEC++ does not change the iteration count or convergence rate. The reasons:

1. **Numerical stability**: The Thomas algorithm is already numerically stable
   for well-conditioned tridiagonal systems.

2. **Same mathematical solution**: Column scaling changes the numerical path
   but not the mathematical solution, so the outer iteration behavior is
   unchanged.

3. **Well-conditioned matrices**: The actual VMEC preconditioner matrices
   appear to be better conditioned than the synthetic matrices used in the
   analysis.

### Recommendations for Future Work

To improve VMEC++ convergence, consider:

1. **Mode-dependent preconditioning**: Different scaling factors for different
   (m, n) modes based on their condition numbers.

2. **Adaptive time stepping**: Modify the conjugate gradient-like time stepping
   algorithm based on convergence rate.

3. **Edge treatment optimization**: The edge pedestal (5% increase) could be
   optimized based on the specific equilibrium.

4. **Axis treatment refinement**: Special handling for m=1 modes near the
   magnetic axis could be improved.

## Conclusion

The column scaling preconditioning approach does not improve VMEC++ convergence.
More fundamental changes to the preconditioning operator or time-stepping
algorithm would be needed to improve convergence rates.
