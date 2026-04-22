# VMEC++ Radial Preconditioner Diagnostics

This directory contains tools for analyzing and diagnosing the radial preconditioner
used in VMEC++ for solving MHD equilibrium equations.

## Background

The VMEC++ radial preconditioner is derived from the highest-order radial derivatives
in the MHD force terms. For each Fourier mode (m, n), the preconditioner forms a
tridiagonal system in the radial direction:

```
a[j] * x[j+1] + d[j] * x[j] + b[j] * x[j-1] = c[j]
```

The preconditioner approximates the inverse of this system using the Thomas algorithm.

## Files

- `analyze_preconditioner.py`: Main analysis module with:
  - Synthetic preconditioner matrix generation based on VMEC physics
  - Condition number computation and analysis
  - Matrix norm calculations (1-norm, inf-norm, 2-norm, Frobenius)
  - Diagonal preconditioning strategy comparison
  - Plot generation and markdown report creation

## Usage

### Basic Usage

Run the analysis with default parameters:

```bash
python analyze_preconditioner.py output_directory
```

This will generate:
- Several diagnostic plots (PNG format)
- A comprehensive markdown report

### Programmatic Usage

```python
from analyze_preconditioner import run_analysis, create_synthetic_vmec_preconditioner

# Run complete analysis
report_path = run_analysis(
    output_dir="my_analysis_output",
    ns=51,      # Number of radial surfaces
    mpol=6,     # Maximum poloidal mode number
    ntor=5,     # Maximum toroidal mode number
    nfp=5       # Number of field periods
)

# Or create diagnostics data for custom analysis
diagnostics = create_synthetic_vmec_preconditioner(ns=101, mpol=12, ntor=10, nfp=5)
```

## Generated Output

### Plots

1. **matrix_structure.png**: Visualization of the tridiagonal matrix structure for
   representative Fourier modes

2. **condition_numbers.png**: Condition number vs mode number analysis showing how
   conditioning varies across the Fourier spectrum

3. **matrix_norms.png**: Heatmaps of different matrix norms (1-norm, inf-norm, 2-norm,
   Frobenius) across all modes

4. **preconditioning_improvement.png**: Comparison of different diagonal preconditioning
   strategies and their improvement factors

5. **tridiagonal_elements.png**: Radial profiles of the tridiagonal elements showing
   the coupling structure

6. **lambda_preconditioner.png**: Profile of the lambda (magnetic stream function)
   preconditioner

### Report

`preconditioner_analysis_report.md`: A comprehensive markdown report including:
- Executive summary with key findings
- Background on the physics of the radial preconditioner
- Analysis results with embedded plots
- Recommendations for improvements
- Technical details

## Physics Background

The radial preconditioner captures the dominant terms from the MHD forces:

```
F_R = d/ds (Z_theta * P) + ...
F_Z = -d/ds (R_theta * P) + ...
```

where P = R * (p + |B|^2 / (2 * mu_0)) is the total pressure times the major radius.

The second-order radial derivatives appear through:

```
d(sqrt(g))/ds = R * (R_theta * d^2Z/ds^2 - Z_theta * d^2R/ds^2) + ...
```

This leads to coupling between adjacent radial surfaces, resulting in the
tridiagonal structure.

## Key Findings

Based on the analysis:

1. **Condition numbers** vary significantly (10^2 to 10^5) across Fourier modes,
   with higher m modes generally having worse conditioning

2. **Column scaling** provides the best diagonal preconditioning improvement
   (5-10x reduction in condition number on average)

3. **Edge and axis treatment** significantly affect the overall conditioning

4. **Mode-dependent preconditioning** could provide further improvements by
   tailoring the scaling to each (m, n) mode's characteristics

## Future Work

- Integration with actual VMEC++ runs to extract real matrices
- Adaptive preconditioning based on mode-dependent analysis
- Investigation of more sophisticated preconditioners (ILU, multigrid, etc.)

## Experimental Results: Column Scaling Preconditioning

### Background

The analysis above suggested that column scaling (dividing each column by its
1-norm) could improve the condition number of the tridiagonal preconditioner
matrices by 5-6x on average.

### Implementation

Column scaling was implemented in the tridiagonal solver by:
1. Computing the 1-norm of each column
2. Scaling the matrix elements accordingly
3. Solving with the Thomas algorithm
4. Un-scaling the solution

### Results

Testing on standard VMEC++ test cases showed **no improvement** in convergence:

| Test Case | Iterations (baseline) | Iterations (with column scaling) |
|-----------|----------------------|----------------------------------|
| W7X       | 2961                 | 2961                             |
| CTH-like  | 3025                 | 3025                             |
| Solovev   | 903                  | 903                              |

### Explanation

Column scaling in the tridiagonal solver does not affect convergence because:

1. **Numerical stability**: The Thomas algorithm is already numerically stable
   for well-conditioned tridiagonal systems.

2. **Same mathematical solution**: Column scaling changes the numerical path
   but not the mathematical solution, so the outer iteration behavior is
   unchanged.

3. **Well-conditioned matrices**: The actual VMEC preconditioner matrices
   appear to be better conditioned than the synthetic matrices used in the
   analysis.

### Conclusion

To improve VMEC++ convergence, more fundamental changes to the preconditioning
operator would be needed, such as:
- Mode-dependent scaling in the preconditioner assembly
- Adaptive time stepping based on convergence rate
- Improved edge and axis treatment

See `compare_convergence.py` for the testing script and
`output/convergence_comparison_report.md` for detailed results.
