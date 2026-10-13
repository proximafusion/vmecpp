# Tutorials

These tutorials provide step-by-step guides for running VMEC++ and troubleshooting common issues:

- {doc}`fixed_boundary`: Learn how to set up and run fixed-boundary equilibria, with recommended settings for stability analysis (`lbsubs`), boundary representation (spectral condensation), and weakly shaped configurations (like QA).
- {doc}`free_boundary`: Learn how to set up and run free-boundary equilibria with external coil fields, including both MGRID files and in-memory response tables.
- {doc}`debugging_failures`: Practical advice for diagnosing stalled or unstable solves, including tuning `delt`, handling strongly shaped free boundaries near islands or separatrices, and checking Jacobian errors.

```{toctree}
:maxdepth: 1
:hidden:

fixed_boundary
free_boundary
debugging_failures
```
