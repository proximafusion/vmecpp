# Test Data for VMEC++

All combinations of the following choices should be tested:

0. fixed-boundary (F) and free-boundary (T)
1. constrained-iota (F) and constrained-current (T)
2. axisymmetric Tokamak (F) and three-dimensional stellarator (T)
3. stellarator-symmetric (F) and non-stellarator-symmetric (T)

| case name                      | 3 | 2 | 1 | 0 |
|--------------------------------|---|---|---|---|
| `solovev`                      | F | F | F | F |
| `solovev_free_bdy`             | F | F | F | T |
| `solovev_analytical`           | F | F | T | F |
| TODO                           | F | F | T | T |
| `cth_like_fixed_bdy_iota`      | F | T | F | F |
| TODO                           | F | T | F | T |
| `cth_like_fixed_bdy`           | F | T | T | F |
| `cth_like_free_bdy`            | F | T | T | T |
| `up_down_asym`                 | T | F | F | F |
| TODO                           | T | F | F | T |
| `up_down_asym_current`         | T | F | T | F |
| TODO                           | T | F | T | T |
| `cth_like_fixed_bdy_asym_iota` | T | T | F | F |
| TODO                           | T | T | F | T |
| `cth_like_fixed_bdy_asym`      | T | T | T | F |
| `cth_like_free_bdy_asym`       | T | T | T | T |

The remaining four are free-boundary and each needs an mgrid file for its
symmetry class.
