# Test Data for VMEC++

All combinations of the following choices should be tested:

0. fixed-boundary (F) and free-boundary (T)
1. constrained-iota (F) and constrained-current (T)
2. axisymmetric Tokamak (F) and three-dimensional stellarator (T)
3. stellarator-symmetric (F) and non-stellarator-symmetric (T)

| case name                       | 3 | 2 | 1 | 0 |
|---------------------------------|---|---|---|---|
| `solovev`                       | F | F | F | F |
| `solovev_free_bdy`              | F | F | F | T |
| `solovev_analytical`            | F | F | T | F |
| `solovev_free_bdy_current`      | F | F | T | T |
| `cth_like_fixed_bdy_iota`       | F | T | F | F |
| `cth_like_free_bdy_iota`        | F | T | F | T |
| `cth_like_fixed_bdy`            | F | T | T | F |
| `cth_like_free_bdy`             | F | T | T | T |
| `up_down_asym`                  | T | F | F | F |
| `up_down_asym_free_bdy`         | T | F | F | T |
| `up_down_asym_current`          | T | F | T | F |
| `up_down_asym_current_free_bdy` | T | F | T | T |
| `cth_like_fixed_bdy_asym_iota`  | T | T | F | F |
| `cth_like_free_bdy_asym_iota`   | T | T | F | T |
| `cth_like_fixed_bdy_asym`       | T | T | T | F |
| `cth_like_free_bdy_asym`        | T | T | T | T |

The two `up_down_asym` free-boundary cases use `mgrid_solovev`, and get their
asymmetry from different currents in its upper and lower PF_02 and PF_03 coils.
