# Test Data for VMEC++

All combinations of the following choices should be tested:

0. fixed-boundary (F) and free-boundary (T)
1. constrained-iota (F) and constrained-current (T)
2. axisymmetric Tokamak (F) and three-dimensional stellarator (T)
3. stellarator-symmetric (F) and non-stellarator-symmetric (T)

| case name            | 3 | 2 | 1 | 0 |
|----------------------|---|---|---|---|
| `solovev`            | F | F | F | F |
| TODO                 | F | F | F | T |
| TODO                 | F | F | T | F |
| TODO                 | F | F | T | T |
| TODO                 | F | T | F | F |
| TODO                 | F | T | F | T |
| `cth_like_fixed_bdy` | F | T | T | F |
| TODO                 | F | T | T | T |
| TODO                 | T | F | F | F |
| TODO                 | T | F | F | T |
| TODO                 | T | F | T | F |
| TODO                 | T | F | T | T |
| TODO                 | T | T | F | F |
| TODO                 | T | T | F | T |
| TODO                 | T | T | T | F |
| TODO                 | T | T | T | T |
