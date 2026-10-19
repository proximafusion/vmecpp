# lbsubs current reconstruction

`lbsubs=True` now uses the same radial derivative when solving for B_s and
calculating currents in C++, JAX, and wout. It also supports asymmetric
configurations. Vacuum uses the same solve as every other case.

The equilibrium iteration and output formats are unchanged. The existing
W7-X plotting example now reads exported currents instead of taking its own
finite differences.

## What the comparison shows

The old run used `lbsubs=True`, but its two current outputs disagreed. The
solve and jxbout used one radial derivative; the wout export used another.
This produced the large axis artifacts in the original before plots.

Replaying the saved baseline gives these maximum errors on the same points:

| Case | ns | Old wout [A/m^2] | Old jxbout [A/m^2] |
| --- | --- | --- | --- |
| Analytical 3D case | 13 | 70894 | 876 |
| Vacuum | 17 | 295465 | 284 |

![Old current outputs with lbsubs enabled and disabled](baseline_current_paths.png)

Most of the large improvement in the symmetric case is a correction to the
exported current, rather than an improvement to the original solve. The old
asymmetric run reported that lbsubs was unsupported and used B_s from the
geometry instead.

## Accuracy and remaining errors

For the analytical cases, the reference current is the curl of the known
magnetic field at the computed positions. We sampled every interior surface
at 96 poloidal angles and four toroidal planes. The axis and boundary were
excluded. Halving the reference derivative step changed the reference by less
than 4.6e-5 A/m^2.

The asymmetric comparison below uses the plane with the largest old error.
The top panels share a color scale; the lower-left panel expands the remaining
error. The hole in the center is the unsampled region inside the first surface.

![Asymmetric analytical current error before and after](sheared-asym_current.png)

Increasing ns from 13 to 49 reduces the new maximum error from 876 to
415 A/m^2 in the symmetric case and from 893 to 411 A/m^2 in the asymmetric
case. The remaining error is largest near the axis.

![Maximum current error under radial refinement](current_convergence.png)

Vacuum does not show the same convergence. At force tolerance 1e-16, the new
peak residual is 281, 302, and 1079 A/m^2 for ns=17, 33, and 65. The finest
run's largest error is at s=0.578125, away from the axis. The signed traces show
that oscillations and a sharp interior peak remain.

![Signed current errors at fixed angular positions](signed_residuals.png)

At ns=33, tightening the force tolerance to 1e-20 reduces the vacuum maximum
to 239 A/m^2; increasing the angular resolution leaves it at 248 A/m^2.
The current is closer to zero than the old export, but this scan does not
establish zero-current convergence.

The 5.9%-beta W7-X case also remains unresolved: its outer-surface peak grows
from 1.83e6 to 1.74e7 A/m^2 between ns=17 and 33. The first-surface Mercier
value still grows with refinement; its correctness has not been established.
If a B_s solve fails, C++ logs a warning and keeps the geometry-based value
on that surface. JAX currently does so silently.

## Checks and reproduction

CMake and Bazel builds passed. All six C++ lbsubs tests, 19 Python output and
JAX checks, and eight symmetry checks passed. The full C++ suite passed 27 of
28 targets; the existing singular-integral polynomial test still fails at a
2.069e-11 difference against a 1e-11 tolerance. Formatting and type checks passed.

[validate.py](validate.py) reproduces the runs and four figures. It lives with
this report because it is a validation tool, rather than a usage example.
Run the before stage with the original build and the after stage with the fix:

```bash
python docs/reports/lbsubs/validate.py --stage before --cases sheared sheared-asym --ns 13 25 49
python docs/reports/lbsubs/validate.py --stage after --cases sheared sheared-asym --ns 13 25 49
python docs/reports/lbsubs/validate.py --stage before --cases vacuum --ns 17 33 65
python docs/reports/lbsubs/validate.py --stage after --cases vacuum --ns 17 33 65
python docs/reports/lbsubs/validate.py --plot
```

Metrics are in [data_final](data_final/). The refreshed vacuum runs restarted
from saved equilibria and converged again to the requested tolerance.
[baseline_audit.json](baseline_audit.json) records the baseline replay and
library checksum; [validation.json](validation.json) records the reference
checks. The numerical checks used an isolated build without Enzyme and left
the installed Enzyme build unchanged; Enzyme derivatives were not validated.
