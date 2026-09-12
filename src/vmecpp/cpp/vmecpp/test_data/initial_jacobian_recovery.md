The boundary in `initial_jacobian_recovery.json` is the outermost surface of
[QUASR configuration 14802](https://quasr.flatironinstitute.org/simsopt_serials/0014/serial0014802.json),
converted with SIMSOPT to `mpol=ntor=6`. The enclosed toroidal flux is evaluated
from the prescribed coil vector potential. This fixture imposes that boundary
with zero pressure and prescribed zero toroidal current.

The requested radial sequence is `[8, 16, 31]`. Its initial interpolated geometry
has a bad Jacobian after the axis search. A cold three-surface bootstrap reaches
a geometry from which the requested sequence converges. Every boundary Fourier
coefficient is retained throughout the recovery.

`wout_initial_jacobian_recovery.nc` was regenerated with educational_VMEC 8.52
at commit `ce663b20746524f391848f2841fa488bf718c923`
([reference fix](https://github.com/jonathanschilling/educational_VMEC/pull/29)),
using the same input and `delt=0.25`. The reference
checks the full geometry, lambda and iota arrays as well as pressure and flux.
