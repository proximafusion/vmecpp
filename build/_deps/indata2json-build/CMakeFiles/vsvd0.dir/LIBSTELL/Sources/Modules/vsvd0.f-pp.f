# 1 "/home/runner/work/vmecpp/vmecpp/build/_deps/indata2json-src/LIBSTELL/Sources/Modules/vsvd0.f"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/home/runner/work/vmecpp/vmecpp/build/_deps/indata2json-src/LIBSTELL/Sources/Modules/vsvd0.f"
      MODULE vsvd0
      USE stel_kinds
      IMPLICIT NONE
      REAL(rprec), PARAMETER :: fturnon_axis = 3.e-9_dp
      REAL(rprec), PARAMETER :: fopt_axis = 3.e-2_dp*fturnon_axis
      INTEGER, PARAMETER :: isamecoil = -2
      INTEGER, PARAMETER :: needit = -1
      INTEGER, PARAMETER :: idontneed = 0
      INTEGER, PARAMETER :: isymcoil = 1
      INTEGER, PARAMETER :: ithom0 = 1
      INTEGER, PARAMETER :: istark0 = 2
      INTEGER, PARAMETER :: islope0 = 3
      INTEGER, PARAMETER :: icurr0 = 4
      INTEGER, PARAMETER :: idiam0 = 5
      INTEGER, PARAMETER :: iflxs0 = 6
      INTEGER, PARAMETER :: ibrzfld = 7
      INTEGER, PARAMETER :: natur = 0
      INTEGER, PARAMETER :: ideriv = 1
      INTEGER, PARAMETER :: intder = 1
      INTEGER, PARAMETER :: intfun = 2
      INTEGER, PARAMETER :: nmse = 100        !number of mse measurements
      INTEGER, PARAMETER :: ntse = 100        !number of thompson scattering measurements
      INTEGER, PARAMETER :: nfloops = 100     !number of external poloidal flux loops
      INTEGER, PARAMETER :: nbsetsp = 5       !number of external b-field loop sets allowed
      INTEGER, PARAMETER :: nbcoilsp = 100    !number of external b-field coils per set
      INTEGER, PARAMETER :: nbctotp = nbsetsp*nbcoilsp
      INTEGER, PARAMETER :: jngrn = 1001      !number of "Greens Function" points
      INTEGER, PARAMETER :: jchix = 7         !number of data types contributing to rms error match
      INTEGER, PARAMETER :: mstp = 100        !number of time steps to store chisq error
      INTEGER, PARAMETER :: jchix1 = jchix + 1
      INTEGER, PARAMETER :: nparts = 4        !number of items needed to specify pf coils
      INTEGER, PARAMETER :: npfcoil = 40      !number of filaments in pf coil pack (for plotting)
      INTEGER, PARAMETER :: nigroup = 300     !number of external current groups
      INTEGER, PARAMETER :: ipedsvd = 8
      END MODULE vsvd0
