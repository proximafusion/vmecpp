# 1 "/home/runner/work/vmecpp/vmecpp/build/_deps/indata2json-src/LIBSTELL/Sources/Modules/vparams.f"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/home/runner/work/vmecpp/vmecpp/build/_deps/indata2json-src/LIBSTELL/Sources/Modules/vparams.f"
      MODULE vparams
      USE stel_kinds
      USE stel_constants, ONLY: zero,twopi,mu0,one
      IMPLICIT NONE
!-----------------------------------------------
!   L o c a l   P a r a m e t e r s
!-----------------------------------------------
!
!     MAXIMUM PARAMETERS FOR VMEC CODE (FOR READING INPUT)
!     USER SHOULD NOT ALTER THESE
!
      INTEGER, PARAMETER :: nsd = 10001    !maximum number of radial nodes
      INTEGER, PARAMETER :: mpold = 101    !maximum number of poloidal harmonics (in r,z,lam fourier series)
      INTEGER, PARAMETER :: ntord = 101    !maximum number of toroidal harmonics
      INTEGER, PARAMETER :: ndatafmax  = nsd
      INTEGER, PARAMETER :: nstore_seq = 100

!
!     CONSTANTS
!
      INTEGER, PARAMETER :: nthreed0=9, nmac0=nthreed0+1,
     1                      indata0=nthreed0+2, nwout0=nthreed0+3,
     2                      jxbout0=nthreed0+4
      INTEGER, PARAMETER :: nfort8=8, nfort18=18
      INTEGER, PARAMETER :: nlog0=51, nmercier0=52
      INTEGER            :: nthreed, nmac, nlog=nlog0

!
!     DERIVED (FROM FUNDAMENTAL) PARAMETERS FOR VMEC CODE
!
      INTEGER, PARAMETER :: mpol1d = mpold - 1
      INTEGER, PARAMETER :: ntor1d = 1 + ntord

!
!     MISCELLANEOUS PARAMETERS
!
      REAL(rprec), PARAMETER :: c1pm2  =  1.e-2_dp
      REAL(rprec), PARAMETER :: cp15   =  0.15_dp
      REAL(rprec), PARAMETER :: cp25   = 0.25_dp
      REAL(rprec), PARAMETER :: cp5    = 0.50_dp
      REAL(rprec), PARAMETER :: c1pm8  = 1.0e-8_dp
      REAL(rprec), PARAMETER :: cbig   = 0.9e30_dp
      REAL(rprec), PARAMETER :: c2p0   = 2
      REAL(rprec), PARAMETER :: c3p0   = 3
      REAL(rprec), PARAMETER :: cp05   = 0.05_dp
      REAL(rprec), PARAMETER :: c1pm13 = 1.0e-13_dp
      REAL(rprec), PARAMETER :: osqrt2 = 0.707106781186547462_dp
      REAL(rprec), PARAMETER :: epstan = EPSILON(zero)

      END MODULE vparams
