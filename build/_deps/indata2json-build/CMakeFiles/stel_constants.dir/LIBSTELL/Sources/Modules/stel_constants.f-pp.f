# 1 "/home/runner/work/vmecpp/vmecpp/build/_deps/indata2json-src/LIBSTELL/Sources/Modules/stel_constants.f"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/home/runner/work/vmecpp/vmecpp/build/_deps/indata2json-src/LIBSTELL/Sources/Modules/stel_constants.f"
      MODULE stel_constants

      USE stel_kinds, ONLY: rprec, dp

!----------------------------------------------------------------------
!  Mathematical constants
!----------------------------------------------------------------------

      REAL(dp), PARAMETER :: pi=3.14159265358979323846264338328_dp
      REAL(dp), PARAMETER :: pio2=pi/2
      REAL(dp), PARAMETER :: twopi=2*pi
      REAL(dp), PARAMETER :: sqrt2=1.41421356237309504880168872_dp
      REAL(dp), PARAMETER :: degree=twopi / 360
      REAL(dp), PARAMETER :: one=1
      REAL(dp), PARAMETER :: zero=0
 
!----------------------------------------------------------------------
!  Physical constants
!------------------------------------------------------------------

      REAL(dp), PARAMETER :: mu0 = 2 * twopi * 1.0e-7_dp

      END MODULE stel_constants
