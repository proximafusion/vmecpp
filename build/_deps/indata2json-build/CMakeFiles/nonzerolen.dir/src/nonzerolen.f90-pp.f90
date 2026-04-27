# 1 "/home/runner/work/vmecpp/vmecpp/build/_deps/indata2json-src/src/nonzerolen.f90"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/home/runner/work/vmecpp/vmecpp/build/_deps/indata2json-src/src/nonzerolen.f90"
module nzl
implicit none

contains

FUNCTION NonZeroLen(array, n)
  USE stel_kinds, ONLY: dp
  use stel_constants, only: zero
  IMPLICIT NONE

  integer :: NonZeroLen
  INTEGER, INTENT(IN)      :: n
  REAL(dp), INTENT(IN)  :: array(n)
  INTEGER :: k

  DO k = n, 1, -1
    IF (array(k) .NE. zero) EXIT
  END DO

  NonZeroLen = k

END ! FUNCTION NonZeroLen

end ! module nzl
