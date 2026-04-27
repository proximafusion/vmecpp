# 1 "/home/runner/work/vmecpp/vmecpp/build/_deps/indata2json-src/LIBSTELL/Sources/Miscel/getcarg.f90"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/home/runner/work/vmecpp/vmecpp/build/_deps/indata2json-src/LIBSTELL/Sources/Miscel/getcarg.f90"
      SUBROUTINE getcarg(narg, arg, numargs)
      IMPLICIT NONE
!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------
      INTEGER, INTENT(in)  :: narg
      INTEGER, INTENT(out) :: numargs
      CHARACTER(LEN=*), INTENT(out) :: arg
!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------
      INTEGER :: numchars
!-----------------------------------------------





      INTEGER iargc
      numargs = iargc()
      CALL getarg(narg, arg)
# 38 "/home/runner/work/vmecpp/vmecpp/build/_deps/indata2json-src/LIBSTELL/Sources/Miscel/getcarg.f90"
      END SUBROUTINE getcarg
