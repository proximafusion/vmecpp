# 1 "/home/runner/work/vmecpp/vmecpp/build/_deps/indata2json-src/LIBSTELL/Sources/Miscel/tolower.f90"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/home/runner/work/vmecpp/vmecpp/build/_deps/indata2json-src/LIBSTELL/Sources/Miscel/tolower.f90"
      SUBROUTINE tolower(string)
! convert a string to lower case
! note: string should NOT be a parameter (constant) or this seg faults
      CHARACTER (len=*) , INTENT(inout) :: string 
      INTEGER :: i,ic,nlen,icLow,icHigh,icDel
      
      icLow = ICHAR('A')
      icHigh = ICHAR('Z')
      icDel = ICHAR('a') - ICHAR('A')
      nlen = LEN(string) 
      DO i=1,nlen
         ic = ICHAR(string(i:i))
         IF (ic >= icLow .and. ic < icHigh) string(i:i) = CHAR(ic+icDel)
      END DO 

      END SUBROUTINE tolower
