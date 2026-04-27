# 1 "/home/runner/work/vmecpp/vmecpp/build/_deps/indata2json-src/LIBSTELL/Sources/Modules/safe_open_mod.f90"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/home/runner/work/vmecpp/vmecpp/build/_deps/indata2json-src/LIBSTELL/Sources/Modules/safe_open_mod.f90"
      MODULE safe_open_mod
!
!     Module for performing a "safe" open of a file for
!     a Fortran read/write operation. Makes sure the requested file
!     unit number is not in use, and increments it until an unused
!     unit is found
!
      CONTAINS

      SUBROUTINE safe_open(iunit, istat, filename, filestat,                   &
     &           fileform, record_in, access_in, delim_in)
!
!     Module for performing a "safe" open of a file for
!     a Fortran read/write operation. Makes sure the requested file
!     unit number is not in use, and increments it until an unused
!     unit is found
!
!  Note that:
! 1)  the actual i/o unit number used is returned in the first argument.
! 2)  the status variable from the OPEN command is returned as the second
!     argument.

!  Here are some examples of usage:
!
!   To open an existing namelist input file:
!      CALL safe_open(iou,istat,nli_file_name,'old','formatted')
!
!   To create a file, in order to write to it:
!      CALL safe_open(iou,istat,my_output_file_name,'replace','formatted')
!
!   To create an output file, with 'NONE' as delimiter for characters for
!   list-directed output and Namelist output
!      CALL safe_open(iou,istat,my_output_file_name,'replace',
!     &   'formatted',delim_in='none')

!  JDH 08-30-2004. 
!     Based on Steve Hirshman's original safe_open routine
!     Rearranged comments, continuation lines, some statement ordering.
!     Should be NO change in functionality.
!
!  JDH 2010-06-09
!     Added coding for DELIM specification


      IMPLICIT NONE
!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------
      INTEGER, INTENT(inout) :: iunit
      INTEGER, INTENT(out) :: istat
      CHARACTER(LEN=*), INTENT(in) :: filename, filestat, fileform
      INTEGER, INTENT(in), OPTIONAL :: record_in
      CHARACTER(LEN=*), INTENT(in), OPTIONAL :: access_in
      CHARACTER(LEN=*), INTENT(in), OPTIONAL :: delim_in
!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------
      CHARACTER(LEN=*), PARAMETER :: cdelim = "apostrophe",             &
           cform="formatted", cunform="unformatted",                    &
           cscratch="scratch", cseq="sequential"                        
      CHARACTER(LEN=10) :: acc_type
      CHARACTER(LEN=10) :: delim_type
      LOGICAL :: lopen, lexist, linvalid
!-----------------------------------------------
!  Start of Executable Code
!-----------------------------------------------

!-----------------------------------------------
!
!     Check that unit is not already opened
!     Increment iunit until find one that is not in use
!
      linvalid = .true.
      IF (iunit < 0) THEN
         WRITE (6, *) 'In safe_open, requested unit was uninitialized: IUNIT=', iunit
         iunit = 10
      END IF
      DO WHILE (linvalid)
         INQUIRE(iunit, exist=lexist, opened=lopen, iostat=istat)
         linvalid = (istat.ne.0 .or. .not.lexist) .or. lopen
         IF (.not.linvalid) EXIT
         iunit = iunit + 1
      END DO

!  JDH 08-24-2004 This next IF(Present) clause seems to be duplicated below. 
!  I think one of the two should be eliminated, for clarity.

      IF (PRESENT(access_in)) THEN
         acc_type = TRIM(access_in)
      ELSE
         acc_type = cseq
      END IF

!  Why not call this variable lscratch?
      lexist = (filestat(1:1).eq.'s') .or. (filestat(1:1).eq.'S')        !Scratch file

!  JDH 08-24-2004 Below is nearly exact duplicate of IF(Present) clause 
!  from above

      IF (PRESENT(access_in)) THEN
         acc_type = TRIM(access_in)
      ELSE
         acc_type = 'SEQUENTIAL'
      END IF

!  JDH 2010-06-09. Coding for DELIM
      IF (PRESENT(delim_in)) THEN
         SELECT CASE (delim_in(1:1))
         CASE ('n', 'N')
            delim_type = 'none'
         CASE ('q', 'Q')
            delim_type = 'quote'
         CASE DEFAULT
            delim_type = cdelim
         END SELECT
      ELSE
         delim_type = cdelim
      ENDIF

! Here are the actual OPEN commands. Eight different cases.
      SELECT CASE (fileform(1:1))
      CASE ('u', 'U')
         IF (PRESENT(record_in)) THEN
            IF (lexist) THEN     ! unformatted, record length specified, scratch 
               OPEN(unit=iunit, form=cunform, status=cscratch,                 &
     &              recl=record_in, access=acc_type, iostat=istat)
            ELSE             ! unformatted, record length specified, non-scratch 
               OPEN(unit=iunit, file=TRIM(filename), form=cunform,             &
     &              status=TRIM(filestat), recl=record_in,                     &
     &              access=acc_type, iostat=istat)
            END IF
         ELSE
            IF (lexist) THEN   ! unformatted, record length unspecified, scratch 
               OPEN(unit=iunit, form=cunform, status=cscratch,                 &
     &              access=acc_type, iostat=istat)
            ELSE           ! unformatted, record length unspecified, non-scratch 
               OPEN(unit=iunit, file=TRIM(filename), form=cunform,             &
     &              status=TRIM(filestat), access=acc_type,iostat=istat)
            END IF
         END IF

      CASE DEFAULT
         IF (PRESENT(record_in)) THEN
            IF (lexist) THEN       ! formatted, record length specified, scratch 
               OPEN(unit=iunit, form=cform, status=cscratch,                   &
                    delim=TRIM(delim_type), recl=record_in,                    &
                    access=acc_type, iostat=istat)
            ELSE               ! formatted, record length specified, non-scratch 
               OPEN(unit=iunit, file=TRIM(filename), form=cform,               &
                    status=TRIM(filestat), delim=TRIM(delim_type),             &
                    recl=record_in, access=acc_type, iostat=istat)
            END IF
         ELSE
            IF (lexist) THEN     ! formatted, record length unspecified, scratch 
               OPEN(unit=iunit, form=cform, status=cscratch,                   &
                    delim=TRIM(delim_type), access=acc_type,                   &
                    iostat=istat)
            ELSE             ! formatted, record length unspecified, non-scratch 
               OPEN(unit=iunit, file=TRIM(filename), form=cform,               &
                   status=TRIM(filestat), delim=TRIM(delim_type),              &
                   access=acc_type, iostat=istat)
            END IF
         END IF

      END SELECT

      END SUBROUTINE safe_open

      END MODULE safe_open_mod
