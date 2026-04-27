# 1 "/home/runner/work/vmecpp/vmecpp/build/_deps/indata2json-src/json-fortran/json.f90"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/home/runner/work/vmecpp/vmecpp/build/_deps/indata2json-src/json-fortran/json.f90"
module json

implicit none

integer, parameter :: dbg_unit = 42
integer, parameter, private :: dp = selected_real_kind(15, 300)
logical  :: has_previous
logical  :: json_pretty_print = .false.

contains

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-

subroutine open_dbg_out(filename)
  character(len=*), intent(in) :: filename

  open(unit=dbg_unit, file=trim(filename), status="unknown")
  if (json_pretty_print) then
    write(dbg_unit, '(A)') "{"
  else
    write(dbg_unit, '(A)', advance="no") "{"
  end if
  has_previous = .false.
end subroutine open_dbg_out

subroutine close_dbg_out
  if (json_pretty_print) then
    write(dbg_unit, '(A)') "}"
  else
    write(dbg_unit, '(A)', advance="no") "}"
  end if
  close(dbg_unit)
end subroutine close_dbg_out

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-

subroutine add_element(varname, content)
  character(len=*), intent(in) :: varname
  character(len=*), intent(in) :: content

  if (json_pretty_print) then
    if (has_previous) then
      write(dbg_unit, '(A)') ','
    end if
  else
    if (has_previous) then
      write(dbg_unit, '(A)', advance="no") ','
    end if
  end if
  write(dbg_unit, '(4A)', advance="no") '"',trim(adjustl(varname)),'":',trim(adjustl(content))

  has_previous = .true.
end subroutine add_element

subroutine add_array_1d(varname, n, content)
  character(len=*), intent(in) :: varname
  integer, intent(in) :: n
  character(len=*), dimension(n), intent(in) :: content

  integer :: i

  if (json_pretty_print) then
    if (has_previous) then
      write(dbg_unit, '(A)') ','
    end if
    write(dbg_unit, '(3A)') '"',trim(adjustl(varname)),'":['
    do i = 1, n-1
      write(dbg_unit, '(2A)') trim(adjustl(content(i))),','
    end do
  else
    if (has_previous) then
      write(dbg_unit, '(A)', advance="no") ','
    end if
    write(dbg_unit, '(3A)', advance="no") '"',trim(adjustl(varname)),'":['
    do i = 1, n-1
      write(dbg_unit, '(2A)', advance="no") trim(adjustl(content(i))),','
    end do
  end if
  write(dbg_unit, '(2A)', advance="no") trim(adjustl(content(n))),']'

  has_previous = .true.
end subroutine add_array_1d

subroutine add_array_2d(varname, n1, n2, content)
  character(len=*), intent(in) :: varname
  integer, intent(in) :: n1, n2
  character(len=*), dimension(n1,n2), intent(in) :: content

  integer :: i, j

  if (json_pretty_print) then
    if (has_previous) then
      write(dbg_unit, '(A)') ','
    end if
    write(dbg_unit, '(3A)') '"',trim(adjustl(varname)),'":['
    do i = 1, n1
      write(dbg_unit, '(A)', advance="no") '['
      do j = 1, n2
        write(dbg_unit, '(2A)', advance="no") trim(adjustl(content(i,j)))
        if (j .lt. n2) then
          write(dbg_unit, '(2A)') ','
        end if
      end do
      write(dbg_unit, '(2A)', advance="no") ']'
      if (i .lt. n1) then
        write(dbg_unit, '(2A)') ','
      end if
    end do
  else
    if (has_previous) then
      write(dbg_unit, '(A)', advance="no") ','
    end if
    write(dbg_unit, '(3A)', advance="no") '"',trim(adjustl(varname)),'":['
    do i = 1, n1
      write(dbg_unit, '(A)', advance="no") '['
      do j = 1, n2
        write(dbg_unit, '(2A)', advance="no") trim(adjustl(content(i,j)))
        if (j .lt. n2) then
          write(dbg_unit, '(2A)', advance="no") ','
        end if
      end do
      write(dbg_unit, '(2A)', advance="no") ']'
      if (i .lt. n1) then
        write(dbg_unit, '(2A)', advance="no") ','
      end if
    end do
  end if
  write(dbg_unit, '(2A)', advance="no") ']'

  has_previous = .true.
end subroutine add_array_2d

subroutine add_array_3d(varname, n1, n2, n3, content)
  character(len=*), intent(in) :: varname
  integer, intent(in) :: n1, n2, n3
  character(len=*), dimension(n1,n2,n3), intent(in) :: content

  integer :: i, j, k

  if (json_pretty_print) then
    if (has_previous) then
      write(dbg_unit, '(A)') ','
    end if
    write(dbg_unit, '(3A)') '"',trim(adjustl(varname)),'":['
    do i = 1, n1
      write(dbg_unit, '(A)', advance="no") '['
      do j = 1, n2
        write(dbg_unit, '(A)', advance="no") '['
        do k = 1, n3
          write(dbg_unit, '(2A)', advance="no") trim(adjustl(content(i,j,k)))
          if (k .lt. n3) then
            write(dbg_unit, '(2A)') ','
          end if
        end do
        write(dbg_unit, '(2A)', advance="no") ']'
        if (j .lt. n2) then
          write(dbg_unit, '(2A)') ','
        end if
      end do
      write(dbg_unit, '(2A)', advance="no") ']'
      if (i .lt. n1) then
        write(dbg_unit, '(2A)') ','
      end if
    end do
  else
    if (has_previous) then
      write(dbg_unit, '(A)', advance="no") ','
    end if
    write(dbg_unit, '(3A)', advance="no") '"',trim(adjustl(varname)),'":['
    do i = 1, n1
      write(dbg_unit, '(A)', advance="no") '['
      do j = 1, n2
        write(dbg_unit, '(A)', advance="no") '['
        do k = 1, n3
          write(dbg_unit, '(2A)', advance="no") trim(adjustl(content(i,j,k)))
          if (k .lt. n3) then
            write(dbg_unit, '(2A)', advance="no") ','
          end if
        end do
        write(dbg_unit, '(2A)', advance="no") ']'
        if (j .lt. n2) then
          write(dbg_unit, '(2A)', advance="no") ','
        end if
      end do
      write(dbg_unit, '(2A)', advance="no") ']'
      if (i .lt. n1) then
        write(dbg_unit, '(2A)', advance="no") ','
      end if
    end do
  end if
  write(dbg_unit, '(2A)', advance="no") ']'

  has_previous = .true.
end subroutine add_array_3d

subroutine add_array_4d(varname, n1, n2, n3, n4, content)
  character(len=*), intent(in) :: varname
  integer, intent(in) :: n1, n2, n3, n4
  character(len=*), dimension(n1,n2,n3,n4), intent(in) :: content

  integer :: i, j, k, l

  if (json_pretty_print) then
    if (has_previous) then
      write(dbg_unit, '(A)') ','
    end if
    write(dbg_unit, '(3A)') '"',trim(adjustl(varname)),'":['
    do i = 1, n1
      write(dbg_unit, '(A)', advance="no") '['
      do j = 1, n2
        write(dbg_unit, '(A)', advance="no") '['
        do k = 1, n3
          write(dbg_unit, '(A)', advance="no") '['
          do l = 1, n4
            write(dbg_unit, '(2A)', advance="no") trim(adjustl(content(i,j,k,l)))
            if (l .lt. n4) then
              write(dbg_unit, '(2A)') ','
            end if
          end do
          write(dbg_unit, '(2A)', advance="no") ']'
          if (k .lt. n3) then
            write(dbg_unit, '(2A)') ','
          end if
        end do
        write(dbg_unit, '(2A)', advance="no") ']'
        if (j .lt. n2) then
          write(dbg_unit, '(2A)') ','
        end if
      end do
      write(dbg_unit, '(2A)', advance="no") ']'
      if (i .lt. n1) then
        write(dbg_unit, '(2A)') ','
      end if
    end do
  else
    if (has_previous) then
      write(dbg_unit, '(A)', advance="no") ','
    end if
    write(dbg_unit, '(3A)', advance="no") '"',trim(adjustl(varname)),'":['
    do i = 1, n1
      write(dbg_unit, '(A)', advance="no") '['
      do j = 1, n2
        write(dbg_unit, '(A)', advance="no") '['
        do k = 1, n3
          write(dbg_unit, '(A)', advance="no") '['
          do l = 1, n4
            write(dbg_unit, '(2A)', advance="no") trim(adjustl(content(i,j,k,l)))
            if (l .lt. n4) then
              write(dbg_unit, '(2A)', advance="no") ','
            end if
          end do
          write(dbg_unit, '(2A)', advance="no") ']'
          if (k .lt. n3) then
            write(dbg_unit, '(2A)', advance="no") ','
          end if
        end do
        write(dbg_unit, '(2A)', advance="no") ']'
        if (j .lt. n2) then
          write(dbg_unit, '(2A)', advance="no") ','
        end if
      end do
      write(dbg_unit, '(2A)', advance="no") ']'
      if (i .lt. n1) then
        write(dbg_unit, '(2A)', advance="no") ','
      end if
    end do
  end if
  write(dbg_unit, '(2A)', advance="no") ']'

  has_previous = .true.
end subroutine add_array_4d

subroutine add_array_5d(varname, n1, n2, n3, n4, n5, content)
  character(len=*), intent(in) :: varname
  integer, intent(in) :: n1, n2, n3, n4, n5
  character(len=*), dimension(n1,n2,n3,n4,n5), intent(in) :: content

  integer :: i, j, k, l, m

  if (json_pretty_print) then
    if (has_previous) then
      write(dbg_unit, '(A)') ','
    end if
    write(dbg_unit, '(3A)') '"',trim(adjustl(varname)),'":['
    do i = 1, n1
      write(dbg_unit, '(A)', advance="no") '['
      do j = 1, n2
        write(dbg_unit, '(A)', advance="no") '['
        do k = 1, n3
          write(dbg_unit, '(A)', advance="no") '['
          do l = 1, n4
            write(dbg_unit, '(A)', advance="no") '['
            do m = 1, n5
              write(dbg_unit, '(2A)', advance="no") trim(adjustl(content(i,j,k,l,m)))
              if (m .lt. n5) then
                write(dbg_unit, '(2A)') ','
              end if
            end do
            write(dbg_unit, '(2A)', advance="no") ']'
            if (l .lt. n4) then
              write(dbg_unit, '(2A)') ','
            end if
          end do
          write(dbg_unit, '(2A)', advance="no") ']'
          if (k .lt. n3) then
            write(dbg_unit, '(2A)') ','
          end if
        end do
        write(dbg_unit, '(2A)', advance="no") ']'
        if (j .lt. n2) then
          write(dbg_unit, '(2A)') ','
        end if
      end do
      write(dbg_unit, '(2A)', advance="no") ']'
      if (i .lt. n1) then
        write(dbg_unit, '(2A)') ','
      end if
    end do
  else
    if (has_previous) then
      write(dbg_unit, '(A)', advance="no") ','
    end if
    write(dbg_unit, '(3A)', advance="no") '"',trim(adjustl(varname)),'":['
    do i = 1, n1
      write(dbg_unit, '(A)', advance="no") '['
      do j = 1, n2
        write(dbg_unit, '(A)', advance="no") '['
        do k = 1, n3
          write(dbg_unit, '(A)', advance="no") '['
          do l = 1, n4
            write(dbg_unit, '(A)', advance="no") '['
            do m = 1, n5
              write(dbg_unit, '(2A)', advance="no") trim(adjustl(content(i,j,k,l,m)))
              if (m .lt. n5) then
                write(dbg_unit, '(2A)', advance="no") ','
              end if
            end do
            write(dbg_unit, '(2A)', advance="no") ']'
            if (l .lt. n4) then
              write(dbg_unit, '(2A)', advance="no") ','
            end if
          end do
          write(dbg_unit, '(2A)', advance="no") ']'
          if (k .lt. n3) then
            write(dbg_unit, '(2A)', advance="no") ','
          end if
        end do
        write(dbg_unit, '(2A)', advance="no") ']'
        if (j .lt. n2) then
          write(dbg_unit, '(2A)', advance="no") ','
        end if
      end do
      write(dbg_unit, '(2A)', advance="no") ']'
      if (i .lt. n1) then
        write(dbg_unit, '(2A)', advance="no") ','
      end if
    end do
  end if
  write(dbg_unit, '(2A)', advance="no") ']'

  has_previous = .true.
end subroutine add_array_5d

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-

subroutine add_null(varname)
  character(len=*), intent(in) :: varname

  call add_element(varname, "null")
end subroutine add_null

subroutine add_null_1d(varname)
  character(len=*), intent(in) :: varname

  call add_array_1d(varname, 1, "null")
end subroutine add_null_1d

subroutine add_null_2d(varname)
  character(len=*), intent(in) :: varname

  call add_array_2d(varname, 1, 1, "null")
end subroutine add_null_2d

subroutine add_null_3d(varname)
  character(len=*), intent(in) :: varname

  call add_array_3d(varname, 1, 1, 1, "null")
end subroutine add_null_3d

subroutine add_null_4d(varname)
  character(len=*), intent(in) :: varname

  call add_array_4d(varname, 1, 1, 1, 1, "null")
end subroutine add_null_4d

subroutine add_null_5d(varname)
  character(len=*), intent(in) :: varname

  call add_array_5d(varname, 1, 1, 1, 1, 1, "null")
end subroutine add_null_5d

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-

subroutine add_none(varname)
  character(len=*), intent(in) :: varname

  call add_element(varname, "")
end subroutine add_none

subroutine add_none_1d(varname)
  character(len=*), intent(in) :: varname

  call add_array_1d(varname, 1, "")
end subroutine add_none_1d

subroutine add_none_2d(varname)
  character(len=*), intent(in) :: varname

  call add_array_2d(varname, 1, 1, "")
end subroutine add_none_2d

subroutine add_none_3d(varname)
  character(len=*), intent(in) :: varname

  call add_array_3d(varname, 1, 1, 1, "")
end subroutine add_none_3d

subroutine add_none_4d(varname)
  character(len=*), intent(in) :: varname

  call add_array_4d(varname, 1, 1, 1, 1, "")
end subroutine add_none_4d

subroutine add_none_5d(varname)
  character(len=*), intent(in) :: varname

  call add_array_5d(varname, 1, 1, 1, 1, 1, "")
end subroutine add_none_5d

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-

subroutine add_logical(varname, val)
  character(len=*), intent(in) :: varname
  logical         , intent(in) :: val

  if (val) then
    call add_element(varname, "true")
  else
    call add_element(varname, "false")
  end if
end subroutine add_logical

subroutine add_logical_1d(varname, n, arr)
  character(len=*), intent(in) :: varname
  integer, intent(in) :: n
  logical, dimension(n), intent(in) :: arr

  character(len=64), dimension(:), allocatable :: temp
  integer :: i

  allocate(temp(n))
  do i = 1, n
    if (arr(i)) then
      temp(i) = "true"
    else
      temp(i) = "false"
    end if
  end do
  call add_array_1d(varname, n, temp)

  deallocate(temp)
end subroutine add_logical_1d

subroutine add_logical_2d(varname, n1, n2, arr, order)
  character(len=*), intent(in) :: varname
  integer, intent(in) :: n1, n2
  logical, dimension(n1, n2), intent(in) :: arr
  integer, dimension(2), intent(in), optional :: order

  character(len=64), dimension(:,:), allocatable :: temp
  integer :: i, j

  allocate(temp(n1, n2))
  do i = 1, n1
    do j = 1, n2
      if (arr(i, j)) then
        temp(i, j) = "true"
      else
        temp(i, j) = "false"
      end if
    end do
  end do
  if (present(order)) then
    call add_array_2d(varname, n1, n2, &
           reshape(temp, (/ n1, n2 /), order=order))
  else
    call add_array_2d(varname, n1, n2, temp)
  end if

  deallocate(temp)
end subroutine add_logical_2d

subroutine add_logical_3d(varname, n1, n2, n3, arr, order)
  character(len=*), intent(in) :: varname
  integer, intent(in) :: n1, n2, n3
  logical, dimension(n1, n2, n3), intent(in) :: arr
  integer, dimension(3), intent(in), optional :: order

  character(len=64), dimension(:,:,:), allocatable :: temp
  integer :: i, j, k

  allocate(temp(n1,n2,n3))
  do i = 1, n1
    do j = 1, n2
      do k = 1, n3
        if (arr(i, j, k)) then
          temp(i, j, k) = "true"
        else
          temp(i, j, k) = "false"
        end if
      end do
    end do
  end do
  if (present(order)) then
    call add_array_3d(varname, n1, n2, n3, &
           reshape(temp, (/ n1, n2, n3 /), order=order))
  else
    call add_array_3d(varname, n1, n2, n3, temp)
  end if

  deallocate(temp)
end subroutine add_logical_3d

subroutine add_logical_4d(varname, n1, n2, n3, n4, arr, order)
  character(len=*), intent(in) :: varname
  integer, intent(in) :: n1, n2, n3, n4
  logical, dimension(n1,n2,n3,n4), intent(in) :: arr
  integer, dimension(4), intent(in), optional :: order

  character(len=64), dimension(:,:,:,:), allocatable :: temp
  integer :: i, j, k, l

  allocate(temp(n1,n2,n3,n4))
  do i = 1, n1
    do j = 1, n2
      do k = 1, n3
        do l = 1, n4
          if (arr(i, j, k, l)) then
            temp(i, j, k, l) = "true"
          else
            temp(i, j, k, l) = "false"
          end if
        end do
      end do
    end do
  end do
  if (present(order)) then
    call add_array_4d(varname, n1, n2, n3, n4, &
           reshape(temp, (/ n1, n2, n3, n4 /), order=order))
  else
    call add_array_4d(varname, n1, n2, n3, n4, temp)
  end if

  deallocate(temp)
end subroutine add_logical_4d

subroutine add_logical_5d(varname, n1, n2, n3, n4, n5, arr, order)
  character(len=*), intent(in) :: varname
  integer, intent(in) :: n1, n2, n3, n4, n5
  logical, dimension(n1,n2,n3,n4,n5), intent(in) :: arr
  integer, dimension(5), intent(in), optional :: order

  character(len=64), dimension(:,:,:,:,:), allocatable :: temp
  integer :: i, j, k, l, m

  allocate(temp(n1,n2,n3,n4,n5))
  do i = 1, n1
    do j = 1, n2
      do k = 1, n3
        do l = 1, n4
          do m = 1, n5
            if (arr(i, j, k, l, m)) then
              temp(i, j, k, l, m) = "true"
            else
              temp(i, j, k, l, m) = "false"
            end if
          end do
        end do
      end do
    end do
  end do
  if (present(order)) then
    call add_array_5d(varname, n1, n2, n3, n4, n5, &
           reshape(temp, (/ n1, n2, n3, n4, n5 /), order=order))
  else
    call add_array_5d(varname, n1, n2, n3, n4, n5, temp)
  end if

  deallocate(temp)
end subroutine add_logical_5d

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-

subroutine add_int(varname, val)
  character(len=*), intent(in) :: varname
  integer         , intent(in) :: val

  character(len=64) :: temp

  write(temp, *) val
  call add_element(varname, temp)
end subroutine add_int

subroutine add_int_1d(varname, n, arr)
  character(len=*), intent(in) :: varname
  integer, intent(in) :: n
  integer, dimension(n), intent(in) :: arr

  character(len=64), dimension(:), allocatable :: temp
  integer :: i

  allocate(temp(n))
  do i = 1, n
    write(temp(i), *) arr(i)
  end do
  call add_array_1d(varname, n, temp)

  deallocate(temp)
end subroutine add_int_1d

subroutine add_int_2d(varname, n1, n2, arr, order)
  character(len=*), intent(in) :: varname
  integer, intent(in) :: n1, n2
  integer, dimension(n1, n2), intent(in) :: arr
  integer, dimension(2), intent(in), optional :: order

  character(len=64), dimension(:,:), allocatable :: temp
  integer :: i, j

  allocate(temp(n1, n2))
  do i = 1, n1
    do j = 1, n2
      write(temp(i,j), *) arr(i,j)
    end do
  end do
  if (present(order)) then
    call add_array_2d(varname, n1, n2, &
           reshape(temp, (/ n1, n2 /), order=order))
  else
    call add_array_2d(varname, n1, n2, temp)
  end if

  deallocate(temp)
end subroutine add_int_2d

subroutine add_int_3d(varname, n1, n2, n3, arr, order)
  character(len=*), intent(in) :: varname
  integer, intent(in) :: n1, n2, n3
  integer, dimension(n1, n2, n3), intent(in) :: arr
  integer, dimension(3), intent(in), optional :: order

  character(len=64), dimension(:,:,:), allocatable :: temp
  integer :: i, j, k

  allocate(temp(n1,n2,n3))
  do i = 1, n1
    do j = 1, n2
      do k = 1, n3
        write(temp(i,j,k), *) arr(i,j,k)
      end do
    end do
  end do
  if (present(order)) then
    call add_array_3d(varname, n1, n2, n3, &
           reshape(temp, (/ n1, n2, n3 /), order=order))
  else
    call add_array_3d(varname, n1, n2, n3, temp)
  end if

  deallocate(temp)
end subroutine add_int_3d

subroutine add_int_4d(varname, n1, n2, n3, n4, arr, order)
  character(len=*), intent(in) :: varname
  integer, intent(in) :: n1, n2, n3, n4
  integer, dimension(n1,n2,n3,n4), intent(in) :: arr
  integer, dimension(4), intent(in), optional :: order

  character(len=64), dimension(:,:,:,:), allocatable :: temp
  integer :: i, j, k, l

  allocate(temp(n1,n2,n3,n4))
  do i = 1, n1
    do j = 1, n2
      do k = 1, n3
        do l = 1, n4
          write(temp(i,j,k,l), *) arr(i,j,k,l)
        end do
      end do
    end do
  end do
  if (present(order)) then
    call add_array_4d(varname, n1, n2, n3, n4, &
           reshape(temp, (/ n1, n2, n3, n4 /), order=order))
  else
    call add_array_4d(varname, n1, n2, n3, n4, temp)
  end if

  deallocate(temp)
end subroutine add_int_4d

subroutine add_int_5d(varname, n1, n2, n3, n4, n5, arr, order)
  character(len=*), intent(in) :: varname
  integer, intent(in) :: n1, n2, n3, n4, n5
  integer, dimension(n1,n2,n3,n4,n5), intent(in) :: arr
  integer, dimension(5), intent(in), optional :: order

  character(len=64), dimension(:,:,:,:,:), allocatable :: temp
  integer :: i, j, k, l, m

  allocate(temp(n1,n2,n3,n4,n5))
  do i = 1, n1
    do j = 1, n2
      do k = 1, n3
        do l = 1, n4
          do m = 1, n5
            write(temp(i,j,k,l,m), *) arr(i,j,k,l,m)
          end do
        end do
      end do
    end do
  end do
  if (present(order)) then
    call add_array_5d(varname, n1, n2, n3, n4, n5, &
           reshape(temp, (/ n1, n2, n3, n4, n5 /), order=order))
  else
    call add_array_5d(varname, n1, n2, n3, n4, n5, temp)
  end if

  deallocate(temp)
end subroutine add_int_5d

!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-

subroutine add_real(varname, val)
  character(len=*), intent(in) :: varname
  real(dp)        , intent(in) :: val

  character(len=64) :: temp

  write(temp, *) val
  call add_element(varname, temp)
end subroutine add_real

subroutine add_real_1d(varname, n, arr)
  character(len=*), intent(in) :: varname
  integer, intent(in) :: n
  real(dp), dimension(n), intent(in) :: arr

  character(len=64), dimension(:), allocatable :: temp
  integer :: i

  allocate(temp(n))
  do i = 1, n
    write(temp(i), *) arr(i)
  end do
  call add_array_1d(varname, n, temp)

  deallocate(temp)
end subroutine add_real_1d

subroutine add_real_2d(varname, n1, n2, arr, order)
  character(len=*), intent(in) :: varname
  integer, intent(in) :: n1, n2
  real(dp), dimension(n1, n2), intent(in) :: arr
  integer, dimension(2), intent(in), optional :: order

  character(len=64), dimension(:,:), allocatable :: temp
  integer :: i, j

  allocate(temp(n1, n2))
  do i = 1, n1
    do j = 1, n2
      write(temp(i,j), *) arr(i,j)
    end do
  end do
  if (present(order)) then
    call add_array_2d(varname, n1, n2, &
           reshape(temp, (/ n1, n2 /), order=order))
  else
    call add_array_2d(varname, n1, n2, temp)
  end if

  deallocate(temp)
end subroutine add_real_2d

subroutine add_real_3d(varname, n1, n2, n3, arr, order)
  character(len=*), intent(in) :: varname
  integer, intent(in) :: n1, n2, n3
  real(dp), dimension(n1, n2, n3), intent(in) :: arr
  integer, dimension(3), intent(in), optional :: order

  character(len=64), dimension(:,:,:), allocatable :: temp
  integer :: i, j, k

  allocate(temp(n1,n2,n3))
  do i = 1, n1
    do j = 1, n2
      do k = 1, n3
        write(temp(i,j,k), *) arr(i,j,k)
      end do
    end do
  end do
  if (present(order)) then
    call add_array_3d(varname, n1, n2, n3, &
           reshape(temp, (/ n1, n2, n3 /), order=order))
  else
    call add_array_3d(varname, n1, n2, n3, temp)
  end if

  deallocate(temp)
end subroutine add_real_3d

subroutine add_real_4d(varname, n1, n2, n3, n4, arr, order)
  character(len=*), intent(in) :: varname
  integer, intent(in) :: n1, n2, n3, n4
  real(dp), dimension(n1,n2,n3,n4), intent(in) :: arr
  integer, dimension(4), intent(in), optional :: order

  character(len=64), dimension(:,:,:,:), allocatable :: temp
  integer :: i, j, k, l

  allocate(temp(n1,n2,n3,n4))
  do i = 1, n1
    do j = 1, n2
      do k = 1, n3
        do l = 1, n4
          write(temp(i,j,k,l), *) arr(i,j,k,l)
        end do
      end do
    end do
  end do
  if (present(order)) then
    call add_array_4d(varname, n1, n2, n3, n4, &
           reshape(temp, (/ n1, n2, n3, n4 /), order=order))
  else
    call add_array_4d(varname, n1, n2, n3, n4, temp)
  end if

  deallocate(temp)
end subroutine add_real_4d

subroutine add_real_5d(varname, n1, n2, n3, n4, n5, arr, order)
  character(len=*), intent(in) :: varname
  integer, intent(in) :: n1, n2, n3, n4, n5
  real(dp), dimension(n1,n2,n3,n4,n5), intent(in) :: arr
  integer, dimension(5), intent(in), optional :: order

  character(len=64), dimension(:,:,:,:,:), allocatable :: temp
  integer :: i, j, k, l, m

  allocate(temp(n1,n2,n3,n4,n5))
  do i = 1, n1
    do j = 1, n2
      do k = 1, n3
        do l = 1, n4
          do m = 1, n5
            write(temp(i,j,k,l,m), *) arr(i,j,k,l,m)
          end do
        end do
      end do
    end do
  end do
  if (present(order)) then
    call add_array_5d(varname, n1, n2, n3, n4, n5, &
           reshape(temp, (/ n1, n2, n3, n4, n5 /), order=order))
  else
    call add_array_5d(varname, n1, n2, n3, n4, n5, temp)
  end if

  deallocate(temp)
end subroutine add_real_5d


end module json
