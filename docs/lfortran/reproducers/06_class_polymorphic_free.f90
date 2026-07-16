subroutine s(x)
  type t
    integer :: n
  end type
  class(t) :: x
  x%n = 1
end subroutine
