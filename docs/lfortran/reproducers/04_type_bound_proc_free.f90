module m
  type t
    integer :: n
  contains
    procedure :: eval => myeval
  end type
contains
  subroutine myeval(self)
    class(t) :: self
    self%n = 0
  end subroutine
end module m
