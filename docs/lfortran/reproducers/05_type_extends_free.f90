module m
  type :: base
    integer :: n
  end type
  type, extends(base) :: child
    integer :: m
  end type
end module m
