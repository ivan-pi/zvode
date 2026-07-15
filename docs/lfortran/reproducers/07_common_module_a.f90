module a_mod
  implicit none
contains
  subroutine seta(v)
    real, intent(in) :: v
    real :: shared
    common /blk/ shared
    shared = v
  end subroutine
  function geta() result(r)
    real :: r
    real :: shared
    common /blk/ shared
    r = shared
  end function
end module a_mod
