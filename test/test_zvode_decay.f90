! test_zvode_decay.f90
! ============================================================
!  Fortran-native ZVODE test (real exponential decay).
!  Replicates test_zvode_scalar_real_decay from the Python suite.
!
!  Problem : dy/dt = -y,  y(0) = 1 + 0i  =>  y(t) = exp(-t)
!  Method  : Adams (MF = 10), no Jacobian (MITER = 0), NEQ = 1.
!
!  This driver uses the functor-based public API of `zvode_mod`:
!  the right-hand side and Jacobian are passed as objects extending
!  the abstract `zvode_fun` / `zvode_jac` types.
!
!  Assertions (counted by the integer passed to `error stop`):
!    1. ISTATE == 2 on return
!    2. T == TOUT on return
!    3. |Re[y(1)] - exp(-10)| <= 1e-4 * exp(-10) + atol
!    4. |Im[y(1)]| < 1e-12
! ============================================================

module test_zvode_decay_mod
  use zvode_mod, only: zvode_fun, zvode_jac
  implicit none
  private
  public :: decay_fun, no_jac
  integer, parameter, public :: dp = kind(1.0d0)

  ! Right-hand side functor: dy/dt = -y.
  type, extends(zvode_fun) :: decay_fun
  contains
    procedure :: eval => decay_eval
  end type

  ! Jacobian stub: MF = 10 (MITER = 0) never calls the Jacobian.
  type, extends(zvode_jac) :: no_jac
  contains
    procedure :: eval => no_jac_eval
  end type

contains

  subroutine decay_eval(fun, t, y, ydot)
    class(decay_fun) :: fun
    real(dp), intent(in) :: t
    complex(dp), intent(in) :: y(fun%neq)
    complex(dp), intent(out) :: ydot(fun%neq)
    ydot = -y
  end subroutine decay_eval

  subroutine no_jac_eval(jac, t, y, ml, mu, pd, nrowpd)
    class(no_jac) :: jac
    integer, intent(in) :: ml, mu, nrowpd
    real(dp), intent(in) :: t
    complex(dp), intent(in) :: y(jac%neq)
    complex(dp), intent(inout) :: pd(nrowpd,*)
    ! This must never be reached for MF = 10.
    write(*,'(a)') 'BUG: Jacobian called for MF = 10 (MITER = 0)'
    error stop 5
  end subroutine no_jac_eval

end module test_zvode_decay_mod

program test_zvode_decay
  use zvode_mod, only: zvode
  use test_zvode_decay_mod, only: decay_fun, no_jac, dp
  implicit none

  integer, parameter :: neq = 1, mf = 10
  integer, parameter :: lzw = 15, lrw = 21, liw = 30

  complex(dp) :: y(neq), zwork(lzw)
  real(dp) :: rwork(lrw), t, tout, rtol(1), atol(1)
  integer :: iwork(liw), itol, itask, istate, iopt

  real(dp) :: exact, abserr, tol, imag_part

  ! --- Initial condition and solver settings ------------------------
  y(1)   = cmplx(1.0_dp, 0.0_dp, dp)   ! y(0) = 1 + 0 i
  t      = 0.0_dp
  tout   = 10.0_dp
  itol   = 1
  rtol   = 1.0e-6_dp
  atol   = 1.0e-8_dp
  itask  = 1
  istate = 1
  iopt   = 0
  zwork  = 0.0_dp
  rwork  = 0.0_dp
  iwork  = 0

  call zvode(decay_fun(neq), neq, y, t, tout, itol, rtol, atol, itask, &
             istate, iopt, zwork, lzw, rwork, lrw, iwork, liw, &
             no_jac(neq), mf)

  ! --- Assertion 1: successful return -------------------------------
  if (istate /= 2) then
    write(*,'(a,i0)') 'FAIL: ZVODE returned istate = ', istate
    error stop 1
  end if

  ! --- Assertion 2: reached TOUT ------------------------------------
  if (t < tout .or. t > tout) then
    write(*,'(a,es22.14)') 'FAIL: ZVODE did not reach TOUT, T = ', t
    error stop 2
  end if

  ! --- Assertion 3: real-part accuracy ------------------------------
  exact  = exp(-tout)
  abserr = abs(real(y(1), dp) - exact)
  tol    = 1.0e-4_dp * abs(exact) + atol(1)

  write(*,'(a)') 'ZVODE decay test  (dy/dt = -y, y(0) = 1)'
  write(*,'(a,es22.15)') '  Re[y(1)] = ', real(y(1), dp)
  write(*,'(a,es22.15)') '  exp(-10) = ', exact
  write(*,'(a,es10.2)')  '  abserr   = ', abserr

  if (abserr > tol) then
    write(*,'(a,es16.8,a,es16.8)') 'FAIL: Re[y(1)] = ', real(y(1), dp), '  exact = ', exact
    error stop 3
  end if

  ! --- Assertion 4: imaginary part stays zero -----------------------
  imag_part = abs(aimag(y(1)))
  if (imag_part > 1.0e-12_dp) then
    write(*,'(a,es10.2)') 'FAIL: Im[y(1)] = ', imag_part
    error stop 4
  end if

  write(*,'(a)') 'PASS: test_zvode_decay'

end program test_zvode_decay
