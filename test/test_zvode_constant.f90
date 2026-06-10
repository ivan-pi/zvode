! test_zvode_constant.f90
! ============================================================
!  Fortran-native ZVODE test (constant right-hand side).
!
!  Problem : dy/dt = i,  y(0) = i,  t in [0, 1]
!  Exact   : y(t) = (1 + t) * i   =>   y(1) = 2 i
!  Method  : Adams (MF = 10), no Jacobian (MITER = 0), NEQ = 1.
!
!  Mirrors the SciPy call sequence
!      r = complex_ode(lambda t, x, arg: 1j)
!      r.set_integrator('vode'); r.set_initial_value(1j); r.integrate(1)
!
!  This driver uses the functor-based public API of `zvode_mod`:
!  the right-hand side and Jacobian are passed as objects extending
!  the abstract `zvode_fun` / `zvode_jac` types.
!
!  Assertions (counted by the integer passed to `error stop`):
!    1. ISTATE == 2 on return
!    2. |Re[y(1) - 2i]| + |Im[y(1) - 2i]| <= 1.0d-8
! ============================================================

module test_zvode_constant_mod
  use zvode_mod, only: zvode_fun, zvode_jac
  implicit none
  private
  public :: const_fun, no_jac
  integer, parameter, public :: dp = kind(1.0d0)

  ! Right-hand side functor: dy/dt = i (constant imaginary unit).
  type, extends(zvode_fun) :: const_fun
  contains
    procedure :: eval => const_eval
  end type

  ! Jacobian stub: MF = 10 (MITER = 0) never calls the Jacobian.
  type, extends(zvode_jac) :: no_jac
  contains
    procedure :: eval => no_jac_eval
  end type

contains

  subroutine const_eval(fun, t, y, ydot)
    class(const_fun) :: fun
    real(dp), intent(in) :: t
    complex(dp), intent(in) :: y(fun%neq)
    complex(dp), intent(out) :: ydot(fun%neq)
    ydot = cmplx(0.0_dp, 1.0_dp, dp)
  end subroutine const_eval

  subroutine no_jac_eval(jac, t, y, ml, mu, pd, nrowpd)
    class(no_jac) :: jac
    integer, intent(in) :: ml, mu, nrowpd
    real(dp), intent(in) :: t
    complex(dp), intent(in) :: y(jac%neq)
    complex(dp), intent(inout) :: pd(nrowpd,*)
    ! This must never be reached for MF = 10.
    write(*,'(a)') 'BUG: Jacobian called for MF = 10 (MITER = 0)'
    error stop 3
  end subroutine no_jac_eval

end module test_zvode_constant_mod

program test_zvode_constant
  use zvode_mod, only: zvode
  use test_zvode_constant_mod, only: const_fun, no_jac, dp
  implicit none

  integer, parameter :: neq = 1, mf = 10
  integer, parameter :: lzw = 15, lrw = 21, liw = 30

  complex(dp) :: y(neq), zwork(lzw)
  real(dp) :: rwork(lrw), t, tout, rtol(1), atol(1)
  integer :: iwork(liw), itol, itask, istate, iopt

  complex(dp) :: yexact, err
  real(dp) :: aberr
  real(dp), parameter :: tol = 1.0e-8_dp

  ! --- Initial condition and solver settings ------------------------
  y(1)   = cmplx(0.0_dp, 1.0_dp, dp)   ! y(0) = i
  t      = 0.0_dp
  tout   = 1.0_dp
  itol   = 1
  rtol   = 1.0e-10_dp
  atol   = 1.0e-10_dp
  itask  = 1
  istate = 1
  iopt   = 0
  zwork  = 0.0_dp
  rwork  = 0.0_dp
  iwork  = 0

  call zvode(const_fun(neq), neq, y, t, tout, itol, rtol, atol, itask, &
             istate, iopt, zwork, lzw, rwork, lrw, iwork, liw, &
             no_jac(neq), mf)

  ! --- Assertion 1: successful return -------------------------------
  if (istate /= 2) then
    write(*,'(a,i0)') 'FAIL: ZVODE returned istate = ', istate
    error stop 1
  end if

  ! --- Assertion 2: solution accuracy -------------------------------
  yexact = cmplx(0.0_dp, 2.0_dp, dp)
  err    = y(1) - yexact
  aberr  = abs(real(err, dp)) + abs(aimag(err))

  write(*,'(a)') 'ZVODE constant-RHS test  (dy/dt = i, y(0) = i)'
  write(*,'(a,f14.10,sp,f14.10,a)') '  computed y(1) = ', real(y(1), dp), aimag(y(1)), 'i'
  write(*,'(a,f14.10,sp,f14.10,a)') '  exact    y(1) = ', real(yexact, dp), aimag(yexact), 'i'
  write(*,'(a,es10.3)') '  |error|       = ', aberr

  if (aberr > tol) then
    write(*,'(a,es10.3,a,es10.3)') 'FAIL: |error| = ', aberr, ' > tol = ', tol
    error stop 2
  end if

  write(*,'(a)') 'PASS: test_zvode_constant'

end program test_zvode_constant
