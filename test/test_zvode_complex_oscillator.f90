! test_zvode_complex_oscillator.f90
! ============================================================
!  Fortran-native ZVODE test (complex oscillator).
!
!  Problem : dy/dt = -i*y,  y(0) = 1 + 0i  =>  y(t) = exp(-i*t)
!  Method  : BDF (MF = 22), internally generated dense Jacobian
!            (MITER = 2), NEQ = 1.
!
!  This driver uses the functor-based public API of `zvode_mod`:
!  the right-hand side and Jacobian are passed as objects extending
!  the abstract `zvode_fun` / `zvode_jac` types.  For MF = 22 the
!  Jacobian is formed internally by finite differences, so the
!  supplied Jacobian functor is never invoked.
!
!  Assertions (counted by the integer passed to `error stop`):
!    1. ISTATE == 2 on return
!    2. T == TOUT on return
!    3. |y(TOUT) - exp(-i*TOUT)| <= 1e-5  (global-error bound)
!
!  Note: the per-step tolerances (rtol, atol) control the *local* error,
!  so the accumulated global error at TOUT = 10 is larger than rtol.  The
!  assertion therefore checks against a fixed global bound comfortably
!  above the observed error (~1e-7 at these tolerances).
! ============================================================

module test_zvode_complex_oscillator_mod
  use zvode_mod, only: zvode_fun, zvode_jac
  implicit none
  private
  public :: osc_fun, no_jac
  integer, parameter, public :: dp = kind(1.0d0)

  ! Right-hand side functor: dy/dt = -i*y.
  type, extends(zvode_fun) :: osc_fun
  contains
    procedure :: eval => osc_eval
  end type

  ! Jacobian stub: MF = 22 (MITER = 2) generates the Jacobian
  ! internally and never calls the user functor.
  type, extends(zvode_jac) :: no_jac
  contains
    procedure :: eval => no_jac_eval
  end type

contains

  subroutine osc_eval(fun, t, y, ydot)
    class(osc_fun) :: fun
    real(dp), intent(in) :: t
    complex(dp), intent(in) :: y(fun%neq)
    complex(dp), intent(out) :: ydot(fun%neq)
    ydot = cmplx(0.0_dp, -1.0_dp, dp) * y
  end subroutine osc_eval

  subroutine no_jac_eval(jac, t, y, ml, mu, pd, nrowpd)
    class(no_jac) :: jac
    integer, intent(in) :: ml, mu, nrowpd
    real(dp), intent(in) :: t
    complex(dp), intent(in) :: y(jac%neq)
    complex(dp), intent(inout) :: pd(nrowpd,*)
    ! This must never be reached for MF = 22 (internally generated).
    write(*,'(a)') 'BUG: Jacobian called for MF = 22 (MITER = 2)'
    error stop 4
  end subroutine no_jac_eval

end module test_zvode_complex_oscillator_mod

program test_zvode_complex_oscillator
  use zvode_mod, only: zvode
  use test_zvode_complex_oscillator_mod, only: osc_fun, no_jac, dp
  implicit none

  ! Work-array minimums for MF = 22 (BDF, MITER = 2), NEQ = 1:
  !   LZW >= 8*NEQ + 2*NEQ**2 = 10,  LRW >= 20 + NEQ = 21,
  !   LIW >= 30 + NEQ = 31 (MITER /= 0 needs the extra NEQ words).
  integer, parameter :: neq = 1, mf = 22
  integer, parameter :: lzw = 15, lrw = 21, liw = 31

  complex(dp) :: y(neq), zwork(lzw)
  real(dp) :: rwork(lrw), t, tout, rtol(1), atol(1)
  integer :: iwork(liw), itol, itask, istate, iopt

  complex(dp) :: exact
  real(dp) :: abserr
  real(dp), parameter :: tol = 1.0e-5_dp   ! global-error bound at TOUT

  ! --- Initial condition and solver settings ------------------------
  y(1)   = cmplx(1.0_dp, 0.0_dp, dp)   ! y(0) = 1 + 0 i
  t      = 0.0_dp
  tout   = 10.0_dp
  itol   = 1
  rtol   = 1.0e-8_dp
  atol   = 1.0e-10_dp
  itask  = 1
  istate = 1
  iopt   = 0
  zwork  = 0.0_dp
  rwork  = 0.0_dp
  iwork  = 0

  call zvode(osc_fun(neq), neq, y, t, tout, itol, rtol, atol, itask, &
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

  ! --- Assertion 3: solution accuracy -------------------------------
  exact  = cmplx(cos(tout), -sin(tout), dp)   ! exp(-i*tout)
  abserr = abs(y(1) - exact)

  write(*,'(a)') 'ZVODE complex-oscillator test  (dy/dt = -i*y, y(0) = 1)'
  write(*,'(a,es16.8,a,es16.8,a)') '  y(1)   = (', real(y(1), dp), ' , ', aimag(y(1)), 'i)'
  write(*,'(a,es16.8,a,es16.8,a)') '  exact  = (', real(exact, dp), ' , ', aimag(exact), 'i)'
  write(*,'(a,es10.2)') '  abserr = ', abserr

  if (abserr > tol) then
    write(*,'(a,es10.2,a,es10.2)') 'FAIL: abserr = ', abserr, ' > tol = ', tol
    error stop 3
  end if

  write(*,'(a)') 'PASS: test_zvode_complex_oscillator'

end program test_zvode_complex_oscillator
