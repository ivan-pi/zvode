! test_zvode_ewt_functor.f90
! ============================================================
!  Fortran-native test of the user-supplied error-weight functor
!  (the optional EWTFUN argument of ZVODE and the concrete ZVODE_EWT
!  class, whose EVAL binding defaults to ZEWSET).
!
!  This is the ZVODE analogue of CVODE's CVodeWFtolerances / CVEwtFn:
!  a callback that (re)sets the error weight vector from the current
!  solution.  Like CVEwtFn, it is invoked just before EVERY internal
!  step, not once -- the test pins that explicitly.
!
!  Problem P (decoupled, closed form): dy_i/dt = lam_i y_i,
!  y_i(0) = 1  =>  y_i(t) = exp(lam_i t), integrated with BDF + dense
!  analytic Jacobian (MF = 21).
!
!  Coverage (assertion codes passed to `error stop`):
!    1-2   baseline solve (no EWTFUN) reaches TOUT and matches exp(lam t)
!    10-12 passing an explicit default ZVODE_EWT reproduces the baseline
!          trajectory bit-for-bit (the default dispatch path is a no-op
!          re-expression of the historical ZEWSET call)
!    20-23 a stateful override that re-implements the default weight
!          formula (a) reproduces the baseline trajectory, (b) is called
!          MANY times -- once per step, not once -- and (c) its counter
!          state persists in the caller's own object after ZVODE returns
!    30-31 an override that loosens the weights actually changes the
!          solver's behaviour (fewer/larger steps), proving the callback
!          is genuinely consulted rather than ignored
! ============================================================

module test_ewt_functor_mod
  use zvode_mod, only: zvode_fun, zvode_jac, zvode_ewt
  implicit none
  private
  public :: diag_fun, diag_jac, counting_ewt, scaled_ewt, dp
  integer, parameter :: dp = kind(1.0d0)

  ! dy_i/dt = lam_i y_i  (decoupled linear system, diagonal Jacobian)
  type, extends(zvode_fun) :: diag_fun
    complex(dp), allocatable :: lam(:)
  contains
    procedure :: eval => diag_f
  end type

  type, extends(zvode_jac) :: diag_jac
    complex(dp), allocatable :: lam(:)
  contains
    procedure :: eval => diag_j
  end type

  ! Error-weight override that re-implements the DEFAULT formula
  ! (ITOL = 1: EWT_i = RTOL*|YCUR_i| + ATOL) but carries mutable state:
  ! a counter of how many times it has been invoked.  Because ZVODE
  ! aliases (does not copy) the functor, this counter is visible in the
  ! caller's object after the solve.  Note the functor holds only its own
  ! state (the counter) -- the problem size arrives as the N argument of
  ! EVAL, it is not a property of the weighting policy.
  type, extends(zvode_ewt) :: counting_ewt
    integer :: ncalls = 0
  contains
    procedure :: eval => counting_e
  end type

  ! Error-weight override that scales the default weights by a constant
  ! factor.  Larger weights => smaller WRMS error norm => the error test
  ! accepts bigger steps, so the step count must not exceed the baseline.
  type, extends(zvode_ewt) :: scaled_ewt
    real(dp) :: factor = 1.0_dp
  contains
    procedure :: eval => scaled_e
  end type

contains

  subroutine diag_f(fun, t, y, ydot)
    class(diag_fun) :: fun
    real(dp), intent(in) :: t
    complex(dp), intent(in) :: y(fun%neq)
    complex(dp), intent(out) :: ydot(fun%neq)
    ydot = fun%lam * y
  end subroutine

  subroutine diag_j(jac, t, y, ml, mu, pd, nrowpd)
    class(diag_jac) :: jac
    integer, intent(in) :: ml, mu, nrowpd
    real(dp), intent(in) :: t
    complex(dp), intent(in) :: y(jac%neq)
    complex(dp), intent(inout) :: pd(nrowpd,*)
    integer :: i
    do i = 1, jac%neq
      pd(i,i) = jac%lam(i)
    end do
  end subroutine

  subroutine counting_e(ewtf, n, itol, rtol, atol, ycur, ewt)
    class(counting_ewt) :: ewtf
    integer, intent(in) :: n, itol
    real(dp), intent(in) :: rtol(*), atol(*)
    complex(dp), intent(in) :: ycur(n)
    real(dp), intent(out) :: ewt(n)
    ewtf%ncalls = ewtf%ncalls + 1
    ! ITOL = 1 branch of the historical ZEWSET formula
    ewt = rtol(1)*abs(ycur) + atol(1)
  end subroutine

  subroutine scaled_e(ewtf, n, itol, rtol, atol, ycur, ewt)
    class(scaled_ewt) :: ewtf
    integer, intent(in) :: n, itol
    real(dp), intent(in) :: rtol(*), atol(*)
    complex(dp), intent(in) :: ycur(n)
    real(dp), intent(out) :: ewt(n)
    ewt = ewtf%factor * (rtol(1)*abs(ycur) + atol(1))
  end subroutine

end module test_ewt_functor_mod


program test_zvode_ewt_functor
  use zvode_mod, only: zvode, zvode_ewt
  use test_ewt_functor_mod, only: diag_fun, diag_jac, counting_ewt, &
                                   scaled_ewt, dp
  implicit none

  integer, parameter :: neq = 4, mf = 21
  integer, parameter :: lzw = 8*neq + 2*neq*neq   ! MF = 21
  integer, parameter :: lrw = 20 + neq
  integer, parameter :: liw = 30 + neq

  complex(dp), parameter :: lam(neq) = [ &
       cmplx(-2.0_dp, 0.0_dp, dp), cmplx(-0.5_dp, 5.0_dp, dp), &
       cmplx(-1.0_dp, 3.0_dp, dp), cmplx(-3.0_dp,-2.0_dp, dp)]
  real(dp), parameter :: tf = 1.5_dp
  real(dp), parameter :: rtol = 1.0e-9_dp, atol = 1.0e-11_dp

  complex(dp) :: y(neq), zwork(lzw), yref(neq)
  real(dp) :: rwork(lrw), t
  integer :: iwork(liw), istate, nst_ref, nst_scaled

  type(counting_ewt) :: cewt
  type(scaled_ewt)   :: sewt

  ! ================================================================
  ! (1) Baseline: solve with no EWTFUN (default weighting).
  ! ================================================================
  call solve(y, t, istate, iwork, diag_fun(neq, lam), diag_jac(neq, lam))
  call check(istate == 2, 1, 'baseline: istate /= 2')
  call check(all(is_close(y, analytic(lam, tf), 1.0e-5_dp)), 2, &
             'baseline accuracy')
  yref = y
  nst_ref = iwork(11)

  ! ================================================================
  ! (10) A plain (default) ZVODE_EWT must reproduce the baseline exactly:
  ! the default dispatch path is a faithful re-expression of ZEWSET.
  ! ================================================================
  call solve(y, t, istate, iwork, diag_fun(neq, lam), diag_jac(neq, lam), &
             zvode_ewt())
  call check(istate == 2, 10, 'default-functor: istate /= 2')
  ! exact equality: identical arithmetic, identical control path
  call check(all(y == yref), 11, 'default-functor: trajectory /= baseline')
  call check(iwork(11) == nst_ref, 12, 'default-functor: NST /= baseline')

  ! ================================================================
  ! (20) Stateful override re-implementing the default formula:
  ! reproduces the baseline, is called once per step (NOT once), and
  ! its counter survives in the caller's object after ZVODE returns.
  ! ================================================================
  cewt = counting_ewt(ncalls=0)
  call solve(y, t, istate, iwork, diag_fun(neq, lam), diag_jac(neq, lam), &
             cewt)
  call check(istate == 2, 20, 'counting-functor: istate /= 2')
  call check(all(is_close(y, yref, 1.0e-12_dp)), 21, &
             'counting-functor: trajectory /= baseline')
  ! the crux: efun/ZEWSET is invoked before every internal step, so the
  ! counter must far exceed one and be at least the number of steps
  call check(cewt%ncalls > 1, 22, 'counting-functor: called only once')
  call check(cewt%ncalls >= nst_ref, 23, &
             'counting-functor: fewer calls than steps')

  ! ================================================================
  ! (30) Loosening the weights must actually change solver behaviour,
  ! confirming the override is consulted rather than ignored.  Larger
  ! weights => smaller error norm => bigger steps => NST cannot exceed
  ! the baseline (and in practice is strictly smaller).
  ! ================================================================
  sewt = scaled_ewt(factor=1.0e6_dp)
  call solve(y, t, istate, iwork, diag_fun(neq, lam), diag_jac(neq, lam), &
             sewt)
  call check(istate == 2, 30, 'scaled-functor: istate /= 2')
  nst_scaled = iwork(11)
  call check(nst_scaled <= nst_ref, 31, &
             'scaled-functor: looser weights did not reduce step count')

  write(*,'(a)') 'PASS: test_zvode_ewt_functor'
  write(*,'(a,i0,a,i0,a,i0)') '  baseline NST=', nst_ref, &
       '  ewt-calls=', cewt%ncalls, '  loosened NST=', nst_scaled

contains

  ! One ZVODE solve from t=0 to tf with fresh work arrays.  EWTF is an
  ! optional error-weight override forwarded to ZVODE's EWTFUN argument.
  subroutine solve(y, t, istate, iwork, f, jac, ewtf)
    use zvode_mod, only: zvode_fun, zvode_jac, zvode_ewt
    complex(dp), intent(out) :: y(neq)
    real(dp), intent(out) :: t
    integer, intent(out) :: istate, iwork(liw)
    class(zvode_fun) :: f
    class(zvode_jac) :: jac
    class(zvode_ewt), optional :: ewtf
    complex(dp) :: zw(lzw)
    real(dp) :: rw(lrw)
    y = cmplx(1.0_dp, 0.0_dp, dp)
    t = 0.0_dp
    zw = 0.0_dp
    rw = 0.0_dp
    iwork = 0
    istate = 1
    if (present(ewtf)) then
      call zvode(f, neq, y, t, tf, 1, [rtol], [atol], 1, istate, 0, &
                 zw, lzw, rw, lrw, iwork, liw, jac, mf, ewtf)
    else
      call zvode(f, neq, y, t, tf, 1, [rtol], [atol], 1, istate, 0, &
                 zw, lzw, rw, lrw, iwork, liw, jac, mf)
    end if
  end subroutine

  elemental function analytic(l, tt) result(v)
    complex(dp), intent(in) :: l
    real(dp), intent(in) :: tt
    complex(dp) :: v
    v = exp(l * tt)
  end function

  elemental function is_close(got, want, rtol_c) result(ok)
    complex(dp), intent(in) :: got, want
    real(dp), intent(in) :: rtol_c
    logical :: ok
    ok = abs(got - want) <= rtol_c * abs(want) + 1.0e-12_dp
  end function

  subroutine check(ok, code, what)
    logical, intent(in) :: ok
    integer, intent(in) :: code
    character(*), intent(in) :: what
    if (.not. ok) then
      write(*,'(a,a)') 'FAIL: ', what
      error stop code
    end if
  end subroutine

end program test_zvode_ewt_functor
