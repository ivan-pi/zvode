! test_zvode_public_api.f90
! ============================================================
!  Fortran-native test of the three public ZVODE_MOD routines
!  ZVODE, ZVINDY and ZVSRCO, added when the internal /ZVOD01/
!  and /ZVOD02/ COMMON blocks were replaced by module variables.
!
!  Problem P (decoupled, so each component has a closed form):
!      dy_i/dt = lam_i * y_i,   y_i(0) = 1     =>   y_i(t) = exp(lam_i t)
!  with NEQ = 8 distinct eigenvalues (a mix of decaying and oscillatory
!  modes), integrated with BDF + a dense analytic Jacobian (MF = 21),
!  which populates the Newton/Jacobian/LU counters that ZVSRCO must
!  round-trip.  Decoupling only makes the reference solution analytic --
!  ZVODE still forms and LU-factors the full NEQ x NEQ Jacobian.
!
!  Coverage (assertion codes passed to `error stop`, unique per check):
!    ZVODE   1-5   integration reaches TOUT and matches exp(lam t)
!    ZVINDY  10-14 interpolation reproduces y and dy/dt, and out-of-range
!                  K returns IFLAG = -1
!    ZVSRCO  20-32 save slots map to the SAME variables as ZVODE's
!                  documented IWORK/RWORK outputs -- i.e. the save/restore
!                  ORDER matches the historical COMMON-block layout, so
!                  this is not a breaking change
!    ZVSRCO  40-43 save -> (solve an unrelated problem, clobbering the
!                  module state) -> restore -> resume reproduces the
!                  uninterrupted reference solution
! ============================================================

module test_public_api_mod
  use zvode_mod, only: zvode_fun, zvode_jac
  implicit none
  private
  public :: diag_fun, diag_jac, dp
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

end module test_public_api_mod


program test_zvode_public_api
  use zvode_mod, only: zvode, zvindy, zvsrco, xsetf
  use test_public_api_mod, only: diag_fun, diag_jac, dp
  implicit none

  ! Solver settings with defaults, so a fresh `solver_settings()` resets
  ! everything (in particular ISTATE = 1) via the default structure
  ! constructor.  ISTATE is updated in place by ZVODE.
  type :: solver_settings
    integer  :: itol   = 1
    real(dp) :: rtol   = 1.0e-9_dp
    real(dp) :: atol   = 1.0e-11_dp
    integer  :: itask  = 1
    integer  :: istate = 1
    integer  :: iopt   = 0
  end type

  integer, parameter :: neq = 8, mf = 21
  integer, parameter :: lzw = 8*neq + 2*neq*neq   ! MF = 21
  integer, parameter :: lrw = 20 + neq
  integer, parameter :: liw = 30 + neq

  ! Named ZVSRCO JOB flags, so the call sites read as save/restore rather
  ! than the bare 1/2.
  integer, parameter :: save_state = 1, restore_state = 2

  ! Problem P and (unrelated) problem Q eigenvalues -- all with negative
  ! real part, spanning pure-decay and decaying-oscillatory modes.
  complex(dp), parameter :: lam(neq) = [ &
       cmplx(-2.0_dp, 0.0_dp, dp), cmplx(-5.0_dp,  0.0_dp, dp), &
       cmplx(-0.5_dp, 5.0_dp, dp), cmplx(-1.0_dp,  3.0_dp, dp), &
       cmplx(-0.3_dp, 8.0_dp, dp), cmplx(-0.1_dp,  1.0_dp, dp), &
       cmplx(-3.0_dp,-2.0_dp, dp), cmplx(-1.5_dp,  4.0_dp, dp)]
  complex(dp), parameter :: lamq(neq) = [ &
       cmplx(-4.0_dp, 1.0_dp, dp), cmplx(-1.0_dp, -3.0_dp, dp), &
       cmplx(-2.5_dp, 2.0_dp, dp), cmplx(-0.8_dp,  6.0_dp, dp), &
       cmplx(-6.0_dp, 0.0_dp, dp), cmplx(-0.2_dp, -1.0_dp, dp), &
       cmplx(-3.5_dp, 3.0_dp, dp), cmplx(-1.2_dp, -4.0_dp, dp)]

  real(dp), parameter :: tmid = 0.6_dp, tf = 1.5_dp

  type(solver_settings) :: s, sq
  complex(dp) :: y(neq), zwork(lzw), yref(neq), dky(neq), ysav(neq)
  real(dp) :: rwork(lrw), t, rsav(51), rsavq(51), tsav
  integer :: iwork(liw), iflag, isav(41), isavq(41)

  complex(dp) :: yq(neq), zworkq(lzw)
  real(dp) :: rworkq(lrw), tq
  integer :: iworkq(liw)

  ! ================================================================
  ! Reference: solve P from 0 to tf in a single call.
  ! ================================================================
  s = solver_settings()
  y = cmplx(1.0_dp, 0.0_dp, dp)
  t = 0.0_dp
  zwork = 0.0_dp
  rwork = 0.0_dp
  iwork = 0
  call zvode(diag_fun(neq, lam), neq, y, t, tf, s%itol, [s%rtol], [s%atol], &
             s%itask, s%istate, s%iopt, zwork, lzw, rwork, lrw, iwork, liw, &
             diag_jac(neq, lam), mf)
  call check(s%istate == 2, 1, 'ZVODE reference: istate /= 2')
  ! exact comparison is intended: on a successful ITASK = 1 return,
  ! ZVODE assigns T = TOUT verbatim
  call check(t == tf,       2, 'ZVODE reference: T /= tf')
  call check(all(is_close(y, analytic(lam, tf), 1.0e-5_dp)), 3, &
             'ZVODE reference accuracy')
  yref = y

  ! ================================================================
  ! Interrupted solve: leg 1 from 0 to tmid.
  ! ================================================================
  s = solver_settings()
  y = cmplx(1.0_dp, 0.0_dp, dp)
  t = 0.0_dp
  zwork = 0.0_dp
  rwork = 0.0_dp
  iwork = 0
  call zvode(diag_fun(neq, lam), neq, y, t, tmid, s%itol, [s%rtol], [s%atol], &
             s%itask, s%istate, s%iopt, zwork, lzw, rwork, lrw, iwork, liw, &
             diag_jac(neq, lam), mf)
  call check(s%istate == 2, 4, 'ZVODE leg1: istate /= 2')
  call check(all(is_close(y, analytic(lam, tmid), 1.0e-5_dp)), 5, &
             'ZVODE leg1 accuracy')

  ! ================================================================
  ! ZVINDY: interpolate at t = tmid using the Nordsieck array in ZWORK.
  ! For the standard setup the YH array starts at ZWORK(1) with column
  ! length NYH = NEQ.
  ! ================================================================
  ! K = 0 must reproduce the returned solution vector.
  call zvindy(tmid, 0, zwork, neq, dky, iflag)
  call check(iflag == 0, 10, 'ZVINDY K=0: iflag /= 0')
  call check(all(is_close(dky, y, 1.0e-10_dp)), 11, 'ZVINDY K=0 vs y')

  ! K = 1 gives dy/dt, compared against the analytic derivative lam*y.
  call zvindy(tmid, 1, zwork, neq, dky, iflag)
  call check(iflag == 0, 12, 'ZVINDY K=1: iflag /= 0')
  call check(all(is_close(dky, lam*analytic(lam, tmid), 1.0e-3_dp)), 13, &
             'ZVINDY K=1 vs lam*y')

  ! Out-of-range derivative order must be reported, not computed.
  call xsetf(0)                       ! silence the informational message
  call zvindy(tmid, 13, zwork, neq, dky, iflag)
  call xsetf(1)
  call check(iflag == -1, 14, 'ZVINDY K=13: expected iflag = -1')

  ! ================================================================
  ! ZVSRCO save: the saved slots must hold the SAME quantities that
  ! ZVODE reported through its documented public IWORK/RWORK outputs.
  ! This pins the save/restore ORDER to the historical COMMON layout:
  !   RSAV(1:50) = /ZVOD01/ reals, RSAV(51) = HU (/ZVOD02/)
  !   ISAV(1:33) = /ZVOD01/ ints,  ISAV(34:41) = /ZVOD02/ ints
  ! ================================================================
  call zvsrco(rsav, isav, job=save_state)

  ! exact equality is intended throughout this block: both sides are
  ! verbatim copies of the same internal variable
  call check(rsav(51) == rwork(11), 20, 'RSAV(51) = HU  = RWORK(11)')
  call check(rsav(49) == rwork(13), 21, 'RSAV(49) = TN  = RWORK(13)')
  call check(isav(41) == iwork(11), 22, 'ISAV(41) = NST  = IWORK(11)')
  call check(isav(36) == iwork(12), 23, 'ISAV(36) = NFE  = IWORK(12)')
  call check(isav(37) == iwork(13), 24, 'ISAV(37) = NJE  = IWORK(13)')
  call check(isav(40) == iwork(14), 25, 'ISAV(40) = NQU  = IWORK(14)')
  call check(isav(26) == iwork(15), 26, 'ISAV(26) = NEWQ = IWORK(15)')
  call check(isav(38) == iwork(20), 27, 'ISAV(38) = NLU  = IWORK(20)')
  call check(isav(39) == iwork(21), 28, 'ISAV(39) = NNI  = IWORK(21)')
  call check(isav(34) == iwork(22), 29, 'ISAV(34) = NCFN = IWORK(22)')
  call check(isav(35) == iwork(23), 30, 'ISAV(35) = NETF = IWORK(23)')
  call check(isav(24) == neq,       31, 'ISAV(24) = N    = NEQ')

  ! sanity: the run must actually have exercised Jacobian/LU machinery,
  ! otherwise the counter checks above would be vacuously comparing zeros
  call check(iwork(11) > 0 .and. iwork(13) > 0 .and. iwork(20) > 0, 32, &
             'counters not populated (test would be vacuous)')

  ! stash P's solution and time for the resume test (module state is in
  ! RSAV/ISAV, and P's ZWORK/RWORK/IWORK are left untouched below)
  ysav = y
  tsav = t

  ! ================================================================
  ! ZVSRCO restore: clobber the shared module state by solving an
  ! unrelated problem Q (separate work arrays), then restore P's state
  ! and resume.  P's ZWORK/RWORK/IWORK are untouched by Q, so a correct
  ! resume depends entirely on ZVSRCO having restored the module state.
  ! ================================================================
  sq = solver_settings()
  yq = cmplx(1.0_dp, 0.5_dp, dp)
  tq = 0.0_dp
  zworkq = 0.0_dp
  rworkq = 0.0_dp
  iworkq = 0
  call zvode(diag_fun(neq, lamq), neq, yq, tq, 1.0_dp, sq%itol, [sq%rtol], &
             [sq%atol], sq%itask, sq%istate, sq%iopt, zworkq, lzw, rworkq, &
             lrw, iworkq, liw, diag_jac(neq, lamq), mf)
  call check(sq%istate == 2, 40, 'ZVODE problem Q: istate /= 2')

  ! sanity: Q must actually have changed the module state, otherwise the
  ! restore below would be vacuous
  call zvsrco(rsavq, isavq, job=save_state)
  call check(any(rsavq /= rsav) .or. any(isavq /= isav), 41, &
             'problem Q did not change the module state (test vacuous)')

  ! restore P's internal state and pick up exactly where leg 1 stopped
  call zvsrco(rsav, isav, job=restore_state)
  y = ysav
  t = tsav
  s%istate = 2
  call zvode(diag_fun(neq, lam), neq, y, t, tf, s%itol, [s%rtol], [s%atol], &
             s%itask, s%istate, s%iopt, zwork, lzw, rwork, lrw, iwork, liw, &
             diag_jac(neq, lam), mf)
  call check(s%istate == 2, 42, 'ZVODE resume: istate /= 2')
  call check(all(is_close(y, yref, 1.0e-6_dp)), 43, &
             'resume matches uninterrupted reference')

  write(*,'(a)') 'PASS: test_zvode_public_api'
  write(*,'(a,i0,a,i0,a,i0)') '  resumed-run NST=', iwork(11), &
       ' NJE=', iwork(13), ' NLU=', iwork(20)

contains

  ! Closed-form solution of each decoupled mode, y_i(t) = exp(lam_i t).
  elemental function analytic(l, tt) result(v)
    complex(dp), intent(in) :: l
    real(dp), intent(in) :: tt
    complex(dp) :: v
    v = exp(l * tt)
  end function

  ! Mixed absolute/relative closeness predicate, elemental so it applies
  ! componentwise and reduces with ALL/ANY at the call site.  ATOL is the
  ! absolute floor and defaults to 1.0e-12 when omitted.
  elemental function is_close(got, want, rtol, atol) result(ok)
    complex(dp), intent(in) :: got, want
    real(dp), intent(in) :: rtol
    real(dp), intent(in), optional :: atol
    logical :: ok
    real(dp) :: a
    a = 1.0e-12_dp
    if (present(atol)) a = atol
    ok = abs(got - want) <= rtol * abs(want) + a
  end function

  ! Single reporting sink: consumes an already-reduced logical (it cannot
  ! be elemental, since it does I/O and stops).
  subroutine check(ok, code, what)
    logical, intent(in) :: ok
    integer, intent(in) :: code
    character(*), intent(in) :: what
    if (.not. ok) then
      write(*,'(a,a)') 'FAIL: ', what
      error stop code
    end if
  end subroutine

end program test_zvode_public_api
