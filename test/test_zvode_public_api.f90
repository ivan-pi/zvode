! test_zvode_public_api.f90
! ============================================================
!  Fortran-native test of the three public ZVODE_MOD routines
!  ZVODE, ZVINDY and ZVSRCO, added when the internal /ZVOD01/
!  and /ZVOD02/ COMMON blocks were replaced by module variables.
!
!  Problem P (decoupled, so each component has a closed form):
!      dy_i/dt = lam_i * y_i,   y_i(0) = 1     =>   y_i(t) = exp(lam_i t)
!  with lam = (-2, 0) and (-0.5, 5), integrated with BDF + a dense
!  analytic Jacobian (MF = 21), which populates the Newton/Jacobian/LU
!  counters that ZVSRCO must round-trip.
!
!  Coverage (assertion codes passed to `error stop`):
!    ZVODE   1-4   integration reaches TOUT and matches exp(lam t)
!    ZVINDY  10-13 interpolation reproduces y and dy/dt; out-of-range
!                  K returns IFLAG = -1
!    ZVSRCO  20-33 save slots map to the SAME variables as ZVODE's
!                  documented IWORK/RWORK outputs -- i.e. the save/restore
!                  ORDER matches the historical COMMON-block layout, so
!                  this is not a breaking change
!    ZVSRCO  40-42 save -> (solve an unrelated problem, clobbering the
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
    complex(dp) :: lam(8) = (0.0_dp, 0.0_dp)
  contains
    procedure :: eval => diag_f
  end type

  type, extends(zvode_jac) :: diag_jac
    complex(dp) :: lam(8) = (0.0_dp, 0.0_dp)
  contains
    procedure :: eval => diag_j
  end type

contains

  subroutine diag_f(fun, t, y, ydot)
    class(diag_fun) :: fun
    real(dp), intent(in) :: t
    complex(dp), intent(in) :: y(fun%neq)
    complex(dp), intent(out) :: ydot(fun%neq)
    integer :: i
    do i = 1, fun%neq
      ydot(i) = fun%lam(i) * y(i)
    end do
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

  integer, parameter :: neq = 2, mf = 21
  integer, parameter :: lzw = 8*neq + 2*neq*neq   ! MF = 21
  integer, parameter :: lrw = 20 + neq
  integer, parameter :: liw = 30 + neq

  complex(dp), parameter :: lam(neq) = &
       [cmplx(-2.0_dp, 0.0_dp, dp), cmplx(-0.5_dp, 5.0_dp, dp)]

  complex(dp) :: y(neq), zwork(lzw), yref(neq), dky(neq)
  real(dp) :: rwork(lrw), t, tout, rtol(1), atol(1)
  integer :: iwork(liw), itol, itask, istate, iopt, iflag

  ! saved copies for the interleaved save/restore/resume test
  complex(dp) :: ysav(neq)
  real(dp) :: rsav(51), tsav
  integer :: isav(41)

  ! second, unrelated problem used to clobber the shared module state
  complex(dp) :: yq(neq), zworkq(lzw)
  real(dp) :: rworkq(lrw), tq, toutq
  integer :: iworkq(liw), istateq
  complex(dp), parameter :: lamq(neq) = &
       [cmplx(-5.0_dp, 1.0_dp, dp), cmplx(-1.0_dp, -3.0_dp, dp)]

  real(dp), parameter :: tmid = 0.6_dp, tf = 1.5_dp
  integer :: i

  ! ================================================================
  ! Reference: solve P from 0 to tf in a single call.
  ! ================================================================
  call init_p(y, t)
  tout = tf
  call zvode(diag_fun(neq, lam), neq, y, t, tout, itol, rtol, atol, itask, &
             istate, iopt, zwork, lzw, rwork, lrw, iwork, liw, &
             diag_jac(neq, lam), mf)
  if (istate /= 2) call fail('ZVODE reference: istate /= 2', 1)
  if (t /= tf)     call fail('ZVODE reference: T /= tf', 2)
  do i = 1, neq
    call assert_close(y(i), analytic(lam(i), tf), 1.0e-5_dp, &
                      'ZVODE reference accuracy', 3)
  end do
  yref = y

  ! ================================================================
  ! Interrupted solve: leg 1 from 0 to tmid.
  ! ================================================================
  call init_p(y, t)
  tout = tmid
  call zvode(diag_fun(neq, lam), neq, y, t, tout, itol, rtol, atol, itask, &
             istate, iopt, zwork, lzw, rwork, lrw, iwork, liw, &
             diag_jac(neq, lam), mf)
  if (istate /= 2) call fail('ZVODE leg1: istate /= 2', 4)
  do i = 1, neq
    call assert_close(y(i), analytic(lam(i), tmid), 1.0e-5_dp, &
                      'ZVODE leg1 accuracy', 4)
  end do

  ! ================================================================
  ! ZVINDY: interpolate at t = tmid using the Nordsieck array in ZWORK.
  ! For the standard setup the YH array starts at ZWORK(1) with column
  ! length NYH = NEQ.
  ! ================================================================
  ! K = 0 must reproduce the returned solution vector exactly.
  call zvindy(tmid, 0, zwork, neq, dky, iflag)
  if (iflag /= 0) call fail('ZVINDY K=0: iflag /= 0', 10)
  do i = 1, neq
    call assert_close(dky(i), y(i), 1.0e-10_dp, 'ZVINDY K=0 vs y', 11)
  end do

  ! K = 1 gives dy/dt; compare to the analytic derivative lam*y.
  call zvindy(tmid, 1, zwork, neq, dky, iflag)
  if (iflag /= 0) call fail('ZVINDY K=1: iflag /= 0', 12)
  do i = 1, neq
    call assert_close(dky(i), lam(i)*analytic(lam(i), tmid), 1.0e-3_dp, &
                      'ZVINDY K=1 vs lam*y', 12)
  end do

  ! Out-of-range derivative order must be reported, not computed.
  call xsetf(0)                       ! silence the informational message
  call zvindy(tmid, 13, zwork, neq, dky, iflag)
  call xsetf(1)
  if (iflag /= -1) call fail('ZVINDY K=13: expected iflag = -1', 13)

  ! ================================================================
  ! ZVSRCO save: the saved slots must hold the SAME quantities that
  ! ZVODE reported through its documented public IWORK/RWORK outputs.
  ! This pins the save/restore ORDER to the historical COMMON layout:
  !   RSAV(1:50) = /ZVOD01/ reals, RSAV(51) = HU (/ZVOD02/)
  !   ISAV(1:33) = /ZVOD01/ ints,  ISAV(34:41) = /ZVOD02/ ints
  ! ================================================================
  call zvsrco(rsav, isav, 1)

  call assert_real(rsav(51), rwork(11), 'RSAV(51) = HU  = RWORK(11)', 20)
  call assert_real(rsav(49), rwork(13), 'RSAV(49) = TN  = RWORK(13)', 21)
  call assert_int(isav(41), iwork(11), 'ISAV(41) = NST  = IWORK(11)', 22)
  call assert_int(isav(36), iwork(12), 'ISAV(36) = NFE  = IWORK(12)', 23)
  call assert_int(isav(37), iwork(13), 'ISAV(37) = NJE  = IWORK(13)', 24)
  call assert_int(isav(40), iwork(14), 'ISAV(40) = NQU  = IWORK(14)', 25)
  call assert_int(isav(26), iwork(15), 'ISAV(26) = NEWQ = IWORK(15)', 26)
  call assert_int(isav(38), iwork(20), 'ISAV(38) = NLU  = IWORK(20)', 27)
  call assert_int(isav(39), iwork(21), 'ISAV(39) = NNI  = IWORK(21)', 28)
  call assert_int(isav(34), iwork(22), 'ISAV(34) = NCFN = IWORK(22)', 29)
  call assert_int(isav(35), iwork(23), 'ISAV(35) = NETF = IWORK(23)', 30)
  call assert_int(isav(24), neq,       'ISAV(24) = N    = NEQ',       31)

  ! sanity: the run must actually have exercised Jacobian/LU machinery,
  ! otherwise the counter checks above would be vacuously comparing zeros
  if (iwork(11) <= 0 .or. iwork(13) <= 0 .or. iwork(20) <= 0) &
       call fail('counters not populated (test would be vacuous)', 32)

  ! stash P's state (module state + solution + time) for the resume test
  ysav = y
  tsav = t

  ! ================================================================
  ! ZVSRCO restore: clobber the shared module state by solving an
  ! unrelated problem Q (separate work arrays), then restore P's state
  ! and resume.  P's ZWORK/RWORK/IWORK are untouched by Q, so a correct
  ! resume depends entirely on ZVSRCO having restored the module state.
  ! ================================================================
  ! initialise Q's OWN arrays directly -- must not touch P's zwork/rwork/iwork
  yq = [cmplx(2.0_dp,0.0_dp,dp), cmplx(0.0_dp,3.0_dp,dp)]
  tq = 0.0_dp; toutq = 1.0_dp; istateq = 1
  zworkq = 0.0_dp; rworkq = 0.0_dp; iworkq = 0
  call zvode(diag_fun(neq, lamq), neq, yq, tq, toutq, itol, rtol, atol, &
             itask, istateq, iopt, zworkq, lzw, rworkq, lrw, iworkq, liw, &
             diag_jac(neq, lamq), mf)
  if (istateq /= 2) call fail('ZVODE problem Q: istate /= 2', 40)

  ! restore P's internal state and pick up exactly where leg 1 stopped
  call zvsrco(rsav, isav, 2)
  y = ysav
  t = tsav
  istate = 2

  tout = tf
  call zvode(diag_fun(neq, lam), neq, y, t, tout, itol, rtol, atol, itask, &
             istate, iopt, zwork, lzw, rwork, lrw, iwork, liw, &
             diag_jac(neq, lam), mf)
  if (istate /= 2) call fail('ZVODE resume: istate /= 2', 41)
  do i = 1, neq
    call assert_close(y(i), yref(i), 1.0e-6_dp, &
                      'resume matches uninterrupted reference', 42)
  end do

  write(*,'(a)') 'PASS: test_zvode_public_api'
  write(*,'(a,i0,a,i0,a,i0)') '  reference NST=', iwork(11), &
       ' NJE=', iwork(13), ' NLU=', iwork(20)

contains

  subroutine init_p(yy, tt)
    complex(dp), intent(out) :: yy(neq)
    real(dp), intent(out) :: tt
    yy = cmplx(1.0_dp, 0.0_dp, dp)
    tt = 0.0_dp
    itol = 1; rtol = 1.0e-9_dp; atol = 1.0e-11_dp
    itask = 1; istate = 1; iopt = 0
    zwork = 0.0_dp; rwork = 0.0_dp; iwork = 0
  end subroutine

  pure function analytic(l, tt) result(v)
    complex(dp), intent(in) :: l
    real(dp), intent(in) :: tt
    complex(dp) :: v
    v = exp(l * tt)
  end function

  subroutine assert_close(got, want, rtol_, what, code)
    complex(dp), intent(in) :: got, want
    real(dp), intent(in) :: rtol_
    character(*), intent(in) :: what
    integer, intent(in) :: code
    if (abs(got - want) > rtol_ * abs(want) + 1.0e-12_dp) then
      write(*,'(a,a)') 'FAIL: ', what
      write(*,'(a,2es24.15)') '  got  = ', got
      write(*,'(a,2es24.15)') '  want = ', want
      write(*,'(a,es12.3)')   '  |err|= ', abs(got - want)
      error stop code
    end if
  end subroutine

  subroutine assert_real(got, want, what, code)
    real(dp), intent(in) :: got, want
    character(*), intent(in) :: what
    integer, intent(in) :: code
    if (got /= want) then
      write(*,'(a,a)') 'FAIL: ', what
      write(*,'(a,es24.15,a,es24.15)') '  got=', got, '  want=', want
      error stop code
    end if
  end subroutine

  subroutine assert_int(got, want, what, code)
    integer, intent(in) :: got, want
    character(*), intent(in) :: what
    integer, intent(in) :: code
    if (got /= want) then
      write(*,'(a,a)') 'FAIL: ', what
      write(*,'(a,i0,a,i0)') '  got=', got, '  want=', want
      error stop code
    end if
  end subroutine

  subroutine fail(what, code)
    character(*), intent(in) :: what
    integer, intent(in) :: code
    write(*,'(a,a)') 'FAIL: ', what
    error stop code
  end subroutine

end program test_zvode_public_api
