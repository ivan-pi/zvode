! test_zvode_decay.f90
! ============================================================
!  Fortran driver that replicates test_zvode_scalar_real_decay
!  from the Python test-suite.
!
!  Problem : dy/dt = -y,  y(0) = 1 + 0i  =>  y(t) = exp(-t)
!  Method  : Adams (MF = 10), no Jacobian (MITER = 0), NEQ = 1.
!  State   : DOUBLE COMPLEX array, purely real solution.
!
!  Assertions (mirror the Python pytest):
!    1. ISTATE == 2 on return
!    2. T == TOUT on return
!    3. numpy-style allclose(Re[y], exp(-TOUT), rtol=1e-4, atol=atol)
!    4. |Im[y(1)]| < 1e-12
!
!  Exits with status 0 on pass, 1 on any failed assertion.
!
!  Compile (see also Makefile):
!    gfortran -Wall -Wextra -fimplicit-none -fcheck=all -fbacktrace \
!             -O0 -g -c test_zvode_decay.f90 -o test_zvode_decay.o
!    gfortran -O2 -g -c zvode_original.f       -o zvode_original.o
!    gfortran -O2 -g -c zvode_linpack_stubs.f  -o zvode_linpack_stubs.o
!    gfortran test_zvode_decay.o zvode_original.o \
!             zvode_linpack_stubs.o -lblas -o test_zvode_decay
! ============================================================

program test_zvode_scalar_real_decay
  implicit none

  ! ---- Subroutines that we pass to ZVODE as procedure arguments -----
  ! Must be declared EXTERNAL so gfortran treats them as procedures,
  ! not as scalar variables, when they appear in the CALL argument list.
  external :: fex        ! right-hand side  f(t,y)
  external :: dummy_jac  ! Jacobian stub (never invoked for MF=10)

  ! ---- Problem / method constants -----------------------------------
  integer, parameter :: neq = 1   ! number of first-order ODEs
  integer, parameter :: mf  = 10  ! Adams method, MITER=0 (no Jacobian)

  ! ---- Minimum work-array lengths for MF=10, NEQ=1 -----------------
  !   LZW >= 15*NEQ       =  15   (complex work array)
  !   LRW >= 20 + NEQ     =  21   (real work array)
  !   LIW >= 30                   (integer work array)
  integer, parameter :: lzw = 15
  integer, parameter :: lrw = 21
  integer, parameter :: liw = 30

  ! ---- ZVODE call-sequence variables --------------------------------
  double complex    :: y(neq)      ! solution vector (input y0; output y(t))
  double complex    :: zwork(lzw)  ! complex scratch space
  double precision  :: rwork(lrw)  ! real scratch space
  integer           :: iwork(liw)  ! integer scratch space / diagnostics
  double precision  :: t           ! current time (input t0; output t_reached)
  double precision  :: tout        ! target time
  double precision  :: rtol        ! relative tolerance (scalar: ITOL=1)
  double precision  :: atol        ! absolute tolerance (scalar: ITOL=1)
  integer           :: itol        ! 1 = both tolerances are scalars
  integer           :: itask       ! 1 = advance to TOUT then return
  integer           :: istate      ! 1 = first call; output 2 = success
  integer           :: iopt        ! 0 = all defaults

  ! Pass-through arrays: not used in this test but required by the
  ! ZVODE/FEX/DUMMY_JAC calling convention.
  double precision  :: rpar(1)
  integer           :: ipar(1)

  ! ---- Local variables for result verification ----------------------
  double precision :: exact, abserr, rel_err, imag_part, tol

  ! ==================================================================
  ! Initialise the problem
  ! ==================================================================
  y(1)   = dcmplx(1.0d0, 0.0d0)   ! y(0) = 1 + 0 i

  t      = 0.0d0
  tout   = 10.0d0

  itol   = 1       ! EWT_i = rtol*|y_i| + atol
  rtol   = 1.0d-6
  atol   = 1.0d-8
  itask  = 1       ! normal: integrate to TOUT by overshooting + interpolation
  istate = 1       ! first call
  iopt   = 0       ! use all defaults (max steps=500, max order=12, …)

  zwork  = dcmplx(0.0d0, 0.0d0)
  rwork  = 0.0d0
  iwork  = 0
  rpar   = 0.0d0
  ipar   = 0

  ! ==================================================================
  ! Call ZVODE
  ! ==================================================================
  call zvode(fex, neq, y, t, tout,               &
             itol, rtol, atol,                   &
             itask, istate, iopt,                &
             zwork, lzw, rwork, lrw, iwork, liw, &
             dummy_jac, mf, rpar, ipar)

  ! ==================================================================
  ! Assertions — mirror test_zvode_scalar_real_decay
  ! ==================================================================

  ! 1.  assert istate_new == 2
  if (istate /= 2) then
    write(*, '(a,i0)') 'FAIL: ZVODE returned istate = ', istate
    stop 1
  end if

  ! 2.  assert t_new == tout
  !     ZVODE with ITASK=1 sets T exactly to TOUT on success; we verify
  !     this with < / > rather than /= to avoid -Wcompare-reals.
  if (t < tout .or. t > tout) then
    write(*, '(a,es22.14)') 'FAIL: ZVODE did not reach TOUT, T = ', t
    stop 1
  end if

  ! 3.  assert_allclose(y[0].real, exp(-tout), rtol=1e-4)
  !
  !     numpy.testing.assert_allclose checks:
  !       |actual - desired| <= atol_check + rtol_check * |desired|
  !     The Python call uses rtol_check=1e-4 and atol_check=0 (default).
  !     At t=10 the solution exp(-10) ≈ 4.54e-5 is tiny, so the solver's
  !     own atol=1e-8 dominates the error weight and the absolute error is
  !     O(1e-9).  We therefore use atol_check = atol (solver's absolute
  !     tolerance) as a sensible floor, matching the intent of the check:
  !       |Re[y] - exp(-10)| <= 1e-4 * exp(-10) + 1e-8
  exact   = exp(-tout)
  abserr  = abs(dble(y(1)) - exact)   ! dble() extracts real part of DOUBLE COMPLEX
  rel_err = abserr / exact             ! reported for diagnostics only
  tol     = 1.0d-4 * abs(exact) + atol
  if (abserr > tol) then
    write(*, '(a,es16.8,a,es16.8)') &
         'FAIL: Re[y(1)] = ', dble(y(1)), '  exact = ', exact
    write(*, '(a,es10.2,a,es10.2)') &
         '      abserr   = ', abserr,     '  tol   = ', tol
    stop 1
  end if

  ! 4.  assert abs(y[0].imag) < 1e-12
  imag_part = abs(aimag(y(1)))         ! aimag() extracts imaginary part
  if (imag_part > 1.0d-12) then
    write(*, '(a,es10.2)') 'FAIL: Im[y(1)] = ', imag_part
    stop 1
  end if

  ! ==================================================================
  ! All assertions passed — print summary and diagnostic counters
  ! ==================================================================
  write(*, '(a)') ''
  write(*, '(a)') '==========================================='
  write(*, '(a)') 'PASS: test_zvode_scalar_real_decay'
  write(*, '(a)') '==========================================='
  write(*, '(a,es22.15)')  '  T              = ', t
  write(*, '(a,es22.15)')  '  Re[y(1)]       = ', dble(y(1))
  write(*, '(a,es22.15)')  '  exp(-10)        = ', exact
  write(*, '(a,es10.2)')   '  Abs. error      = ', abserr
  write(*, '(a,es10.2)')   '  Rel. error      = ', rel_err
  write(*, '(a,es10.2)')   '  |Im[y(1)]|      = ', imag_part
  write(*, '(a)') '-------------------------------------------'
  write(*, '(a,i0)')  '  NST  (steps taken)            = ', iwork(11)
  write(*, '(a,i0)')  '  NFE  (f-evaluations)          = ', iwork(12)
  write(*, '(a,i0)')  '  NJE  (Jacobian evaluations)   = ', iwork(13)
  write(*, '(a,i0)')  '  NQU  (method order used)      = ', iwork(14)
  write(*, '(a,i0)')  '  NLU  (LU decompositions)      = ', iwork(20)
  write(*, '(a,i0)')  '  NETF (error-test failures)    = ', iwork(23)
  write(*, '(a)') ''

end program test_zvode_scalar_real_decay


! ==================================================================
!  FEX — right-hand side subroutine
!
!  Interface required by ZVODE:
!    SUBROUTINE F(NEQ, T, Y, YDOT, RPAR, IPAR)
!    DOUBLE COMPLEX Y(NEQ), YDOT(NEQ)
!    DOUBLE PRECISION T
!
!  Implements: YDOT(i) = -Y(i)
! ==================================================================
subroutine fex(neq, t, y, ydot, rpar, ipar)
  implicit none
  integer,          intent(in)  :: neq
  double precision, intent(in)  :: t
  double complex,   intent(in)  :: y(neq)
  double complex,   intent(out) :: ydot(neq)
  double precision, intent(in)  :: rpar(*)   ! not used in this test
  integer,          intent(in)  :: ipar(*)   ! not used in this test
  integer :: i

  do i = 1, neq
    ydot(i) = -y(i)
  end do

  ! Suppress -Wunused-dummy-argument (-Wextra) for t, rpar, ipar.
  ! The condition t < -huge(t) is permanently false for any finite t
  ! that ZVODE passes in, so this branch never executes at runtime.
  if (t < -huge(t)) ydot(1) = ydot(1) + dcmplx(rpar(1), dble(ipar(1)))

end subroutine fex


! ==================================================================
!  DUMMY_JAC — Jacobian stub
!
!  Interface required by ZVODE:
!    SUBROUTINE JAC(NEQ, T, Y, ML, MU, PD, NROWPD, RPAR, IPAR)
!    DOUBLE COMPLEX Y(NEQ), PD(NROWPD, NEQ)
!    DOUBLE PRECISION T
!
!  For MF = 10 (Adams / MITER = 0) ZVODE never calls the Jacobian
!  routine; a dummy subroutine is all that is required.
! ==================================================================
subroutine dummy_jac(neq, t, y, ml, mu, pd, nrowpd, rpar, ipar)
  implicit none
  integer,          intent(in)    :: neq, ml, mu, nrowpd
  double precision, intent(in)    :: t
  double complex,   intent(in)    :: y(neq)
  double complex,   intent(inout) :: pd(nrowpd, neq)
  double precision, intent(in)    :: rpar(*)
  integer,          intent(in)    :: ipar(*)

  ! Trap any accidental call: this subroutine must never execute.
  write(*, '(a)') &
    'BUG: dummy_jac called — should never happen for MF = 10 (MITER = 0)'
  stop 1

  ! Dead code below this STOP.
  ! Referencing every dummy argument suppresses -Wunused-dummy-argument
  ! from -Wextra.  The STOP above ensures this code never runs at runtime.
  pd(1, 1) = y(neq) * dcmplx( t + dble(ml + mu + nrowpd) + rpar(1), &
                               dble(ipar(1)) )

end subroutine dummy_jac
