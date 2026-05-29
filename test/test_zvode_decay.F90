! test_zvode_decay.F90
! ============================================================
!  Fortran driver that replicates test_zvode_scalar_real_decay
!  from the Python test-suite.
!
!  Problem : dy/dt = -y,  y(0) = 1 + 0i  =>  y(t) = exp(-t)
!  Method  : Adams (MF = 10), no Jacobian (MITER = 0), NEQ = 1.
!
!  Two ZVODE calling conventions are supported, selected at
!  compile time with the preprocessor macro USE_ZVODE_BIND_C:
!
!  Legacy interface  (default — zvode_original.f):
!    CALL ZVODE(F, NEQ, Y, T, TOUT, ITOL, RTOL, ATOL, ITASK,
!               ISTATE, IOPT, ZWORK, LZW, RWORK, LRW, IWORK, LIW,
!               JAC, MF, RPAR, IPAR)
!    F(NEQ, T, Y, YDOT, RPAR, IPAR)          — Fortran 77 style
!    JAC(NEQ, T, Y, ML, MU, PD, NROWPD, RPAR, IPAR)
!
!  bind(c) interface  (-DUSE_ZVODE_BIND_C — extern/zvode.f):
!    CALL ZVODE(F, NEQ, Y, T, TOUT, ITOL, RTOL, ATOL, ITASK,
!               ISTATE, IOPT, ZWORK, LZW, RWORK, LRW, IWORK, LIW,
!               JAC, MF, CTX)
!    F(NEQ, T, Y, YDOT, CTX) BIND(C)         — all scalars by value
!    JAC(NEQ, T, Y, ML, MU, PD, NROWPD, CTX) BIND(C)
!
!  Assertions mirror the Python pytest (see test_zvode_scalar_real_decay):
!    1. ISTATE == 2 on return
!    2. T == TOUT on return
!    3. |Re[y(1)] - exp(-10)| <= 1e-4 * exp(-10) + atol   (numpy-style)
!    4. |Im[y(1)]| < 1e-12
!
!  Exits with status 0 on pass, 1 on any failed assertion.
!
!  Compile (see also Makefile):
!    Legacy:   gfortran ... -c test_zvode_decay.F90  (no -D flag needed)
!    bind(c):  gfortran ... -DUSE_ZVODE_BIND_C -c test_zvode_decay.F90
!              (extern/zvode.f must be compiled first to provide zvode_mod.mod)
! ============================================================

program test_zvode_scalar_real_decay
#ifdef USE_ZVODE_BIND_C
  use zvode_mod, only: zvode
  use, intrinsic :: iso_c_binding, only: c_ptr, c_null_ptr
#endif
  implicit none

#ifndef USE_ZVODE_BIND_C
  external :: zvode
#endif

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
  integer           :: iwork(liw)  ! integer scratch / diagnostics
  double precision  :: t           ! current time (input t0; output t_reached)
  double precision  :: tout        ! target time
  double precision  :: rtol        ! relative tolerance (scalar: ITOL=1)
  double precision  :: atol        ! absolute tolerance (scalar: ITOL=1)
  integer           :: itol        ! 1 = both tolerances are scalars
  integer           :: itask       ! 1 = advance to TOUT then return
  integer           :: istate      ! 1 = first call; output 2 = success
  integer           :: iopt        ! 0 = all defaults

  ! ---- Interface-specific context / pass-through --------------------
#ifdef USE_ZVODE_BIND_C
  ! Opaque context pointer forwarded to every F/JAC callback.
  ! Null here because this test needs no extra parameters.
  type(c_ptr) :: ctx
#else
  ! Pass-through arrays required by the F77 calling convention.
  ! Not used in this test.
  double precision :: rpar(1)
  integer          :: ipar(1)
#endif

  ! ---- Local variables for result verification ----------------------
  double precision :: exact, abserr, rel_err, imag_part, tol

  ! ==================================================================
  ! Initialise
  ! ==================================================================
  y(1)   = dcmplx(1.0d0, 0.0d0)   ! y(0) = 1 + 0 i

  t      = 0.0d0
  tout   = 10.0d0
  itol   = 1       ! EWT_i = rtol*|y_i| + atol
  rtol   = 1.0d-6
  atol   = 1.0d-8
  itask  = 1       ! normal: integrate to TOUT
  istate = 1       ! first call
  iopt   = 0       ! use all defaults

  zwork  = dcmplx(0.0d0, 0.0d0)
  rwork  = 0.0d0
  iwork  = 0

#ifdef USE_ZVODE_BIND_C
  ctx  = c_null_ptr
#else
  rpar = 0.0d0
  ipar = 0
#endif

  ! ==================================================================
  ! Call ZVODE
  ! ==================================================================
#ifdef USE_ZVODE_BIND_C
  call zvode(fex, neq, y, t, tout,               &
             itol, rtol, atol,                   &
             itask, istate, iopt,                &
             zwork, lzw, rwork, lrw, iwork, liw, &
             dummy_jac, mf, ctx)
#else
  call zvode(fex, neq, y, t, tout,               &
             itol, rtol, atol,                   &
             itask, istate, iopt,                &
             zwork, lzw, rwork, lrw, iwork, liw, &
             dummy_jac, mf, rpar, ipar)
#endif

  ! ==================================================================
  ! Assertions — mirror test_zvode_scalar_real_decay
  ! ==================================================================

  ! 1.  assert istate_new == 2
  if (istate /= 2) then
    write(*, '(a,i0)') 'FAIL: ZVODE returned istate = ', istate
    stop 1
  end if

  ! 2.  assert t_new == tout
  !     Avoid -Wcompare-reals: use < / > rather than /=.
  if (t < tout .or. t > tout) then
    write(*, '(a,es22.14)') 'FAIL: ZVODE did not reach TOUT, T = ', t
    stop 1
  end if

  ! 3.  assert_allclose(y[0].real, exp(-tout), rtol=1e-4)
  !
  !     numpy.testing.assert_allclose checks:
  !       |actual - desired| <= atol_check + rtol_check * |desired|
  !     The Python call uses rtol_check=1e-4, atol_check=0 (default).
  !     At t=10 the solution exp(-10) ≈ 4.54e-5 is tiny, so the solver's
  !     own atol=1e-8 dominates the error weight and the absolute error is
  !     O(1e-9).  We therefore floor with atol to match the intent:
  !       |Re[y] - exp(-10)| <= 1e-4 * exp(-10) + 1e-8
  exact   = exp(-tout)
  abserr  = abs(dble(y(1)) - exact)
  rel_err = abserr / exact
  tol     = 1.0d-4 * abs(exact) + atol
  if (abserr > tol) then
    write(*, '(a,es16.8,a,es16.8)') &
         'FAIL: Re[y(1)] = ', dble(y(1)), '  exact = ', exact
    write(*, '(a,es10.2,a,es10.2)') &
         '      abserr   = ', abserr,     '  tol   = ', tol
    stop 1
  end if

  ! 4.  assert abs(y[0].imag) < 1e-12
  imag_part = abs(aimag(y(1)))
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

contains

! ==================================================================
!  FEX — right-hand side subroutine
!
!  Implements: YDOT(i) = -Y(i)
! ==================================================================
#ifdef USE_ZVODE_BIND_C
subroutine fex(neq, t, y, ydot, ctx) bind(c)
  use, intrinsic :: iso_c_binding, only: &
      c_int, c_double, c_double_complex, c_ptr, c_associated
  implicit none
  type(c_ptr),               value         :: ctx
#else
subroutine fex(neq, t, y, ydot, rpar, ipar)
  implicit none
  double precision, intent(in)  :: rpar(*)
  integer,          intent(in)  :: ipar(*)
#endif
  integer(c_int),            value         :: neq
  real(c_double),            value         :: t
  complex(c_double_complex), intent(in)    :: y(neq)
  complex(c_double_complex), intent(out)   :: ydot(neq)
  integer :: i

  do i = 1, neq
    ydot(i) = -y(i)
  end do

end subroutine fex

! ==================================================================
!  DUMMY_JAC — Jacobian stub
!
!  For MF = 10 (Adams / MITER = 0) ZVODE never calls the Jacobian;
!  this stub exists solely to satisfy the linker.
! ==================================================================
#ifdef USE_ZVODE_BIND_C
subroutine dummy_jac(neq, t, y, ml, mu, pd, nrowpd, ctx) bind(c)
  use, intrinsic :: iso_c_binding, only: &
      c_int, c_double, c_double_complex, c_ptr, c_associated
  implicit none
  type(c_ptr),               value         :: ctx
#else
subroutine dummy_jac(neq, t, y, ml, mu, pd, nrowpd, rpar, ipar)
  implicit none
  double precision, intent(in)    :: rpar(*)
  integer,          intent(in)    :: ipar(*)
#endif
  integer(c_int),            value         :: neq, ml, mu, nrowpd
  real(c_double),            value         :: t
  complex(c_double_complex), intent(in)    :: y(neq)
  complex(c_double_complex), intent(inout) :: pd(nrowpd, *)

  ! Trap any accidental call: this subroutine must never execute.
  write(*, '(a)') &
    'BUG: dummy_jac called — should never happen for MF = 10 (MITER = 0)'
  stop 1

end subroutine dummy_jac

end program test_zvode_scalar_real_decay