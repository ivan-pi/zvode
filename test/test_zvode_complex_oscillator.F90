! test_zvode_complex_oscillator.F90
! ============================================================
!  Fortran driver for a complex oscillator test using ZVODE.
!
!  Problem : dy/dt = -i*y,  y(0) = 1+0i  =>  y(t) = exp(-i*t)
!  Method  : BDF (MF = 22), internally generated dense Jacobian.
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
!  Assertions:
!    1. ISTATE == 2 on return
!    2. T == TOUT on return
!    3. |y(TOUT) - exp(-i*TOUT)| <= rtol*|exp(-i*TOUT)| + atol
!
!  Exits with status 0 on pass, 1 on any failed assertion.
!
!  Compile (see also Makefile):
!    Legacy:   gfortran ... -c test_zvode_complex_oscillator.F90
!    bind(c):  gfortran ... -DUSE_ZVODE_BIND_C -c test_zvode_complex_oscillator.F90
!              (extern/zvode.f must be compiled first to provide zvode_mod.mod)
! ============================================================

program test_zvode_complex_oscillator
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
  integer, parameter :: mf  = 22  ! BDF method, MITER=2 (internally generated Jacobian)

  ! ---- Minimum work-array lengths for MF=22, NEQ=1 -----------------
  !   LZW >= 15*NEQ       =  15   (complex work array)
  !   LRW >= 20 + NEQ     =  21   (real work array)
  !   LIW >= 30           =  30   (integer work array)
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
  double precision  :: rtol(1)     ! relative tolerance (scalar: ITOL=1)
  double precision  :: atol(1)     ! absolute tolerance (scalar: ITOL=1)
  integer           :: itol        ! 1 = both tolerances are scalars
  integer           :: itask       ! 1 = advance to TOUT then return
  integer           :: istate      ! 1 = first call; output 2 = success
  integer           :: iopt        ! 0 = all defaults

  ! ---- Interface-specific context / pass-through --------------------
#ifdef USE_ZVODE_BIND_C
  type(c_ptr) :: ctx
#else
  double precision :: rpar(1)
  integer          :: ipar(1)
#endif

  ! ---- Local variables for result verification ----------------------
  double complex   :: exact
  double precision :: abserr, rel_err, tol

  ! ==================================================================
  ! Initialise
  ! ==================================================================
  y(1)   = dcmplx(1.0d0, 0.0d0)   ! y(0) = 1 + 0 i

  t      = 0.0d0
  tout   = 10.0d0
  itol   = 1       ! EWT_i = rtol*|y_i| + atol
  
  ! Python solve_ivp default tolerances mapped from the intercepted args
  rtol   = 1.0d-3
  atol   = 1.0d-6
  
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
  call zvode(fex, neq, y, t, tout,                &
             itol, rtol, atol,                &
             itask, istate, iopt,                 &
             zwork, lzw, rwork, lrw, iwork, liw,  &
             dummy_jac, mf, ctx)
#else
  call zvode(fex, neq, y, t, tout,                &
             itol, rtol, atol,                    &
             itask, istate, iopt,                 &
             zwork, lzw, rwork, lrw, iwork, liw,  &
             dummy_jac, mf, rpar, ipar)
#endif

  ! ==================================================================
  ! Assertions
  ! ==================================================================

  ! 1.  assert istate_new == 2
  if (istate /= 2) then
    write(*, '(a,i0)') 'FAIL: ZVODE returned istate = ', istate
    stop 1
  end if

  ! 2.  assert t_new == tout
  if (t < tout .or. t > tout) then
    write(*, '(a,es22.14)') 'FAIL: ZVODE did not reach TOUT, T = ', t
    stop 1
  end if

  ! 3.  Evaluate solution accuracy
  !     Exact solution is y(t) = exp(-i*t) = cos(t) - i*sin(t)
  exact   = dcmplx(cos(tout), -sin(tout))
  abserr  = abs(y(1) - exact)
  rel_err = abserr / abs(exact)
  tol     = rtol * abs(exact) + atol
  
  if (abserr > tol) then
    write(*, '(a,es16.8,es16.8)') 'FAIL: y(1) = ', dble(y(1)), aimag(y(1))
    write(*, '(a,es16.8,es16.8)') '      exact = ', dble(exact), aimag(exact)
    write(*, '(a,es10.2,a,es10.2)') '      abserr   = ', abserr, '  tol   = ', tol
    stop 1
  end if

  ! ==================================================================
  ! All assertions passed — print summary and diagnostic counters
  ! ==================================================================
  write(*, '(a)') ''
  write(*, '(a)') '==========================================='
  write(*, '(a)') 'PASS: test_zvode_complex_oscillator'
  write(*, '(a)') '==========================================='
  write(*, '(a,es22.15)')  '  T              = ', t
  write(*, '(a,es16.8,a,es16.8,a)')  '  y(1)           = (', dble(y(1)), ' , ', aimag(y(1)), 'i)'
  write(*, '(a,es16.8,a,es16.8,a)')  '  exact          = (', dble(exact), ' , ', aimag(exact), 'i)'
  write(*, '(a,es10.2)')   '  Abs. error     = ', abserr
  write(*, '(a,es10.2)')   '  Rel. error     = ', rel_err
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
!  Implements: YDOT(i) = -1j * Y(i)
! ==================================================================
#ifdef USE_ZVODE_BIND_C
subroutine fex(neq, t, y, ydot, ctx) bind(c)
  use, intrinsic :: iso_c_binding, only: &
      c_int, c_double, c_double_complex, c_ptr, c_associated
  implicit none
  integer(c_int),            value         :: neq
  real(c_double),            value         :: t
  complex(c_double_complex), intent(in)    :: y(neq)
  complex(c_double_complex), intent(out)   :: ydot(neq)
  type(c_ptr),               value         :: ctx
#else
subroutine fex(neq, t, y, ydot, rpar, ipar)
  implicit none
  integer, intent(in)           :: neq
  double precision, intent(in)  :: t
  double complex, intent(in)    :: y(neq)
  double complex, intent(out)   :: ydot(neq)
  double precision, intent(in)  :: rpar(*)
  integer,          intent(in)  :: ipar(*)
#endif
  integer :: i

  do i = 1, neq
    ydot(i) = dcmplx(0.0d0, -1.0d0) * y(i)
  end do

end subroutine fex

! ==================================================================
!  DUMMY_JAC — Jacobian stub
!
!  For MF = 22 (BDF / MITER = 2) ZVODE generates the Jacobian
!  internally and never calls the user-supplied JAC;
!  this stub exists solely to satisfy the linker.
! ==================================================================
#ifdef USE_ZVODE_BIND_C
subroutine dummy_jac(neq, t, y, ml, mu, pd, nrowpd, ctx) bind(c)
  use, intrinsic :: iso_c_binding, only: &
      c_int, c_double, c_double_complex, c_ptr, c_associated
  implicit none
  integer(c_int),            value         :: neq, ml, mu, nrowpd
  real(c_double),            value         :: t
  complex(c_double_complex), intent(in)    :: y(neq)
  complex(c_double_complex), intent(inout) :: pd(nrowpd, *)
  type(c_ptr),               value         :: ctx
#else
subroutine dummy_jac(neq, t, y, ml, mu, pd, nrowpd, rpar, ipar)
  implicit none
  integer,          intent(in)    :: neq, ml, mu, nrowpd
  double precision, intent(in)    :: t
  double complex,   intent(in)    :: y(neq)
  double complex,   intent(inout) :: pd(nrowpd, *)
  double precision, intent(in)    :: rpar(*)
  integer,          intent(in)    :: ipar(*)
#endif

  ! Trap any accidental call: this subroutine must never execute.
  write(*, '(a)') &
    'BUG: dummy_jac called — should never happen for MF = 22 (MITER = 2)'
  stop 1

end subroutine dummy_jac

end program test_zvode_complex_oscillator
