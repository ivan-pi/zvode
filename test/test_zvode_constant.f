*DECK TEST_ZVODE_CONST
      PROGRAM TEST_ZVODE_CONST
C-----------------------------------------------------------------------
C  Test the original ZVODE interface (RPAR/IPAR convention).
C
C  Problem:
C    dy/dt = i,   y(0) = i,   t in [0, 1]
C
C  Exact solution:  y(t) = (1 + t) * i
C  At t = 1:        y(1) = 2 * i
C
C  This mirrors the SciPy call sequence:
C
C    from scipy.integrate import complex_ode
C    r = complex_ode(lambda t, x, arg: 1j)
C    r.set_integrator('vode')
C    r.set_initial_value(1j)
C    r.set_f_params(1j)
C    r.integrate(1)
C
C  The f_params argument is not forwarded to the RHS (the lambda
C  ignores its third parameter), so RPAR/IPAR are unused here too.
C
C  Assertions:
C    1. ISTATE == 2 on return
C    2. |Re[y(1) - 2i]| + |Im[y(1) - 2i]| <= 1.0d-8
C
C  Exits with status 0 on pass, 1 on any failed assertion.
C
C  Compile together with zvode_original.f, for example:
C    gfortran -O2 zvode_original.f test_zvode_const.f -o test_zvode_const
C-----------------------------------------------------------------------
      IMPLICIT NONE
      EXTERNAL FEX, JEX
C
C     --- Dimensions -------------------------------------------------------
C     NEQ = 1  :  scalar ODE
C     MF  = 10 :  non-stiff Adams (METH=1, MITER=0), no Jacobian needed
C
C     Required work-array sizes for MF = 10, NEQ = 1:
C       LZW >= 15 * NEQ  = 15
C       LRW >= 20 + NEQ  = 21
C       LIW >= 30        = 30
C     (see ZVODE documentation, Part i, ZWORK/RWORK/IWORK sections)
C     ----------------------------------------------------------------------
      INTEGER  NEQ, LZW, LRW, LIW
      PARAMETER (NEQ = 1, LZW = 15, LRW = 21, LIW = 30)
C
      DOUBLE COMPLEX   Y(NEQ), ZWORK(LZW)
      DOUBLE PRECISION T, TOUT, RTOL, ATOL, RWORK(LRW)
      INTEGER          ITOL, ITASK, ISTATE, IOPT, MF, IWORK(LIW)
C
C     RPAR/IPAR are unused but must be passed; declare as size-1 arrays
C     so the assumed-size (*) declaration inside FEX/JEX is valid.
      DOUBLE COMPLEX   RPAR(1)
      INTEGER          IPAR(1)
C
      DOUBLE COMPLEX   YEXACT, ERR
      DOUBLE PRECISION ABERR, TOL
      PARAMETER (TOL = 1.0D-8)
C
C     --- Initial condition ------------------------------------------------
      Y(1)   = DCMPLX(0.0D0, 1.0D0)   !  y(0) = i
      T      = 0.0D0
      TOUT   = 1.0D0
C
C     --- Solver settings --------------------------------------------------
      ITOL   = 1                        !  scalar RTOL and ATOL
      RTOL   = 1.0D-10
      ATOL   = 1.0D-10
      ITASK  = 1                        !  normal output at TOUT
      ISTATE = 1                        !  first call
      IOPT   = 0                        !  no optional input
      MF     = 10                       !  Adams, functional iteration
C
C     --- Unused context ---------------------------------------------------
      RPAR(1) = DCMPLX(0.0D0, 0.0D0)
      IPAR(1) = 0
C
C     --- Integrate from t = 0 to t = 1 -----------------------------------
      CALL ZVODE(FEX, NEQ, Y, T, TOUT,
     1           ITOL, RTOL, ATOL,
     2           ITASK, ISTATE, IOPT,
     3           ZWORK, LZW, RWORK, LRW, IWORK, LIW,
     4           JEX, MF, RPAR, IPAR)
C
C     --- Check ZVODE return status ----------------------------------------
      IF (ISTATE .NE. 2) THEN
        WRITE(*,'(A,I3)') 'ERROR: ZVODE returned ISTATE =', ISTATE
        STOP 1
      END IF
C
C     --- Verify result against exact solution y(1) = 2i ------------------
      YEXACT = DCMPLX(0.0D0, 2.0D0)
      ERR    = Y(1) - YEXACT
      ABERR  = ABS(DREAL(ERR)) + ABS(DIMAG(ERR))
C
      WRITE(*,'(A)')
      WRITE(*,'(A)') '  ZVODE constant-RHS test  (dy/dt=i, y(0)=i)'
      WRITE(*,'(A)') '  -------------------------------------------'
      WRITE(*,'(A,F14.10,SP,F14.10,A)')
     1   '  Computed  y(1) = ', DREAL(Y(1)), DIMAG(Y(1)), 'i'
      WRITE(*,'(A,F14.10,SP,F14.10,A)')
     1   '  Exact     y(1) = ', DREAL(YEXACT), DIMAG(YEXACT), 'i'
      WRITE(*,'(A,ES10.3)')  '  |error|        = ', ABERR
      WRITE(*,'(A)') '  -------------------------------------------'
      WRITE(*,'(A,I5)')      '  Steps taken    = ', IWORK(11)
      WRITE(*,'(A,I5)')      '  f evaluations  = ', IWORK(12)
      WRITE(*,'(A,F10.6)')   '  t reached      = ', T
      WRITE(*,'(A)')
C
      IF (ABERR .GT. TOL) THEN
        WRITE(*,'(A,ES10.3,A,ES10.3)')
     1    '  FAIL  |error| = ', ABERR, '  > tol = ', TOL
        STOP 1
      END IF
C
      WRITE(*,'(A)') '  PASS'
      WRITE(*,'(A)')
      STOP
      END
C
C=======================================================================
C  Right-hand side:  dy/dt = i  (constant imaginary unit)
C
C  The RPAR/IPAR arguments are accepted but not consulted, matching
C  the SciPy lambda that ignores its third parameter (arg).
C=======================================================================
      SUBROUTINE FEX(NEQ, T, Y, YDOT, RPAR, IPAR)
      IMPLICIT NONE
      INTEGER          NEQ, IPAR(*)
      DOUBLE PRECISION T
      DOUBLE COMPLEX   Y(NEQ), YDOT(NEQ), RPAR(*)
C
      YDOT(1) = DCMPLX(0.0D0, 1.0D0)
C
      RETURN
      END
C
C=======================================================================
C  Dummy Jacobian -- never called when MF = 10 (functional iteration).
C  Must be declared EXTERNAL and passed to ZVODE; ZVODE never invokes
C  it for this method flag.
C=======================================================================
      SUBROUTINE JEX(NEQ, T, Y, ML, MU, PD, NROWPD, RPAR, IPAR)
      IMPLICIT NONE
      INTEGER          NEQ, ML, MU, NROWPD, IPAR(*)
      DOUBLE PRECISION T
      DOUBLE COMPLEX   Y(NEQ), PD(NROWPD,*), RPAR(*)
C
      RETURN
      END
      