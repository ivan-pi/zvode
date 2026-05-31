module c_zvode_mod

    use, intrinsic :: iso_c_binding, only: &
        c_int, c_double, c_double_complex, c_ptr, c_null_ptr

    use zvode_mod, only: zvode, zvode_fun, zvode_jac

    implicit none
    private

    public :: c_zvode
    public :: c_zvode_fun
    public :: c_zvode_jac
    public :: c_zvindy

    ! Struct to hold information needed by zvindy
    type, bind(c) :: step_t
        real(c_double) :: h, tn, hu
        integer(c_int) :: nq
    end type

    !
    ! C callback interface
    !
    abstract interface
        subroutine c_zvode_fun(neq,t,y,ydot,ctx) bind(c)
           import c_int, c_double, c_double_complex, c_ptr
           implicit none
           integer(c_int), value :: neq
           real(c_double), value :: t
           complex(c_double_complex), intent(in) :: y(neq)
           complex(c_double_complex), intent(out) :: ydot(neq)
           type(c_ptr), value :: ctx
        end subroutine
        subroutine c_zvode_jac(neq,t,y,ml,mu,pd,nrowpd,ctx) bind(c)
           import c_int, c_double, c_double_complex, c_ptr
           integer(c_int), value :: neq, ml, mu, nrowpd
           real(c_double), value :: t
           complex(c_double_complex), intent(in) :: y(neq)
           complex(c_double_complex), intent(inout) :: pd(nrowpd,*)
           type(c_ptr), value :: ctx
        end subroutine
    end interface

    !
    ! Child classes implementing the ZVODE interface
    !
    type, extends(zvode_fun), private :: c_fun_wrapper
        procedure(c_zvode_fun), pointer, nopass :: fun => null()
        type(c_ptr) :: ctx = c_null_ptr
    contains
        procedure :: eval => c_fun_eval
    end type

    type, extends(zvode_jac), private :: c_jac_wrapper
        procedure(c_zvode_jac), pointer, nopass :: jac => null()
        type(c_ptr) :: ctx = c_null_ptr
    contains
        procedure :: eval => c_jac_eval
    end type

contains

    ! The main C driver for ZVODE
    subroutine c_zvode (f, neq, y, t, tout, itol, rtol, atol, itask, &
          istate, iopt, zwork, lzw, rwork, lrw, iwork, liw, &
          jac, mf, ctx) bind(c,name="c_zvode")

        procedure(c_zvode_fun) :: f
        procedure(c_zvode_jac) :: jac

        integer(c_int), intent(in), value :: neq, itol, itask, iopt, lzw, &
                                             lrw, liw, mf
        real(c_double), intent(in), value :: tout
        real(c_double), intent(inout) :: t
        complex(c_double_complex), intent(inout) :: y(neq), zwork(lzw)
        real(c_double), intent(inout) :: rwork(lrw)
        real(c_double), intent(in) :: rtol(*), atol(*)
        integer(c_int), intent(inout) :: istate, iwork(liw)
        type(c_ptr), value :: ctx

        call zvode(&
            c_fun_wrapper(neq,f,ctx), &
            neq,y,t,tout,itol,rtol,atol,itask,istate,iopt, &
            zwork,lzw,rwork,lrw,iwork,liw,&
            c_jac_wrapper(neq,jac,ctx),mf)

    end subroutine c_zvode

    subroutine c_fun_eval(fun,t,y,ydot)
        class(c_fun_wrapper) :: fun
        real(c_double), intent(in) :: t
        complex(c_double_complex), intent(in) :: y(fun%neq)
        complex(c_double_complex), intent(out) :: ydot(fun%neq)
        call fun%fun(fun%neq,t,y,ydot,fun%ctx)
    end subroutine c_fun_eval

    subroutine c_jac_eval(jac,t,y,ml,mu,pd,nrowpd)
        class(c_jac_wrapper) :: jac
        integer, intent(in) :: ml, mu, nrowpd
        real(c_double), intent(in) :: t
        complex(c_double_complex), intent(in) :: y(jac%neq)
        complex(c_double_complex), intent(inout) :: pd(nrowpd,*)
        call jac%jac(jac%neq,t,y,ml,mu,pd(1,1),nrowpd,jac%ctx)
    end subroutine c_jac_eval

    !-----------------------------------------------------------------------
    ! ZVINDY computes interpolated values of the K-th derivative of the
    ! dependent variable vector y, and stores it in DKY.  This routine
    ! is called within the package with K = 0 and T = TOUT, but may
    ! also be called by the user for any K up to the current order.
    ! (See detailed instructions in the usage documentation.)
    !-----------------------------------------------------------------------
    ! The computed values in DKY are gotten by interpolation using the
    ! Nordsieck history array YH.  This array corresponds uniquely to a
    ! vector-valued polynomial of degree NQCUR or less, and DKY is set
    ! to the K-th derivative of this polynomial at T.
    ! The formula for DKY is:
    !              q
    !  DKY(i)  =  sum  c(j,K) * (T - TN)**(j-K) * H**(-j) * YH(i,j+1)
    !             j=K
    ! where  c(j,K) = j*(j-1)*...*(j-K+1), q = NQCUR, TN = TCUR, H = HCUR.
    ! The quantities  NQ = NQCUR, L = NQ+1, N, TN, and H are
    ! communicated by COMMON.  The above sum is done in reverse order.
    ! IFLAG is returned negative if either K or T is out of bounds.
    !
    ! Discussion above and comments in driver explain all variables.
    !-----------------------------------------------------------------------
    function c_zvindy(n, t, yh, ldyh, k, dky, step) result(iflag) bind(c)
        use zvode_mod, only: dzscal, xerrwd
        implicit none
        integer, parameter :: dp = kind(1.0d0)

        integer(c_int), value :: n, ldyh, k
        real(c_double), value :: t
        complex(c_double_complex), intent(in) :: yh(ldyh,*) ! LDYH >= N
        complex(c_double_complex), intent(out) :: dky(n)
        type(step_t), intent(in) :: step
        integer(c_int) :: iflag

        real(dp) ::  c, s, tfuzz, tn1, tp
        integer :: j
        character(len=80) :: msg

        real(dp), parameter :: hun = 100, one = 1, zero = 0

        associate(h=>step%h, tn=>step%tn, hu=>step%hu, nq=>step%nq)

            iflag = 0

            if (k .lt. 0 .or. k .gt. step%nq) then
                msg = 'zvindy-- k (=i1) illegal      '
                call xerrwd (msg, 30, 51, 1, 1, k, 0, 0, zero, zero)
                iflag = -1
                return
            end if

            tfuzz = hun*epsilon(1.0_dp)*sign(abs(tn) + abs(hu), hu)
            tp = tn - hu - tfuzz
            tn1 = tn + tfuzz
            if ((t-tp)*(t-tn1) .gt. zero) then
                msg = 'zvindy-- t (=r1) illegal      '
                call xerrwd (msg, 30, 52, 1, 0, 0, 0, 1, t, zero)
                msg = '      t not in interval tcur - hu (= r1) to tcur (=r2)      '
                call xerrwd (msg, 60, 52, 1, 0, 0, 0, 2, tp, tn)
                iflag = -2
                return
            end if

            s = (t - tn)/h
            c = falling_factorial(nq, k)
            dky(1:n) = c*yh(1:n,nq+1)
            do j = nq-1,k,-1
                c = falling_factorial(j, k)
                dky(1:n) = c*yh(1:n,j+1) + s*dky(1:n)
            end do

            if (k == 0) return

            call dzscal (n, (one/h)**k, dky, 1)

        end associate

    contains

        pure function falling_factorial(j, k) result(c)
            integer, intent(in) :: j, k
            real(dp) :: c
            integer :: ic, jj
            ic = 1
            do jj = j - k + 1, j
              ic = ic*jj
            end do
            c = ic
        end function

    end function c_zvindy

end module c_zvode_mod
