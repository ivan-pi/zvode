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

    function c_zvindy(t,k,yh,ldyh,dky) result(iflag) bind(c)
        real(c_double), value :: t
        integer(c_int), value :: k, ldyh
        complex(c_double_complex), intent(in) :: yh(ldyh,*)
        complex(c_double_complex), intent(out) :: dky(*)
        integer(c_int) :: iflag
        call zvindy(t,k,yh,ldyh,dky,iflag)
    end function c_zvindy

end module c_zvode_mod
