! c_bdf.f90 -- C ABI wrapper around bdf_module.
!
! Exposes an opaque-handle C interface (see bdf.h).  C callers pass C function
! pointers for the right-hand side and (optionally) Jacobian; these are stored
! in a per-handle box together with the user context pointer and adapted to the
! native Fortran callback interfaces.  The integrator core stays free of any
! FFI concerns.

module c_bdf_module

    use, intrinsic :: iso_c_binding
    use bdf_module, only: dp, bdf_solver, bdf_rhs, bdf_jac, &
                          BDF_JAC_FD, BDF_JAC_USER, BDF_JAC_CONSTANT

    implicit none
    private

    public :: bdf_create, bdf_c_init, bdf_c_step, bdf_c_integrate, &
              bdf_c_set_t_bound, bdf_c_get_t, bdf_c_get_y, &
              bdf_c_get_stats, bdf_c_destroy

    !> C callback signatures (interoperable mirrors of the native interfaces).
    abstract interface
        subroutine c_rhs_t(n, t, y, f, ctx) bind(c)
            import :: c_int, c_double, c_ptr
            integer(c_int), value      :: n
            real(c_double), value      :: t
            real(c_double), intent(in)  :: y(n)
            real(c_double), intent(out) :: f(n)
            type(c_ptr),    value       :: ctx
        end subroutine
        subroutine c_jac_t(n, t, y, ml, mu, pd, ldpd, ctx) bind(c)
            import :: c_int, c_double, c_ptr
            integer(c_int), value         :: n, ml, mu, ldpd
            real(c_double), value         :: t
            real(c_double), intent(in)    :: y(n)
            real(c_double), intent(inout) :: pd(ldpd, n)
            type(c_ptr),    value         :: ctx
        end subroutine
    end interface

    !> Per-handle callback box; its address is handed to the core as `ctx`.
    type :: callbacks_t
        type(c_funptr) :: cfun = c_null_funptr
        type(c_funptr) :: cjac = c_null_funptr
        type(c_ptr)    :: user_ctx = c_null_ptr
    end type

    !> Opaque handle: the solver plus the (target) callback box it points at.
    type :: handle_t
        type(bdf_solver)         :: solver
        type(callbacks_t), pointer :: box => null()
    end type

contains

    !--- lifecycle ----------------------------------------------------------

    function bdf_create() result(h) bind(c, name="bdf_create")
        type(c_ptr) :: h
        type(handle_t), pointer :: p
        allocate(p)
        allocate(p%box)
        h = c_loc_handle(p)
    end function

    subroutine bdf_c_destroy(h) bind(c, name="bdf_destroy")
        type(c_ptr), value :: h
        type(handle_t), pointer :: p
        if (.not. c_associated(h)) return
        call c_f_pointer(h, p)
        call p%solver%destroy()
        if (associated(p%box)) deallocate(p%box)
        deallocate(p)
    end subroutine

    !--- configuration ------------------------------------------------------

    subroutine bdf_c_init(h, n, t0, y0, t_bound, fun, rtol, atol, jac_mode, &
                          jac, ml, mu, max_step, first_step, reuse_jac, ctx) &
                          bind(c, name="bdf_init")
        type(c_ptr),    value      :: h
        integer(c_int), value      :: n, jac_mode, ml, mu, reuse_jac
        real(c_double), value      :: t0, t_bound, rtol, max_step, first_step
        real(c_double), intent(in) :: y0(n), atol(n)
        type(c_funptr), value      :: fun, jac
        type(c_ptr),    value      :: ctx

        type(handle_t), pointer :: p
        logical :: reuse
        call c_f_pointer(h, p)

        p%box%cfun     = fun
        p%box%cjac     = jac
        p%box%user_ctx = ctx
        reuse = (reuse_jac /= 0)

        call init_dispatch(p, n, t0, y0, t_bound, rtol, atol, jac_mode, &
                           ml, mu, max_step, first_step, reuse)
    end subroutine

    !> Helper that turns the C sentinels (max_step/first_step <= 0, ml/mu < 0)
    !> into the core's optional arguments.
    subroutine init_dispatch(p, n, t0, y0, t_bound, rtol, atol, jac_mode, &
                             ml, mu, max_step, first_step, reuse)
        type(handle_t), intent(inout) :: p
        integer,  intent(in) :: n, jac_mode, ml, mu
        real(dp), intent(in) :: t0, y0(n), t_bound, rtol, atol(n), &
                                max_step, first_step
        logical,  intent(in) :: reuse

        real(dp) :: ms, fs
        logical  :: has_ms, has_fs, banded

        has_ms = (max_step > 0.0_dp);   ms = max_step
        has_fs = (first_step > 0.0_dp); fs = first_step
        banded = (ml >= 0 .and. mu >= 0)

        if (banded) then
            if (has_ms .and. has_fs) then
                call p%solver%init(n, t0, y0, t_bound, rhs_adapter, rtol, atol, &
                    jac_mode, jac=jac_adapter, ml=ml, mu=mu, max_step=ms, &
                    first_step=fs, reuse_jac=reuse, ctx=c_loc(p%box))
            else if (has_ms) then
                call p%solver%init(n, t0, y0, t_bound, rhs_adapter, rtol, atol, &
                    jac_mode, jac=jac_adapter, ml=ml, mu=mu, max_step=ms, &
                    reuse_jac=reuse, ctx=c_loc(p%box))
            else if (has_fs) then
                call p%solver%init(n, t0, y0, t_bound, rhs_adapter, rtol, atol, &
                    jac_mode, jac=jac_adapter, ml=ml, mu=mu, first_step=fs, &
                    reuse_jac=reuse, ctx=c_loc(p%box))
            else
                call p%solver%init(n, t0, y0, t_bound, rhs_adapter, rtol, atol, &
                    jac_mode, jac=jac_adapter, ml=ml, mu=mu, &
                    reuse_jac=reuse, ctx=c_loc(p%box))
            end if
        else
            if (has_ms .and. has_fs) then
                call p%solver%init(n, t0, y0, t_bound, rhs_adapter, rtol, atol, &
                    jac_mode, jac=jac_adapter, max_step=ms, first_step=fs, &
                    reuse_jac=reuse, ctx=c_loc(p%box))
            else if (has_ms) then
                call p%solver%init(n, t0, y0, t_bound, rhs_adapter, rtol, atol, &
                    jac_mode, jac=jac_adapter, max_step=ms, &
                    reuse_jac=reuse, ctx=c_loc(p%box))
            else if (has_fs) then
                call p%solver%init(n, t0, y0, t_bound, rhs_adapter, rtol, atol, &
                    jac_mode, jac=jac_adapter, first_step=fs, &
                    reuse_jac=reuse, ctx=c_loc(p%box))
            else
                call p%solver%init(n, t0, y0, t_bound, rhs_adapter, rtol, atol, &
                    jac_mode, jac=jac_adapter, &
                    reuse_jac=reuse, ctx=c_loc(p%box))
            end if
        end if
    end subroutine

    subroutine bdf_c_set_t_bound(h, t_bound) bind(c, name="bdf_set_t_bound")
        type(c_ptr),    value :: h
        real(c_double), value :: t_bound
        type(handle_t), pointer :: p
        call c_f_pointer(h, p)
        call p%solver%set_t_bound(t_bound)
    end subroutine

    !--- advance ------------------------------------------------------------

    subroutine bdf_c_step(h, code) bind(c, name="bdf_step")
        type(c_ptr),    value         :: h
        integer(c_int), intent(out)   :: code
        type(handle_t), pointer :: p
        call c_f_pointer(h, p)
        call p%solver%step(code)
    end subroutine

    subroutine bdf_c_integrate(h, code) bind(c, name="bdf_integrate")
        type(c_ptr),    value       :: h
        integer(c_int), intent(out) :: code
        type(handle_t), pointer :: p
        call c_f_pointer(h, p)
        call p%solver%integrate(code)
    end subroutine

    !--- accessors ----------------------------------------------------------

    function bdf_c_get_t(h) result(t) bind(c, name="bdf_get_t")
        type(c_ptr), value :: h
        real(c_double) :: t
        type(handle_t), pointer :: p
        call c_f_pointer(h, p)
        t = p%solver%get_t()
    end function

    subroutine bdf_c_get_y(h, n, y) bind(c, name="bdf_get_y")
        type(c_ptr),    value       :: h
        integer(c_int), value       :: n
        real(c_double), intent(out) :: y(n)
        type(handle_t), pointer :: p
        call c_f_pointer(h, p)
        call p%solver%get_y(y)
    end subroutine

    subroutine bdf_c_get_stats(h, nfev, njev, nlu, nsteps) &
                               bind(c, name="bdf_get_stats")
        type(c_ptr),    value       :: h
        integer(c_int), intent(out) :: nfev, njev, nlu, nsteps
        type(handle_t), pointer :: p
        call c_f_pointer(h, p)
        call p%solver%get_stats(nfev, njev, nlu, nsteps)
    end subroutine

    !--- callback adapters (native interface -> stored C function pointer) ---

    subroutine rhs_adapter(n, t, y, f, ctx)
        integer,     intent(in)  :: n
        real(dp),    intent(in)  :: t
        real(dp),    intent(in)  :: y(n)
        real(dp),    intent(out) :: f(n)
        type(c_ptr), value       :: ctx
        type(callbacks_t), pointer :: box
        procedure(c_rhs_t), pointer :: cf
        call c_f_pointer(ctx, box)
        call c_f_procpointer(box%cfun, cf)
        call cf(n, t, y, f, box%user_ctx)
    end subroutine

    subroutine jac_adapter(n, t, y, ml, mu, pd, ldpd, ctx)
        integer,     intent(in)    :: n, ml, mu, ldpd
        real(dp),    intent(in)    :: t
        real(dp),    intent(in)    :: y(n)
        real(dp),    intent(inout) :: pd(ldpd, n)
        type(c_ptr), value         :: ctx
        type(callbacks_t), pointer :: box
        procedure(c_jac_t), pointer :: cj
        call c_f_pointer(ctx, box)
        call c_f_procpointer(box%cjac, cj)
        call cj(n, t, y, ml, mu, pd, ldpd, box%user_ctx)
    end subroutine

    !> c_loc on the (noninteroperable but target-able) handle.
    function c_loc_handle(p) result(h)
        type(handle_t), pointer, intent(in) :: p
        type(c_ptr) :: h
        h = c_loc(p)
    end function

end module c_bdf_module
