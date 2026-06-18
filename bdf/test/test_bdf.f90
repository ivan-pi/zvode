! Native smoke tests for the BDF integrator.
!
! Covers: scalar decay against the analytic solution, the stiff Robertson
! problem with a user-supplied dense Jacobian, finite-difference vs analytic
! Jacobian agreement, and a banded finite-difference Jacobian path.

module test_bdf_problems
    use bdf_module, only: dp
    use iso_c_binding, only: c_ptr
    implicit none
contains

    ! y' = -y,  y(0) = 1  ->  y(t) = exp(-t)
    subroutine decay_rhs(n, t, y, f, ctx)
        integer, intent(in) :: n
        real(dp), intent(in) :: t, y(n)
        real(dp), intent(out) :: f(n)
        type(c_ptr), value :: ctx
        f(1) = -y(1)
    end subroutine

    ! Robertson stiff kinetics.
    subroutine rober_rhs(n, t, y, f, ctx)
        integer, intent(in) :: n
        real(dp), intent(in) :: t, y(n)
        real(dp), intent(out) :: f(n)
        type(c_ptr), value :: ctx
        f(1) = -0.04_dp*y(1) + 1.0e4_dp*y(2)*y(3)
        f(2) =  0.04_dp*y(1) - 1.0e4_dp*y(2)*y(3) - 3.0e7_dp*y(2)**2
        f(3) =  3.0e7_dp*y(2)**2
    end subroutine

    subroutine rober_jac(n, t, y, ml, mu, pd, ldpd, ctx)
        integer, intent(in) :: n, ml, mu, ldpd
        real(dp), intent(in) :: t, y(n)
        real(dp), intent(inout) :: pd(ldpd, n)
        type(c_ptr), value :: ctx
        pd(1,1) = -0.04_dp
        pd(1,2) =  1.0e4_dp*y(3)
        pd(1,3) =  1.0e4_dp*y(2)
        pd(2,1) =  0.04_dp
        pd(2,2) = -1.0e4_dp*y(3) - 6.0e7_dp*y(2)
        pd(2,3) = -1.0e4_dp*y(2)
        pd(3,1) =  0.0_dp
        pd(3,2) =  6.0e7_dp*y(2)
        pd(3,3) =  0.0_dp
    end subroutine

    ! Tridiagonal linear system y' = A y (1D heat-equation stencil).
    subroutine band_rhs(n, t, y, f, ctx)
        integer, intent(in) :: n
        real(dp), intent(in) :: t, y(n)
        real(dp), intent(out) :: f(n)
        type(c_ptr), value :: ctx
        integer :: i
        do i = 1, n
            f(i) = -2.0_dp*y(i)
            if (i > 1) f(i) = f(i) + y(i-1)
            if (i < n) f(i) = f(i) + y(i+1)
        end do
    end subroutine

end module test_bdf_problems


program test_bdf
    use bdf_module
    use test_bdf_problems
    use iso_c_binding, only: c_null_ptr
    implicit none

    integer :: nfail
    nfail = 0

    call test_decay(nfail)
    call test_rober_user_jac(nfail)
    call test_rober_fd_vs_user(nfail)
    call test_band(nfail)

    if (nfail == 0) then
        print '(A)', 'ALL TESTS PASSED'
    else
        print '(A,I0,A)', 'FAILED: ', nfail, ' test(s)'
        error stop 1
    end if

contains

    subroutine check(name, ok, nfail)
        character(*), intent(in) :: name
        logical, intent(in) :: ok
        integer, intent(inout) :: nfail
        if (ok) then
            print '(A,A)', 'PASS  ', name
        else
            print '(A,A)', 'FAIL  ', name
            nfail = nfail + 1
        end if
    end subroutine

    subroutine test_decay(nfail)
        integer, intent(inout) :: nfail
        type(bdf_solver) :: s
        real(dp) :: y(1), tout, err
        integer :: code
        tout = 5.0_dp
        call s%init(1, 0.0_dp, [1.0_dp], tout, decay_rhs, &
                    rtol=1.0e-8_dp, atol=[1.0e-10_dp], jac_mode=BDF_JAC_FD)
        call s%integrate(code)
        call s%get_y(y)
        err = abs(y(1) - exp(-tout))
        print '(A,ES12.4,A,ES12.4)', '   decay y=', y(1), ' exact=', exp(-tout)
        call check('decay: scalar exp(-t)', code == BDF_FINISHED .and. err < 1.0e-6_dp, nfail)
        call s%destroy()
    end subroutine

    subroutine test_rober_user_jac(nfail)
        integer, intent(inout) :: nfail
        type(bdf_solver) :: s
        real(dp) :: y(3), summ
        integer :: code, nfev, njev, nlu, nsteps
        call s%init(3, 0.0_dp, [1.0_dp, 0.0_dp, 0.0_dp], 40.0_dp, rober_rhs, &
                    rtol=1.0e-6_dp, atol=[1.0e-8_dp, 1.0e-10_dp, 1.0e-8_dp], &
                    jac_mode=BDF_JAC_USER, jac=rober_jac)
        call s%integrate(code)
        call s%get_y(y)
        call s%get_stats(nfev, njev, nlu, nsteps)
        summ = sum(y)
        print '(A,3ES13.5)', '   rober y=', y
        print '(A,F12.9,A,I0,A,I0,A,I0)', '   sum=', summ, &
            '  nsteps=', nsteps, ' njev=', njev, ' nlu=', nlu
        ! Reference values (scipy BDF) ~ [0.7158, 9.19e-6, 0.2842].
        call check('rober: mass conservation', abs(summ - 1.0_dp) < 1.0e-6_dp, nfail)
        call check('rober: y1 reference', abs(y(1) - 0.7158_dp) < 5.0e-3_dp, nfail)
        call check('rober: y3 reference', abs(y(3) - 0.2842_dp) < 5.0e-3_dp, nfail)
        call s%destroy()
    end subroutine

    subroutine test_rober_fd_vs_user(nfail)
        integer, intent(inout) :: nfail
        type(bdf_solver) :: s
        real(dp) :: y_user(3), y_fd(3), diff
        integer :: code
        call s%init(3, 0.0_dp, [1.0_dp, 0.0_dp, 0.0_dp], 40.0_dp, rober_rhs, &
                    rtol=1.0e-6_dp, atol=[1.0e-8_dp, 1.0e-10_dp, 1.0e-8_dp], &
                    jac_mode=BDF_JAC_USER, jac=rober_jac)
        call s%integrate(code)
        call s%get_y(y_user)
        call s%destroy()

        call s%init(3, 0.0_dp, [1.0_dp, 0.0_dp, 0.0_dp], 40.0_dp, rober_rhs, &
                    rtol=1.0e-6_dp, atol=[1.0e-8_dp, 1.0e-10_dp, 1.0e-8_dp], &
                    jac_mode=BDF_JAC_FD)
        call s%integrate(code)
        call s%get_y(y_fd)
        call s%destroy()

        diff = maxval(abs(y_user - y_fd))
        print '(A,ES12.4)', '   |user - fd| =', diff
        call check('rober: FD agrees with user Jacobian', diff < 1.0e-3_dp, nfail)
    end subroutine

    subroutine test_band(nfail)
        integer, intent(inout) :: nfail
        integer, parameter :: n = 20
        type(bdf_solver) :: s
        real(dp) :: y0(n), yb(n), yd(n), diff
        integer :: code, i
        do i = 1, n
            y0(i) = sin(real(i, dp))
        end do

        ! Banded FD Jacobian (ml = mu = 1).
        call s%init(n, 0.0_dp, y0, 1.0_dp, band_rhs, &
                    rtol=1.0e-8_dp, atol=spread(1.0e-10_dp, 1, n), &
                    jac_mode=BDF_JAC_FD, ml=1, mu=1)
        call s%integrate(code)
        call s%get_y(yb)
        call s%destroy()

        ! Same problem with a dense FD Jacobian as the reference.
        call s%init(n, 0.0_dp, y0, 1.0_dp, band_rhs, &
                    rtol=1.0e-8_dp, atol=spread(1.0e-10_dp, 1, n), &
                    jac_mode=BDF_JAC_FD)
        call s%integrate(code)
        call s%get_y(yd)
        call s%destroy()

        diff = maxval(abs(yb - yd))
        print '(A,ES12.4)', '   |band - dense| =', diff
        call check('band: banded FD matches dense FD', &
                   code == BDF_FINISHED .and. diff < 1.0e-6_dp, nfail)
    end subroutine

end program test_bdf
