! bdf.f90 -- Variable-order (1..5) BDF integrator for stiff ODE systems.
!
! This is a faithful Fortran port of the quasi-constant-step BDF method
! implemented in scipy.integrate.BDF (the NDF-enhanced backward
! differentiation formulas of Shampine & Reichelt).  The scope is
! deliberately narrow:
!
!   * real, double-precision state only;
!   * dense or banded Jacobians;
!   * user-supplied or internally generated finite-difference Jacobians;
!   * LAPACK (xGETRF/xGETRS, xGBTRF/xGBTRS) for the linear algebra;
!   * Jacobian reuse ("saving") between steps, as in SciPy;
!   * no dense output / interpolation.
!
! The integrator is a stateful object (`bdf_solver`) that owns and caches
! all of its workspace.  The workspace is allocated once, dynamically, in
! `init` from the problem size and never reallocated during a run.
!
! References
!   [1] L. F. Shampine, M. W. Reichelt, "The MATLAB ODE Suite", SIAM J.
!       Sci. Comput., 18(1), 1997.
!   [2] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations I".

module bdf_module

    use, intrinsic :: iso_fortran_env, only: dp => real64
    use, intrinsic :: iso_c_binding, only: c_ptr, c_null_ptr
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite

    implicit none
    private

    public :: dp
    public :: bdf_solver
    public :: bdf_rhs, bdf_jac

    ! Method constants (identical to SciPy).
    integer,  parameter :: MAX_ORDER      = 5
    integer,  parameter :: NEWTON_MAXITER = 4
    real(dp), parameter :: MIN_FACTOR     = 0.2_dp
    real(dp), parameter :: MAX_FACTOR     = 10.0_dp

    ! Jacobian sourcing.
    integer, parameter, public :: BDF_JAC_FD       = 0  ! finite differences
    integer, parameter, public :: BDF_JAC_USER     = 1  ! user callback, t/y dependent
    integer, parameter, public :: BDF_JAC_CONSTANT = 2  ! user supplied, constant

    ! Status / return codes from `step` and `integrate`.
    integer, parameter, public :: BDF_OK             =  0  ! step taken, still running
    integer, parameter, public :: BDF_FINISHED       =  1  ! reached t_bound
    integer, parameter, public :: BDF_TOO_SMALL_STEP = -1  ! required step underflowed
    integer, parameter, public :: BDF_TOO_MANY_STEPS = -2  ! step budget exhausted

    !> Right-hand side callback: f = dy/dt at (t, y).
    abstract interface
        subroutine bdf_rhs(n, t, y, f, ctx)
            import :: dp, c_ptr
            integer,      intent(in)  :: n
            real(dp),     intent(in)  :: t
            real(dp),     intent(in)  :: y(n)
            real(dp),     intent(out) :: f(n)
            type(c_ptr),  value       :: ctx
        end subroutine
    end interface

    !> Jacobian callback.  For a dense problem (`ml < 0`) `pd` is (n, n) with
    !> pd(i, j) = df_i/dy_j.  For a banded problem `pd` is (ml+mu+1, n) in the
    !> LSODE/LAPACK band layout, pd(i-j+mu+1, j) = df_i/dy_j.
    abstract interface
        subroutine bdf_jac(n, t, y, ml, mu, pd, ldpd, ctx)
            import :: dp, c_ptr
            integer,      intent(in)    :: n, ml, mu, ldpd
            real(dp),     intent(in)    :: t
            real(dp),     intent(in)    :: y(n)
            real(dp),     intent(inout) :: pd(ldpd, n)
            type(c_ptr),  value         :: ctx
        end subroutine
    end interface

    !> Stateful BDF integrator.  Construct with `init`, advance with `step`
    !> or `integrate`, release with `destroy`.
    type :: bdf_solver
        private

        ! --- problem definition -------------------------------------------
        integer  :: n = 0                 ! number of equations
        real(dp) :: t = 0.0_dp            ! current time
        real(dp) :: t_bound = 0.0_dp      ! integration boundary
        real(dp) :: direction = 1.0_dp    ! +1 / -1
        real(dp) :: max_step = huge(1.0_dp)
        real(dp) :: rtol = 1.0e-3_dp
        real(dp), allocatable :: atol(:)  ! length n

        procedure(bdf_rhs), pointer, nopass :: fun => null()
        procedure(bdf_jac), pointer, nopass :: jac => null()
        type(c_ptr) :: ctx = c_null_ptr

        ! --- Jacobian configuration ---------------------------------------
        integer :: jac_mode = BDF_JAC_FD  ! BDF_JAC_*
        logical :: banded   = .false.
        integer :: ml = -1, mu = -1       ! band half-widths (banded only)
        integer :: ldj = 0                ! leading dim of stored Jacobian
        integer :: ldlu = 0               ! leading dim of factored matrix
        logical :: reuse_jac = .true.     ! cache/reuse Jacobian between steps

        ! --- integrator state ---------------------------------------------
        integer  :: order = 1
        integer  :: n_equal_steps = 0
        real(dp) :: h_abs = 0.0_dp
        real(dp) :: h_abs_old = -1.0_dp   ! < 0 == "None"
        real(dp) :: error_norm_old = -1.0_dp
        real(dp) :: newton_tol = 0.0_dp
        logical  :: lu_valid = .false.    ! factored matrix matches current J,c
        real(dp) :: c_lu = 0.0_dp         ! c used when matrix was factored

        ! --- diagnostics --------------------------------------------------
        integer :: nfev = 0, njev = 0, nlu = 0, nsteps = 0
        integer :: max_steps = 100000

        ! --- method coefficients (index 0..MAX_ORDER) ---------------------
        real(dp) :: gamma(0:MAX_ORDER)
        real(dp) :: alpha(0:MAX_ORDER)
        real(dp) :: error_const(0:MAX_ORDER)

        ! --- cached workspace (allocated once in init) --------------------
        real(dp), allocatable :: D(:,:)        ! (n, 0:MAX_ORDER+2) differences
        real(dp), allocatable :: Dtmp(:,:)     ! (n, 0:MAX_ORDER)   change_D temp
        real(dp), allocatable :: Jac_store(:,:) ! Jacobian (dense or band)
        real(dp), allocatable :: lu(:,:)       ! factored Newton matrix
        integer,  allocatable :: ipiv(:)       ! LAPACK pivots

        real(dp), allocatable :: y(:)          ! current state
        real(dp), allocatable :: f(:)          ! scratch rhs
        real(dp), allocatable :: y_predict(:)
        real(dp), allocatable :: psi(:)
        real(dp), allocatable :: scale(:)
        real(dp), allocatable :: y_new(:)
        real(dp), allocatable :: d_vec(:)      ! accumulated Newton correction
        real(dp), allocatable :: dy(:)
        real(dp), allocatable :: f_pert(:)     ! FD Jacobian scratch
        real(dp), allocatable :: yp(:)         ! FD Jacobian scratch
        real(dp), allocatable :: hstep(:)      ! FD Jacobian scratch

    contains
        procedure :: init        => bdf_init
        procedure :: step        => bdf_step
        procedure :: integrate   => bdf_integrate
        procedure :: set_t_bound => bdf_set_t_bound
        procedure :: destroy     => bdf_destroy
        ! Accessors.
        procedure :: get_t     => bdf_get_t
        procedure :: get_y     => bdf_get_y
        procedure :: get_stats => bdf_get_stats
    end type bdf_solver

contains

    !========================================================================
    ! Construction
    !========================================================================

    !> Initialise the solver.  Allocates all workspace from `n`.
    !>
    !> jac_mode selects BDF_JAC_FD / _USER / _CONSTANT.  For _USER/_CONSTANT a
    !> jac callback must be supplied; _CONSTANT means the callback returns a
    !> t/y-independent matrix and is invoked at most once.  ml/mu >= 0 select a
    !> banded Jacobian; ml < 0 selects dense.
    subroutine bdf_init(self, n, t0, y0, t_bound, fun, rtol, atol, &
                        jac_mode, jac, ml, mu, max_step, first_step, &
                        reuse_jac, ctx)
        class(bdf_solver), intent(inout) :: self
        integer,           intent(in)    :: n
        real(dp),          intent(in)    :: t0, y0(n), t_bound
        procedure(bdf_rhs)               :: fun
        real(dp),          intent(in)    :: rtol, atol(n)
        integer,           intent(in)    :: jac_mode
        procedure(bdf_jac), optional     :: jac
        integer,  optional, intent(in)   :: ml, mu
        real(dp), optional, intent(in)   :: max_step, first_step
        logical,  optional, intent(in)   :: reuse_jac
        type(c_ptr), optional, intent(in) :: ctx

        real(dp) :: kappa(0:MAX_ORDER)
        integer  :: k

        call self%destroy()

        self%n         = n
        self%t         = t0
        self%t_bound   = t_bound
        self%fun       => fun
        self%jac_mode  = jac_mode
        self%nfev = 0; self%njev = 0; self%nlu = 0; self%nsteps = 0

        if (present(jac))       self%jac => jac
        if (present(ctx))       self%ctx = ctx
        if (present(reuse_jac)) self%reuse_jac = reuse_jac
        self%rtol = rtol
        self%max_step = huge(1.0_dp)
        if (present(max_step)) self%max_step = max_step

        self%direction = sign(1.0_dp, t_bound - t0)
        if (t_bound == t0) self%direction = 1.0_dp

        ! Band configuration.
        self%banded = .false.; self%ml = -1; self%mu = -1
        if (present(ml) .and. present(mu)) then
            if (ml >= 0 .and. mu >= 0) then
                self%banded = .true.
                self%ml = ml
                self%mu = mu
            end if
        end if

        allocate(self%atol(n));  self%atol = atol
        allocate(self%y(n));     self%y = y0
        allocate(self%f(n))
        allocate(self%y_predict(n), self%psi(n), self%scale(n))
        allocate(self%y_new(n), self%d_vec(n), self%dy(n))
        allocate(self%f_pert(n), self%yp(n), self%hstep(n))
        allocate(self%D(n, 0:MAX_ORDER+2))
        allocate(self%Dtmp(n, 0:MAX_ORDER))

        ! Jacobian and factored-matrix storage.
        if (self%banded) then
            self%ldj  = self%ml + self%mu + 1
            self%ldlu = 2*self%ml + self%mu + 1
        else
            self%ldj  = n
            self%ldlu = n
        end if
        allocate(self%Jac_store(self%ldj, n))
        allocate(self%lu(self%ldlu, n))
        allocate(self%ipiv(n))

        ! Method coefficients.  kappa, gamma, alpha, error_const as in SciPy.
        kappa = [0.0_dp, -0.1850_dp, -1.0_dp/9.0_dp, -0.0823_dp, -0.0415_dp, 0.0_dp]
        self%gamma(0) = 0.0_dp
        do k = 1, MAX_ORDER
            self%gamma(k) = self%gamma(k-1) + 1.0_dp/real(k, dp)
        end do
        do k = 0, MAX_ORDER
            self%alpha(k)       = (1.0_dp - kappa(k)) * self%gamma(k)
            self%error_const(k) = kappa(k) * self%gamma(k) + 1.0_dp/real(k+1, dp)
        end do

        ! Newton tolerance (SciPy: max(10*EPS/rtol, min(0.03, sqrt(rtol)))).
        self%newton_tol = max(10.0_dp*epsilon(1.0_dp)/rtol, &
                              min(0.03_dp, sqrt(rtol)))

        ! Initial right-hand side and first step.
        call eval_fun(self, self%t, self%y, self%f)
        if (present(first_step)) then
            self%h_abs = abs(first_step)
        else
            self%h_abs = select_initial_step(self, self%t, self%y, self%f)
        end if

        ! Seed the differences array: D0 = y, D1 = h*y'.
        self%D = 0.0_dp
        self%D(:, 0) = self%y
        self%D(:, 1) = self%f * self%h_abs * self%direction

        self%order = 1
        self%n_equal_steps = 0
        self%lu_valid = .false.
        self%h_abs_old = -1.0_dp
        self%error_norm_old = -1.0_dp
    end subroutine bdf_init

    subroutine bdf_destroy(self)
        class(bdf_solver), intent(inout) :: self
        if (allocated(self%atol))      deallocate(self%atol)
        if (allocated(self%y))         deallocate(self%y)
        if (allocated(self%f))         deallocate(self%f)
        if (allocated(self%y_predict)) deallocate(self%y_predict)
        if (allocated(self%psi))       deallocate(self%psi)
        if (allocated(self%scale))     deallocate(self%scale)
        if (allocated(self%y_new))     deallocate(self%y_new)
        if (allocated(self%d_vec))     deallocate(self%d_vec)
        if (allocated(self%dy))        deallocate(self%dy)
        if (allocated(self%f_pert))    deallocate(self%f_pert)
        if (allocated(self%yp))        deallocate(self%yp)
        if (allocated(self%hstep))     deallocate(self%hstep)
        if (allocated(self%D))         deallocate(self%D)
        if (allocated(self%Dtmp))      deallocate(self%Dtmp)
        if (allocated(self%Jac_store)) deallocate(self%Jac_store)
        if (allocated(self%lu))        deallocate(self%lu)
        if (allocated(self%ipiv))      deallocate(self%ipiv)
        self%fun => null()
        self%jac => null()
        self%n = 0
    end subroutine bdf_destroy

    !========================================================================
    ! Accessors
    !========================================================================

    pure function bdf_get_t(self) result(t)
        class(bdf_solver), intent(in) :: self
        real(dp) :: t
        t = self%t
    end function

    subroutine bdf_get_y(self, y)
        class(bdf_solver), intent(in)  :: self
        real(dp),          intent(out) :: y(self%n)
        y = self%y
    end subroutine

    subroutine bdf_get_stats(self, nfev, njev, nlu, nsteps)
        class(bdf_solver), intent(in)  :: self
        integer,           intent(out) :: nfev, njev, nlu, nsteps
        nfev = self%nfev; njev = self%njev; nlu = self%nlu; nsteps = self%nsteps
    end subroutine

    !> Move the integration boundary (e.g. to advance to the next output
    !> time).  The integration direction is left unchanged, so the new bound
    !> must lie ahead of the current time in the original direction.
    subroutine bdf_set_t_bound(self, t_bound)
        class(bdf_solver), intent(inout) :: self
        real(dp),          intent(in)    :: t_bound
        self%t_bound = t_bound
    end subroutine

    !========================================================================
    ! Driver
    !========================================================================

    !> Integrate to `self%t_bound`, taking internal steps.  Returns
    !> BDF_FINISHED on success or a negative failure code.
    subroutine bdf_integrate(self, code)
        class(bdf_solver), intent(inout) :: self
        integer,           intent(out)   :: code
        code = BDF_OK
        do
            if (self%direction*(self%t - self%t_bound) >= 0.0_dp) then
                code = BDF_FINISHED
                return
            end if
            call self%step(code)
            if (code < 0) return
        end do
    end subroutine bdf_integrate

    !========================================================================
    ! A single internal step.  Direct port of BDF._step_impl.
    !========================================================================

    subroutine bdf_step(self, code)
        class(bdf_solver), intent(inout) :: self
        integer,           intent(out)   :: code

        real(dp) :: t, h, t_new, h_abs, min_step, c, factor, safety
        real(dp) :: dy_norm, error_norm, error_m_norm, error_p_norm
        real(dp) :: factors(3), best
        integer  :: order, k, n_iter, i, delta_order, ibest
        logical  :: step_accepted, converged, current_jac

        associate (n => self%n, D => self%D, alpha => self%alpha, &
                   gamma => self%gamma, error_const => self%error_const)

        ! Already at (or past) the boundary: nothing to do.
        if (self%direction*(self%t - self%t_bound) >= 0.0_dp) then
            code = BDF_FINISHED
            return
        end if

        if (self%nsteps >= self%max_steps) then
            code = BDF_TOO_MANY_STEPS
            return
        end if

        t = self%t
        min_step = 10.0_dp * spacing(t)

        ! Clamp h to [min_step, max_step], adjusting the differences array.
        if (self%h_abs > self%max_step) then
            h_abs = self%max_step
            call change_D(self, self%order, self%max_step/self%h_abs)
            self%n_equal_steps = 0
        else if (self%h_abs < min_step) then
            h_abs = min_step
            call change_D(self, self%order, min_step/self%h_abs)
            self%n_equal_steps = 0
        else
            h_abs = self%h_abs
        end if

        order = self%order
        current_jac = (self%jac_mode == BDF_JAC_CONSTANT)

        step_accepted = .false.
        do while (.not. step_accepted)

            if (h_abs < min_step) then
                code = BDF_TOO_SMALL_STEP
                return
            end if

            h = h_abs * self%direction
            t_new = t + h

            ! Do not step past the boundary.
            if (self%direction*(t_new - self%t_bound) > 0.0_dp) then
                t_new = self%t_bound
                call change_D(self, order, abs(t_new - t)/h_abs)
                self%n_equal_steps = 0
                self%lu_valid = .false.
            end if

            h = t_new - t
            h_abs = abs(h)

            ! Predictor and BDF auxiliary quantities.
            do i = 1, n
                self%y_predict(i) = sum(D(i, 0:order))
            end do
            self%scale = self%atol + self%rtol * abs(self%y_predict)
            do i = 1, n
                self%psi(i) = dot_product(D(i, 1:order), gamma(1:order)) / alpha(order)
            end do

            c = h / alpha(order)
            converged = .false.
            do while (.not. converged)
                if (.not. (self%lu_valid .and. self%c_lu == c)) then
                    call factor_newton_matrix(self, c)
                end if

                call solve_bdf_system(self, t_new, c, n_iter, dy_norm, &
                                      converged)

                if (.not. converged) then
                    if (current_jac) exit
                    ! Refresh the Jacobian at the predicted point and retry.
                    call eval_jacobian(self, t_new, self%y_predict)
                    self%lu_valid = .false.
                    current_jac = .true.
                end if
            end do

            if (.not. converged) then
                ! Convergence failure: shrink the step and rebuild.
                factor = 0.5_dp
                h_abs = h_abs * factor
                call change_D(self, order, factor)
                self%n_equal_steps = 0
                self%lu_valid = .false.
                cycle
            end if

            safety = 0.9_dp * real(2*NEWTON_MAXITER + 1, dp) / &
                     real(2*NEWTON_MAXITER + n_iter, dp)

            self%scale = self%atol + self%rtol * abs(self%y_new)
            error_norm = weighted_rms(error_const(order) * self%d_vec, &
                                      self%scale, n)

            if (error_norm > 1.0_dp) then
                factor = max(MIN_FACTOR, &
                             safety * error_norm**(-1.0_dp/real(order+1, dp)))
                h_abs = h_abs * factor
                call change_D(self, order, factor)
                self%n_equal_steps = 0
                ! Convergence was fine, so keep the factored matrix.
            else
                step_accepted = .true.
            end if
        end do

        ! --- accept the step -------------------------------------------------
        self%nsteps = self%nsteps + 1
        self%n_equal_steps = self%n_equal_steps + 1
        self%t = t_new
        self%y = self%y_new
        self%h_abs = h_abs

        ! Update the differences array (D^{j+1} y_n = D^j y_n - D^j y_{n-1}).
        D(:, order+2) = self%d_vec - D(:, order+1)
        D(:, order+1) = self%d_vec
        do i = order, 0, -1
            D(:, i) = D(:, i) + D(:, i+1)
        end do

        code = BDF_OK

        ! Hold order/step until enough equal steps have accumulated.
        if (self%n_equal_steps < order + 1) return

        ! --- order selection -------------------------------------------------
        if (order > 1) then
            error_m_norm = weighted_rms(error_const(order-1) * D(:, order), &
                                        self%scale, n)
        else
            error_m_norm = huge(1.0_dp)
        end if
        if (order < MAX_ORDER) then
            error_p_norm = weighted_rms(error_const(order+1) * D(:, order+2), &
                                        self%scale, n)
        else
            error_p_norm = huge(1.0_dp)
        end if

        factors(1) = safe_pow(error_m_norm, -1.0_dp/real(order,   dp))
        factors(2) = safe_pow(error_norm,   -1.0_dp/real(order+1, dp))
        factors(3) = safe_pow(error_p_norm, -1.0_dp/real(order+2, dp))

        ibest = 1; best = factors(1)
        do k = 2, 3
            if (factors(k) > best) then
                best = factors(k); ibest = k
            end if
        end do
        delta_order = ibest - 2          ! argmax - 1, with 0-based argmax
        order = order + delta_order
        self%order = order

        factor = min(MAX_FACTOR, safety * best)
        self%h_abs = self%h_abs * factor
        call change_D(self, order, factor)
        self%n_equal_steps = 0
        self%lu_valid = .false.

        end associate
    end subroutine bdf_step

    !========================================================================
    ! Simplified Newton iteration for the BDF algebraic system.
    ! Port of solve_bdf_system; results land in self%y_new / self%d_vec.
    !========================================================================

    subroutine solve_bdf_system(self, t_new, c, n_iter, dy_norm, converged)
        class(bdf_solver), intent(inout) :: self
        real(dp),          intent(in)    :: t_new, c
        integer,           intent(out)   :: n_iter
        real(dp),          intent(out)   :: dy_norm
        logical,           intent(out)   :: converged

        real(dp) :: dy_norm_old, rate
        integer  :: k
        logical  :: have_rate

        associate (n => self%n)

        self%d_vec = 0.0_dp
        self%y_new = self%y_predict
        rate        = 0.0_dp
        dy_norm_old = -1.0_dp
        have_rate   = .false.
        converged   = .false.
        dy_norm     = 0.0_dp
        n_iter      = 0

        do k = 0, NEWTON_MAXITER - 1
            n_iter = k + 1
            call eval_fun(self, t_new, self%y_new, self%f)
            if (.not. all_finite(self%f, n)) return

            ! Solve (I - c J) dy = c f - psi - d.
            self%dy = c*self%f - self%psi - self%d_vec
            call solve_factored(self, self%dy)
            dy_norm = weighted_rms(self%dy, self%scale, n)

            if (have_rate) then
                rate = dy_norm / dy_norm_old
                if (rate >= 1.0_dp .or. &
                    rate**(NEWTON_MAXITER - k) / (1.0_dp - rate) * dy_norm &
                        > self%newton_tol) return
            end if

            self%y_new = self%y_new + self%dy
            self%d_vec = self%d_vec + self%dy

            if (dy_norm == 0.0_dp) then
                converged = .true.
                return
            end if
            if (have_rate) then
                if (rate/(1.0_dp - rate) * dy_norm < self%newton_tol) then
                    converged = .true.
                    return
                end if
            end if

            dy_norm_old = dy_norm
            have_rate   = .true.
        end do

        end associate
    end subroutine solve_bdf_system

    !========================================================================
    ! Difference-array rescaling on step-size change (change_D / compute_R).
    !========================================================================

    subroutine change_D(self, order, factor)
        class(bdf_solver), intent(inout) :: self
        integer,           intent(in)    :: order
        real(dp),          intent(in)    :: factor

        real(dp) :: R(0:order, 0:order), U(0:order, 0:order)
        real(dp) :: RU(0:order, 0:order)

        call compute_R(order, factor, R)
        call compute_R(order, 1.0_dp, U)
        RU = matmul(R, U)
        ! new D(:,0:order) = D(:,0:order) @ RU
        self%Dtmp(:, 0:order) = matmul(self%D(:, 0:order), RU)
        self%D(:, 0:order) = self%Dtmp(:, 0:order)
    end subroutine change_D

    !> R(0:order, 0:order): column-cumulative-product matrix used to rescale
    !> the differences array (see SciPy compute_R).
    subroutine compute_R(order, factor, R)
        integer,  intent(in)  :: order
        real(dp), intent(in)  :: factor
        real(dp), intent(out) :: R(0:order, 0:order)
        integer :: i, j
        R = 0.0_dp
        R(0, :) = 1.0_dp
        do j = 1, order
            do i = 1, order
                R(i, j) = (real(i, dp) - 1.0_dp - factor*real(j, dp)) / real(i, dp)
            end do
        end do
        ! Cumulative product down each column.
        do j = 0, order
            do i = 1, order
                R(i, j) = R(i-1, j) * R(i, j)
            end do
        end do
    end subroutine compute_R

    !========================================================================
    ! Jacobian evaluation (user callback or finite differences) and the
    ! factorization of the Newton iteration matrix I - c*J.
    !========================================================================

    !> Fill self%Jac_store with the Jacobian at (t, y).
    subroutine eval_jacobian(self, t, y)
        class(bdf_solver), intent(inout) :: self
        real(dp),          intent(in)    :: t, y(self%n)
        if (self%jac_mode == BDF_JAC_FD) then
            call fd_jacobian(self, t, y)
        else
            call self%jac(self%n, t, y, self%ml, self%mu, self%Jac_store, &
                          self%ldj, self%ctx)
            self%njev = self%njev + 1
        end if
    end subroutine eval_jacobian

    !> Forward-difference Jacobian.  Dense problems perturb one column at a
    !> time; banded problems use Curtis-Powell-Reid column grouping so that a
    !> single group of mutually non-overlapping columns is perturbed at once.
    subroutine fd_jacobian(self, t, y)
        class(bdf_solver), intent(inout) :: self
        real(dp),          intent(in)    :: t, y(self%n)
        real(dp) :: h, sqrt_eps
        integer  :: j, i, g, ngroup, r, i0, i1

        sqrt_eps = sqrt(epsilon(1.0_dp))
        call eval_fun(self, t, y, self%f)   ! base point

        if (.not. self%banded) then
            self%yp = y
            do j = 1, self%n
                h = sqrt_eps * max(abs(y(j)), 1.0_dp)
                self%yp(j) = y(j) + h
                call eval_fun(self, t, self%yp, self%f_pert)
                self%yp(j) = y(j)
                self%Jac_store(:, j) = (self%f_pert - self%f) / h
            end do
        else
            ngroup = min(self%n, self%ml + self%mu + 1)
            self%Jac_store = 0.0_dp
            do g = 0, ngroup - 1
                self%yp = y
                do j = 1, self%n
                    if (mod(j-1, ngroup) == g) then
                        h = sqrt_eps * max(abs(y(j)), 1.0_dp)
                        self%hstep(j) = h
                        self%yp(j) = y(j) + h
                    end if
                end do
                call eval_fun(self, t, self%yp, self%f_pert)
                do j = 1, self%n
                    if (mod(j-1, ngroup) == g) then
                        i0 = max(1, j - self%mu)
                        i1 = min(self%n, j + self%ml)
                        do i = i0, i1
                            r = i - j + self%mu + 1
                            self%Jac_store(r, j) = &
                                (self%f_pert(i) - self%f(i)) / self%hstep(j)
                        end do
                    end if
                end do
            end do
        end if
        self%njev = self%njev + 1
    end subroutine fd_jacobian

    !> Build and LU-factor the Newton matrix I - c*J, refreshing the stored
    !> Jacobian first unless a valid cached one may be reused.
    subroutine factor_newton_matrix(self, c)
        class(bdf_solver), intent(inout) :: self
        real(dp),          intent(in)    :: c
        integer :: info, j, i, r, mdiag
        logical :: need_eval

        ! Decide whether to (re)evaluate the Jacobian before factoring:
        !   * always on the first factorization (njev == 0);
        !   * on every factorization when Jacobian saving is disabled, except
        !     for a constant Jacobian which is only ever evaluated once.
        ! Otherwise the cached Jacobian in Jac_store is reused (the "saving"
        ! option), and only the I - c*J matrix is rebuilt and refactored.
        need_eval = (self%njev == 0)
        if (.not. self%reuse_jac .and. self%jac_mode /= BDF_JAC_CONSTANT) &
            need_eval = .true.
        if (need_eval) call eval_jacobian(self, self%t, self%y)

        if (.not. self%banded) then
            ! Dense: lu = I - c*J.
            do j = 1, self%n
                do i = 1, self%n
                    self%lu(i, j) = -c * self%Jac_store(i, j)
                end do
                self%lu(j, j) = self%lu(j, j) + 1.0_dp
            end do
            call dgetrf(self%n, self%n, self%lu, self%ldlu, self%ipiv, info)
        else
            ! Banded: shift band rows down by ml to leave room for fill-in.
            self%lu = 0.0_dp
            mdiag = self%mu + 1                ! diagonal row in Jac_store
            do j = 1, self%n
                do r = 1, self%ldj
                    self%lu(r + self%ml, j) = -c * self%Jac_store(r, j)
                end do
                ! Add the identity on the diagonal.
                self%lu(mdiag + self%ml, j) = self%lu(mdiag + self%ml, j) + 1.0_dp
            end do
            call dgbtrf(self%n, self%n, self%ml, self%mu, self%lu, self%ldlu, &
                        self%ipiv, info)
        end if

        self%nlu = self%nlu + 1
        self%lu_valid = .true.
        self%c_lu = c
    end subroutine factor_newton_matrix

    !> Solve (I - c*J) x = b in place, reusing the cached factorization.
    subroutine solve_factored(self, b)
        class(bdf_solver), intent(inout) :: self
        real(dp),          intent(inout) :: b(self%n)
        integer :: info
        if (.not. self%banded) then
            call dgetrs('N', self%n, 1, self%lu, self%ldlu, self%ipiv, b, &
                        self%n, info)
        else
            call dgbtrs('N', self%n, self%ml, self%mu, 1, self%lu, self%ldlu, &
                        self%ipiv, b, self%n, info)
        end if
    end subroutine solve_factored

    !========================================================================
    ! Small helpers
    !========================================================================

    subroutine eval_fun(self, t, y, f)
        class(bdf_solver), intent(inout) :: self
        real(dp),          intent(in)    :: t, y(self%n)
        real(dp),          intent(out)   :: f(self%n)
        call self%fun(self%n, t, y, f, self%ctx)
        self%nfev = self%nfev + 1
    end subroutine eval_fun

    !> Root-mean-square norm of x/scale (SciPy `norm`).
    pure function weighted_rms(x, scale, n) result(r)
        integer,  intent(in) :: n
        real(dp), intent(in) :: x(n), scale(n)
        real(dp) :: r
        r = sqrt(sum((x/scale)**2) / real(n, dp))
    end function

    pure function all_finite(x, n) result(ok)
        integer,  intent(in) :: n
        real(dp), intent(in) :: x(n)
        logical :: ok
        integer :: i
        ok = .true.
        do i = 1, n
            if (.not. ieee_is_finite(x(i))) then
                ok = .false.
                return
            end if
        end do
    end function

    !> x**p that returns 0 for non-finite/zero base (mirrors SciPy's
    !> errstate(divide='ignore') giving inf -> factor 0 after argmax).
    pure function safe_pow(x, p) result(r)
        real(dp), intent(in) :: x, p
        real(dp) :: r
        if (x <= 0.0_dp .or. .not. ieee_is_finite(x)) then
            r = 0.0_dp
        else
            r = x**p
        end if
    end function

    !> Initial step-size heuristic (SciPy select_initial_step).
    function select_initial_step(self, t0, y0, f0) result(h)
        class(bdf_solver), intent(inout) :: self
        real(dp),          intent(in)    :: t0, y0(self%n), f0(self%n)
        real(dp) :: h, interval, d0, d1, d2, h0, h1
        integer  :: order

        order = 1
        interval = abs(self%t_bound - t0)
        if (self%n == 0 .or. interval == 0.0_dp) then
            h = 0.0_dp
            return
        end if

        self%scale = self%atol + abs(y0)*self%rtol
        d0 = weighted_rms(y0, self%scale, self%n)
        d1 = weighted_rms(f0, self%scale, self%n)
        if (d0 < 1.0e-5_dp .or. d1 < 1.0e-5_dp) then
            h0 = 1.0e-6_dp
        else
            h0 = 0.01_dp * d0 / d1
        end if
        h0 = min(h0, interval)

        self%yp = y0 + h0*self%direction*f0
        call eval_fun(self, t0 + h0*self%direction, self%yp, self%f_pert)
        d2 = weighted_rms(self%f_pert - f0, self%scale, self%n) / h0

        if (d1 <= 1.0e-15_dp .and. d2 <= 1.0e-15_dp) then
            h1 = max(1.0e-6_dp, h0*1.0e-3_dp)
        else
            h1 = (0.01_dp / max(d1, d2))**(1.0_dp/real(order+1, dp))
        end if

        h = min(100.0_dp*h0, h1, interval, self%max_step)
    end function select_initial_step

end module bdf_module
