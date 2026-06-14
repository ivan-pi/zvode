! test_zvode_numjac.f90
! ============================================================
!  Fortran-native test for the finite-difference Jacobian helpers
!  in module `zvode_numjac` (finite_diff_dense / finite_diff_band).
!
!  Strategy: pick right-hand sides with known analytic Jacobians and
!  check that the forward-difference approximation reproduces them to
!  first-order accuracy.
!
!    Dense problem (n = 4):
!       f_i(y) = sum_j A_ij y_j + b_i y_i^2
!       => J_ij = A_ij + delta_ij * 2 b_i y_i
!    A is a dense complex matrix, so every column is exercised.
!
!    Banded problem (n = 6, ml = mu = 1, tridiagonal):
!       f_i(y) = sub_i y_{i-1} + dia_i y_i + sup_i y_{i+1} + b_i y_i^2
!       => banded J with the same diagonal nonlinear term.
!
!  Assertions (code passed to error stop):
!    1. dense   : max |a_ij - J_ij|        within tolerance
!    2. dense   : nfev incremented by n
!    3. banded  : max band entry error     within tolerance
!    4. banded  : nfev incremented by min(mband, n)
!    5. both    : y restored to its input value
! ============================================================

module test_numjac_mod
  use zvode_mod, only: zvode_fun
  implicit none
  private
  public :: dense_fun, band_fun, dp

  integer, parameter :: dp = kind(1.0d0)
  integer, parameter :: ND = 4, NB = 6

  ! Dense RHS: f = A y + b .* y**2
  type, extends(zvode_fun) :: dense_fun
    complex(dp) :: a(ND,ND)
    complex(dp) :: b(ND)
  contains
    procedure :: eval => dense_eval
  end type

  ! Tridiagonal RHS: f_i = sub y_{i-1} + dia y_i + sup y_{i+1} + b y_i**2
  type, extends(zvode_fun) :: band_fun
    complex(dp) :: sub(NB), dia(NB), sup(NB), b(NB)
  contains
    procedure :: eval => band_eval
  end type

contains

  subroutine dense_eval(fun, t, y, ydot)
    class(dense_fun) :: fun
    real(dp), intent(in) :: t
    complex(dp), intent(in) :: y(fun%neq)
    complex(dp), intent(out) :: ydot(fun%neq)
    integer :: i
    ydot = matmul(fun%a, y)
    do i = 1, fun%neq
      ydot(i) = ydot(i) + fun%b(i)*y(i)**2
    end do
  end subroutine dense_eval

  subroutine band_eval(fun, t, y, ydot)
    class(band_fun) :: fun
    real(dp), intent(in) :: t
    complex(dp), intent(in) :: y(fun%neq)
    complex(dp), intent(out) :: ydot(fun%neq)
    integer :: i, n
    n = fun%neq
    do i = 1, n
      ydot(i) = fun%dia(i)*y(i) + fun%b(i)*y(i)**2
      if (i > 1) ydot(i) = ydot(i) + fun%sub(i)*y(i-1)
      if (i < n) ydot(i) = ydot(i) + fun%sup(i)*y(i+1)
    end do
  end subroutine band_eval

end module test_numjac_mod


program test_zvode_numjac
  use test_numjac_mod
  use zvode_numjac, only: finite_diff_dense, finite_diff_band
  implicit none

  real(dp), parameter :: uround = epsilon(1.0_dp)
  real(dp), parameter :: srur = sqrt(uround)
  ! Forward differences are O(h) ~ O(sqrt(uround)); allow a generous bound.
  real(dp), parameter :: tol = 1.0e-6_dp

  call test_dense()
  call test_band()
  write(*,'(a)') 'All zvode_numjac tests passed.'

contains

  real(dp) function step_r0(n, savf, ewt) result(r0)
    integer, intent(in) :: n
    complex(dp), intent(in) :: savf(n)
    real(dp), intent(in) :: ewt(n)
    real(dp), parameter :: hstep = 0.1_dp
    real(dp) :: nrm
    nrm = sqrt(sum((savf%re**2 + savf%im**2)*ewt**2)/n)
    r0 = 1000.0_dp*abs(hstep)*uround*real(n, dp)*nrm
    if (r0 == 0.0_dp) r0 = 1.0_dp
  end function step_r0

  subroutine test_dense()
    integer, parameter :: n = 4
    type(dense_fun) :: f
    complex(dp) :: y(n), y0(n), savf(n), a(n,n), jexact(n,n)
    real(dp) :: ewt(n), r0, err
    integer :: i, j, nfev

    f%neq = n
    ! A deliberately non-symmetric, fully populated complex matrix.
    do j = 1, n
      do i = 1, n
        f%a(i,j) = cmplx(0.5_dp*(i - j) + 1.0_dp, 0.2_dp*i - 0.1_dp*j, dp)
      end do
    end do
    f%b = [(cmplx(0.3_dp*i, -0.15_dp*i, dp), i = 1, n)]

    y0 = [(cmplx(1.0_dp + 0.25_dp*i, 0.5_dp - 0.1_dp*i, dp), i = 1, n)]
    y = y0
    call f%eval(0.0_dp, y, savf)
    ewt = 1.0_dp
    r0 = step_r0(n, savf, ewt)

    nfev = 0
    call finite_diff_dense(f, 0.0_dp, y, savf, ewt, srur, r0, a, n, n, nfev)

    ! Exact Jacobian: A + diag(2 b_i y_i).
    jexact = f%a
    do i = 1, n
      jexact(i,i) = jexact(i,i) + 2.0_dp*f%b(i)*y0(i)
    end do

    err = maxval(abs(a - jexact))
    if (err > tol) then
      write(*,'(a,es12.4)') 'dense Jacobian error too large: ', err
      error stop 1
    end if
    if (nfev /= n) error stop 2
    if (maxval(abs(y - y0)) /= 0.0_dp) error stop 5
  end subroutine test_dense

  subroutine test_band()
    integer, parameter :: n = 6, ml = 1, mu = 1
    integer, parameter :: mband = ml + mu + 1
    integer, parameter :: meband = 2*ml + mu + 1
    type(band_fun) :: f
    complex(dp) :: y(n), y0(n), savf(n), ab(meband,n)
    real(dp) :: ewt(n), r0, err, eij
    complex(dp) :: jij
    integer :: i, j, nfev

    f%neq = n
    do i = 1, n
      f%dia(i) = cmplx(-2.0_dp - 0.1_dp*i, 0.3_dp, dp)
      f%sub(i) = cmplx(1.0_dp, -0.2_dp*i, dp)
      f%sup(i) = cmplx(0.7_dp, 0.1_dp*i, dp)
      f%b(i)   = cmplx(0.2_dp*i, -0.1_dp, dp)
    end do

    y0 = [(cmplx(0.5_dp + 0.2_dp*i, 0.3_dp - 0.05_dp*i, dp), i = 1, n)]
    y = y0
    call f%eval(0.0_dp, y, savf)
    ewt = 1.0_dp
    r0 = step_r0(n, savf, ewt)

    ab = (0.0_dp, 0.0_dp)
    nfev = 0
    call finite_diff_band(f, 0.0_dp, y, savf, ewt, srur, r0, ml, mu, ab, meband, n, nfev)

    ! Compare every in-band entry against the analytic Jacobian.
    err = 0.0_dp
    do j = 1, n
      do i = max(j - mu, 1), min(j + ml, n)
        jij = (0.0_dp, 0.0_dp)
        if (i == j)     jij = f%dia(i) + 2.0_dp*f%b(i)*y0(i)
        if (i == j + 1) jij = f%sub(i)         ! sub-diagonal: f_i depends on y_{i-1}=y_j
        if (i == j - 1) jij = f%sup(i)         ! super-diagonal: f_i depends on y_{i+1}=y_j
        eij = abs(ab(mband + i - j, j) - jij)
        err = max(err, eij)
      end do
    end do

    if (err > tol) then
      write(*,'(a,es12.4)') 'band Jacobian error too large: ', err
      error stop 3
    end if
    if (nfev /= min(mband, n)) error stop 4
    if (maxval(abs(y - y0)) /= 0.0_dp) error stop 5
  end subroutine test_band

end program test_zvode_numjac
