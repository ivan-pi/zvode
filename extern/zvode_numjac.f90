!> Finite-difference Jacobian approximations for ZVODE.
!>
!> These helpers extract the forward-difference Jacobian code that was
!> previously inlined in ZVJAC (the MITER = 2 dense and MITER = 5 banded
!> branches) into self-contained, COMMON-free routines that can be unit
!> tested on their own.
!>
!> Two references inform the implementation:
!>
!>   1. The original ZVODE/DVODE step magnitude
!>         r = max( srur*|y_j| , r0/ewt_j )
!>      with srur = sqrt(uround) and
!>         r0 = 1000 * |h| * uround * n * ||savf||_ewt   (r0 = 1 if zero).
!>
!>   2. SciPy's `num_jac` (scipy.integrate._ivp.common), which directs the
!>      perturbation along the sign of the real part of f and rounds the
!>      increment to a machine-representable value via
!>         h = (y_j + step) - y_j .
!>      For a complex state the step is taken along the real axis, so h is
!>      real and the difference quotient (f(y + h) - f(y))/h stays complex.
!>
!> The adaptive per-column `factor` feedback from SciPy's num_jac is *not*
!> reproduced here; only the directional, representable step is adopted on
!> top of ZVODE's step magnitude.
module zvode_numjac

   use zvode_mod, only: dp, zvode_fun
   implicit none
   private

   public :: finite_diff_dense
   public :: finite_diff_band

contains

   !> Machine-representable, real-axis, signed perturbation for column j.
   !>
   !> Returns a step h with magnitude approximately `rmag`, directed by the
   !> sign of `fre` (the real part of f_j), and rounded so that y_j + h is
   !> exactly representable.  The loop guards against the rare case where the
   !> nominal step underflows to zero relative to Re(y_j).
   pure real(dp) function repr_step(yj, rmag, fre) result(h)
      complex(dp), intent(in) :: yj
      real(dp), intent(in) :: rmag, fre
      real(dp) :: yr, s, m
      yr = real(yj, dp)
      s = sign(1.0_dp, fre)        ! +1 when fre >= 0 (matches SciPy's >= 0)
      m = rmag
      h = (yr + s*m) - yr
      do while (h == 0.0_dp)
         m = m*10.0_dp
         h = (yr + s*m) - yr
      end do
   end function repr_step

   !> Forward-difference dense Jacobian, P(i,j) = d f_i / d y_j.
   !>
   !> Columns are perturbed one at a time (n calls to f).  Replaces the
   !> MITER = 2 branch of ZVJAC.
   !>
   !>   f     : right-hand side functor, f%eval(t, y, ydot)
   !>   t     : current time
   !>   y     : current state (perturbed in place, restored on exit)
   !>   savf  : f(t, y), evaluated by the caller
   !>   ewt   : error weights, length n
   !>   srur  : sqrt(uround)
   !>   r0    : ZVODE base step (see module header), already guarded /= 0
   !>   a     : output Jacobian, leading dimension lda (>= n)
   !>   nfev  : f-evaluation counter, incremented by n
   subroutine finite_diff_dense(f, t, y, savf, ewt, srur, r0, a, lda, n, nfev)
      class(zvode_fun), intent(in) :: f
      real(dp), intent(in) :: t
      complex(dp), intent(inout) :: y(n)
      complex(dp), intent(in) :: savf(n)
      real(dp), intent(in) :: ewt(n)
      real(dp), intent(in) :: srur, r0
      integer, intent(in) :: lda, n
      complex(dp), intent(out) :: a(lda,n)
      integer, intent(inout) :: nfev

      integer :: j
      real(dp) :: rmag, h
      complex(dp) :: yj, ftem(n)

      do j = 1, n
         yj = y(j)
         rmag = max(srur*abs(yj), r0/ewt(j))
         h = repr_step(yj, rmag, real(savf(j), dp))
         y(j) = yj + h
         call f%eval(t, y, ftem)
         a(1:n,j) = (ftem - savf)/h
         y(j) = yj
      end do
      nfev = nfev + n
   end subroutine finite_diff_dense

   !> Forward-difference banded Jacobian in LAPACK band storage.
   !>
   !> Columns spaced `mband = ml + mu + 1` apart have disjoint bands, so they
   !> are perturbed together and recovered from a single f call; this needs
   !> only min(mband, n) evaluations.  Replaces the MITER = 5 branch of ZVJAC.
   !>
   !> The Jacobian entry A(i,j) is written to ab(mband + i - j, j), the LAPACK
   !> ZGBTRF convention (diagonal on row mband; rows 1..ml left as fill-in
   !> workspace).  `ldab` must be at least meband = 2*ml + mu + 1.
   subroutine finite_diff_band(f, t, y, savf, ewt, srur, r0, ml, mu, ab, ldab, n, nfev)
      class(zvode_fun), intent(in) :: f
      real(dp), intent(in) :: t
      complex(dp), intent(inout) :: y(n)
      complex(dp), intent(in) :: savf(n)
      real(dp), intent(in) :: ewt(n)
      real(dp), intent(in) :: srur, r0
      integer, intent(in) :: ml, mu, ldab, n
      complex(dp), intent(out) :: ab(ldab,n)
      integer, intent(inout) :: nfev

      integer :: g, i, i1, i2, jj, mband, mba
      real(dp) :: rmag
      complex(dp) :: ftem(n), yold(n)
      real(dp) :: hcol(n)

      mband = ml + mu + 1
      mba = min(mband, n)

      do g = 1, mba
         ! Perturb every column in this group along the real axis.
         do jj = g, n, mband
            yold(jj) = y(jj)
            rmag = max(srur*abs(y(jj)), r0/ewt(jj))
            hcol(jj) = repr_step(y(jj), rmag, real(savf(jj), dp))
            y(jj) = yold(jj) + hcol(jj)
         end do
         call f%eval(t, y, ftem)
         ! Recover the band of each perturbed column and restore y.
         do jj = g, n, mband
            y(jj) = yold(jj)
            i1 = max(jj - mu, 1)
            i2 = min(jj + ml, n)
            do i = i1, i2
               ab(mband + i - jj, jj) = (ftem(i) - savf(i))/hcol(jj)
            end do
         end do
      end do
      nfev = nfev + mba
   end subroutine finite_diff_band

end module zvode_numjac
