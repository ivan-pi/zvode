module zvode_linalg_mod
  implicit none
  private

  integer, parameter :: dp = kind(1.0d0)

  public :: zcopy, zacopy, dzscal, dzaxpy, dzaxpynrm

contains

  ! Copy N complex elements from ZX (stride INCX) to ZY (stride INCY).
  subroutine zcopy(n, zx, incx, zy, incy)
    integer,     intent(in)    :: n, incx, incy
    complex(dp), intent(in)    :: zx(*)
    complex(dp), intent(inout) :: zy(*)
    integer :: i, ix, iy
    if (n <= 0) return
    if (incx == 1 .and. incy == 1) then
      do i = 1, n
        zy(i) = zx(i)
      end do
    else
      ix = 1; iy = 1
      if (incx < 0) ix = (-n + 1)*incx + 1
      if (incy < 0) iy = (-n + 1)*incy + 1
      do i = 1, n
        zy(iy) = zx(ix)
        ix = ix + incx
        iy = iy + incy
      end do
    end if
  end subroutine zcopy

  ! Copy NROW x NCOL complex array A (row dim NROWA) to B (row dim NROWB).
  subroutine zacopy(nrow, ncol, a, nrowa, b, nrowb)
    integer,     intent(in)    :: nrow, ncol, nrowa, nrowb
    complex(dp), intent(in)    :: a(nrowa, ncol)
    complex(dp), intent(inout) :: b(nrowb, ncol)
    b(:nrow,:) = a(:nrow,:)
  end subroutine zacopy

  ! Scale complex vector ZX by double precision scalar DA.
  ! Variant of ZSCAL with a real scalar: 2 multiplies/element instead of 4.
  ! Note: gfortran requires -ffast-math to exploit this; without it the compiler
  ! emits a full complex multiply pattern to preserve IEEE 754 NaN/Inf semantics.
  ! Ifort applies this optimisation by default due to unsafe math being enabled.
  subroutine dzscal(n, da, zx, incx)
    integer,     intent(in)    :: n, incx
    real(dp),    intent(in)    :: da
    complex(dp), intent(inout) :: zx(*)
    integer :: i, ix
    if (n <= 0 .or. incx <= 0) return
    if (da == 1.0d0) return
    if (incx == 1) then
      do i = 1, n
        zx(i) = da * zx(i)
      end do
    else
      ix = 1
      do i = 1, n
        zx(ix) = da * zx(ix)
        ix = ix + incx
      end do
    end if
  end subroutine dzscal

  ! Add double precision scalar DA times ZX to ZY.
  ! Variant of ZAXPY with a real scalar: 2 multiplies/element instead of 4.
  ! Note: same fast-math caveat as dzscal applies here.
  subroutine dzaxpy(n, da, zx, incx, zy, incy)
    integer,     intent(in)    :: n, incx, incy
    real(dp),    intent(in)    :: da
    complex(dp), intent(in)    :: zx(*)
    complex(dp), intent(inout) :: zy(*)
    integer :: i, ix, iy
    if (n <= 0) return
    if (da == 0.0d0) return
    if (incx == 1 .and. incy == 1) then
      do i = 1, n
        zy(i) = zy(i) + da * zx(i)
      end do
    else
      ix = 1; iy = 1
      if (incx < 0) ix = (-n + 1)*incx + 1
      if (incy < 0) iy = (-n + 1)*incy + 1
      do i = 1, n
        zy(iy) = zy(iy) + da * zx(ix)
        ix = ix + incx
        iy = iy + incy
      end do
    end if
  end subroutine dzaxpy

  ! Fused dzaxpy + weighted RMS norm: computes ZY += DA*ZX and returns
  ! SQRT((1/N) * SUM(|DA*ZX(i)|^2 * W(i)^2)).  Saves a full memory pass
  ! compared to a separate DZSCAL/DZAXPY followed by ZVNORM.
  ! Stride-1 only (matches ZVNORM's interface).
  real(dp) function dzaxpynrm(n, da, zx, zy, w) result(nrm)
    integer,     intent(in)    :: n
    real(dp),    intent(in)    :: da
    complex(dp), intent(in)    :: zx(n)
    complex(dp), intent(inout) :: zy(n)
    real(dp),    intent(in)    :: w(n)
    real(dp) :: s
    integer  :: i
    if (n <= 0) then
      nrm = 0.0d0
      return
    end if
    s = 0.0d0
    if (da == 1.0d0) then
      do i = 1, n
        zy(i) = zy(i) + zx(i)
        s = s + (real(zx(i))**2 + aimag(zx(i))**2) * w(i)**2
      end do
      nrm = sqrt(s / n)
    else
      do i = 1, n
        zy(i) = zy(i) + da * zx(i)
        s = s + (real(zx(i))**2 + aimag(zx(i))**2) * w(i)**2
      end do
      nrm = abs(da) * sqrt(s / n)
    end if
  end function dzaxpynrm

end module zvode_linalg_mod
