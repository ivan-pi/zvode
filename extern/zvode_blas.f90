module zvode_blas_mod
  implicit none
  private

  integer, parameter :: dp = kind(1.0d0)

  public :: zcopy, zacopy, dzscal, dzaxpy

contains

  ! Copy N complex elements from ZX to ZY (unit stride only).
  subroutine zcopy(n, zx, incx, zy, incy)
    integer,     intent(in)    :: n, incx, incy
    complex(dp), intent(in)    :: zx(*)
    complex(dp), intent(inout) :: zy(*)
    integer :: i
    do i = 1, n
      zy(i) = zx(i)
    end do
  end subroutine zcopy

  ! Copy NROW x NCOL complex array A (row dim NROWA) to B (row dim NROWB).
  subroutine zacopy(nrow, ncol, a, nrowa, b, nrowb)
    integer,     intent(in)    :: nrow, ncol, nrowa, nrowb
    complex(dp), intent(in)    :: a(nrowa, ncol)
    complex(dp), intent(inout) :: b(nrowb, ncol)
    integer :: ic
    do ic = 1, ncol
      call zcopy(nrow, a(1,ic), 1, b(1,ic), 1)
    end do
  end subroutine zacopy

  ! Scale complex vector ZX by double precision scalar DA.
  ! (Variant of ZSCAL where the scalar is real, saving 2 multiplies/element.)
  subroutine dzscal(n, da, zx, incx)
    integer,     intent(in)    :: n, incx
    real(dp),    intent(in)    :: da
    complex(dp), intent(inout) :: zx(*)
    integer :: i
    do i = 1, n
      zx(i) = da * zx(i)
    end do
  end subroutine dzscal

  ! Add DA*ZX to ZY where DA is double precision (not complex).
  ! (Variant of ZAXPY where the scalar is real, saving 2 multiplies/element.)
  subroutine dzaxpy(n, da, zx, incx, zy, incy)
    integer,     intent(in)    :: n, incx, incy
    real(dp),    intent(in)    :: da
    complex(dp), intent(in)    :: zx(*)
    complex(dp), intent(inout) :: zy(*)
    integer :: i
    if (n <= 0 .or. da == 0.0d0) return
    do i = 1, n
      zy(i) = zy(i) + da * zx(i)
    end do
  end subroutine dzaxpy

end module zvode_blas_mod
