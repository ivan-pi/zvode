module lapack_interfaces
  implicit none
  private

  integer, parameter :: dp = kind(1.0d0)

  public :: zgbtrf, zgbtrs, zgetrf, zgetrs

  interface

    ! LU factorization of a complex banded matrix
    subroutine ZGBTRF(m, n, kl, ku, ab, ldab, ipiv, info)
      import dp
      integer,     intent(in)    :: m, n, kl, ku, ldab
      complex(dp), intent(inout) :: ab(ldab,*)
      integer,     intent(out)   :: ipiv(*), info
    end subroutine ZGBTRF

    ! Solve a banded system using the LU factors from ZGBTRF
    subroutine ZGBTRS(trans, n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb, info)
      import dp
      character(len=1), intent(in)    :: trans
      integer,          intent(in)    :: n, kl, ku, nrhs, ldab, ldb
      complex(dp),      intent(in)    :: ab(ldab,*)
      integer,          intent(in)    :: ipiv(*)
      complex(dp),      intent(inout) :: b(ldb,*)
      integer,          intent(out)   :: info
    end subroutine ZGBTRS

    ! LU factorization of a general complex matrix
    subroutine ZGETRF(m, n, a, lda, ipiv, info)
      import dp
      integer,     intent(in)    :: m, n, lda
      complex(dp), intent(inout) :: a(lda,*)
      integer,     intent(out)   :: ipiv(*), info
    end subroutine ZGETRF

    ! Solve a general system using the LU factors from ZGETRF
    subroutine ZGETRS(trans, n, nrhs, a, lda, ipiv, b, ldb, info)
      import dp
      character(len=1), intent(in)    :: trans
      integer,          intent(in)    :: n, nrhs, lda, ldb
      complex(dp),      intent(in)    :: a(lda,*)
      integer,          intent(in)    :: ipiv(*)
      complex(dp),      intent(inout) :: b(ldb,*)
      integer,          intent(out)   :: info
    end subroutine ZGETRS

  end interface

end module lapack_interfaces
