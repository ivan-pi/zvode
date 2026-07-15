      MODULE M
        IMPLICIT NONE
        INTEGER, PARAMETER :: DP = KIND(1.0D0)
      CONTAINS
        SUBROUTINE S(N, X)
          INTEGER, INTENT(IN) :: N
          REAL(DP), INTENT(INOUT) :: X(*)
          INTEGER :: I
          REAL(DP) :: A, V(3)
          COMMON /BLK/ A
          SAVE /BLK/
          DATA V /1.0D0, 2.0D0, 3.0D0/
          DO 10 I = 1, N
            X(I) = X(I) + A + V(1)
 10       CONTINUE
          GO TO (100, 200), N
 100      CONTINUE
 200      CONTINUE
        END SUBROUTINE
      END MODULE M
