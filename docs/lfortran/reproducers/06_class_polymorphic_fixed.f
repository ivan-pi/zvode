      SUBROUTINE S(X)
        TYPE T
          INTEGER :: N
        END TYPE
        CLASS(T) :: X
        X%N = 1
      END SUBROUTINE
