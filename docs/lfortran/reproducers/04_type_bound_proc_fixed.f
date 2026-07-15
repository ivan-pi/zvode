      MODULE M
        TYPE T
          INTEGER :: N
        CONTAINS
          PROCEDURE :: EVAL => MYEVAL
        END TYPE
      CONTAINS
        SUBROUTINE MYEVAL(SELF)
          CLASS(T) :: SELF
          SELF%N = 0
        END SUBROUTINE
      END MODULE M
