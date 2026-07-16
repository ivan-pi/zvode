      MODULE M
        TYPE :: BASE
          INTEGER :: N
        END TYPE
        TYPE, EXTENDS(BASE) :: CHILD
          INTEGER :: M
        END TYPE
      END MODULE M
