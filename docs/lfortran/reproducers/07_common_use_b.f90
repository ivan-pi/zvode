program b
  use a_mod, only: seta, geta
  implicit none
  call seta(3.5)
  print *, geta()
end program b
