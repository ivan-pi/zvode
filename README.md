# zvode

Python bindings to the classic ZVODE library in Fortran

Limitations:
- complex floats only
- no event-handling/root-finding capabilities
- not thread-safe (ZVODE uses global data)
- no solution back-tracking available

