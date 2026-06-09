"""Python bindings to the ZVODE ODE solver"""

from .solve import solve_complex_ivp, ZVODEResult, ZVODE_FUN_CTYPE, ZVODE_JAC_CTYPE

__all__ = [
    "solve_complex_ivp",
    "ZVODEResult",
    "ZVODE_FUN_CTYPE",
    "ZVODE_JAC_CTYPE",
    "zvode_fun_sig",
    "zvode_jac_sig",
]

try:
    from .zvode_impl import ZVODE, ZVODE_Adams, ZVODE_BDF

    __all__ += ["ZVODE", "ZVODE_Adams", "ZVODE_BDF"]
except ImportError:
    pass


def __getattr__(name):
    # Python calls __getattr__ only when normal attribute lookup has already
    # failed, so this function only runs when 'name' is not defined in the
    # module (i.e. scipy was absent and the try/except above skipped the
    # ZVODE class imports).  We intercept the known names to surface a
    # helpful install hint instead of the default AttributeError.
    if name in ("ZVODE", "ZVODE_Adams", "ZVODE_BDF"):
        raise ImportError(
            f"{name!r} requires SciPy. Install it with: pip install 'zvode[scipy]'"
        )
    # Lazy numba signature objects — constructed on first access so that numba
    # remains an optional dependency and is never imported at module level.
    if name == "zvode_fun_sig":
        from numba import types  # ImportError propagates if numba not installed

        return types.void(
            types.int32,  # neq
            types.float64,  # t
            types.CPointer(types.complex128),  # const double complex *y
            types.CPointer(types.complex128),  # double complex *dy
            types.voidptr,  # void *ctx
        )
    if name == "zvode_jac_sig":
        from numba import types

        return types.void(
            types.int32,  # neq
            types.float64,  # t
            types.CPointer(types.complex128),  # const double complex *y
            types.int32,  # ml
            types.int32,  # mu
            types.CPointer(types.complex128),  # double complex *pd
            types.int32,  # nrowpd
            types.voidptr,  # void *ctx
        )
    raise AttributeError(f"module 'zvode' has no attribute {name!r}")
