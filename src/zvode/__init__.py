"""Python bindings to the ZVODE ODE solver"""

from .solve import solve_complex_ivp

__all__ = ["solve_complex_ivp"]

try:
    from .zvode_impl import ZVODE, ZVODE_Adams, ZVODE_BDF
    __all__ += ["ZVODE", "ZVODE_Adams", "ZVODE_BDF"]
except ImportError:
    pass


def __getattr__(name):
    if name in ("ZVODE", "ZVODE_Adams", "ZVODE_BDF"):
        raise ImportError(
            f"{name!r} requires SciPy. "
            "Install it with: pip install 'zvode[scipy]'"
        )
    raise AttributeError(f"module 'zvode' has no attribute {name!r}")
