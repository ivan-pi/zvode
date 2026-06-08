"""Python bindings to the ZVODE ODE solver"""

import sys as _sys

from .solve import solve_complex_ivp
from ._helpers import ZVODE_FUN_CTYPE, ZVODE_JAC_CTYPE

__all__ = [
    "solve_complex_ivp",
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
    # failed, so this runs only for names not yet bound in the module dict.

    if name in ("ZVODE", "ZVODE_Adams", "ZVODE_BDF"):
        raise ImportError(
            f"{name!r} requires SciPy. Install it with: pip install 'zvode[scipy]'"
        )

    if name in ("zvode_fun_sig", "zvode_jac_sig"):
        try:
            from ._helpers import _make_numba_sigs
            fun_sig, jac_sig = _make_numba_sigs()
            # Cache so subsequent accesses skip __getattr__ entirely.
            _sys.modules[__name__].zvode_fun_sig = fun_sig
            _sys.modules[__name__].zvode_jac_sig = jac_sig
            return fun_sig if name == "zvode_fun_sig" else jac_sig
        except ImportError:
            raise ImportError(
                f"{name!r} requires numba. Install it with: pip install numba"
            )

    raise AttributeError(f"module 'zvode' has no attribute {name!r}")
