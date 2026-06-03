"""Pytest configuration for ZVODE benchmarks."""

import sys
from pathlib import Path

# Make problems.py importable from benchmark files
sys.path.insert(0, str(Path(__file__).parent))


def pytest_configure(config):
    """Suppress routine ZVODE warnings that would otherwise spam benchmark output."""
    config.addinivalue_line("filterwarnings", "ignore::UserWarning:zvode")
    config.addinivalue_line("filterwarnings", "ignore:The following arguments have no effect")
    config.addinivalue_line("filterwarnings", "ignore:Bandwidth lband")
    config.addinivalue_line("filterwarnings", "ignore:y0 has a real dtype")
