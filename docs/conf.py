import tomllib
from pathlib import Path

with open(Path(__file__).parent.parent / "pyproject.toml", "rb") as f:
    _meta = tomllib.load(f)

project = "zvode"
author = "Ivan Pribec"
copyright = "2026, Ivan Pribec"
release = _meta["project"]["version"]

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

autodoc_member_order = "bysource"
autodoc_typehints = "none"
# NumPy-style References sections define citations without in-text usage;
# suppress the resulting "not referenced" false positive.
suppress_warnings = ["ref.citation"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

html_theme = "furo"
html_title = "zvode"
html_baseurl = "https://ivan-pi.github.io/zvode/"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]
