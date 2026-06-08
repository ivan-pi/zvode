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
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

html_theme = "furo"
html_title = "zvode"
html_baseurl = "https://ivan-pi.github.io/zvode/"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]
