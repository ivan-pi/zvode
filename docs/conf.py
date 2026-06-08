project = "zvode"
author = "Ivan Pribec"
copyright = "2024, Ivan Pribec"
release = "0.2.0"

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
