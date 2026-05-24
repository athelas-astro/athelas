# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Athelas"
copyright = "2024-2026, Brandon L. Barker"
author = "Brandon L. Barker"
release = "v26.03"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx_multiversion",
    "sphinx.ext.todo",
    "sphinx_copybutton",
    "sphinx_sitemap",
]

templates_path = ["_templates"]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"

html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
math_number_all = True

# configuration for sphinx_multiversion
smv_remote_whitelist = r"^(origin)$"

todo_include_todos = True

# baseurl for sitemap
html_baseurl = "https://athelas-astro.github.io/athelas"
