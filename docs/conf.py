"""Sphinx configuration for the DerivKit documentation."""

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "DerivKit"
copyright = "2025, Nikolina Šarčević, Matthijs van der Wild, Cynthia Trendafilova"
author = "Nikolina Šarčević et al."

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

html_favicon = "assets/favicon.png"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
    "sphinx_design",
]

autoclass_content = "both"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#3b9ab2",
        "color-brand-content": "#3b9ab2",

        # Link color vars that Furo uses
        "color-link": "#3b9ab2",
        "color-link--hover": "#f21901",
        "color-link--visited": "#e1af00",
    },
    "dark_css_variables": {
        "color-brand-primary": "#3b9ab2",
        "color-brand-content": "#3b9ab2",

        "color-link": "#3b9ab2",
        "color-link--hover": "#f21901",
        "color-link--visited": "#e1af00",
    },
}

html_static_path = ["_static"]
html_css_files = [
    "derivkit.css",  # keep LAST; bump v to bust cache
]
