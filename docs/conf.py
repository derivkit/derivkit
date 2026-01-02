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
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx_design",
    "sphinx_multiversion",
]

intersphinx_mapping = {
    "getdist": ("https://getdist.readthedocs.io/en/stable/", None),
    "emcee": ("https://emcee.readthedocs.io/en/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}

autoclass_content = "both"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_sidebar = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/variant-selector.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/scroll-end.html",
    ],
}

# -- Sphinx Multiversion --------------------------------------------------
# https://sphinx-contrib.github.io/multiversion/main/configuration.html

smv_tag_whitelist = r"^v\d+\.\d+\.\d+$"
smv_branch_whitelist = "main"

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
