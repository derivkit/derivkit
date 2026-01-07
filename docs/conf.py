"""Sphinx configuration for the DerivKit documentation."""

# -----------------------------------------------------------------------------
# Standard library imports
# -----------------------------------------------------------------------------
import logging

# -----------------------------------------------------------------------------
# Third-party imports
# -----------------------------------------------------------------------------
import matplotlib
from sphinx.ext.doctest import doctest

# -----------------------------------------------------------------------------
# Global setup
# -----------------------------------------------------------------------------
matplotlib.use("Agg")

# Silence emcee progress-bar / logging noise during Sphinx builds
logging.getLogger("emcee").setLevel(logging.ERROR)
logging.getLogger("emcee.pbar").setLevel(logging.ERROR)

# -----------------------------------------------------------------------------
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = "DerivKit"
copyright = "2025, Nikolina Šarčević, Matthijs van der Wild, Cynthia Trendafilova"
author = "Nikolina Šarčević et al."

# -----------------------------------------------------------------------------
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

html_favicon = "assets/favicon.png"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_design",
    "sphinx_multiversion",
    "sphinx_copybutton",
]

# -----------------------------------------------------------------------------
# Doctest configuration
# -----------------------------------------------------------------------------
doctest_global_setup = r"""
import numpy as np
np.set_printoptions(precision=12, suppress=True)

# Silence noisy libraries during doctest execution.
import io
import contextlib
import warnings
import logging

# Silence GetDist informational prints (e.g. "Removed no burn in")
try:
    from getdist import chains as _getdist_chains
    _getdist_chains.print_load_details = False
except Exception:
    pass

# Redirect stdout/stderr to avoid doctest failures from unexpected prints.
_doctest_stdout = io.StringIO()
_doctest_stderr = io.StringIO()
_doctest_redirect = contextlib.ExitStack()
_doctest_redirect.enter_context(contextlib.redirect_stdout(_doctest_stdout))
_doctest_redirect.enter_context(contextlib.redirect_stderr(_doctest_stderr))

# Silence warnings and logger chatter (emcee, tqdm, etc.).
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("emcee").setLevel(logging.ERROR)
logging.getLogger("emcee.pbar").setLevel(logging.ERROR)
"""

doctest_default_flags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE

# -----------------------------------------------------------------------------
# Copybutton configuration
# -----------------------------------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True
copybutton_copy_empty_lines = False

# -----------------------------------------------------------------------------
# Matplotlib / plot directive configuration
# -----------------------------------------------------------------------------
plot_html_show_source_link = False
plot_formats = [("png", 300)]
plot_rcparams = {
    # Figure defaults
    "figure.figsize": (4.5, 4.5),
    "figure.dpi": 150,
    "savefig.dpi": 150,

    # DerivKit color scheme
    "axes.edgecolor": "#3b9ab2",
    "axes.labelcolor": "#3b9ab2",
    "axes.titlecolor": "#3b9ab2",
    "xtick.color": "#3b9ab2",
    "ytick.color": "#3b9ab2",
    "text.color": "#3b9ab2",

    # Default color cycle
    "axes.prop_cycle": "cycler(color=['#f21901', '#3b9ab2', '#e1af00'])",

    # Styling tweaks
    "axes.linewidth": 1.0,
    "font.size": 10,
}

# -----------------------------------------------------------------------------
# Intersphinx configuration
# -----------------------------------------------------------------------------
intersphinx_mapping = {
    "getdist": ("https://getdist.readthedocs.io/en/stable/", None),
    "emcee": ("https://emcee.readthedocs.io/en/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}

# -----------------------------------------------------------------------------
# Autodoc / templates
# -----------------------------------------------------------------------------
autoclass_content = "both"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -----------------------------------------------------------------------------
# Sidebar layout
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Sphinx Multiversion
# -----------------------------------------------------------------------------
smv_tag_whitelist = r"^v\d+\.\d+\.\d+$"
smv_branch_whitelist = "main"

# -----------------------------------------------------------------------------
# HTML output
# -----------------------------------------------------------------------------
html_theme = "furo"
html_permalinks_icon = "<span>#</span>"

if html_theme == "furo":
    html_theme_options = {
        "light_css_variables": {
            "color-brand-primary": "#3b9ab2",
            "color-brand-content": "#3b9ab2",
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
else:
    html_theme_options = {}

html_static_path = ["_static"]
html_css_files = [
    "derivkit.css",  # keep LAST; bump version to bust cache
]
