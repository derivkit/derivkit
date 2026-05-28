"""Sphinx configuration for the DerivKit documentation."""

# -----------------------------------------------------------------------------
# Standard library imports
# -----------------------------------------------------------------------------
import logging
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# Third-party imports
# -----------------------------------------------------------------------------
import matplotlib
from sphinx.ext.doctest import doctest

# -----------------------------------------------------------------------------
# Global setup
# -----------------------------------------------------------------------------
sys.path.append(str(Path("_ext").resolve()))
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
    "adoption",
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
        "source_repository": "https://github.com/derivkit-org/derivkit/",
        "source_branch": "main",
        "source_directory": "docs/",
        "light_logo": "logos/logo-black.png",
        "dark_logo": "logos/logo-blue.png",
        "footer_icons": [
            {
                "name": "GitHub",
                "url": "https://github.com/derivkit/derivkit",
                "html": """
                <svg stroke="var(--color-foreground-primary)"
                     fill="var(--color-foreground-primary)"
                     stroke-width="0"
                     viewBox="0 0 16 16">
                    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
                    0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13
                    -.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87
                    2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95
                    0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21
                    2.2.82A7.65 7.65 0 0 1 8 3.87c.68 0 1.36.09 2 .26
                    1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12
                    .51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65
                    3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01
                    2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16
                    8c0-4.42-3.58-8-8-8z"></path>
                </svg>
                """,
                "class": "",
            },
        ],
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

html_static_path = ["_static"]
html_css_files = [
    "derivkit.css",  # keep LAST; bump version to bust cache
]
