# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information

project = 'easyPheno'
copyright = '2022, GrimmLab @ TUM Campus Straubing (https://bit.cs.tum.de/)'
author = 'Florian Haselbeck, Maura John, Dominik G. Grimm'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'autoapi.extension',
    'sphinx.ext.intersphinx',
    "sphinx.ext.imgconverter",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    'myst_parser',
    'sphinxcontrib.youtube',
    'sphinx.ext.autosectionlabel',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

html_theme_options = {"logo_only": True}

html_logo = "../image/Logo_easyPheno_Text.png"

# -- Options for EPUB output
epub_show_urls = 'footnote'

exclude_patterns = ['*/docs*']

autoapi_type = 'python'
autoapi_dirs = ['../../']
autodoc_typehints = 'description'
autoapi_ignore = ['*conf*', '*setup*', '*run*']
