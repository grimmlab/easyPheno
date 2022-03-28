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
    'sphinx.ext.intersphinx',
    'myst_parser',
    'autoapi.extension',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

exclude_patterns = ['*/docs*']

autoapi_type = 'python'
autoapi_dirs = ['../../evaluation', '../../model', '../../optimization', '../../preprocess', '../../utils', '../../']
