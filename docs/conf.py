# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from pathlib import Path

# This song and dance enables builds from outside the docs directory
srcpath = os.path.abspath(Path(os.path.dirname(__file__)) / "../")
sys.path.insert(0, srcpath)

# sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyAMPACT'
copyright = '2024, AMPACT Research Team'
author = 'AMPACT Research Team'
release = '0.0.4'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode'
]

autosummary_generate = True
autosummary_generate_overwrite = False
templates_path = ['_templates']
exclude_patterns = ['docs', '_build', 'Thumbs.db', '.DS_Store', 'sandbox.py']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = []
html_baseurl = 'https://pyampact.github.io'
