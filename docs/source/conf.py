# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'Optimal Pointing Point Code for NICER'
copyright = '2023, Pierre Lambin'
author = 'Pierre Lambin'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.doctest',
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

# Configuration spécifique pour autodoc
autodoc_default_options = {
    'members': True,  # Documente tous les membres de la classe
    'member-order': 'bysource',  # L'ordre des membres comme dans le source
    'special-members': '__init__',  # Inclut __init__
    'undoc-members': True,  # Documente les membres non documentés
    'exclude-members': '__weakref__',  # Exclut certains membres
}

# -- Napoleon options (pour les docstrings Google et NumPy) -------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True
