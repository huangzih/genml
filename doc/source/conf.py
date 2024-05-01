# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GenML'
copyright = '2024, QuXiang2333'
author = 'QuXiang2333'
release = '0.4.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

mathjax_config = {
    'tex2jax': {
        'inlineMath': [ ['$','$'], ["\\(","\\)"] ],
        'displayMath': [ ['$$','$$'], ["\\[","\\]"] ],
        'processEscapes': True,
    }
}


extensions = ['recommonmark',
              'sphinx_markdown_tables',
              'sphinx.ext.mathjax'
              # 'myst_parser'
            #   'sphinx_markdown_tablesmyst_parser'
            ]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'en – English'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
