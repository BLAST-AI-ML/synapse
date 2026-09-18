# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Synapse"
# Rendered by sphinx-book-theme as "© Copyright ..." in the page footer, so
# this is the copyright holder from LICENSE.txt, not the license identifier.
copyright = (
    "2025, The Regents of the University of California, "
    "through Lawrence Berkeley National Laboratory"
)
# An empty value prevents sphinx-book-theme from rendering an author footer.
author = ""

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["myst_parser", "sphinx.ext.extlinks", "sphinx_copybutton"]
myst_heading_anchors = 4

# Roles that turn a repository-relative path into a link to GitHub, so that
# source files and directories mentioned in the docs stay navigable:
#   {repo}`ml/train_model.py`                  -> "ml/train_model.py"
#   {repo}`train_model.py <ml/train_model.py>` -> "train_model.py"
#   {repo-dir}`dashboard/`                     -> "dashboard/"
_repo_url = "https://github.com/BLAST-AI-ML/synapse"
_repo_ref = "main"
extlinks = {
    "repo": (f"{_repo_url}/blob/{_repo_ref}/%s", "%s"),
    "repo-dir": (f"{_repo_url}/tree/{_repo_ref}/%s", "%s"),
}
# Warn when a hardcoded URL could be written with one of the roles above.
extlinks_detect_hardcoded_links = True

exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_theme_options = {
    # The left sidebar lists pages only: sphinx-book-theme resolves it with
    # titles_only=True, so navbar depth cannot expose in-page headings there.
    "show_navbar_depth": 1,
    "max_navbar_depth": 1,
    # In-page navigation lives in the right-hand "Contents" sidebar. The level
    # counts from the h2 entries below the page title, so this expands it to h4
    # on load: on the long pages, dashboard.md and ml-training.md, the steps a
    # reader scans for ("Prepare the conda environment", "Run the training",
    # "Build the Docker image") are h4, and stopping at h3 hides them until the
    # surrounding section is scrolled into view. This matches the h4 depth of
    # myst_heading_anchors above, so every entry in the sidebar is linkable.
    "show_toc_level": 3,
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
