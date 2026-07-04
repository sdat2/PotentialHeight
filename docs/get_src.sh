#!/bin/bash
# Regenerate the sphinx-apidoc stubs for the packages in the repo root.
# -T: do not create modules.rst (index.rst lists the packages explicitly).
# setup.py and tests/ are excluded so no orphan 'setup'/'tests' pages are
# generated (a stale modules.rst previously referenced a nonexistent 'setup'
# document).
sphinx-apidoc -f -T -o . .. ../setup.py ../tests
