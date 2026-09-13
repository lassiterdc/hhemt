#!/usr/bin/env bash
# The documentation gate list, stated ONCE.
#
# `.github/workflows/docs-build.yml` runs this script bare: one checkout, one
# editable install, so plain `python` is the tree under test and RUN is empty.
# `just docs-check` runs it with RUN="uv run --locked --extra docs": the per-tree
# uv .venv, whose editable .pth names THIS checkout. A second copy of this list in
# either caller is the drift class scripts/check_autodoc_coverage.py records at its
# head (two hand-maintained lists that silently diverged), so neither caller
# carries one. Never `uv run --active`: under a foreign VIRTUAL_ENV it re-points
# ANOTHER tree's editable install at this one.
#
# Each gate is announced on stdout before it runs, so a local log keeps the
# per-step legibility the Actions summary loses when four steps become one.
set -euo pipefail

cd "$(dirname "$0")/.."
RUN="${RUN:-}"

gate() {
  echo "== docs gate: $*"
  # RUN is deliberately unquoted: it is zero or more words, not one argument.
  # shellcheck disable=SC2086
  $RUN "$@"
}

# Strict build, including the htmlproofer internal-link check. The three gates
# below read the site this produces, so it runs first.
gate mkdocs build --strict

# Two independent assertions: every public symbol renders an anchor, AND
# every public symbol carries a docstring. The second was added after 25
# symbols, including the central `TRITONSWMM_analysis` class, were found
# rendering as bare names while this gate was green, because an anchor
# proves the symbol reached the page and nothing more.
gate python scripts/check_autodoc_coverage.py --site-dir site

# Placeholder leakage and bare `path:line` citations. Needs no built SITE,
# so it cannot false-red on a rendering issue, but it does need the docs
# extra, because it derives its population from `griffe` rather than from a
# model of what the renderer does. That is a deliberate trade: the model it
# replaced was wrong in three ways at once and each was silent. This gate
# must therefore run on an interpreter that carries the docs extra, and it is
# no longer runnable against a bare checkout.
gate python scripts/check_docs_content.py

# The two mirrors between gate A's DERIVED population and what the built
# site actually renders, plus the pinned public surface. Needs the SITE, so
# it is a third leaf rather than a clause inside the gate above; that gate's
# "needs no built SITE" property is the whole reason it cannot live there.
# Imports both gates and is imported by neither, so it inverts no import
# direction.
gate python scripts/check_published_surface.py --site-dir site
