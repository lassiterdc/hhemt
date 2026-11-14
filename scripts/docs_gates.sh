#!/usr/bin/env bash
# The documentation gate list, stated ONCE.
#
# `.github/workflows/docs-build.yml` runs this script bare: one checkout, one
# editable install, so plain `python` is the tree under test and RUN is empty.
# `just docs-check` runs it with RUN="scripts/hook_env.sh --python 3.12": the shared
# environment that wrapper keeps outside every tree, with THIS checkout's `src` first on
# the import path. A second copy of this list in
# either caller is the drift class scripts/check_autodoc_coverage.py records at its
# head (two hand-maintained lists that silently diverged), so neither caller
# carries one. Never `uv run --active`: under a foreign VIRTUAL_ENV it re-points
# ANOTHER tree's editable install at this one.
#
# Each gate is announced on stdout before it runs, so a local log keeps the
# per-step legibility the Actions summary loses when five steps become one.
set -euo pipefail

cd "$(dirname "$0")/.."
RUN="${RUN:-}"

gate() {
  echo "== docs gate: $*"
  # RUN is deliberately unquoted: it is zero or more words, not one argument.
  # shellcheck disable=SC2086
  $RUN "$@"
}

# Strict build, including the htmlproofer internal-link check. Two of the three
# gates below read the site this produces, so it runs first.
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

# Docstring dialect: a rendered public docstring written in a section style the configured
# numpy handler does not parse still passes the two gates above and renders its parameters as
# run-on prose. Needs no built site. Wired here and in the list in docs-build.yml in one landing.
gate python scripts/check_docstring_dialect.py

# The two mirrors between gate A's DERIVED population and what the built
# site actually renders, plus the pinned public surface. Needs the SITE, so
# it is a third leaf rather than a clause inside the gate above; that gate's
# "needs no built SITE" property is the whole reason it cannot live there.
# Imports both gates and is imported by neither, so it inverts no import
# direction.
gate python scripts/check_published_surface.py --site-dir site

# Wording of the published pages: the Vale rules of `.vale/styles/`, which every page must satisfy with no
# alert. Needs the `vale` executable on PATH; the CI job installs it.
gate python scripts/check_vale.py

# A green run certifies less than six gate names suggest, and the Actions page shows only
# the green. State what the run does NOT reach on the stream a reader of that page sees, so
# nobody reads this exit as "the docs are verified". Each line is a fact about the gate it
# names, as that gate's own source or output states it: edit the line when the gate moves.
# (No line begins with the word the gate function is named for, so the list of gates that
# tests/docs/test_retrofit_code_facts.py reads from this file is unchanged.)
cat <<'NOTES'
== docs gates passed. What this run does not certify:
   - strict build: links between pages only; external URLs are not fetched (validate_external_urls: false in mkdocs.yml).
   - autodoc coverage: symbols of the modules the API reference pages name. A module dropped from those pages is caught by the published-surface baseline, not here.
   - docstring dialect: that every rendered public docstring parses as numpy sections. It grades no wording and no accuracy.
   - docs content: placeholders, bare line citations and banned wording in published pages and the API source lines they render. Files its output lists as personal-voice or repo-internal are advisory and are not gated.
   - published surface: that the docs content gate's derived population matches what the built site renders, and that the surface has not widened past the pinned baseline. It grades no docstring text.
   - Vale: the wording of tracked Markdown under docs/ and README.md, never its accuracy. The generated configuration schema pages are untracked, so Vale does not see them; hooks/config_reference.py applies the content patterns to them as it writes them.
NOTES
