"""suite/ -- the toolkit's own pytest suite, run as a cluster campaign.

Three modules, promoted from the private estate so that the couplings between
them and `tests/` sit in ONE repo where the toolkit's own gates can watch them:

- `partition.py`  derives chunk membership from the fixture-consumer graph. It
  reads `tests/` as DATA (an AST walk over paths), never as an import, so the
  strict `tests -> src` direction is preserved. Its `HEAVY_FIXTURES` tuple is a
  hand-maintained mirror of names in `tests/conftest.py` and drifts silently --
  that mirror is exactly what co-location makes testable.
- `aggregate.py`  cross-chunk verdict and scope-bearing summary.
- `_runner.py`    drive / chunk / triage entry points, and a pytest plugin half.

The CLI front is `suite/_cli.py`, mounted at `hhemt test toolkit`.
"""
