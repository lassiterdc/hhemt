"""Every caller of a dry-run-aware tree-hygiene helper must forward the caller's own dry-run flag.

Problem this file solves: ``TRITONSWMM_analysis._prune_settled_markers`` takes ``dry_run`` with a default
of ``False``, so a call that simply omits the keyword deletes settled ``_status`` markers even when the
caller is a dry run. ``run()`` and ``reprocess()`` did exactly that, and a dry run (``hhemt run-experiment
--dry-run``) deleted files from the analysis tree. The behaviour of each helper is covered in
``test_stale_metadata_cleanup_dry_run.py``; that file calls the helpers directly and so cannot see whether
``run()`` / ``reprocess()`` hand them the flag. Exercising ``run()`` for that needs a compiled solver
fixture, so this file reads the wiring from source instead.

Not established: this proves the flag is FORWARDED at every call site, never that the helper behind it
honours the flag (the sibling file does that), and the helper-name set below is closed, so a new
tree-deleting helper added later is not covered until its name is added.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import hhemt

_DRY_RUN_AWARE_HELPERS = frozenset(
    {
        "_prune_settled_markers",
        "_run_settled_marker_prune_gate",
        "_run_stale_metadata_cleanup_gate",
    }
)


def _dry_run_blind_calls(source: str) -> list[str]:
    """Return ``attr@line`` for each call of a dry-run-aware helper that does not forward a flag.

    A call is blind when it carries no ``dry_run=`` keyword, splats ``**kwargs`` (the flag cannot be
    seen), or passes a literal constant (``dry_run=False`` is the omitted-keyword bug spelled out).
    """
    blind: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _DRY_RUN_AWARE_HELPERS
        ):
            continue
        kws = {k.arg: k.value for k in node.keywords}
        splat = any(k.arg is None for k in node.keywords)
        value = kws.get("dry_run")
        if splat or value is None or isinstance(value, ast.Constant):
            blind.append(f"{node.func.attr}@{node.lineno}")
    return blind


def _call_sites(source: str) -> int:
    """Count calls of the dry-run-aware helpers in ``source`` (the control's denominator)."""
    return sum(
        1
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in _DRY_RUN_AWARE_HELPERS
    )


def _src_files() -> list[Path]:
    return sorted(Path(hhemt.__file__).parent.rglob("*.py"))


@pytest.mark.parametrize(
    "snippet",
    [
        "a._prune_settled_markers()",
        "a._prune_settled_markers(dry_run=False)",
        "a._run_settled_marker_prune_gate(prune_settled_markers=x, **opts)",
        "a._run_stale_metadata_cleanup_gate(cleanup_stale_metadata=x, from_scratch=y, verbose=z)",
    ],
)
def test_matcher_flags_each_shape_of_blind_call(snippet):
    """Property: the matcher reports a call that omits, splats away, or hardcodes the dry-run flag.

    Class: the four blind-call shapes; the control that the matcher can return non-zero at all.
    Why required: a matcher that never fires makes the repository-wide test below vacuously green.
    What kills it: weakening ``_dry_run_blind_calls`` to inspect only the keyword's presence.
    Second correct implementation: a regex over source text that flags the same four shapes.
    """
    assert len(_dry_run_blind_calls(snippet)) == 1


@pytest.mark.parametrize(
    "snippet",
    [
        "a._prune_settled_markers(dry_run=dry_run)",
        "a._prune_settled_markers(dry_run=self._dry)",
        "a._run_settled_marker_prune_gate(prune_settled_markers=x, dry_run=flag, verbose=v)",
        "a._run_stale_metadata_cleanup_gate(cleanup_stale_metadata=x, from_scratch=y, dry_run=not live, verbose=z)",
    ],
)
def test_matcher_accepts_differently_named_forwarded_flags(snippet):
    """Property: the matcher accepts any non-constant expression bound to ``dry_run``.

    Class: forwarded flags under a different local name or expression than ``dry_run`` itself.
    Why required: pinning the local variable's NAME would reject a correct refactor of the caller.
    What kills it: tightening the matcher to require ``ast.Name`` with id ``dry_run``.
    Second correct implementation: any caller that computes its flag under another name.
    """
    assert _dry_run_blind_calls(snippet) == []


def test_every_dry_run_aware_helper_call_in_src_forwards_the_flag():
    """Property: no call of a dry-run-aware helper anywhere under ``src/hhemt`` omits or hardcodes the flag.

    Class: every ``Call`` of the three helper names in every module of the installed package; the
    population is discovered by glob, so a new caller in any module is covered without an edit here.
    Why required: ``_prune_settled_markers`` defaults ``dry_run`` to False, so one omitted keyword at
    one caller makes a dry run delete settled ``_status`` markers (the defect this guards).
    What kills it: restoring ``self._prune_settled_markers()`` in ``run()`` or ``reprocess()``, or
    calling it with ``dry_run=False``.
    Second correct implementation: ``run()`` and ``reprocess()`` call ``_prune_settled_markers``
    directly with ``dry_run=dry_run`` instead of through ``_run_settled_marker_prune_gate``.
    """
    offenders = {}
    for path in _src_files():
        blind = _dry_run_blind_calls(path.read_text(encoding="utf-8"))
        if blind:
            offenders[str(path)] = blind
    assert offenders == {}


def test_the_population_the_property_ranges_over_is_not_empty():
    """Property: the repository-wide scan actually sees callers, including the two hygiene call sites.

    Class: the census control. A scan over zero call sites is green whatever the code does.
    Why required: the glob resolves relative to the imported package, so a wrong root reads nothing.
    What kills it: pointing ``_src_files`` at a directory that holds no ``analysis.py``.
    Second correct implementation: counting call sites with a regex instead of ``ast``.
    """
    files = _src_files()
    assert any(p.name == "analysis.py" for p in files)
    analysis = next(p for p in files if p.name == "analysis.py")
    assert _call_sites(analysis.read_text(encoding="utf-8")) >= 2
