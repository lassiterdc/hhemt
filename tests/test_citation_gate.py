"""Paired tests for the published-source-line gate and the citation ledger.

Each test names, in its own docstring, whether it DISCRIMINATES between the
pre-fix and post-fix states or is a REGRESSION GUARD that must be green in both.
Only the discriminating ones are evidence that the change works.

The fixture is synthetic and that is the whole point: pinning a live citation
would be green only until somebody repairs it, which turns a repair into a red
build -- the hazard `test_citations_in_docstrings_are_gated_not_advisory` already
names in the sibling module.

The five tests that need `citation_ledger` import it INSIDE the function body.
That is deliberate: under the pre-fix tree the module still COLLECTS and each test
reports its own verdict, instead of the whole module erroring at collection and
hiding the one failure that carries behavioural evidence.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "check_docs_content.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_docs_content", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_docs_content"] = mod
    spec.loader.exec_module(mod)
    return mod


#: Loading the gate is also what puts `scripts/` on `sys.path`, which is why the
#: deferred `import citation_ledger` below resolves at all.
cdc = _load()

_PROBE = '''"""Probe module docstring citing workflow.py:11 for the docstring limb."""

__all__ = ["probe_fn"]


def probe_fn():
    """Probe function docstring, carrying no citation."""
    # body comment citing workflow.py:22 for the span limb
    return 1
'''


@pytest.fixture
def probe(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "probe_mod.py").write_text(_PROBE, encoding="utf-8")
    api = tmp_path / "api.md"
    api.write_text("# API Reference\n\n::: probe_mod\n", encoding="utf-8")
    return src, api


def _citations(src, api):
    return {e for c, _p, _l, e in cdc.scan_rendered_docstrings(src, api) if c == "bare-line-citation"}


def test_body_comments_in_published_members_are_scanned(probe):
    """DISCRIMINATING. Pre-fix this FAILS: the population is docstring VALUES, and a
    body comment is in no docstring, so `workflow.py:22` is invisible while the page
    publishes it inside the member's source block. Measured pre-fix red:
    `assert 'workflow.py:22' in {'workflow.py:11'}`."""
    assert "workflow.py:22" in _citations(*probe), (
        "a bare citation in a published member's BODY is unscanned; the gate is "
        "covering the renderer's member set rather than its output"
    )


def test_module_docstrings_stay_in_the_population(probe):
    """REGRESSION GUARD, green in BOTH states by design. A `griffe` Module carries no
    `lineno`/`endlineno`, so a span-only widening drops every module docstring -- 116
    lines across 7 files on the live corpus. This goes red if the population is ever
    narrowed to the span limb alone."""
    assert "workflow.py:11" in _citations(*probe), (
        "a module-docstring citation left the population; the span limb has "
        "replaced the docstring limb instead of joining it"
    )


def test_published_population_is_a_strict_superset_of_the_docstring_population(probe):
    """REGRESSION GUARD. States the post-condition as a superset assertion rather than
    as a size comparison, because a widening described by the attribute it ADDS can
    subtract wherever that attribute is absent on some members."""
    src, api = probe
    doc_only = {
        (h, dl + ln - 1)
        for _q, h, dl, d in cdc.rendered_docstrings(src, api)
        for _c, _p, ln, _e in cdc._binary_findings(h, d)
    }
    scanned = {(p, ln) for _c, p, ln, _e in cdc.scan_rendered_docstrings(src, api)}
    assert doc_only <= scanned, f"the widened population dropped {sorted(doc_only - scanned)}"


def test_repair_is_silent_and_a_dissolved_address_is_a_finding(tmp_path):
    """DISCRIMINATING on behaviour, and the one that carries the ratchet.

    All three pinned entries below are absent from `live` -- structurally identical
    inputs. A symmetric baseline returns all three as findings and this test goes red;
    that is exactly what copying `check_published_surface.baseline_findings` would
    ship, and it would fail the gate on every successful repair.
    """
    import citation_ledger

    (tmp_path / "live.py").write_text("x = 1\n", encoding="utf-8")
    repaired = citation_ledger.key("live.py", "workflow.py:6684", "pkg.Cls.meth")
    orphaned = citation_ledger.key("live.py", "workflow.py:6684", "pkg.Cls.renamed_away")
    gone_file = citation_ledger.key("deleted.py", "workflow.py:1", "pkg.Cls.meth")

    added, retired, orphans = citation_ledger.classify(
        live=set(),
        pinned={repaired, orphaned, gone_file},
        repo_root=tmp_path,
        live_qualnames={"pkg.Cls.meth"},
    )
    assert added == set(), "nothing was added; the ratchet must not invent an addition"
    assert retired == {repaired}, "a repair whose file and member still resolve must be SILENT"
    assert orphans == {orphaned, gone_file}, "a dissolved address must be a finding, not a silent drop"


def test_a_new_citation_fails_while_a_pinned_one_does_not(tmp_path):
    """DISCRIMINATING on behaviour. The pre-seed arithmetic: seeding the WHOLE live set
    must leave `added` empty, and one unpinned token must not be swallowed by its
    pinned neighbours."""
    import citation_ledger

    pinned_tok = citation_ledger.key("a.py", "workflow.py:10", "pkg.f")
    new_tok = citation_ledger.key("a.py", "workflow.py:99", "pkg.f")

    added, _r, orphans = citation_ledger.classify(
        live={pinned_tok}, pinned={pinned_tok}, repo_root=tmp_path, live_qualnames={"pkg.f"}
    )
    assert added == set() and orphans == set(), "a fully seeded ledger must go green"

    added, _r, _o = citation_ledger.classify(
        live={pinned_tok, new_tok}, pinned={pinned_tok}, repo_root=tmp_path, live_qualnames={"pkg.f"}
    )
    assert added == {new_tok}, "a new citation beside a pinned one must still fail the gate"


def test_ledger_normalizes_whitespace_around_the_colon(tmp_path):
    """DISCRIMINATING on behaviour. `LINE_CITATION` tolerates `workflow.py : 6684` and
    keeps the spaces in `m.group(0)`, so without normalization a reformat of a cited
    line reads as one retirement plus one addition."""
    import citation_ledger

    assert citation_ledger.key("a.py", "workflow.py : 6684", "pkg.f") == citation_ledger.key(
        "a.py", "workflow.py:6684", "pkg.f"
    )


def test_the_seeding_path_is_callable_with_the_signature_the_gate_declares():
    """DISCRIMINATING on behaviour, and the one test the landing procedure depends on.

    `--write-citation-ledger` is the acknowledgement gesture: without it there is no
    route from the red landing commit to green. It is also the only entry point in
    this change that no other test calls, which is how an ARITY mismatch between the
    seeding call site and `_live_citation_keys`' signature survived every anchor
    check -- `TypeError: missing 1 required positional argument: 'spans'`, raised
    ABOVE `try: return _run(args)` and therefore escaping as a raw traceback rather
    than any documented exit code.

    Asserting on the SIGNATURE rather than on a full run keeps this fast and keeps
    it honest: the property is that the gate's own call site and the function it
    calls agree, which is true or false independently of any corpus.
    """
    import ast
    import inspect

    import citation_ledger  # noqa: F401  -- the seeding path imports it too

    params = list(inspect.signature(cdc._live_citation_keys).parameters)
    assert params == ["published", "repo_root", "spans"], (
        f"_live_citation_keys takes {params}; the seeding path in `main()` passes "
        f"(published, repo_root, spans) and any divergence makes the ledger unseedable"
    )
    # Count the call's arguments on the AST NODE, never on the source text. A
    # textual recovery -- split on `_live_citation_keys(`, then on the first `)` --
    # is closed by the first NESTED call's paren, and this call site's first
    # argument IS a call. That assertion reddens on correct code and its message
    # tells a maintainer to change it, which is worse than having no test.
    tree = ast.parse(inspect.getsource(cdc.main))
    calls = [
        n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_live_citation_keys"
    ]
    assert calls, "the seeding path no longer calls the keying helper"
    for call in calls:
        assert len(call.args) == 3, (
            f"the seeding call site passes {len(call.args)} positional argument(s) where "
            f"`_live_citation_keys` declares 3 -- the shape that made the acknowledgement "
            f"gesture unrunnable, raised ABOVE `try: return _run(args)` and so escaping as a "
            f"raw traceback rather than any documented exit code"
        )


def test_a_missing_ledger_is_empty_rather_than_a_skip(tmp_path):
    """DISCRIMINATING on behaviour. An absent ledger must make every live citation an
    ADDITION -- the loud state on the landing commit -- never a silent pass. A skip
    branch would make an unseeded ledger byte-identical to a clean one."""
    import citation_ledger

    assert citation_ledger.parse(tmp_path / "absent.tsv") == set()
    live = {citation_ledger.key("a.py", "workflow.py:10", "pkg.f")}
    added, _r, _o = citation_ledger.classify(
        live=live,
        pinned=citation_ledger.parse(tmp_path / "absent.tsv"),
        repo_root=tmp_path,
        live_qualnames={"pkg.f"},
    )
    assert added == live, "an absent ledger tolerated a citation instead of failing on it"
