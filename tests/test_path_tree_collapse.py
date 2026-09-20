"""Guards for the `_path_tree_html` sibling-run collapse (Iteration 11, item 14).

The collapse renders ONE exemplar in place of N structurally identical siblings. It is
correct only while the siblings really ARE identical, and a wrong collapse is INVISIBLE
in the output -- the page shows a confident sentinel and nothing contradicts it. These
tests make that a CI failure instead of a reading error.
"""

from __future__ import annotations

from hhemt.report_renderers import metadata

_SENTINEL_MARK = "×"  # the multiplier glyph in `{stem…} × N`


def _relpaths(member_ids: list[str]) -> list[str]:
    """The shape `sensitivity_analysis.py` writes into the master crate's hasPart."""
    return ["sensitivity_datatree.zarr"] + [f"members/member_{i}/analysis_datatree.zarr" for i in member_ids]


def _tree(html: str) -> str:
    """Just the `<pre>` tree, without the `<details>` roster that follows it.

    The roster REPEATS each sentinel label by design, so a sentinel count taken over the
    whole document is always tree-count plus roster-count. These tests are about the
    shape of the tree, so they must look at the tree.
    """
    return html.split("</pre>")[0]


def test_identical_sibling_run_collapses_to_one_exemplar():
    html = metadata._path_tree_html(_relpaths([f"gpu_{n}_r1" for n in range(28)]))
    assert _tree(html).count(_SENTINEL_MARK) == 1, "expected exactly one sentinel line for one run of 28"
    assert "× 28" in _tree(html)
    # The whole point of the collapse: the shared structure is rendered ONCE.
    assert _tree(html).count("analysis_datatree.zarr") == 1
    # And the roster must still carry every collapsed name -- the collapse hides noise,
    # never information that was in the crate.
    for n in range(28):
        assert f"member_gpu_{n}_r1" in html


def test_a_divergent_sibling_breaks_the_run_and_renders_under_its_own_name():
    """The detector, not the writer's naming convention, decides identity.

    27 siblings share `analysis_datatree.zarr`; one carries a different child. The
    divergent member must survive as its own line -- if it were folded into the
    exemplar, the page would assert a structure that member does not have.
    """
    paths = _relpaths([f"gpu_{n}_r1" for n in range(27)])
    paths.append("members/member_odd_one/partial_datatree.zarr")
    html = metadata._path_tree_html(paths)
    assert "member_odd_one" in _tree(html)
    assert "partial_datatree.zarr" in _tree(html)
    assert _tree(html).count(_SENTINEL_MARK) == 1, "the 27 identical siblings collapse; the outlier does not"
    assert "× 27" in _tree(html)


def test_a_leaf_run_collapses_when_it_is_an_index_family():
    """Reverses `test_a_run_of_leaves_is_never_collapsed`.

    That test's rationale was true of the code it guarded: no roster was reachable for a
    leaf run, because `collapsed.append` sat inside the branch the `child` conjunct gated.
    Lifting the conjunct supplies the roster in the same edit, so the names survive the
    collapse -- which is asserted here rather than assumed.
    """
    html = metadata._path_tree_html([f"flat_{n}.json" for n in range(5)])
    assert _SENTINEL_MARK in _tree(html), "an index family of leaves must collapse to one sentinel"
    assert "{flat_{i}.json}" in _tree(html), "the label must name the varying index, not a bare prefix"
    assert "<details>" in html, "a collapsed leaf run must still disclose every name"
    for n in range(5):
        assert f"flat_{n}.json" in html, "no name may be lost by the collapse"


def test_a_heterogeneous_leaf_run_stays_expanded():
    """The differently-positioned satisfying arm: NOT an index family, so no collapse.

    These names share no skeleton -- the varying token is not a digit run -- so the
    predicate must refuse them. This is the arm that catches an over-collapse, which is
    the failure the old `child` conjunct used to prevent for free: under the pre-change
    signature every leaf compared equal to every other, so five unrelated files formed
    one run of five and would collapse behind an empty-stemmed sentinel.
    """
    names = ["alpha.json", "bravo.json", "charlie.json", "delta.json", "echo.json"]
    html = metadata._path_tree_html(names)
    assert _SENTINEL_MARK not in _tree(html), "unrelated leaves must not collapse"
    for n in names:
        assert n in html


def test_two_identical_siblings_stay_expanded():
    """Below the threshold the collapse costs a line rather than saving one."""
    html = metadata._path_tree_html(_relpaths(["gpu_0_r1", "gpu_1_r1"]))
    assert _SENTINEL_MARK not in html
    assert "member_gpu_0_r1" in html
    assert "member_gpu_1_r1" in html


def _assert_declared_attrs_are_fields():
    """Every name `_resolve_consolidated_tree` looks up reflectively is a real field.

    Extracted from its test node so the rename demonstration below can assert that THIS
    raises. Both components of that demonstration are ordinary in this suite; a test
    function calling another test function is not, and an unprecedented shape invites a
    cleanup whose most natural form would inline this body and silently remove the check.

    Pure introspection: no store, no analysis, no I/O, no solver.
    """
    import dataclasses

    # IMPORT POSITION IS LOAD-BEARING -- do not hoist to module scope. The name is
    # resolved from `hhemt.paths` at CALL time, which is what lets
    # test_metadata_reflective_guard_fires_under_a_rename patch the module attribute and
    # observe this raise. A module-scope binding is resolved once at import and the patch
    # would not reach it: measured, this helper then does NOT raise under the patch and
    # that test goes red. The comment explains; the test is what stops you.
    from hhemt.paths import AnalysisPaths

    declared = {f.name for f in dataclasses.fields(AnalysisPaths)}
    missing = [a for a in metadata._DECLARED_TREE_PATH_ATTRS if a not in declared]
    assert not missing, (
        f"_resolve_consolidated_tree looks up {missing} on AnalysisPaths, which no longer "
        f"declares them. getattr's default will mask this: the lookup yields None and the "
        f"ROOT_TREE_NAMES fallback returns a path anyway, so nothing fails. Update "
        f"metadata._DECLARED_TREE_PATH_ATTRS to the new field name(s)."
    )


def test_metadata_reflective_attr_names_are_real_fields():
    """The shipped guard. Delegates to the helper so the rename check has a subject.

    Guards a MASKED failure rather than a loud one: `_resolve_consolidated_tree` reads
    `AnalysisPaths` attributes by string via `getattr(..., None)`, so a renamed or split
    field yields None, the `is not None` filter drops it, and the ROOT_TREE_NAMES fallback
    answers anyway -- identically to the healthy call when the declared path equals the
    fallback, and differently or as None otherwise. Nothing raises in any of those.
    """
    _assert_declared_attrs_are_fields()


def test_metadata_reflective_tuple_resolves_sensitivity_first():
    """The tuple's FIRST entry is the precedence, and it is pinned by VALUE.

    Three clauses, none subsuming the others -- measured over four states:

      * tuple inverted, fields unchanged        -> value clause RED, order clause RED
      * BOTH tuple and fields reordered         -> value clause RED, order clause GREEN
      * a field renamed or split                -> value clause GREEN, set clause RED

    SINGLE-POINT COVERAGE, disclosed because nothing else in this file says it: a field
    SPLIT is caught by the SET clause ALONE. Both order clauses pass, and the helper's own
    set-guard passes too, because both original names still exist alongside the new one.
    The set clause reads as a weaker restatement of the helper and is therefore the clause
    a simplifier deletes first -- and it is the sole cover for the split half of the
    hazard `_DECLARED_TREE_PATH_ATTRS`'s own comment names.
    """
    import dataclasses

    from hhemt.paths import AnalysisPaths

    assert metadata._DECLARED_TREE_PATH_ATTRS[0] == "sensitivity_datatree_zarr", (
        f"the reflective lookup now resolves {metadata._DECLARED_TREE_PATH_ATTRS[0]!r} FIRST. "
        f"`_resolve_consolidated_tree`'s docstring declares the sensitivity-master name is "
        f"tried first, mirroring `_combine_merge._resolve_root_tree`. Either the precedence "
        f"was inverted by accident, or it changed deliberately and both the docstring and "
        f"this test must change with it."
    )

    field_order = [f.name for f in dataclasses.fields(AnalysisPaths) if f.name.endswith("_datatree_zarr")]
    assert set(metadata._DECLARED_TREE_PATH_ATTRS) == set(field_order), (
        "the reflective tuple and the dataclass no longer name the same fields"
    )
    assert list(metadata._DECLARED_TREE_PATH_ATTRS) != field_order, (
        "the reflective tuple now matches AnalysisPaths declaration order. That is the "
        "signature of a comprehension having replaced the hand-ordered constant. The tuple "
        "must stay sensitivity-FIRST while the dataclass declares analysis-first; if the "
        "dataclass was deliberately reordered instead, retire this clause explicitly rather "
        "than letting it pass by coincidence."
    )


def test_metadata_reflective_guard_fires_under_a_rename(monkeypatch):
    """The field-contract guard actually RAISES when a field is renamed.

    Without this the guard is only known to be GREEN on a healthy tree, and a guard never
    observed failing is indistinguishable from one written backwards.

    It also pins the helper's function-local import: the patch below replaces the
    attribute on `hhemt.paths`, a call-time import sees the replacement and a module-scope
    one does not, so hoisting turns this red.
    """
    import dataclasses

    import pytest

    renamed = dataclasses.make_dataclass(
        "AnalysisPaths",
        [
            ("analysis_tree_zarr", object, dataclasses.field(default=None)),
            ("sensitivity_datatree_zarr", object, dataclasses.field(default=None)),
        ],
    )
    monkeypatch.setattr("hhemt.paths.AnalysisPaths", renamed)
    with pytest.raises(AssertionError, match="analysis_datatree_zarr"):
        _assert_declared_attrs_are_fields()
