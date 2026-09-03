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
