"""The crate/deposit relpath DEFAULTS must agree with the regular-analysis producer.

`emit_provenance` and `publish_analysis` both default `consolidated_zarr_relpath`, and
the only callers reaching either default are the REGULAR-analysis paths
(`processing_analysis.py`, `analysis.py`). The sensitivity paths pass the value
explicitly and are unaffected.

SCOPE OF THE CLAIM, stated because the unqualified form is FALSE. What the defaults must
match is what the LIVE PRODUCER writes -- `analysis.py`'s own root-store literal. On a
tree ALREADY MIGRATED by V0021 the on-disk store is `experiment_datatree.zarr` and these
defaults do not describe it; that divergence is a known consequence of the regular
producer not yet being renamed, it is NOT what this guard measures, and this guard must
not be read as asserting the defaults are correct for a migrated tree.

WHAT THIS GUARD DOES NOT COVER, so absence of a failure here is not evidence of health:
it says nothing about which files a deposit actually contains. Deposit completeness is a
separate property with a separate remedy and no assertion here bears on it.

The authority is `hhemt.analysis` itself, read from source. It is deliberately NOT
`_combine_merge.CONSOLIDATED_TREE_NAME`: that constant is documented as what a BUNDLE
ships, a different fact that shares this value today, so pinning to it would make the
guard fail for a reason it does not name.

This is a DRIFT GUARD, not a repair check: it is green today and must stay green until
the regular producer itself is renamed, at which point all three move together. It exists
because a probe measured that NO suite observes these defaults -- every test that touches
the value supplies it explicitly -- so a one-sided change ships silently.

Fixture-free and solver-free: source inspection only, no `tmp_path`, no fixture, no
`compile_TRITON_SWMM` call.
"""

import ast
import inspect

import hhemt.analysis
from hhemt.provenance import emit_provenance
from hhemt.publishing import publish_analysis
from hhemt.utils import ROOT_TREE_NAMES


def _relpath_default(fn) -> str:
    return inspect.signature(fn).parameters["consolidated_zarr_relpath"].default


def _producer_root_store_name() -> str:
    """The root-store literal the REGULAR producer writes, read from its own module.

    Located by the assignment `analysis_paths_kwargs["analysis_datatree_zarr"] = ...`
    rather than by a line number, so it survives edits elsewhere in the file and fails
    loudly -- never silently -- if that assignment is restructured.
    """
    tree = ast.parse(inspect.getsource(hhemt.analysis))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "analysis_paths_kwargs"
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == "analysis_datatree_zarr"
        ):
            continue
        for sub in ast.walk(node.value):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and sub.value.endswith(".zarr"):
                return sub.value
    raise AssertionError(
        "could not locate the regular producer's root-store literal in hhemt.analysis; "
        "the assignment this guard keys on has been restructured and the guard needs updating"
    )


def test_provenance_and_publishing_defaults_agree() -> None:
    """One-sided drift between the two defaults is the failure this catches."""
    assert _relpath_default(emit_provenance) == _relpath_default(publish_analysis)


def test_the_defaults_match_what_the_live_producer_writes() -> None:
    """Both defaults must name the store a REGULAR analysis actually writes.

    Pinned to `hhemt.analysis`'s own literal, never to a bundle-side constant that
    happens to share the value.
    """
    producer = _producer_root_store_name()
    assert _relpath_default(emit_provenance) == producer
    assert _relpath_default(publish_analysis) == producer


def test_the_defaults_name_an_accepted_root_store() -> None:
    """A typo would satisfy the two assertions above if it were made in all three places."""
    assert _relpath_default(emit_provenance) in ROOT_TREE_NAMES
