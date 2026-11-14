"""Reviewer tests for chunk 1q.2: the `AnalysisPaths` attribute is now `experiment_datatree_zarr`.

The rename is a token substitution over 24 files, and its dangerous half is not the substitution but the readers
that fetch the attribute by NAME with a default: `getattr(paths, "<name>", None)` and a tuple of names walked with
`getattr(..., None)` both turn a stale or mistyped name into `None`, and every one of those readers treats `None`
as "this scope has not consolidated yet" and carries on. Nothing raises. The post-condition grep proves the retired
spelling is gone; it cannot prove that every name still read is a real field, and it cannot prove that each
default-`None` reader still FINDS a store when handed one under the new name.

These tests need no solver, no compile and no analysis tree: the first group reads source statically (AST), the
second drives three readers with `SimpleNamespace` stand-ins over tiny zarr stores written into `tmp_path`.

The retired underscore spelling is built from two halves below so that no line of this file spells it, because the
chunk's Definition-of-Done grep over `src tests` must print 0 and a literal here would be a line it counts.

Not established: that a live sensitivity analysis on Rivanna consolidates, resumes and reports under the new name;
the Rivanna-only `tests/test_synth_05_sensitivity_analysis_with_snakemake.py` is the one that exercises it.
"""

from __future__ import annotations

import ast
import dataclasses
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hhemt import recompute
from hhemt.analysis_validation import check_consolidated_metadata
from hhemt.paths import AnalysisPaths, SysPaths
from hhemt.report_renderers.metadata import _resolve_consolidated_tree
from hhemt.utils import append_zarr, write_zarr

_REPO = Path(__file__).resolve().parents[1]
_OLD = "sensitivity" + "_datatree_zarr"
# A stale spelling unrelated to the rename, also built from halves: the live-tree scan below reads this very file, and
# a whole-token literal here is a token it would (correctly) report.
_MISTYPE = "experiments" + "_datatree_zarr"

# A path-attribute-shaped token: lowercase words ending `_datatree_zarr`. The leading-letter rule leaves out the bare
# suffix `_datatree_zarr` that a `name.endswith(...)` filter spells.
_TOKEN = re.compile(r"^[a-z][a-z0-9_]*_datatree_zarr$")


def _tokens_in(source: str) -> list[tuple[str, str]]:
    """Return (token, position) for every path-attribute-shaped token in a string constant, keyword or attribute.

    A bare `Name`, a `def` and an import are left out on purpose: they name functions and local variables
    (`write_datatree_zarr`), which no `getattr(paths, ...)` can reach.
    """
    found: list[tuple[str, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and _TOKEN.match(node.value):
            found.append((node.value, "string"))
        elif isinstance(node, ast.Attribute) and _TOKEN.match(node.attr):
            found.append((node.attr, "attribute"))
        elif isinstance(node, ast.keyword) and node.arg and _TOKEN.match(node.arg):
            found.append((node.arg, "keyword"))
    return found


def _master_store_field() -> str:
    """The one `AnalysisPaths` field, besides `analysis_datatree_zarr`, that names a consolidated store.

    Derived from the dataclass rather than spelled, so the reader tests below turn on "the reader finds a store
    declared under the field the dataclass declares", which holds on the tree before the rename (old field, old
    readers) and after it (new field, new readers), and reds only when a reader and the field disagree.
    """
    others = [
        f.name
        for f in dataclasses.fields(AnalysisPaths)
        if f.name.endswith("_datatree_zarr") and f.name != "analysis_datatree_zarr"
    ]
    assert len(others) == 1, others
    return others[0]


def _real_names() -> set[str]:
    """Every name a reader may legitimately spell: a field of a paths dataclass, or a callable in `hhemt.utils`."""
    from hhemt import utils

    fields = {f.name for cls in (AnalysisPaths, SysPaths) for f in dataclasses.fields(cls)}
    return fields | {n for n in dir(utils) if _TOKEN.match(n) and callable(getattr(utils, n))}


def _stale(source: str) -> list[tuple[str, str]]:
    """Return the (token, position) pairs in `source` that name neither a paths field nor a utils callable."""
    real = _real_names()
    return [(tok, pos) for tok, pos in _tokens_in(source) if tok not in real]


# Four positions a reader can spell the attribute in. The first three are the ones the chunk's four hand-edited
# sites and its test stand-ins actually use; the fourth is a tuple member, the shape `_DECLARED_TREE_PATH_ATTRS` has.
_POSITION_SOURCES = {
    "getattr string": 'x = getattr(paths, "{n}", None)\n',
    "tuple member": 'ATTRS = ("{n}", "analysis_datatree_zarr")\n',
    "keyword": "paths = SimpleNamespace({n}=None)\n",
    "attribute": "p = analysis.analysis_paths.{n}\n",
}


@pytest.mark.parametrize("position", sorted(_POSITION_SOURCES))
@pytest.mark.parametrize("stale_name", [_OLD, _MISTYPE], ids=["retired-name", "mistyped-name"])
def test_the_scanner_flags_a_stale_name_in_every_position(position, stale_name):
    """Property: a stale path-attribute name is reported whichever of the four positions spells it.

    Class: NEW CAPABILITY -- every (position, stale spelling) pair; two spellings, the retired name and an unrelated
    mistype, because a scanner keyed on the retired name alone would pass the first and miss every other stale name.
    Plausible wrong implementation: a scanner that only looks at string constants (the `getattr` position) or only at
    the retired literal; red under it for the three other positions and for the mistype. Why required: the chunk's
    claim that no reader still spells a name that is not a field is only as strong as this scanner. A second correct
    implementation: matching on the token's text through `tokenize` rather than the AST, which reports the same
    pairs.
    """
    source = _POSITION_SOURCES[position].format(n=stale_name)
    assert [tok for tok, _ in _stale(source)] == [stale_name], (position, stale_name, _stale(source))


@pytest.mark.parametrize("position", sorted(_POSITION_SOURCES))
@pytest.mark.parametrize(
    "real_name",
    [_master_store_field(), "analysis_datatree_zarr", "system_datatree_zarr", "write_datatree_zarr"],
    ids=["new-field", "sibling-field", "system-field", "utils-callable"],
)
def test_the_scanner_passes_every_real_name_in_every_position(position, real_name):
    """Property: a real field of either paths dataclass, or a callable of `hhemt.utils`, is never reported.

    Class: NEW CAPABILITY -- the satisfying half of the differential, deliberately in states other than the one the
    chunk was built against: the sibling field, the `SysPaths` field and a utils function, none of which is the
    renamed field. Plausible wrong implementation: a scanner that allow-lists only the renamed field, or flags every
    token ending `_datatree_zarr`; it reds on three of the four names. Why required: an over-firing scanner would
    turn the live-tree test below red on the first correct addition of a field. A second correct implementation:
    deriving the real names from `typing.get_type_hints` instead of `dataclasses.fields`.
    """
    source = _POSITION_SOURCES[position].format(n=real_name)
    assert _stale(source) == [], (position, real_name)


def test_no_source_or_test_spells_a_path_attribute_that_is_not_a_field():
    """Property: every path-attribute-shaped token in `src/hhemt` and `tests` resolves to a real name.

    Class: every string constant, keyword and attribute of that shape in every `.py` file of the two trees (the
    population is read from the tree at run time, so a file added later is inside it). Why required: a name that is
    not a field makes `getattr(paths, name, None)` return `None` and the reader skip its scope silently, which is the
    failure `metadata._DECLARED_TREE_PATH_ATTRS`'s own comment measures; the chunk renamed 56 spellings by one sed and
    nothing but this reads whether any stale one remained in a position the grep cannot tell from prose. Must-fire
    control: the same scan sees the two real field names at least once each, so a matcher that finds nothing cannot
    pass it; the planted-source tests above show it fires. Kills it: re-introducing the retired spelling, or any
    other non-field spelling, at any of the three positions in any file. A second correct implementation: the same
    set of readers reading the names off `dataclasses.fields` instead of literals; it spells none and passes.
    """
    seen: set[str] = set()
    unparsed: list[str] = []
    stale: list[str] = []
    n_files = 0
    for root in (_REPO / "src" / "hhemt", _REPO / "tests"):
        for path in sorted(root.rglob("*.py")):
            n_files += 1
            try:
                found = _tokens_in(path.read_text())
            except (SyntaxError, UnicodeDecodeError):
                unparsed.append(str(path.relative_to(_REPO)))
                continue
            real = _real_names()
            for tok, pos in found:
                seen.add(tok)
                if tok not in real:
                    stale.append(f"{path.relative_to(_REPO)}: {tok} ({pos})")
    assert n_files > 300, f"the scan read {n_files} files; the population is not the repository"
    assert {_master_store_field(), "analysis_datatree_zarr"} <= seen, (
        f"control: the scan never saw the real field names: {sorted(seen)}"
    )
    assert unparsed == [], f"files the scan could not parse, so did not read: {unparsed}"
    assert stale == [], stale


def _write_tree_with_sha(store: Path, sha: str) -> None:
    """Write a one-node DataTree store whose root attrs carry only `hhemt_producing_sha`."""
    store.parent.mkdir(parents=True, exist_ok=True)
    tree = xr.DataTree()
    tree.attrs["hhemt_producing_sha"] = sha
    tree.to_zarr(store)


def _write_indexed_store(store: Path) -> None:
    """Write a two-group zarr store through the toolkit's writers, so it carries a consolidated index."""
    data = xr.Dataset({"a": (("t",), np.arange(4.0))}, coords={"t": np.arange(4)}, attrs={"k": 1})
    write_zarr(data, store, chunks=None, encoding={"a": {"chunks": (2,)}})
    append_zarr(data, store, group="g")


def _drive_scope_stamps(tmp_path: Path, attr: str) -> bool:
    """True when `recompute._iter_scope_stamps` yields the master scope's stamp from a store bound as `attr`."""
    store = tmp_path / "relocated" / "tree.zarr"
    _write_tree_with_sha(store, "abc123def456")
    analysis = SimpleNamespace(
        cfg_analysis=SimpleNamespace(toggle_sensitivity_analysis=True),
        analysis_paths=SimpleNamespace(**{attr: store}),
    )
    stamps = list(recompute._iter_scope_stamps(analysis))
    return stamps == [("sensitivity-master", {"sha": "abc123def456", "semver": None})]


def _drive_resolve_tree(tmp_path: Path, attr: str) -> bool:
    """True when `_resolve_consolidated_tree` returns the relocated store bound as `attr`, not the fallback."""
    store = tmp_path / "relocated" / "tree.zarr"
    _write_tree_with_sha(store, "abc123def456")
    analysis_dir = tmp_path / "analysis_dir"
    analysis_dir.mkdir()
    analysis = SimpleNamespace(
        cfg_analysis=SimpleNamespace(toggle_sensitivity_analysis=True),
        analysis_paths=SimpleNamespace(**{attr: store}),
    )
    return _resolve_consolidated_tree(analysis_dir, analysis) == store


def _drive_consolidated_metadata(tmp_path: Path, attr: str) -> bool:
    """True when `check_consolidated_metadata` finds and passes a store bound only as `attr` (applicable and passed)."""
    store = tmp_path / "relocated" / "indexed.zarr"
    store.parent.mkdir(parents=True)
    _write_indexed_store(store)
    paths = SimpleNamespace(analysis_dir=tmp_path, analysis_datatree_zarr=None, **{attr: store})
    analysis = SimpleNamespace(
        cfg_analysis=SimpleNamespace(toggle_sensitivity_analysis=False), sensitivity=None, analysis_paths=paths
    )
    result = check_consolidated_metadata(analysis)
    return bool(result.passed and result.applicable)


_READERS = {
    "recompute._iter_scope_stamps": _drive_scope_stamps,
    "metadata._resolve_consolidated_tree": _drive_resolve_tree,
    "analysis_validation.check_consolidated_metadata": _drive_consolidated_metadata,
}


@pytest.mark.parametrize("reader", sorted(_READERS))
def test_each_default_none_reader_finds_a_relocated_store_declared_under_the_field_name(reader, tmp_path):
    """Property: handed a stand-in whose ONLY store attribute is the dataclass field, the reader finds the store.

    Class: the three readers that fetch the store attribute by name with a `None` default and have no other test
    that enters them with a store bound to that name -- `recompute._iter_scope_stamps` (sensitivity arm; no test
    enters the function), `metadata._resolve_consolidated_tree` (declared-path arm; the existing callers pass
    `analysis=None`) and `analysis_validation.check_consolidated_metadata` (the existing stub binds the other
    attribute and leaves this one `None`). The store sits in a RELOCATED directory, never at the analysis-dir
    fallback, so a wrong name cannot be rescued by the resolver's second candidate. The attribute is derived from the
    dataclass (`_master_store_field`), so the assertion turns on reader-versus-field agreement, a property of the tree
    before and after the rename alike, and not on the spelling the chunk introduces. Why required: with a stale name
    each reader returns "nothing consolidated yet" and nothing raises. Kills it: any reader whose literal differs
    from the field name; the readers then report no stamp, no store, or `applicable=False`. A second correct
    implementation: a reader that resolves the attribute off `dataclasses.fields(AnalysisPaths)` instead of a
    literal; the stand-in carries the field name, so it passes.
    """
    field = _master_store_field()
    assert _READERS[reader](tmp_path, field), f"{reader} did not find a store declared under the field {field}"


@pytest.mark.parametrize("reader", sorted(_READERS))
def test_no_reader_still_answers_to_the_retired_name(reader, tmp_path):
    """Property: handed a stand-in whose ONLY store attribute is the retired name, the reader finds nothing.

    Class: the same three readers, the other arm of the differential: it varies WHICH name carries the store and
    holds everything else fixed, so the test above cannot be satisfied by a reader that accepts any name. Why
    required: the shape renames the attribute and keeps no alias (clean cut, no compatibility shim); a reader that
    still reads the old name as a fallback would hide a producer that kept writing it. Kills it: a reader that
    tries the retired name after the new one. A second correct implementation: a reader that ignores unknown
    attributes entirely by reading only `dataclasses.fields`; it finds nothing and passes.
    """
    assert not _READERS[reader](tmp_path, _OLD), f"{reader} still found a store declared under the retired name"
