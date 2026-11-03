"""Tests for ``hhemt.utils.experiment_tree_summary`` and the full-tree opens it removes.

``experiment_tree_summary(store)`` returns a root consolidated store's attributes, its
``/member_*`` groups and its model tier from the store's metadata files, where the two
cross-experiment renderers (compatibility, disk utilization) used to call ``xr.open_datatree``,
which reads the ``zarr.json`` of every node. The maps renderer's loader is split so one open tree
serves every event of a child (``_open_experiment_tree`` and ``_load_subs_from_tree``).

Every test writes a small zarr store under ``tmp_path`` and uses only xarray, numpy and the code
under test, so none compiles or runs a solver and all run on a laptop. Run from a worktree as
``PYTHONPATH=${PWD}/src pytest tests/test_experiment_tree_summary.py``. Expected values come from
``xr.open_datatree`` read in the test (the oracle) or from literals, never from the function under test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from hhemt.eda._config_diff import _load_subs, _load_subs_from_tree, _open_experiment_tree
from hhemt.report_renderers.cross_experiment_compatibility import _combine_provenance_rows
from hhemt.report_renderers.cross_experiment_disk_utilization import _child_model
from hhemt.utils import EXPERIMENT_TREE_NAME, experiment_tree_summary

_ROOT_ATTRS = {"triton_producing_sha": "abc123", "hhemt_producing_version": "0.1+test", "note": "root"}


def _ds(shape=(1, 2, 2), name="max_wlevel_m", fill=1.0) -> xr.Dataset:
    dims = ("event_iloc", "y", "x")[: len(shape)]
    coords = {d: np.arange(n) for d, n in zip(dims, shape, strict=True)}
    return xr.Dataset({name: (dims, np.full(shape, fill, dtype="f4"))}, coords=coords)


def _write(store: Path, nodes: dict[str, xr.Dataset], *, zarr_format: int = 3, attrs=None) -> Path:
    nodes = dict(nodes)
    nodes["/"] = xr.Dataset(attrs=dict(_ROOT_ATTRS if attrs is None else attrs))
    xr.DataTree.from_dict(nodes).to_zarr(store, mode="w", consolidated=False, zarr_format=zarr_format)
    return store


def _oracle(store: Path) -> tuple[dict, tuple[str, ...], str | None]:
    """The three answers read from a FULL open, with the logic the renderers used before the helper."""
    dt = xr.open_datatree(str(store), engine="zarr", consolidated=False)
    groups = set(dt.groups)
    members = tuple(sorted(g for g in groups if g.count("/") == 1 and g.startswith("/member_")))
    tier = None
    for g in members:
        if f"{g}/tritonswmm/triton" in groups:
            tier = "tritonswmm"
            break
        if f"{g}/triton_only/triton" in groups:
            tier = "triton_only"
            break
    return dict(dt.attrs), members, tier


def _assert_matches_oracle(store: Path) -> None:
    attrs, members, tier = _oracle(store)
    got = experiment_tree_summary(store)
    assert dict(got.attrs) == attrs
    assert tuple(got.members) == members
    assert got.tier == tier


_COUPLED = {
    "/member_a/tritonswmm/triton": _ds(),
    "/member_a/tritonswmm/swmm_link": _ds(name="max_flow_cms"),
    "/member_b/tritonswmm/triton": _ds(),
}
_PURE = {"/member_a/triton_only/triton": _ds(), "/member_b/triton_only/triton": _ds()}


@pytest.mark.parametrize(
    ("nodes", "tier"),
    [(_COUPLED, "tritonswmm"), (_PURE, "triton_only")],
    ids=["coupled", "pure-triton"],
)
def test_summary_equals_a_full_open_on_coupled_and_pure_trees(tmp_path, nodes, tier):
    """Property: attrs, member groups and tier equal what a full ``open_datatree`` shows.

    Class: REGRESSION (the answers hold before the helper existed, as the old inline logic). The
    input class is a producer-shaped tree with two members, coupled and pure-TRITON. The literal tier
    is asserted too so the oracle and the helper cannot agree on a wrong answer. The plausible wrong
    implementation is a helper that reads members from a different prefix, or returns the tier name
    of the wrong path. Why required: the renderers print these values in the report.

    A second correct implementation is a helper that does the full open and derives the answers
    from it (the fallback path), which gives the same triple.
    """
    store = _write(tmp_path / "t.zarr", nodes)
    _assert_matches_oracle(store)
    got = experiment_tree_summary(store)
    assert got.tier == tier
    assert got.members == ("/member_a", "/member_b")
    assert got.attrs == _ROOT_ATTRS


def test_summary_inspects_every_member_for_the_tier(tmp_path):
    """Property: the tier comes from the first sorted member that HAS one, not from the first member.

    Class: NEW CAPABILITY-shaped discriminator against a first-member-only helper. The tree has
    ``member_a`` with no tier path, ``member_b`` with ``triton_only`` and ``member_c`` coupled, so the
    first sorted member lacks both paths and the answer is ``triton_only`` (member_b's), neither
    ``None`` nor ``tritonswmm``. Why required: both renderers took the first member with a tier.

    A second correct implementation is the full-open fallback applied to every tree.
    """
    store = _write(
        tmp_path / "t.zarr",
        {
            "/member_a/other": _ds(),
            "/member_b/triton_only/triton": _ds(),
            "/member_c/tritonswmm/triton": _ds(),
        },
    )
    _assert_matches_oracle(store)
    assert experiment_tree_summary(store).tier == "triton_only"


def test_summary_ignores_non_member_top_level_groups_and_reports_no_tier_for_a_member_less_tree(tmp_path):
    """Property: only ``member_*`` groups count, and a tree with no tier path reports ``None``.

    Class: REGRESSION. The satisfying input differs from the tests above: a top-level group that is
    not named ``member_*`` (it carries a tier path, which must be ignored) beside one member without a
    tier. The plausible wrong implementation is counting every top-level group. Why required: ``n_subs``
    is the member count. A second correct implementation is the full-open fallback.
    """
    store = _write(
        tmp_path / "t.zarr",
        {"/member_a/other": _ds(), "/stray/tritonswmm/triton": _ds()},
    )
    _assert_matches_oracle(store)
    got = experiment_tree_summary(store)
    assert got.members == ("/member_a",)
    assert got.tier is None


def test_summary_falls_back_to_a_full_open_for_a_zarr_format_2_tree(tmp_path, monkeypatch):
    """Property: a zarr-format-2 root (``.zgroup``, no ``zarr.json``) is read through ``xr.open_datatree``.

    Class: REGRESSION on the answers, with a must-fire arm on the path: the counting wrapper must see
    one ``open_datatree`` call for the v2 tree and none for a v3 tree. The plausible wrong
    implementation is reading only ``zarr.json``, which finds nothing in a v2 tree. Why required: the
    renderers must still work on such a tree. A second correct implementation is parsing ``.zgroup``,
    ``.zattrs`` and the nested ``.zgroup`` files directly.
    """
    v2 = _write(tmp_path / "v2.zarr", _COUPLED, zarr_format=2)
    v3 = _write(tmp_path / "v3.zarr", _COUPLED)
    assert (v2 / ".zgroup").exists()
    assert not (v2 / "zarr.json").exists()
    _assert_matches_oracle(v2)
    calls: list[str] = []
    real = xr.open_datatree

    def _count(*args, **kwargs):
        calls.append(str(args[0]))
        return real(*args, **kwargs)

    monkeypatch.setattr(xr, "open_datatree", _count)
    experiment_tree_summary(v3)
    assert calls == []
    experiment_tree_summary(v2)
    assert calls == [str(v2)]


def test_summary_of_an_absent_store_raises_as_a_full_open_does(tmp_path):
    """Property: a path that is not a store raises, as ``open_datatree`` does, rather than returning an empty summary.

    Class: REGRESSION. The plausible wrong implementation is returning ``(attrs={}, members=(), None)``
    for a missing store, which would let a renderer print a plausible empty row. Why required: both
    renderers rely on their own ``try/except`` to turn the failure into a default.

    A second correct implementation is raising a more specific exception type; the test accepts any
    ``Exception`` and only forbids a silent return.
    """
    missing = tmp_path / "nope.zarr"
    with pytest.raises(Exception):  # noqa: B017, PT011 - the type is the library's, not part of the contract
        xr.open_datatree(str(missing), engine="zarr", consolidated=False)
    with pytest.raises(Exception):  # noqa: B017, PT011
        experiment_tree_summary(missing)


def _child(tmp_path: Path, nodes: dict[str, xr.Dataset], name="child") -> Path:
    child = tmp_path / name
    child.mkdir()
    _write(child / EXPERIMENT_TREE_NAME, nodes)
    return child


def _count_opens(monkeypatch) -> list[str]:
    calls: list[str] = []
    real = xr.open_datatree

    def _count(*args, **kwargs):
        calls.append(str(args[0]))
        return real(*args, **kwargs)

    monkeypatch.setattr(xr, "open_datatree", _count)
    return calls


@pytest.mark.parametrize(("nodes", "label"), [(_COUPLED, "TRITON-SWMM"), (_PURE, "TRITON")], ids=["coupled", "pure"])
def test_disk_utilization_model_label_needs_no_full_open(tmp_path, monkeypatch, nodes, label):
    """Property: ``_child_model`` returns the model label with zero full tree opens.

    Class: REGRESSION on the label, discriminator on the open count: before this change the function
    opens the whole tree once per child and the count reads 1. The input class is a coupled and a
    pure-TRITON child tree. The plausible wrong implementation is keeping the ``open_datatree`` call.
    Why required: a combine's children hold thousands of nodes each. A second correct implementation
    is any metadata-only reader that gives the same label.
    """
    child = _child(tmp_path, nodes)
    calls = _count_opens(monkeypatch)
    assert _child_model(child) == label
    assert calls == []


def test_compatibility_rows_need_no_full_open_and_keep_their_values(tmp_path, monkeypatch):
    """Property: ``_combine_provenance_rows`` reports model, n_subs and solver sha with zero full tree opens.

    Class: REGRESSION on the row values, discriminator on the open count (one per child before this
    change). The input class is one coupled child with two members and a root ``triton_producing_sha``
    attribute. The plausible wrong implementation is dropping the root attributes along with the open,
    which blanks ``solver_sha``. Why required: the row feeds the cross-experiment compatibility table.
    A second correct implementation is any metadata-only reader that gives the same row.
    """
    analysis_dir = tmp_path / "analysis"
    (analysis_dir / "child_crates").mkdir(parents=True)
    child = analysis_dir / "child_crates" / "e0"
    child.mkdir()
    _write(child / EXPERIMENT_TREE_NAME, _COUPLED)
    calls = _count_opens(monkeypatch)
    (row,) = _combine_provenance_rows(analysis_dir)
    assert calls == []
    assert (row["model"], row["n_subs"], row["solver_sha"], row["toolkit_version"]) == (
        "TRITON-SWMM",
        2,
        "abc123",
        "0.1+test",
    )


def _two_event_tree(root: Path) -> None:
    ev = {
        "max_wlevel_m": (("event_iloc", "y", "x"), np.stack([np.full((2, 2), 1.0), np.full((2, 2), 2.0)]).astype("f4")),
    }
    coords = {"event_iloc": [0, 1], "y": [0, 1], "x": [0, 1]}
    wl = xr.Dataset(ev, coords=coords)
    fl = xr.Dataset(
        {
            "max_flow_cms": (
                ("event_iloc", "y", "x"),
                np.stack([np.full((2, 2), 5.0), np.full((2, 2), 6.0)]).astype("f4"),
            )
        },
        coords=coords,
    )
    nodes = {
        "/member_a": xr.Dataset(attrs={"member_id": "a", "run_mode": "serial"}),
        "/member_a/tritonswmm/triton": wl,
        "/member_a/tritonswmm/swmm_link": fl,
        "/member_b": xr.Dataset(attrs={"member_id": "b", "run_mode": "mpi"}),
        "/member_b/triton_only/triton": wl,
    }
    _write(root / EXPERIMENT_TREE_NAME, nodes)
    (root / "scenario_status.csv").write_text("member_id,n_resumes\na,3\nb,0\n")


def _snapshot(subs: dict[str, dict]) -> dict:
    return {
        k: (
            v["attrs"],
            v["label"],
            v["run_mode"],
            v["n_resumes"],
            np.asarray(v["wlevel"].values).tolist(),
            None if v["flow"] is None else np.asarray(v["flow"].values).tolist(),
        )
        for k, v in subs.items()
    }


@pytest.mark.parametrize("event_iloc", [0, 1])
def test_load_subs_from_tree_equals_load_subs_for_each_event(tmp_path, event_iloc):
    """Property: the split loader equals ``_load_subs`` for each event.

    Class: REGRESSION for the EDA caller. The input is a two-event tree with a coupled member (flow
    present) and a pure-TRITON member (flow ``None``), both events, so an event-0 hardcode and a
    dropped ``flow`` each fail one arm. Expected values are literal in the tree (1.0 and 2.0 for
    depth, 5.0 and 6.0 for flow) as well as compared pairwise. The plausible wrong implementation is
    ignoring ``event_iloc`` in the split function. Why required: the maps renderer reads a per-pair
    event. A second correct implementation is ``_load_subs`` kept as it was and the split a copy.
    """
    root = tmp_path / "exp"
    root.mkdir()
    _two_event_tree(root)
    direct = _snapshot(_load_subs(root, event_iloc=event_iloc))
    split = _snapshot(_load_subs_from_tree(_open_experiment_tree(root), root, event_iloc=event_iloc))
    assert direct == split
    assert direct["a"][4] == np.full((2, 2), 1.0 + event_iloc).tolist()
    assert direct["a"][5] == np.full((2, 2), 5.0 + event_iloc).tolist()
    assert direct["b"][5] is None
    assert direct["a"][3] == 3


def test_one_open_tree_serves_two_events_with_one_full_open(tmp_path, monkeypatch):
    """Property: one open plus two split loads costs one ``open_datatree`` call, two ``_load_subs`` calls cost two.

    Class: discriminator for the maps renderer's saving (its inner loop did one ``_load_subs`` per
    (child, event)). The counting wrapper reads 2 for two ``_load_subs`` calls and 1 for one open plus
    two ``_load_subs_from_tree`` calls, and the two events' arrays differ (so one tree is not serving
    a cached event). The plausible wrong implementation is a split function that reopens. Why
    required: the open reads every node's metadata. A second correct implementation is a cache inside
    ``_load_subs`` keyed on the root; this test pins the split API only.

    Not established: this exercises the loader pair, not the renderer's ``_subs_for`` closure, whose
    wiring the stress run over the renderer tests and the reviewer's file must cover.
    """
    root = tmp_path / "exp"
    root.mkdir()
    _two_event_tree(root)
    calls = _count_opens(monkeypatch)
    _load_subs(root, event_iloc=0)
    _load_subs(root, event_iloc=1)
    assert len(calls) == 2
    calls.clear()
    tree = _open_experiment_tree(root)
    e0 = _load_subs_from_tree(tree, root, event_iloc=0)
    e1 = _load_subs_from_tree(tree, root, event_iloc=1)
    assert len(calls) == 1
    assert float(e0["a"]["wlevel"].values.max()) == 1.0
    assert float(e1["a"]["wlevel"].values.max()) == 2.0


# ---- integration with develop's model label: the union of models, SWMM-only included ---------------------


def _label_from_summary(store: Path) -> str:
    from hhemt.report_renderers.cross_experiment_disk_utilization import model_label_from_models

    return model_label_from_models(experiment_tree_summary(store).models)


def _label_from_full_open(store: Path) -> str:
    from hhemt.report_renderers.cross_experiment_disk_utilization import model_from_groups
    from hhemt.utils import open_store

    return model_from_groups(set(open_store(store).groups))


_SWMM = {"/member_s/swmm_only/swmm_node": _ds(name="v")}
_TREES = {
    "coupled-only": (_COUPLED, "TRITON-SWMM"),
    "pure-only": (_PURE, "TRITON"),
    "swmm-only": (_SWMM, "SWMM"),
    "coupled-plus-swmm": ({**_COUPLED, **_SWMM}, "TRITON-SWMM + SWMM"),
    "pure-plus-swmm": ({**_PURE, **_SWMM}, "TRITON + SWMM"),
    "pure-plus-coupled": ({**_PURE, "/member_c/tritonswmm/triton": _ds()}, "TRITON + TRITON-SWMM"),
}


@pytest.mark.parametrize("name", sorted(_TREES))
def test_summary_label_equals_the_full_open_label_and_the_literal(tmp_path, name):
    """Property: the label built from ``summary.models`` equals ``model_from_groups`` over a full open, and a literal.

    Class: differential over generated trees (coupled, pure, SWMM-only, and the three mixtures). The
    literal is asserted too so the two readers cannot agree on a wrong label. The plausible wrong
    implementation is the first-member tier (``summary.tier``), which reads "TRITON-SWMM" for
    coupled plus SWMM where the label is "TRITON-SWMM + SWMM"; the next test is that must-fire
    control. Why required: the page prints the label. A second correct implementation is the full open.
    """
    nodes, literal = _TREES[name]
    store = _write(tmp_path / "t.zarr", nodes)
    assert _label_from_summary(store) == literal
    assert _label_from_full_open(store) == literal


def test_the_first_tier_alone_would_differ_on_coupled_plus_swmm(tmp_path):
    """Must-fire control: the old first-tier answer differs from the label on a coupled-plus-SWMM tree."""
    store = _write(tmp_path / "t.zarr", {**_COUPLED, **_SWMM})
    s = experiment_tree_summary(store)
    old = "TRITON-SWMM" if s.tier == "tritonswmm" else "TRITON"
    assert old != _label_from_summary(store)


def test_a_tier_directory_without_its_own_document_is_not_a_model(tmp_path):
    """Property: a tier or SWMM directory that lost its ``zarr.json`` is not counted, as in a full open.

    Class: REGRESSION on the group rule for the new union, for the coupled tier (parent
    ``tritonswmm/`` document removed) and the SWMM tier (``swmm_only/`` document removed) beside a
    member that does hold a model. The plausible wrong implementation is a leaf-only existence check.
    A second correct implementation is the full open.
    """
    store = _write(tmp_path / "t.zarr", {**_PURE, **_SWMM, "/member_c/tritonswmm/triton": _ds()})
    (store / "member_c" / "tritonswmm" / "zarr.json").unlink()
    (store / "member_s" / "swmm_only" / "zarr.json").unlink()
    assert _label_from_summary(store) == _label_from_full_open(store) == "TRITON"


def test_no_members_and_a_path_that_is_not_a_store(tmp_path):
    """Property: no members reads "model not recorded"; a path that is not a store raises as a full open does."""
    store = _write(tmp_path / "t.zarr", {"/other/x": _ds()})
    assert experiment_tree_summary(store).models == ()
    assert _label_from_summary(store) == _label_from_full_open(store) == "model not recorded"
    with pytest.raises(Exception):  # noqa: B017, PT011
        experiment_tree_summary(tmp_path / "nope.zarr")


def test_the_zarr_format_2_fallback_reports_the_same_models(tmp_path):
    """Property: a zarr-format-2 tree (full-open fallback) reports the same ``models`` as the v3 tree."""
    nodes = {**_COUPLED, **_SWMM}
    v3 = _write(tmp_path / "v3.zarr", nodes)
    v2 = _write(tmp_path / "v2.zarr", nodes, zarr_format=2)
    assert not (v2 / "zarr.json").exists()
    assert experiment_tree_summary(v2).models == experiment_tree_summary(v3).models == ("TRITON-SWMM", "SWMM")


def test_the_summary_tier_table_matches_the_renderers_tier_table():
    """Property: the utils tier table equals the renderer tier table (models, order, paths)."""
    from hhemt.report_renderers.cross_experiment_disk_utilization import _MODEL_TIERS
    from hhemt.utils import _SUMMARY_MODEL_TIERS

    assert [(m, "/".join(path)) for m, path in _SUMMARY_MODEL_TIERS] == list(_MODEL_TIERS)
