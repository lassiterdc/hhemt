"""Reviewer arms for ``hhemt bundle --omit-output-zarrs`` (DR-3, C1(e)), work chunk 1b.2.

Problem this file solves: the coder's own tests drive ``_harvest_sources`` and
``_write_bundle_manifest`` separately, so the line inside ``emit_bundle`` that derives
``omitted_output_zarrs`` from the shallow-store list is covered by nothing, and the
regeneration-Snakefile refusal is exercised only on declarations spelled inside the analysis
tree. These arms drive the REAL ``emit_bundle`` (through the ``_world`` builder of
``tests/test_publish_deposit_paths.py``) and the REAL ``generate_regeneration_snakefile`` over
copies of the checked-in bundle fixtures.

Solver-free: every arm writes ``tmp_path`` files only, no compile, no simulation, and no
fixture from the synthetic-model tier is requested.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from hhemt.bundle import _combine as CB
from hhemt.bundle import _emit
from hhemt.bundle._emit import _zarr_metadata_filenames
from hhemt.bundle.snakefile_generator import generate_regeneration_snakefile
from hhemt.exceptions import ProcessingError
from hhemt.report_renderers._figure_emission import MANIFEST_METADATA_ONLY_KEY
from hhemt.version_migration.constants import BUNDLE_MANIFEST_FILENAME
from tests.test_publish_deposit_paths import _world

FIXTURES = Path(__file__).parent / "fixtures" / "bundles"
STORE = "analysis_datatree.zarr"
OTHER = "other.zarr"


def _write_store(
    store: Path, *, zarr_format: int = 3, consolidated: bool = False, chunks=None, dtype=float, root_attr: str = "v"
) -> None:
    """Write a DataTree with a root attribute, a deep nested group and a sibling group, chunked.

    ``root_attr`` is the root attribute's value; the default writes the same bytes every call. A store
    written with another value differs in its root metadata document, which is the part of a store that an
    emit with ``omit_output_zarrs`` still carries.
    """
    chunks = chunks or {"y": 5, "x": 5}
    ds = xr.Dataset({"h": (("y", "x"), np.arange(400).reshape(20, 20).astype(dtype))}).chunk(chunks)
    tree = xr.DataTree.from_dict({"/": xr.Dataset(attrs={"k": root_attr}), "/a/b/c": ds, "/z": ds})
    tree.to_zarr(store, mode="w", consolidated=consolidated, zarr_format=zarr_format)


def _emit_world(tmp_path: Path, *, sidecars: dict[str, dict], flag: bool, stores=(STORE, OTHER), **store_kw):
    """Run the real ``emit_bundle`` over a world holding ``stores`` and one sidecar per ``sidecars`` key.

    Returns ``(world, zip path, archive member names, parsed bundle_manifest.json)``.
    """
    world = _world(tmp_path)
    for name in stores:
        _write_store(world.ad / name, **store_kw)
    plots = world.ad / "plots"
    plots.mkdir(exist_ok=True)
    for name, body in sidecars.items():
        (plots / f"{name}.manifest.json").write_text(json.dumps(body))
    kwargs = {"omit_output_zarrs": True} if flag else {}
    out = _emit.emit_bundle(world.analysis, world.base / "bundle.zip", **kwargs)
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        manifest = json.loads(zf.read(BUNDLE_MANIFEST_FILENAME))
    return world, out, names, manifest


def _chunk_members(names: list[str], store: str) -> list[str]:
    """Archive members under ``store`` that are not zarr metadata documents (the chunk files)."""
    meta = _zarr_metadata_filenames()
    return [n for n in names if n.startswith(store + "/") and n.rsplit("/", 1)[-1] not in meta]


BOTH_FULL = {"fig": {"source_paths_relative": [STORE, OTHER]}}


def test_real_emit_carries_no_chunk_of_any_store_names_both_keys_and_the_control_carries_chunks(tmp_path):
    """Property: with the flag the real emit carries 0 chunk files per store and names both stores in both keys.

    Class: every harvested ``.zarr`` directory source of a real ``emit_bundle`` (two stores here).
    UNIT: archive members under a store directory whose basename is not a zarr metadata document.
    Why required: the derivation of ``omitted_output_zarrs`` sits between ``_harvest_sources`` and
    ``_write_bundle_manifest`` inside ``emit_bundle``, so only a test through ``emit_bundle`` reads it.
    What kills it: passing ``None`` for ``omitted_output_zarrs`` at the ``_write_bundle_manifest`` call
    inside ``emit_bundle``, or dropping ``omit_output_zarrs=`` from the ``_harvest_sources`` call there.
    Second correct implementation: computing the omitted list inside ``_harvest_sources`` and returning
    it as a fourth tuple element passes unchanged, since the test reads only the archive and the manifest.
    """
    _w, _z, names, manifest = _emit_world(tmp_path / "on", sidecars=BOTH_FULL, flag=True)
    assert _chunk_members(names, STORE) == [] and _chunk_members(names, OTHER) == []
    assert any(n.startswith(STORE + "/") for n in names), "metadata documents must still be carried"
    assert manifest["omitted_output_zarrs"] == sorted([STORE, OTHER])
    assert manifest["metadata_only_stores"] == sorted([STORE, OTHER])

    _w, _z, names, manifest = _emit_world(tmp_path / "off", sidecars=BOTH_FULL, flag=False)
    assert len(_chunk_members(names, STORE)) >= 1, "control: the default emit carries chunk files"
    assert "omitted_output_zarrs" not in manifest
    assert "metadata_only_stores" not in manifest


def test_a_store_a_sidecar_already_qualified_is_not_reported_omitted_without_the_flag(tmp_path):
    """Property: a store shallow only because a sidecar qualified it is never named in ``omitted_output_zarrs``.

    Class: the two ways a store becomes shallow (sidecar qualifier, flag), crossed with flag on and off.
    Why required: ``omitted_output_zarrs`` is what the Snakefile refusal reads; naming a store the flag did
    not omit would refuse figures of an ordinary default bundle.
    What kills it: computing the omitted list as ``metadata_only_stores`` without the flag guard.
    Second correct implementation: recording the omitted list from the flag-forced set only inside the
    harvest passes unchanged, because the test reads the manifest and not the derivation.
    """
    body = {
        "fig": {"source_paths_relative": [STORE]},
        "meta": {"source_paths_relative": [OTHER], MANIFEST_METADATA_ONLY_KEY: [OTHER]},
    }
    _w, _z, _n, manifest = _emit_world(tmp_path / "off", sidecars=body, flag=False)
    assert manifest["metadata_only_stores"] == [OTHER]
    assert "omitted_output_zarrs" not in manifest

    _w, _z, names, manifest = _emit_world(tmp_path / "on", sidecars=body, flag=True)
    assert manifest["omitted_output_zarrs"] == sorted([STORE, OTHER])
    assert manifest["metadata_only_stores"] == sorted([STORE, OTHER])
    assert _chunk_members(names, STORE) == [] and _chunk_members(names, OTHER) == []


@pytest.mark.parametrize(
    "shape",
    [
        dict(zarr_format=3, chunks={"y": 7, "x": 3}),
        dict(zarr_format=3, consolidated=True),
        dict(zarr_format=2, consolidated=True),
        dict(zarr_format=2, consolidated=False),
        dict(zarr_format=3, dtype=int),
    ],
    ids=["v3-uneven-chunks", "v3-consolidated-nested", "v2-consolidated", "v2-plain", "v3-int"],
)
def test_the_dr3_premise_a_shallow_store_reads_fill_values_and_raises_nothing(tmp_path, shape):
    """Property: reading an array of a store whose chunks are absent from the emitted archive returns the fill value.

    Class: store shapes the coder's single fixture did not cover (uneven chunks, nested consolidated v3,
    consolidated and plain v2, an integer dtype), each read from the EMITTED archive.
    Why required: DR-3 says that if a read raises, the Snakefile refusal is dropped; the refusal is kept
    only while this holds for every shape the toolkit writes.
    What kills it: a zarr or xarray version whose missing-chunk read raises (the test then fails with the
    exception), or a harvest that carried chunk files (the values would not be fill values).
    Second correct implementation: reading through ``zarr.open_group`` instead of ``xr.open_datatree``
    passes unchanged, because the assertion is on the returned fill value, not on the reader.
    """
    _w, out, _n, _m = _emit_world(
        tmp_path, sidecars={"fig": {"source_paths_relative": [STORE]}}, flag=True, stores=(STORE,), **shape
    )
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(out) as zf:
        zf.extractall(extracted)
    consolidated = bool(shape.get("consolidated", False))
    tree = xr.open_datatree(str(extracted / STORE), engine="zarr", consolidated=consolidated)
    values = tree["/a/b/c"].dataset["h"].values
    if shape.get("dtype") is int:
        assert (values == 0).all()
    else:
        assert np.isnan(values).all()


def test_combine_refuses_the_extracted_real_emit(tmp_path):
    """Property: combine refuses the bundle the real ``emit_bundle`` wrote under the flag.

    Class: a root consolidated store named in the writer's own ``metadata_only_stores``.
    Why required: DR-3 adds no combine site and relies on the existing refusal reading the key this
    option writes; a hand-built manifest would not show a writer that stopped writing it.
    What kills it: ``emit_bundle`` not passing ``metadata_only_stores`` for a flag-forced shallow store.
    Second correct implementation: a new combine-side check keyed on ``omitted_output_zarrs`` instead of
    ``metadata_only_stores`` passes unchanged, since the test asserts the refusal and its word, not the key.
    """
    _w, out, _n, _m = _emit_world(tmp_path, sidecars=BOTH_FULL, flag=True)
    root = tmp_path / "extracted"
    with zipfile.ZipFile(out) as zf:
        zf.extractall(root)
    with pytest.raises(ProcessingError) as exc:
        CB._load_intercomparison_subs(root)
    assert "metadata-only" in str(exc.value)


# ----------------------------------------------------------------------------------------------
# The regeneration-Snakefile refusal
# ----------------------------------------------------------------------------------------------


def _bundle(tmp_path: Path, which: str = "multi_sim") -> Path:
    """Copy a checked-in bundle fixture into ``tmp_path`` and return the copy."""
    dest = tmp_path / which
    shutil.copytree(FIXTURES / which, dest)
    return dest


def _sidecar(bundle: Path, rel: str, declared: list[str], *, qualified=None, channel_source=None) -> None:
    """Write a figure sidecar under ``plots/`` declaring ``declared`` (and optionally a channel source)."""
    path = bundle / "plots" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    body: dict = {"source_paths_relative": declared}
    if qualified:
        body[MANIFEST_METADATA_ONLY_KEY] = qualified
    if channel_source:
        body["artists"] = [{"channels": [{"channel": "x", "ref": {"source_path": channel_source}}]}]
    path.write_text(json.dumps(body))


def _omit(bundle: Path, stores: list[str]) -> None:
    """Set ``omitted_output_zarrs`` in the bundle's own manifest."""
    path = bundle / BUNDLE_MANIFEST_FILENAME
    manifest = json.loads(path.read_text())
    manifest["omitted_output_zarrs"] = stores
    path.write_text(json.dumps(manifest))


def _rules(text: str) -> dict[str, str]:
    """Split a generated Snakefile into ``{rule name: rule block}``."""
    return {b.split(":", 1)[0].strip(): b for b in text.split("\nrule ")[1:]}


def _generate(bundle: Path) -> str:
    """Generate the regeneration Snakefile of ``bundle`` with the matplotlib backend."""
    return generate_regeneration_snakefile(bundle, static_backend="matplotlib")


def test_a_store_declared_only_through_an_artist_channel_ref_is_still_a_full_declaration(tmp_path):
    """Property: a store reached only through ``artists[].channels[].ref.source_path`` refuses its rule.

    Class: the second declaration channel of a sidecar (beside ``source_paths_relative``).
    Why required: the shared resolver unions both channels; a refusal keyed on one would let a figure read
    fill values through the other.
    What kills it: iterating ``source_paths_relative`` alone instead of ``_iter_manifest_declarations``.
    Second correct implementation: reading the sidecar JSON directly for both keys passes unchanged.
    """
    bundle = _bundle(tmp_path)
    _sidecar(bundle, "system_overview.manifest.json", ["sims/x.inp"], channel_source=STORE)
    _omit(bundle, [STORE])
    assert "exit 1" in _rules(_generate(bundle))["plot_system_overview"]


def test_a_wildcard_family_is_refused_when_any_one_member_declares_the_store_in_full(tmp_path):
    """Property: a wildcard rule is refused when any sidecar of its family declares an omitted store in full.

    Class: a per-event family of two sidecars, one qualified metadata-only and one full.
    Why required: one Snakemake rule renders every event of the family, so one full declarant makes the
    rule unsafe to run.
    What kills it: refusing only when every sidecar of the family declares the store in full.
    Second correct implementation: one refused rule per wildcard instance passes only if the rule name is
    unchanged, so this holds against the rule-granular design DR-3 specifies and not against a finer one.
    """
    bundle = _bundle(tmp_path)
    shutil.rmtree(bundle / "plots" / "per_sim")
    _sidecar(bundle, "per_sim/event_index.0/event_page__evt.event_index.0.manifest.json", [STORE], qualified=[STORE])
    _sidecar(bundle, "per_sim/event_index.1/event_page__evt.event_index.1.manifest.json", [STORE])
    _omit(bundle, [STORE])
    assert "exit 1" in _rules(_generate(bundle))["plot_per_sim_event_page"]


def test_a_wildcard_family_whose_every_declaration_is_qualified_is_not_refused(tmp_path):
    """Property: a family whose every declaration of an omitted store is metadata-only keeps its renderer call.

    Class: the over-refusal side of the predicate, a different correct state than the full declarant.
    Why required: the metadata page declares the store metadata-only on purpose; refusing it would block
    the one page an omitted bundle can still regenerate.
    What kills it: subtracting nothing for ``MANIFEST_METADATA_ONLY_KEY`` (every declaration counts as full).
    Second correct implementation: deciding fullness from a per-path flag in a richer manifest passes
    unchanged, since the test asserts only the emitted rule block.
    """
    bundle = _bundle(tmp_path)
    shutil.rmtree(bundle / "plots" / "per_sim")
    for i in (0, 1):
        _sidecar(
            bundle,
            f"per_sim/event_index.{i}/event_page__evt.event_index.{i}.manifest.json",
            [STORE],
            qualified=[STORE],
        )
    _omit(bundle, [STORE])
    block = _rules(_generate(bundle))["plot_per_sim_event_page"]
    assert "exit 1" not in block and "report_renderers._cli" in block


def test_only_the_refused_rule_differs_and_its_shell_has_no_brace(tmp_path):
    """Property: omitting a store changes exactly the refused rule's block, and that shell body has no brace.

    Class: every rule block of the generated Snakefile (8 in this fixture).
    Why required: Snakemake formats a shell string, so a brace in the refusal message would be read as a
    field and fail the whole Snakefile parse; any other rule changing is collateral.
    What kills it: a message containing ``{`` or ``}``, or a refusal that rewrites more than one rule.
    Second correct implementation: a refusal that also drops the ``log`` directive passes unchanged.
    """
    bundle = _bundle(tmp_path)
    _sidecar(bundle, "system_overview.manifest.json", ["sims/x.inp", STORE])
    before = _generate(bundle)
    _omit(bundle, [STORE])
    after = _generate(bundle)
    b, a = _rules(before), _rules(after)
    assert b.keys() == a.keys()
    assert [k for k in a if a[k] != b[k]] == ["plot_system_overview"]
    refused = a["plot_system_overview"]
    shell = refused[refused.index("    shell:") :]
    assert "{" not in shell and "}" not in shell, "Snakemake formats a shell string; a brace would be read as a field"
    assert shell.count("'") == 2, "the message sits in one pair of single quotes"


def test_a_sensitivity_member_sidecar_that_declares_the_member_store_refuses_the_member_family(tmp_path):
    """Property: a member sidecar's re-rooted store declaration meets the bundle-root-relative omitted entry.

    Class: sensitivity-master bundles, where a sidecar under ``member-{N}/`` declares paths relative to
    ``members/member_{N}`` and the emit names the store with that prefix.
    Why required: the two spellings are produced by different functions and must meet in the predicate.
    What kills it: resolving member sidecars against the master root instead of ``members/member_{N}``.
    Second correct implementation: comparing the declared path's bundle-relative string instead of its
    resolved path passes unchanged, since the test asserts the emitted rule blocks.
    """
    bundle = _bundle(tmp_path, "sensitivity_master")
    shutil.rmtree(bundle / "plots" / "sensitivity" / "per_sim")
    d1 = "sensitivity/per_sim/member-1/event_index.0/peak_flood_depth__member.1__evt.event_index.0.manifest.json"
    d2 = "sensitivity/per_sim/member-2/event_index.0/peak_flood_depth__member.2__evt.event_index.0.manifest.json"
    _sidecar(bundle, d1, [STORE])
    _sidecar(bundle, d2, [STORE])
    _omit(bundle, ["members/member_1/" + STORE])
    rules = _rules(_generate(bundle))
    assert "exit 1" in rules["plot_per_sim_per_member_peak_flood_depth"]
    assert "members/member_1/" + STORE in rules["plot_per_sim_per_member_peak_flood_depth"]
    assert "members/member_2/" not in rules["plot_per_sim_per_member_peak_flood_depth"]
    assert "exit 1" not in rules["plot_per_sim_per_member_conduit_flow"]


def test_a_full_declarant_of_an_omitted_store_outside_the_analysis_tree_is_refused(tmp_path):
    """Property: a plot rule whose sidecar declares, in full, a store carried as ``external/{name}`` is refused.

    Class: declared full sources outside the analysis tree; the emit names them ``external/{name}`` while
    the sidecar keeps its emit-time spelling (``../outside/ext.zarr``).
    Why required: DR-3's guard is "any plot rule whose declared full sources include an omitted store";
    a rule that is not refused re-renders a figure from fill values with no error, which is the exact
    failure the guard exists to prevent.
    What kills it: matching omitted stores to declarations by the resolved path under the bundle root
    alone; a match that also maps an out-of-tree declaration to its ``external/{name}`` entry greens it.
    Second correct implementation: having the emit rewrite such a sidecar's spelling to ``external/{name}``
    at harvest time passes unchanged, since the test drives the real emit and then the real generator.
    """
    world = _world(tmp_path)
    _write_store(world.out / "ext.zarr")
    plots = world.ad / "plots"
    plots.mkdir()
    (plots / "system_overview.manifest.json").write_text(json.dumps({"source_paths_relative": ["../outside/ext.zarr"]}))
    out = _emit.emit_bundle(world.analysis, world.base / "bundle.zip", omit_output_zarrs=True)
    root = tmp_path / "extracted"
    with zipfile.ZipFile(out) as zf:
        zf.extractall(root)
        manifest = json.loads(zf.read(BUNDLE_MANIFEST_FILENAME))
    assert manifest["omitted_output_zarrs"] == ["external/ext.zarr"], (
        "premise: the store is carried shallow as external/"
    )
    assert "exit 1" in _rules(_generate(root))["plot_system_overview"]


# ----------------------------------------------------------------------------------------------
# Round 2: attacks on the out-of-tree mapping
# ----------------------------------------------------------------------------------------------


def _external_world(
    tmp_path: Path, declare: dict[str, list[str]], stores: dict[str, Path], differing: frozenset[str] = frozenset()
):
    """Real emit with the flag over a world whose sidecars (keyed by rule stem) declare ``declare``.

    A store whose key is in ``differing`` has a different root attribute from the rest, so its metadata
    documents differ in bytes; by default every store is written alike and is byte-identical to the others.
    """
    world = _world(tmp_path)
    for key, rel in stores.items():
        _write_store(world.base / rel, root_attr="other" if key in differing else "v")
    plots = world.ad / "plots"
    plots.mkdir()
    for stem, paths in declare.items():
        (plots / f"{stem}.manifest.json").write_text(json.dumps({"source_paths_relative": paths}))
    out = _emit.emit_bundle(world.analysis, world.base / "bundle.zip", omit_output_zarrs=True)
    root = tmp_path / "extracted"
    with zipfile.ZipFile(out) as zf:
        zf.extractall(root)
        manifest = json.loads(zf.read(BUNDLE_MANIFEST_FILENAME))
    return world, root, manifest


def test_an_in_tree_and_an_external_store_of_one_name_are_refused_by_their_own_declarants_only(tmp_path):
    """Property: an in-tree x.zarr and an external x.zarr (both omitted) each refuse exactly their own declaring rule.

    Class: one basename carried at two archive names (x.zarr and external/x.zarr); three rules, one per case.
    Why required: the new mapping matches an out-of-tree declaration by basename, so a same-named in-tree
    store is the input most likely to cross the two entries.
    What kills it: matching an out-of-tree declaration to every omitted entry of that basename, or an
    in-tree declaration to external/{name}.
    Second correct implementation: keying the mapping on the full declared spelling passes unchanged.
    """
    _w, root, manifest = _external_world(
        tmp_path,
        {
            "system_overview": ["../outside/x.zarr"],
            "disk_utilization": ["x.zarr"],
            "metadata": ["plain.txt"],
        },
        {"ext": Path("outside/x.zarr"), "int": Path("analysis/x.zarr")},
    )
    assert manifest["omitted_output_zarrs"] == ["external/x.zarr", "x.zarr"], "premise: both entries are omitted"
    rules = _rules(_generate(root))
    assert "exit 1" in rules["plot_system_overview"] and "external/x.zarr" in rules["plot_system_overview"]
    assert "exit 1" in rules["plot_disk_utilization"] and "external/x.zarr" not in rules["plot_disk_utilization"]
    assert "exit 1" not in rules["plot_metadata"]


def test_two_different_external_stores_of_one_name_are_refused_at_emit_not_merged(tmp_path):
    """Property: two distinct out-of-tree stores sharing a basename cannot both be carried as external/{name}.

    Class: out-of-tree full sources whose basenames collide.
    Why required: the predicate's basename mapping is only unambiguous if the emit never admits both.
    What kills it: an emit that silently keeps one of the two (the mapping would then refuse the wrong figure).
    Second correct implementation: renaming the second to external/x.zarr.1 also passes if the test
    accepts any ConfigurationError-free emit, so this test asserts the refusal the current emit makes.
    """
    from hhemt.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError):
        _external_world(
            tmp_path,
            {"system_overview": ["../a/x.zarr"], "disk_utilization": ["../b/x.zarr"]},
            {"a": Path("a/x.zarr"), "b": Path("b/x.zarr")},
            differing=frozenset({"b"}),
        )


def test_an_out_of_tree_declaration_from_a_sensitivity_member_sidecar_is_refused(tmp_path):
    """Property: a member sidecar declaring an out-of-tree omitted store refuses the member family.

    Class: member sidecars, whose declared paths resolve from members/member_{N}, with an out-of-tree store.
    Why required: the mapping must hold for the re-rooted resolution too, not only for master sidecars.
    What kills it: applying the out-of-tree test to the master root only.
    Second correct implementation: resolving every declaration against the emit-time analysis directory
    recorded in the manifest passes unchanged.
    """
    bundle = _bundle(tmp_path, "sensitivity_master")
    shutil.rmtree(bundle / "plots" / "sensitivity" / "per_sim")
    d = "sensitivity/per_sim/member-1/event_index.0/peak_flood_depth__member.1__evt.event_index.0.manifest.json"
    _sidecar(bundle, d, ["../../../outside/ext.zarr"])
    _omit(bundle, ["external/ext.zarr"])
    assert "exit 1" in _rules(_generate(bundle))["plot_per_sim_per_member_peak_flood_depth"]


def test_an_absolute_out_of_tree_channel_reference_is_refused(tmp_path):
    """Property: an absolute out-of-tree source_path on an artist channel refuses its rule.

    Class: the absolute-path spelling, the other form the shared resolver admits.
    Why required: an absolute path is not relative to the bundle root and must still map to external/{name}.
    What kills it: an out-of-tree test that assumes a relative spelling.
    Second correct implementation: normalising absolute paths to relative ones before matching passes unchanged.
    """
    bundle = _bundle(tmp_path)
    _sidecar(bundle, "system_overview.manifest.json", ["sims/x.inp"], channel_source="/nonexistent/outside/ext.zarr")
    _omit(bundle, ["external/ext.zarr"])
    assert "exit 1" in _rules(_generate(bundle))["plot_system_overview"]
