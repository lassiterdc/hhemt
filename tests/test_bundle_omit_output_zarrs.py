"""``hhemt bundle --omit-output-zarrs`` (DR-3, C1(e)): shallow carriage of every output zarr.

Solver-free and fixture-light: every test writes ``tmp_path`` files only (a tiny chunked
DataTree store, the checked-in ``multi_sim`` bundle fixture copied aside). No compile, no
simulation, no ``synth_*`` fixture.

The emit-side arms drive the real ``_harvest_sources`` / ``_emit_bundle_zip`` /
``_write_bundle_manifest`` chain that ``emit_bundle`` composes; the option's threading
through ``bundle_command`` and ``emit_bundle`` is asserted separately. Every arm fails on the
parent commit (no ``omit_output_zarrs`` keyword, no ``--omit-output-zarrs`` option, no
generator refusal).
"""

from __future__ import annotations

import inspect
import json
import shutil
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hhemt.bundle import _combine as CB
from hhemt.bundle._emit import (
    _emit_bundle_zip,
    _harvest_sources,
    _write_bundle_manifest,
    _zarr_metadata_filenames,
    emit_bundle,
)
from hhemt.bundle.snakefile_generator import generate_regeneration_snakefile
from hhemt.exceptions import ProcessingError
from hhemt.report_renderers._figure_emission import MANIFEST_METADATA_ONLY_KEY
from hhemt.utils import resolve_experiment_tree
from hhemt.version_migration.constants import BUNDLE_MANIFEST_FILENAME

MULTI_SIM_FIXTURE = Path(__file__).parent / "fixtures" / "bundles" / "multi_sim"


def _write_tree(store: Path) -> None:
    """A two-level DataTree with a chunked array, so the store has real chunk files."""
    ds = xr.Dataset({"h": (("y", "x"), np.arange(400.0).reshape(20, 20))}).chunk({"y": 5, "x": 5})
    tree = xr.DataTree.from_dict({"/": xr.Dataset(attrs={"k": "v"}), "/tritonswmm/triton": ds})
    tree.to_zarr(store, mode="w", consolidated=False, zarr_format=3)


def _analysis_with_store(tmp_path: Path) -> tuple[Path, str]:
    """An analysis dir holding the root consolidated store, plus a plain CSV and a plain dir."""
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    store = resolve_experiment_tree(analysis)
    _write_tree(store)
    (analysis / "table.csv").write_text("a,b\n1,2\n")
    (analysis / "plain_dir").mkdir()
    (analysis / "plain_dir" / "payload.bin").write_bytes(b"x" * 64)
    return analysis, store.relative_to(analysis).as_posix()


def _zip_members(analysis: Path, tmp_path: Path, name: str, **kwargs) -> tuple[list[str], list[str], list[str]]:
    """Harvest + zip; return (zip member names, metadata_only_stores, declared_absent)."""
    store = resolve_experiment_tree(analysis)
    sources = {"fig": [store, analysis / "table.csv", analysis / "plain_dir"]}
    streamed, absent, shallow = _harvest_sources(sources, analysis, **kwargs)
    staging = tmp_path / f"staging_{name}"
    staging.mkdir()
    (staging / "cfg_analysis.yaml").write_text("k: v\n")
    out = tmp_path / f"{name}.zip"
    _emit_bundle_zip(staging, out, streamed=streamed)
    with zipfile.ZipFile(out) as zf:
        return zf.namelist(), shallow, absent


def test_flag_carries_no_chunk_file_of_any_harvested_zarr(tmp_path):
    """UNIT: zip members under the store directory that are NOT zarr metadata documents.
    With the flag it is 0 and the store is named shallow; without it, at least 1 (the control
    that makes the zero mean anything). A `.zarr`-less directory source is untouched by the flag.
    """
    analysis, store_rel = _analysis_with_store(tmp_path)
    meta = _zarr_metadata_filenames()

    def _chunks(members: list[str]) -> list[str]:
        return [m for m in members if m.startswith(store_rel + "/") and m.rsplit("/", 1)[-1] not in meta]

    omitted_members, omitted_shallow, _ = _zip_members(analysis, tmp_path, "omit", omit_output_zarrs=True)
    full_members, full_shallow, _ = _zip_members(analysis, tmp_path, "full")

    assert _chunks(omitted_members) == []
    assert any(m.startswith(store_rel + "/") for m in omitted_members), "metadata documents must still be carried"
    assert omitted_shallow == [store_rel]
    assert len(_chunks(full_members)) >= 1, "control: the same emit without the flag carries chunk files"
    assert full_shallow == []
    # A directory that is not a .zarr store is carried whole under the flag.
    assert "plain_dir/payload.bin" in omitted_members
    assert "table.csv" in omitted_members


def test_a_metadata_only_read_of_the_shallow_store_returns_fill_values_not_an_error(tmp_path):
    """The DR-3 premise, measured on this fixture's own shallow consolidated store: reading an
    array of a store carried without chunk files returns the fill value (NaN) with no exception.
    If this ever raises, the Snakefile refusal's stated reason no longer holds and DR-3 says to
    drop the guard and log the deviation with the read.
    """
    analysis, store_rel = _analysis_with_store(tmp_path)
    members, _shallow, _ = _zip_members(analysis, tmp_path, "omit", omit_output_zarrs=True)
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(tmp_path / "omit.zip") as zf:
        zf.extractall(extracted)
    assert (extracted / store_rel / "zarr.json").is_file()
    dt = xr.open_datatree(str(extracted / store_rel), engine="zarr", consolidated=False)
    values = dt["/tritonswmm/triton"].dataset["h"].values
    assert np.isnan(values).all()


def test_manifest_names_each_store_in_both_keys_and_default_carries_neither(tmp_path):
    analysis, store_rel = _analysis_with_store(tmp_path)
    _members, shallow, _ = _zip_members(analysis, tmp_path, "omit", omit_output_zarrs=True)

    with_flag = tmp_path / "with_flag"
    with_flag.mkdir()
    _write_bundle_manifest(
        with_flag,
        sources_by_renderer={},
        analysis_id="a",
        git_sha="sha",
        metadata_only_stores=shallow,
        omitted_output_zarrs=shallow,
        analysis_dir=analysis,
    )
    m = json.loads((with_flag / BUNDLE_MANIFEST_FILENAME).read_text())
    assert m["omitted_output_zarrs"] == [store_rel]
    assert m["metadata_only_stores"] == [store_rel]

    default = tmp_path / "default"
    default.mkdir()
    _write_bundle_manifest(
        default,
        sources_by_renderer={},
        analysis_id="a",
        git_sha="sha",
        metadata_only_stores=None,
        omitted_output_zarrs=None,
        analysis_dir=analysis,
    )
    d = json.loads((default / BUNDLE_MANIFEST_FILENAME).read_text())
    assert "omitted_output_zarrs" not in d
    assert "metadata_only_stores" not in d


def test_combine_refuses_a_bundle_whose_manifest_came_from_the_writer(tmp_path):
    """Combine's existing shallow-store guard refuses the manifest this option writes: no new
    combine site is needed (DR-3). The child root's manifest is the WRITER's output."""
    root = tmp_path / "child"
    root.mkdir()
    store = resolve_experiment_tree(root)
    xr.Dataset({"max_wlevel_m": ("a", [1.0, 2.0])}).to_zarr(store)
    store_rel = store.relative_to(root).as_posix()
    _write_bundle_manifest(
        root,
        sources_by_renderer={},
        analysis_id="a",
        git_sha="sha",
        metadata_only_stores=[store_rel],
        omitted_output_zarrs=[store_rel],
        analysis_dir=root,
    )
    with pytest.raises(ProcessingError) as exc:
        CB._load_intercomparison_subs(root)
    assert "metadata-only" in str(exc.value)


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    dest = tmp_path / "multi_sim"
    shutil.copytree(MULTI_SIM_FIXTURE, dest)
    return dest


def _declare_store_in_system_overview(bundle: Path, *, qualified: bool) -> None:
    sidecar = bundle / "plots" / "system_overview.manifest.json"
    data = json.loads(sidecar.read_text())
    data["source_paths_relative"] = [*data["source_paths_relative"], "analysis_datatree.zarr"]
    if qualified:
        data[MANIFEST_METADATA_ONLY_KEY] = ["analysis_datatree.zarr"]
    sidecar.write_text(json.dumps(data))


def _set_omitted(bundle: Path, stores: list[str] | None) -> None:
    path = bundle / BUNDLE_MANIFEST_FILENAME
    manifest = json.loads(path.read_text())
    if stores is None:
        manifest.pop("omitted_output_zarrs", None)
    else:
        manifest["omitted_output_zarrs"] = stores
    path.write_text(json.dumps(manifest))


def _rule_block(text: str, rule_name: str) -> str:
    blocks = {b.split(":", 1)[0].strip(): b for b in text.split("\nrule ")[1:]}
    return blocks[rule_name]


def test_regeneration_snakefile_refuses_the_rule_of_a_full_declarant_of_an_omitted_store(bundle):
    _declare_store_in_system_overview(bundle, qualified=False)
    _set_omitted(bundle, ["analysis_datatree.zarr"])
    text = generate_regeneration_snakefile(bundle, static_backend="matplotlib")

    refused = _rule_block(text, "plot_system_overview")
    assert "exit 1" in refused
    assert "--omit-output-zarrs" in refused
    assert "analysis_datatree.zarr" in refused
    assert "report_renderers._cli" not in refused
    # Rules whose figures declared no omitted store keep their renderer invocation.
    assert "report_renderers._cli" in _rule_block(text, "plot_per_analysis_summary_table")


def test_regeneration_snakefile_is_unchanged_when_nothing_was_omitted_or_the_declaration_is_metadata_only(
    tmp_path, bundle
):
    """Three differently-positioned arms of the same predicate: no `omitted_output_zarrs` key,
    a key naming a store no figure declares, and a key naming a store its only declarant
    qualified metadata-only. All three leave the Snakefile identical to the no-key text."""
    _declare_store_in_system_overview(bundle, qualified=False)
    baseline = generate_regeneration_snakefile(bundle, static_backend="matplotlib")
    assert "exit 1" not in _rule_block(baseline, "plot_system_overview")

    _set_omitted(bundle, ["some_other.zarr"])
    assert generate_regeneration_snakefile(bundle, static_backend="matplotlib") == baseline

    qualified = tmp_path / "qualified"
    shutil.copytree(MULTI_SIM_FIXTURE, qualified)
    _declare_store_in_system_overview(qualified, qualified=True)
    _set_omitted(qualified, ["analysis_datatree.zarr"])
    assert _rule_block(
        generate_regeneration_snakefile(qualified, static_backend="matplotlib"), "plot_system_overview"
    ) == _rule_block(baseline, "plot_system_overview")


def test_emit_bundle_and_the_cli_thread_the_option(tmp_path, monkeypatch):
    param = inspect.signature(emit_bundle).parameters["omit_output_zarrs"]
    assert param.default is False

    import hhemt.analysis
    import hhemt.bundle
    import hhemt.system
    from hhemt.cli import app

    seen: dict = {}

    def _spy(target, output, exclude_config=None, **kwargs):
        seen.update(kwargs)
        return tmp_path / "out.zip"

    cfg_analysis = SimpleNamespace(toggle_sensitivity_analysis=False)
    monkeypatch.setattr(hhemt.system, "TRITONSWMM_system", lambda *_a, **_k: object())
    monkeypatch.setattr(
        hhemt.analysis, "TRITONSWMM_analysis", lambda *_a, **_k: SimpleNamespace(cfg_analysis=cfg_analysis)
    )
    monkeypatch.setattr(hhemt.bundle, "emit_bundle", _spy)
    sys_cfg, ana_cfg = tmp_path / "s.yaml", tmp_path / "a.yaml"
    sys_cfg.write_text("k: v\n")
    ana_cfg.write_text("k: v\n")

    from typer.testing import CliRunner

    runner = CliRunner()
    base = ["bundle", "--system-config", str(sys_cfg), "--analysis-config", str(ana_cfg)]
    result = runner.invoke(app, [*base, "--omit-output-zarrs"])
    assert result.exit_code == 0, result.output
    assert seen["omit_output_zarrs"] is True
    seen.clear()
    result = runner.invoke(app, base)
    assert result.exit_code == 0, result.output
    assert seen["omit_output_zarrs"] is False


def test_real_emit_bundle_derives_omitted_output_zarrs_from_the_shallow_stores(tmp_path):
    """The line in ``emit_bundle`` that derives ``omitted_output_zarrs`` is read through the real emit.

    UNIT: the manifest keys of the archive a real ``emit_bundle`` writes. Two stores: ``a.zarr``
    declared in full and ``b.zarr`` qualified metadata-only by its sidecar. With the flag both are
    omitted and shallow; without it ``b.zarr`` is shallow (qualified) but NOT omitted and no
    ``omitted_output_zarrs`` key is written. The control varies the flag, not the token. Fails if
    the derivation is dropped (key absent with the flag) or made unconditional (key present
    without it, or naming the qualified store).
    """
    from hhemt.report_renderers._figure_emission import MANIFEST_METADATA_ONLY_KEY
    from tests.test_publish_deposit_paths import _world

    def _emit(base: Path, *, flag: bool) -> dict:
        world = _world(base)
        _write_tree(world.ad / "a.zarr")
        _write_tree(world.ad / "b.zarr")
        plots = world.ad / "plots"
        plots.mkdir(exist_ok=True)
        (plots / "fig_a.manifest.json").write_text(json.dumps({"source_paths_relative": ["a.zarr"]}))
        (plots / "fig_b.manifest.json").write_text(
            json.dumps({"source_paths_relative": ["b.zarr"], MANIFEST_METADATA_ONLY_KEY: ["b.zarr"]})
        )
        out = emit_bundle(world.analysis, world.base / "bundle.zip", omit_output_zarrs=flag)
        with zipfile.ZipFile(out) as zf:
            return json.loads(zf.read(BUNDLE_MANIFEST_FILENAME))

    on = _emit(tmp_path / "on", flag=True)
    assert on["omitted_output_zarrs"] == ["a.zarr", "b.zarr"]
    assert on["metadata_only_stores"] == ["a.zarr", "b.zarr"]
    off = _emit(tmp_path / "off", flag=False)
    assert off["metadata_only_stores"] == ["b.zarr"]
    assert "omitted_output_zarrs" not in off
