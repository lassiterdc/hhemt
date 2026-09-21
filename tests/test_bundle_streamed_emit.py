"""Bundle emit after the INEF-4 fix: metadata-only qualifier, planned harvest, streamed zip.

Four parts, one module. Every test here is fixture-light (`tmp_path`, or the configured-
but-not-run `synth_multi_sim_builder` for the cfg-input copy) and solver-free; no compile,
no simulation. Each test names the pre-fix input that reds it in its docstring.

Determinism is anchored on BYTES (two archives compared whole), never on a wording; the
shallow-store tests are anchored on the metadata FILE SET and on a re-open of the copy,
never on the emitter's own naming of what it copied.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import xarray as xr

matplotlib.use("Agg")

from hhemt.bundle._emit import (  # noqa: E402 -- backend pinned above, before pyplot import
    _emit_bundle_zip,
    _harvest_sources,
    _write_bundle_manifest,
    _zarr_metadata_filenames,
)
from hhemt.exceptions import ProcessingError  # noqa: E402
from hhemt.report_renderers._figure_emission import (  # noqa: E402
    MANIFEST_METADATA_ONLY_KEY,
    emit_plot_with_sources,
    harvest_metadata_only_sources,
)
from hhemt.report_renderers._provenance_audit import _declared_set_from_manifest  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_tree(store: Path, *, zarr_format: int = 3) -> None:
    """A two-level DataTree with a chunked array so the store has real chunk files."""
    ds = xr.Dataset({"h": (("y", "x"), np.arange(400.0).reshape(20, 20))}).chunk({"y": 5, "x": 5})
    tree = xr.DataTree.from_dict({"/": xr.Dataset(attrs={"hhemt_producing_sha": "abc"}), "/tritonswmm/triton": ds})
    tree.attrs["hhemt_producing_sha"] = "abc"
    tree.to_zarr(store, mode="w", consolidated=(zarr_format == 2), zarr_format=zarr_format)


#: Part 1


def test_copy_declared_inputs_never_calls_copy2(synth_multi_sim_builder, tmp_path, monkeypatch):
    """Pre-fix: `_copy_declared_inputs` calls `shutil.copy2` for every file input and this
    test's refusing stub reds it. Post-fix it uses `copyfile` (the zip normalizes every
    stat field copy2 would preserve). The stub restricts nothing about WHICH inputs are
    copied -- the real synth config drives that -- it only refuses the retired primitive.
    """
    from hhemt.bundle._emit import _copy_declared_inputs

    def _refuse(*_a, **_k):
        raise AssertionError("copy2 reached: the emit must not copystat per file")

    monkeypatch.setattr(shutil, "copy2", _refuse)
    staging = tmp_path / "staging"
    staging.mkdir()
    deposits = _copy_declared_inputs(synth_multi_sim_builder, staging)
    assert deposits == []
    assert any(p.is_file() for p in staging.rglob("*")), "at least one cfg-declared input must be carried"


#: Part 2


def test_metadata_only_qualifier_is_recorded_and_must_be_a_subset(tmp_path):
    """Arm (a): a qualifier naming a path NOT in source_paths is refused (ProcessingError).
    Arm (b): a qualifier naming the SECOND declared path (not the first) is recorded under
    the key, `source_paths_relative` is untouched, and the audit's declared set still
    contains the full path (PREFIX semantics unchanged). Pre-fix: TypeError on the kwarg.
    """
    import matplotlib.pyplot as plt

    a = tmp_path / "a.csv"
    a.write_text("x\n")
    store = tmp_path / "experiment_datatree.zarr"
    store.mkdir()
    (store / "zarr.json").write_text("{}")
    other = tmp_path / "other.csv"
    other.write_text("y\n")

    fig, _ = plt.subplots()
    with pytest.raises(ProcessingError):
        emit_plot_with_sources(fig, tmp_path / "bad.png", [a], analysis_dir=tmp_path, metadata_only_sources=[other])

    out = tmp_path / "fig.html"
    emit_plot_with_sources(
        "<html></html>", out, [a, store], analysis_dir=tmp_path, output_format="html", metadata_only_sources=[store]
    )
    manifest = json.loads((tmp_path / "fig.manifest.json").read_text())
    assert manifest["source_paths_relative"] == ["a.csv", "experiment_datatree.zarr"]
    assert manifest[MANIFEST_METADATA_ONLY_KEY] == ["experiment_datatree.zarr"]
    assert store.resolve() in _declared_set_from_manifest(out, tmp_path)

    # Differently-positioned satisfying input: no qualifier -> no key at all (byte-identity).
    out2 = tmp_path / "plain.html"
    emit_plot_with_sources("<html></html>", out2, [a], analysis_dir=tmp_path, output_format="html")
    assert MANIFEST_METADATA_ONLY_KEY not in json.loads((tmp_path / "plain.manifest.json").read_text())


def _sidecar(plots: Path, stem: str, declared: list[str], qualified: list[str] | None = None) -> None:
    plots.mkdir(parents=True, exist_ok=True)
    payload: dict = {"plot_id": stem, "source_paths_relative": declared}
    if qualified:
        payload[MANIFEST_METADATA_ONLY_KEY] = qualified
    (plots / f"{stem}.manifest.json").write_text(json.dumps(payload))


def test_any_full_declaration_wins_across_roots(tmp_path):
    """Invariant: a store is shallow iff EVERY declarant qualified it.
    Arm (a, violating shape caught): plots/ qualifies X, eda/ declares X in full -> X is NOT
    shallow. Arm (b, satisfying, differently positioned): only eda/ exists and qualifies X ->
    X IS shallow. Pre-fix: ImportError on `harvest_metadata_only_sources`.
    """
    root = tmp_path / "analysis"
    x = "experiment_datatree.zarr"
    _sidecar(root / "plots", "metadata", [x], qualified=[x])
    _sidecar(root / "eda", "b4b", [x])
    assert harvest_metadata_only_sources((root / "plots", root / "eda"), root) == frozenset()

    root2 = tmp_path / "analysis2"
    _sidecar(root2 / "eda", "b4b", [x, "weather.nc"], qualified=[x])
    assert harvest_metadata_only_sources((root2 / "plots", root2 / "eda"), root2) == frozenset({(root2 / x).resolve()})


#: Part 3


def test_streamed_archive_is_deterministic_and_equals_a_materialized_one(tmp_path):
    """Two streamed emits are byte-identical, AND a streamed emit equals an emit whose
    staging tree holds the same files materialized -- so WHERE bytes are read from is not a
    determinism input. Pre-fix: `streamed=` is an unexpected keyword (TypeError)."""
    staging = tmp_path / "staging"
    (staging / "plots").mkdir(parents=True)
    (staging / "cfg_analysis.yaml").write_text("k: v\n")
    (staging / "plots" / "a-b.html").write_text("<html>1</html>")
    src_tree = tmp_path / "analysis"
    (src_tree / "eda" / "s.zarr").mkdir(parents=True)
    (src_tree / "eda" / "s.zarr" / "zarr.json").write_text("{}")
    (src_tree / "eda" / "s.zarr" / "0").write_bytes(bytes(range(256)) * 64)
    (src_tree / "plots").mkdir()
    (src_tree / "plots" / "a" / "b.html").parent.mkdir()
    (src_tree / "plots" / "a" / "b.html").write_text("<html>2</html>")
    streamed = {str(p.relative_to(src_tree).as_posix()): p for p in sorted(src_tree.rglob("*")) if p.is_file()}

    z1, z2 = tmp_path / "1.zip", tmp_path / "2.zip"
    _emit_bundle_zip(staging, z1, streamed=streamed)
    _emit_bundle_zip(staging, z2, streamed=streamed)
    assert _sha(z1) == _sha(z2)

    materialized = tmp_path / "materialized"
    shutil.copytree(staging, materialized)
    for arc, p in streamed.items():
        dest = materialized / arc
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    z3 = tmp_path / "3.zip"
    _emit_bundle_zip(materialized, z3)
    assert _sha(z1) == _sha(z3), "reading harvested bytes from source paths changed the archive"
    with zipfile.ZipFile(z1) as zf:
        assert zf.testzip() is None
        assert zf.namelist() == [
            "cfg_analysis.yaml",
            "eda/s.zarr/0",
            "eda/s.zarr/zarr.json",
            "plots/a/b.html",
            "plots/a-b.html",
        ]


def test_harvest_plans_entries_dedupes_across_keys_and_records_absent(tmp_path):
    """The plan walks a path declared by two figures ONCE, keys every file of a directory
    source, and records an absent source without an entry. Pre-fix: ImportError."""
    root = tmp_path / "analysis"
    (root / "eda" / "d.zarr").mkdir(parents=True)
    (root / "eda" / "d.zarr" / "zarr.json").write_text("{}")
    (root / "eda" / "d.zarr" / "0").write_bytes(b"\x00")
    flat = root / "weather.nc"
    flat.write_bytes(b"nc")
    absent = root / "_status" / "_du.json"
    entries, declared_absent, shallow = _harvest_sources(
        {"fig_a": [root / "eda" / "d.zarr", flat, absent], "fig_b": [flat, root / "eda" / "d.zarr"]}, root
    )
    assert entries == {
        "eda/d.zarr/0": root / "eda" / "d.zarr" / "0",
        "eda/d.zarr/zarr.json": root / "eda" / "d.zarr" / "zarr.json",
        "weather.nc": flat,
    }
    assert declared_absent == ["_status/_du.json"]
    assert shallow == []


@pytest.mark.parametrize("zarr_format", [3, 2])
def test_metadata_only_store_is_carried_shallowly_and_reopens(tmp_path, zarr_format):
    """Arm (a): qualified -> the plan carries ONLY zarr metadata documents (no chunk file
    names), the shallow list names the store, and a copy built from the plan re-opens with
    identical attrs and groups. Arm (b, differently positioned satisfying input): the same
    store un-qualified -> every file is carried. Parametrized over BOTH zarr formats so a
    metadata-filename set written from memory of one format cannot pass. Pre-fix: ImportError.
    """
    root = tmp_path / "analysis"
    root.mkdir()
    store = root / "experiment_datatree.zarr"
    _write_tree(store, zarr_format=zarr_format)
    all_files = {p for p in store.rglob("*") if p.is_file()}
    meta_names = _zarr_metadata_filenames()
    chunk_files = {p for p in all_files if p.name not in meta_names}
    assert chunk_files, "fixture must have chunk files or the test proves nothing"

    entries, _absent, shallow = _harvest_sources(
        {"metadata": [store]}, root, metadata_only=frozenset({store.resolve()})
    )
    assert shallow == ["experiment_datatree.zarr"]
    assert set(entries.values()) == all_files - chunk_files
    copy = tmp_path / "copy" / "experiment_datatree.zarr"
    for arc, p in entries.items():
        dest = tmp_path / "copy" / arc
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    full = xr.open_datatree(store, engine="zarr", chunks=None, consolidated=False)
    shallow_tree = xr.open_datatree(copy, engine="zarr", chunks=None, consolidated=False)
    assert dict(shallow_tree.attrs) == dict(full.attrs)
    assert list(shallow_tree.groups) == list(full.groups)

    entries_full, _absent, shallow_none = _harvest_sources({"per_sim": [store]}, root)
    assert shallow_none == []
    assert set(entries_full.values()) == all_files


def test_bundle_manifest_names_shallow_stores_only_when_present(tmp_path):
    """`metadata_only_stores` is present iff non-empty (absent-when-clean byte-identity).
    Pre-fix: TypeError on the kwarg."""
    s1 = tmp_path / "s1"
    s1.mkdir()
    _write_bundle_manifest(
        s1,
        sources_by_renderer={},
        analysis_id="a",
        git_sha="deadbeef",
        metadata_only_stores=["experiment_datatree.zarr"],
    )
    assert json.loads((s1 / "bundle_manifest.json").read_text())["metadata_only_stores"] == ["experiment_datatree.zarr"]
    s2 = tmp_path / "s2"
    s2.mkdir()
    _write_bundle_manifest(s2, sources_by_renderer={}, analysis_id="a", git_sha="deadbeef", metadata_only_stores=[])
    assert "metadata_only_stores" not in json.loads((s2 / "bundle_manifest.json").read_text())
