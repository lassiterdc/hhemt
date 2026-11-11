"""The publish route writes the same relative-path form the render bundle already writes.

A data deposit's ``cfg_analysis.yaml`` and ``cfg_system.yaml`` used to be written as a bare
``model_dump``, so every absolute path the producer's machine held went into a public, immutable
record. The render bundle already rewrote the same two files through ``_PATH_FIELD_POLICY``; this
change makes the data deposit call that one writer (``emit_relative_config_pair``), stages the pair in a
folder ``publish_analysis`` owns instead of the LIVE analysis directory, refuses when two DIFFERENT files
would take one name, makes the toolkit's own reader recognise the form, and spells the bundle
manifest's source paths the way the archive names them.

SOLVER-FREE. Every arm builds REAL ``system_config`` and ``analysis_config`` models and runs the
REAL ``emit_bundle`` / ``_copy_declared_inputs`` / ``_harvest_sources`` / ``_emit_bundle_zip`` over a
temporary tree; nothing compiles and no solver runs.

ONE BODY, TWO TREES. The arms the completion artifact says must fail against the parent commit are
written once and run on both trees, so a red on the parent is the SAME assertion that goes green on the
change. Three idioms make that possible without a ``TypeError`` or a collection error standing in for the
real failure:

- names imported at module top exist on both trees (``emit_relative_config_pair``, the staged-names
  table and ``_plan_declared_inputs`` are imported INSIDE the arm that expects them);
- a parameter the change adds (``staging``, ``analysis_dir``) is passed only when
  ``inspect.signature`` says the function accepts it;
- a refusal arm that finds none does not ``pytest.raises``-fail on "did not raise": it reports what the
  emit PUBLISHED and which file it dropped, as an ``AssertionError`` naming that file.

Every file in a collision arm carries DISTINCT BYTES, or a dropped file is invisible.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import os
import re
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hhemt import publishing
from hhemt.bundle import _emit
from hhemt.bundle._path_policy import enumerate_path_fields
from hhemt.config.analysis import analysis_config
from hhemt.config.system import system_config
from hhemt.exceptions import ConfigurationError, PublishError
from tests.test_config_validation import _minimal_analysis_config_dict, _minimal_system_config_dict

#: Validate with the path-existence assertion off: a declared input that is absent is a case several
#: arms need, and the rewrite and the refusals read only the path STRINGS.
CTX = {"existence": "metadata"}

CFG_FILES = ("cfg_analysis.yaml", "cfg_system.yaml")


# --------------------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------------------


def _put(path: Path, data: bytes | str = b"") -> Path:
    """Write ``data`` (bytes, or str encoded) at ``path``, creating parents; return ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode())
    return path


def _world(
    tmp_path: Path,
    *,
    nested: bool = False,
    system_over: dict | None = None,
    analysis_over: dict | None = None,
    case_manifest: Path | None = None,
    base_name: str = "w",
) -> SimpleNamespace:
    """A REAL pair of config models plus the directories they name.

    ``nested`` puts the analysis directory two levels INSIDE the system directory (the layout the
    relative-value arm needs); otherwise the two are siblings under one base and a path inside
    neither is "outside". ``system_over`` / ``analysis_over`` replace config values after the minimal
    dicts are built; a value of ``None`` clears the field.
    """
    base = tmp_path / base_name
    sd = base / "system"
    ad = sd / "a" / "b" if nested else base / "analysis"
    out = base / "outside"
    for d in (sd, ad, out):
        d.mkdir(parents=True, exist_ok=True)
    sdict = _minimal_system_config_dict(tmp_path)
    sdict["system_directory"] = str(sd)
    adict = _minimal_analysis_config_dict(tmp_path)
    adict["analysis_dir"] = str(ad)
    adict["dataset_license"] = "CC0-1.0"
    sdict.update(system_over or {})
    adict.update(analysis_over or {})
    analysis = SimpleNamespace(
        analysis_paths=SimpleNamespace(analysis_dir=ad),
        cfg_analysis=analysis_config.model_validate(adict, context=CTX),
        _system=SimpleNamespace(cfg_system=system_config.model_validate(sdict, context=CTX)),
        case_manifest_yaml=case_manifest,
        cfg_hpc_system=None,
    )
    return SimpleNamespace(analysis=analysis, base=base, sd=sd, ad=ad, out=out)


def _every_path_field_absolute(tmp_path: Path) -> SimpleNamespace:
    """A world in which EVERY Path field of both models holds an ABSOLUTE path.

    Inside the analysis directory, inside the system directory, and outside both, so each branch of the
    rewrite is exercised. The guard at the end makes the arm total: a Path field added to either model
    that this builder does not set fails here, instead of silently leaving a field unexercised.
    """
    base = tmp_path / "w"
    sd, ad, out = base / "system", base / "analysis", base / "outside"
    # A shapefile input is a set of files: the emit and the ingest refuse a .shp with no .shx or .dbf beside it.
    _put(sd / "gis" / "ws.shx")
    _put(sd / "gis" / "ws.dbf")
    world = _world(
        tmp_path,
        system_over={
            "watershed_gis_polygon": str(_put(sd / "gis" / "ws.shp")),
            "DEM_fullres": str(_put(out / "dem.tif")),
            "landuse_lookup_file": str(_put(sd / "gis" / "lu.csv")),
            "SWMM_hydraulics": str(_put(out / "hyd.inp")),
            "SWMM_hydrology": str(_put(sd / "hy.inp")),
            "SWMM_full": str(_put(out / "full.inp")),
            "swmm_coupling_mode": "list",
            "swmm_coupling_nodes_file": str(_put(sd / "nodes.txt")),
            "landuse_raster": str(_put(out / "lu.tif")),
            "SWMM_software_directory": str(out / "swmm_src"),
            "TRITONSWMM_software_directory": str(out / "triton_src"),
            "subcatchment_raingage_mapping": str(_put(sd / "map.csv")),
            "triton_swmm_configuration_template": str(_put(out / "t.cfg")),
        },
        analysis_over={
            "weather_timeseries": str(_put(ad / "in" / "weather.nc")),
            "dataset_license": None,
            "dataset_license_file": str(_put(out / "lic.yaml", "name: Our License\n")),
            "storm_tide_boundary_line_gis": str(_put(out / "line.gpkg")),
            "weather_event_summary_csv": str(_put(ad / "sum.csv")),
            "toggle_sensitivity_analysis": True,
            "sensitivity_analysis": str(_put(ad / "sens.csv")),
            "weather_events_to_simulate": str(_put(ad / "ev.csv")),
            "weather_event_windows_csv": str(_put(ad / "win.csv")),
            "experiment_cfg_yaml": str(_put(sd / "exp.yaml")),
            "brand_theme": str(_put(out / "brand.yaml")),
            "static_plot_configs": [str(_put(ad / "p" / "s1.yaml")), str(_put(out / "s2.yaml"))],
        },
    )
    for cfg in (world.analysis._system.cfg_system, world.analysis.cfg_analysis):
        for name in enumerate_path_fields(type(cfg)):
            value = getattr(cfg, name)
            assert value not in (None, []), f"the builder left Path field {name!r} unset; extend it"
            for elem in value if isinstance(value, list) else [value]:
                assert Path(elem).is_absolute(), f"{name} is not absolute in the fixture: {elem}"
    return world


def _call_deposit_set(analysis, staging: Path) -> list[Path]:
    """``_deposit_set`` with ``staging`` passed only if the function accepts it (one body, two trees)."""
    kwargs = {"staging": staging} if "staging" in inspect.signature(publishing._deposit_set).parameters else {}
    return publishing._deposit_set(analysis, "analysis_datatree.zarr", **kwargs)


def _staged_cfgs(deposit: list[Path]) -> dict[str, dict]:
    """The parsed config pair, located BY FILE NAME in the returned deposit list."""
    return {p.name: yaml.safe_load(p.read_text()) for p in deposit if p.name in CFG_FILES}


def _path_values(cfg: dict, model: type) -> list[tuple[str, str]]:
    """``(field, string)`` for every string held by a Path field of ``model`` in ``cfg``."""
    out: list[tuple[str, str]] = []
    for name in enumerate_path_fields(model):
        value = cfg.get(name)
        out += [(name, e) for e in (value if isinstance(value, list) else [value]) if isinstance(e, str)]
    return out


def _write_crate(ad: Path, license_id: str = "CC0-1.0") -> None:
    """The consolidated crate and zarr a publish needs to find in the analysis directory."""
    uri = publishing._SPDX_LICENSE_TABLE[license_id]["uri"]
    doc = {
        "@context": "https://w3id.org/ro/crate/1.1/context",
        "@graph": [
            {"@id": "ro-crate-metadata.json", "@type": "CreativeWork"},
            {"@id": "./", "@type": "Dataset", "name": "Test", "license": {"@id": uri}},
        ],
    }
    _put(ad / "ro-crate-metadata.json", json.dumps(doc))
    _put(ad / "analysis_datatree.zarr" / "zarr.json", "{}")


def _install_stub(monkeypatch) -> list[dict]:
    """A recording Zenodo target. Each call records the deposited file NAMES and their parsed text."""
    calls: list[dict] = []

    def _publish(**kw):
        calls.append(
            {
                "names": sorted(p.name for p in kw["deposit"]),
                "text": {p.name: p.read_text() for p in kw["deposit"] if p.is_file() and p.suffix == ".yaml"},
            }
        )
        return {"target": "zenodo", "data_doi": "10.0/x", "software_doi": None, "record_url": "u"}

    monkeypatch.setitem(publishing._TARGETS, "zenodo", lambda: SimpleNamespace(publish=_publish))
    return calls


def _publish(world, monkeypatch, **kw):
    """Run ``publish_analysis`` on the data deposit; return ``("refused", exc)`` or ``("published", call)``."""
    calls = _install_stub(monkeypatch)
    _write_crate(world.ad)
    try:
        publishing.publish_analysis(world.analysis, target="zenodo", **kw)
    except (PublishError, ConfigurationError) as exc:
        assert calls == [], "the target was called although the publish was refused"
        return "refused", exc
    return "published", calls[0]


def _assert_refused_naming(outcome, *paths: Path, exc_type=PublishError) -> None:
    """The publish was refused with ``exc_type`` and BOTH (every) full resolved path is in the message."""
    kind, payload = outcome
    if kind != "refused":
        raise AssertionError(
            f"the publish went through and deposited {payload['names']}; the files {[str(p) for p in paths]} "
            "claim one name and one of them was dropped silently"
        )
    assert isinstance(payload, exc_type), f"refused with {type(payload).__name__}, expected {exc_type.__name__}"
    for p in paths:
        assert str(p.resolve()) in str(payload), f"{p.resolve()} is not named in: {payload}"


def _run_emit(world, harvested: tuple[str, ...] = (), exclude=None) -> Path:
    """The REAL ``emit_bundle`` over the world. ``harvested`` are analysis-relative paths one renderer declares."""
    plots = world.ad / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    (plots / "fig.manifest.json").write_text(json.dumps({"source_paths_relative": list(harvested)}))
    return _emit.emit_bundle(world.analysis, world.base / "bundles" / "b.zip", exclude_config=exclude)


def _zip_names(zpath: Path) -> list[str]:
    with zipfile.ZipFile(zpath) as zf:
        return zf.namelist()


def _zip_bytes(zpath: Path, name: str) -> bytes | None:
    with zipfile.ZipFile(zpath) as zf:
        return zf.read(name) if name in zf.namelist() else None


def _emit_outcome(world, harvested=(), exclude=None):
    """``("refused", exc)`` or ``("published", zip path)`` from the real emit."""
    try:
        return "published", _run_emit(world, harvested, exclude)
    except ConfigurationError as exc:
        return "refused", exc


def _assert_emit_refused_naming(outcome, archive_name: str, files: dict[Path, bytes]) -> None:
    """The emit refused naming every file; or FAIL naming the file the published bundle dropped."""
    kind, payload = outcome
    if kind == "refused":
        for p in files:
            assert str(p.resolve()) in str(payload), f"{p.resolve()} is not named in: {payload}"
        return
    held = _zip_bytes(payload, archive_name)
    dropped = sorted(str(p) for p, b in files.items() if b != held)
    raise AssertionError(
        f"emit PUBLISHED a bundle whose {archive_name!r} holds {held!r}; the file(s) {dropped} were dropped silently"
    )


# --------------------------------------------------------------------------------------
# (a) the data deposit's configs are written in the relative form, into the staging folder
# --------------------------------------------------------------------------------------


def test_data_deposit_configs_are_written_in_the_relative_form(tmp_path):
    """Property: no Path field of either staged config holds an absolute path, and each kind reads right.

    Class: a deposit from a real pair of models with EVERY Path field absolute under the analysis
    directory, the system directory and an outside directory. Why required: the absolute strings are the
    producer's host paths, and they were in a public, immutable record. Killed by: writing ``model_dump``
    as is (the parent), or rewriting only one of the two configs. Second correct implementation: the
    rewrite called directly in ``_deposit_set`` instead of through the shared writer.

    The must-fire control is this body on the parent's ``_deposit_set``, which fails on the assertion
    that lists the absolute values it found.
    """
    world = _every_path_field_absolute(tmp_path)
    staging = tmp_path / "staging"
    staging.mkdir()
    deposit = _call_deposit_set(world.analysis, staging)
    cfgs = _staged_cfgs(deposit)
    assert sorted(cfgs) == sorted(CFG_FILES)
    models = {"cfg_analysis.yaml": analysis_config, "cfg_system.yaml": system_config}
    absolute = [(n, f, v) for n, m in models.items() for f, v in _path_values(cfgs[n], m) if Path(v).is_absolute()]
    assert not absolute, f"the staged configs carry ABSOLUTE producer paths: {absolute}"

    a, s = cfgs["cfg_analysis.yaml"], cfgs["cfg_system.yaml"]
    assert a["analysis_dir"] == "." and s["system_directory"] == "."
    assert a["weather_timeseries"] == "in/weather.nc", "a path inside the analysis directory reads analysis-relative"
    assert s["landuse_lookup_file"] == "gis/lu.csv", "a path inside the system directory reads system-relative"
    assert s["SWMM_full"] == "external/full.inp", "a path inside neither reads external/{filename}"
    assert a["static_plot_configs"] == ["p/s1.yaml", "external/s2.yaml"], "a list field is rewritten element by element"
    assert s["SWMM_software_directory"] is None and s["TRITONSWMM_software_directory"] is None


def test_the_writer_and_the_deposit_set_have_the_signatures_the_totality_pass_keys_on():
    """Property: the destination parameter is NAMED ``staging`` and the deposit set requires it by keyword.

    Class: the extracted writer and its caller. Why required: completion (j) matches
    ``staging / ...`` writes, so a renamed parameter would make the config write invisible to it.
    Killed by: naming the parameter ``dest`` or ``out``. Second correct implementation: ``staging``
    positional-or-keyword on the deposit set (rejected: the row fixes it keyword-only).
    """
    assert list(inspect.signature(_emit.emit_relative_config_pair).parameters) == ["analysis", "staging"]
    param = inspect.signature(publishing._deposit_set).parameters["staging"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY and param.default is inspect.Parameter.empty


# --------------------------------------------------------------------------------------
# (c) nothing is written into the LIVE analysis directory
# --------------------------------------------------------------------------------------


def _live_root_empty_arm(module, tmp_path: Path) -> None:
    """An EMPTY live root holds no config after the deposit set (red on the parent, which creates both)."""
    world = _world(tmp_path)
    staging = tmp_path / "staging"
    staging.mkdir()
    kwargs = {"staging": staging} if "staging" in inspect.signature(module._deposit_set).parameters else {}
    module._deposit_set(world.analysis, "analysis_datatree.zarr", **kwargs)
    created = sorted(f for f in CFG_FILES if (world.ad / f).exists())
    assert not created, f"the deposit set wrote {created} into the LIVE analysis directory {world.ad}"


def _live_root_seeded_arm(module, tmp_path: Path) -> None:
    """A live root seeded with the absolute dumps ``eda()`` leaves keeps both byte-equal."""
    world = _world(tmp_path)
    seeded = {
        "cfg_analysis.yaml": yaml.safe_dump(world.analysis.cfg_analysis.model_dump(mode="json")),
        "cfg_system.yaml": yaml.safe_dump(world.analysis._system.cfg_system.model_dump(mode="json")),
    }
    for name, text in seeded.items():
        (world.ad / name).write_text(text)
    staging = tmp_path / "staging"
    staging.mkdir()
    kwargs = {"staging": staging} if "staging" in inspect.signature(module._deposit_set).parameters else {}
    module._deposit_set(world.analysis, "analysis_datatree.zarr", **kwargs)
    changed = sorted(n for n, text in seeded.items() if (world.ad / n).read_text() != text)
    assert not changed, f"the deposit set overwrote the LIVE copies {changed}"


def _mutant_publishing(tmp_path: Path):
    """A copy of ``publishing.py`` whose deposit set writes the staged pair into the LIVE root."""
    src = Path(publishing.__file__).read_text()
    needle = "emit_relative_config_pair(analysis, staging)"
    assert src.count(needle) == 1, "the mutation site moved; update the mutant"
    path = tmp_path / "mutant" / "publishing_mutant.py"
    _put(path, src.replace(needle, "emit_relative_config_pair(analysis, root)"))
    spec = importlib.util.spec_from_file_location("publishing_mutant", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert Path(module.__file__) == path, "the mutant module is not the mutated copy"
    return module


def test_the_deposit_set_writes_nothing_into_an_empty_live_analysis_directory(tmp_path):
    """Property: (c1) the data deposit never creates ``cfg_*.yaml`` in the live analysis directory.

    Class: an EMPTY live root. Why required: five readers load those live copies, and ``hhemt eda
    --analysis-dir`` builds runnable configs from them; written with relative paths they would hand it a
    ``system_directory`` of ``.`` resolved against the caller's working directory. Killed by: writing the
    pair into the root. Second correct implementation: staging the pair anywhere but the live root.

    RED on the parent, which creates both files there.
    """
    _live_root_empty_arm(publishing, tmp_path)


def test_the_deposit_set_leaves_the_live_copies_of_an_analysis_that_ran_eda_byte_equal(tmp_path):
    """Property: (c2) a live root seeded with the absolute dumps ``eda()`` leaves keeps both byte-equal.

    Class: a live root that already holds both configs. Why required: they are the toolkit's working
    copies and the data deposit must not overwrite them. PERMANENTLY GREEN on the parent: the parent
    rewrites the live copies with the same ``model_dump`` bytes that seeded them, so this arm kept alone
    could not tell it from a correct build. Its must-fail control is the mutant below, which writes the
    staged pair into the live root and must fail this arm.
    """
    _live_root_seeded_arm(publishing, tmp_path)


def test_a_deposit_set_that_writes_the_live_root_fails_both_live_root_arms(tmp_path):
    """Property: both live-root arms FAIL against a mutant that writes the staged pair into the live root.

    Class: the mutation of the one call that names the destination. Why required: (c2) is green on the
    parent, so without this control it could pass on a deposit set that destroyed the live copies.
    Killed by: arms that do not read the live root. Second correct implementation: a mutant that adds a
    second write into the live root instead of redirecting the first.
    """
    if "staging" not in inspect.signature(publishing._deposit_set).parameters:
        pytest.skip("the parent has no staging parameter to mutate")
    mutant = _mutant_publishing(tmp_path)
    with pytest.raises(AssertionError):
        _live_root_empty_arm(mutant, tmp_path / "m1")
    with pytest.raises(AssertionError):
        _live_root_seeded_arm(mutant, tmp_path / "m2")


# --------------------------------------------------------------------------------------
# (b) refusals: two DIFFERENT files may not reach one name
# --------------------------------------------------------------------------------------


def test_publish_refuses_an_analysis_inside_and_a_system_inside_path_with_one_relative_text(tmp_path, monkeypatch):
    """Property: ``{analysis}/x/y.nc`` and ``{system}/x/y.nc`` both read ``x/y.nc``, so the publish refuses.

    Class: two different files, one inside each directory, whose rewritten strings are equal. Why
    required: the string names one place and only one file can be there. Killed by: comparing field
    names rather than the files behind the strings. Second correct implementation: refusing in the
    rewrite itself rather than in the deposit set.
    """
    base = tmp_path / "w"
    a = _put(base / "analysis" / "x" / "y.nc", b"analysis bytes")
    s = _put(base / "system" / "x" / "y.nc", b"system bytes")
    world = _world(
        tmp_path,
        system_over={"landuse_lookup_file": str(s)},
        analysis_over={"storm_tide_boundary_line_gis": str(a)},
    )
    _assert_refused_naming(_publish(world, monkeypatch), a, s)


def test_publish_refuses_one_external_filename_from_the_system_and_analysis_configs(tmp_path, monkeypatch):
    """Property: two different outside files of one name collide at ``external/{filename}``.

    Class: the same filename in two outside directories, one per config. Why required: the fallback
    keeps only the filename, so two outside files of one name are one bundle name. Killed by: keying the
    comparison on the full path. Second correct implementation: giving the second a numbered name
    (rejected: the config string would no longer say where the file came from).
    """
    base = tmp_path / "w"
    one = _put(base / "o1" / "w.nc", b"one")
    two = _put(base / "o2" / "w.nc", b"two")
    world = _world(
        tmp_path, system_over={"landuse_raster": str(one)}, analysis_over={"storm_tide_boundary_line_gis": str(two)}
    )
    _assert_refused_naming(_publish(world, monkeypatch), one, two)


def test_publish_refuses_an_already_relative_value_equal_to_a_rewritten_output(tmp_path, monkeypatch):
    """Property: a relative value is identified by the file it names, so it can collide with a rewritten one.

    Class: ``ext/w.nc`` (relative, so ``{analysis}/ext/w.nc``) beside an absolute ``{system}/ext/w.nc``,
    which rewrites to the SAME string. Why required: before D-199 the relative value passed through
    untouched and the two strings were equal with nobody looking. Killed by: treating a relative value
    as opaque. Second correct implementation: resolving against the system directory (rejected: the
    copy step resolves against the analysis directory).
    """
    base = tmp_path / "w"
    inside = _put(base / "analysis" / "ext" / "w.nc", b"analysis bytes")
    system_file = _put(base / "system" / "ext" / "w.nc", b"system bytes")
    world = _world(
        tmp_path,
        system_over={"landuse_lookup_file": str(system_file)},
        analysis_over={"storm_tide_boundary_line_gis": "ext/w.nc"},
    )
    _assert_refused_naming(_publish(world, monkeypatch), inside, system_file)


def test_publish_accepts_the_same_outside_file_in_two_fields_and_distinct_filenames(tmp_path, monkeypatch):
    """Property: one file named twice is ONE file, and distinct names never collide (the over-fire guards).

    Class: the same outside path in a system and an analysis field; two different outside files of
    DIFFERENT names. Why required: a comparison on strings alone, or on field count, would refuse
    correct deposits. Killed by: counting claims instead of distinct files. Second correct
    implementation: de-duplicating by inode (rejected: a copied file is another file).
    """
    base = tmp_path / "w"
    shared = _put(base / "o1" / "shared.nc", b"shared")
    world = _world(
        tmp_path,
        system_over={"landuse_raster": str(shared)},
        analysis_over={"storm_tide_boundary_line_gis": str(shared)},
    )
    kind, call = _publish(world, monkeypatch)
    assert kind == "published", f"one file named twice was refused: {call}"
    a, s = yaml.safe_load(call["text"]["cfg_analysis.yaml"]), yaml.safe_load(call["text"]["cfg_system.yaml"])
    assert a["storm_tide_boundary_line_gis"] == s["landuse_raster"] == "external/shared.nc"

    distinct = _world(
        tmp_path / "d",
        system_over={"landuse_raster": str(_put(base / "o1" / "one.nc", b"1"))},
        analysis_over={"storm_tide_boundary_line_gis": str(_put(base / "o2" / "two.nc", b"2"))},
    )
    assert _publish(distinct, monkeypatch)[0] == "published"


def test_reprex_bundle_publish_receives_the_emit_s_configuration_error(tmp_path, monkeypatch):
    """Property: through ``deposit_source="reprex_bundle"`` the caller receives the emit's ``ConfigurationError``.

    Class: the colliding pair of the arm above, deposited as a runnable bundle. Why required: the
    refusal lives in the emit for that route, and a wrapper that swallowed or re-typed it would hide it.
    Killed by: catching the emit's error inside ``publish_analysis``. Second correct implementation:
    the emit raising ``PublishError`` (rejected: the emit is not a publishing module).
    """
    base = tmp_path / "w"
    one = _put(base / "o1" / "w.nc", b"one")
    two = _put(base / "o2" / "w.nc", b"two")
    world = _world(
        tmp_path, system_over={"landuse_raster": str(one)}, analysis_over={"storm_tide_boundary_line_gis": str(two)}
    )
    (world.ad / "plots").mkdir()
    (world.ad / "plots" / "fig.manifest.json").write_text(json.dumps({"source_paths_relative": []}))
    outcome = _publish(world, monkeypatch, deposit_source="reprex_bundle")
    _assert_refused_naming(outcome, one, two, exc_type=ConfigurationError)


# --------------------------------------------------------------------------------------
# (b) the archive namespace, measured on the real emit functions
# --------------------------------------------------------------------------------------


def test_gap1_a_relative_and_an_absolute_value_of_one_filename_refuse(tmp_path):
    """Property: GAP1 -- ``../o1/w.nc`` beside ``/o2/w.nc`` collide at ``external/w.nc`` and the emit refuses.

    Class: a relative value that points outside the analysis directory beside an absolute value of the
    same filename. Why required: the parent copies both to ``external/w.nc`` and the second overwrites
    the first. Killed by: skipping relative values in the namespace check. Second correct implementation:
    refusing in ``_plan_declared_inputs`` rather than in ``emit_bundle``.
    """
    base = tmp_path / "w"
    one = _put(base / "o1" / "w.nc", b"one bytes")
    two = _put(base / "o2" / "w.nc", b"two bytes")
    world = _world(
        tmp_path,
        analysis_over={"storm_tide_boundary_line_gis": "../o1/w.nc", "weather_event_summary_csv": str(two)},
    )
    _assert_emit_refused_naming(_emit_outcome(world), "external/w.nc", {one: b"one bytes", two: b"two bytes"})


def test_gap2_a_config_input_and_a_harvested_source_of_one_filename_refuse(tmp_path):
    """Property: GAP2 -- a config ``/o1/dem.tif`` beside a harvested ``/o2/dem.tif`` refuse.

    Class: a declared input and a renderer-declared source that both land at ``external/dem.tif``.
    Why required: ``_emit_bundle_zip`` lets staging win, so the harvested file is dropped. Killed by:
    comparing only the declared inputs among themselves. Second correct implementation: giving the
    harvested file a distinct archive name (rejected: renderers read it at the name they declared).
    """
    base = tmp_path / "w"
    config = _put(base / "o1" / "dem.tif", b"config bytes")
    harvested = _put(base / "o2" / "dem.tif", b"harvested bytes")
    world = _world(tmp_path, system_over={"landuse_raster": str(config)})
    outcome = _emit_outcome(world, harvested=("../o2/dem.tif",))
    _assert_emit_refused_naming(outcome, "external/dem.tif", {config: b"config bytes", harvested: b"harvested bytes"})


def test_gap3_a_system_relative_config_input_and_an_analysis_relative_harvest_refuse(tmp_path):
    """Property: GAP3 -- ``{system}/gis/a.gpkg`` and a harvested ``{analysis}/gis/a.gpkg`` both read ``gis/a.gpkg``.

    Class: a system-relative declared input and an analysis-relative harvested source of one name.
    Why required: the same drop as GAP2, through the two relative branches instead of the external one.
    Killed by: checking only the ``external/`` namespace. Second correct implementation: a single
    claims table over every name, which is what this change builds.
    """
    base = tmp_path / "w"
    system_file = _put(base / "system" / "gis" / "a.gpkg", b"system bytes")
    harvested = _put(base / "analysis" / "gis" / "a.gpkg", b"analysis bytes")
    world = _world(tmp_path, system_over={"landuse_raster": str(system_file)})
    outcome = _emit_outcome(world, harvested=("gis/a.gpkg",))
    _assert_emit_refused_naming(outcome, "gis/a.gpkg", {system_file: b"system bytes", harvested: b"analysis bytes"})


def test_overfire_a_relative_and_an_absolute_value_of_ONE_file_publish_and_lose_nothing(tmp_path):
    """Property: ``x.nc`` and the absolute path of that same file are one file, so the emit publishes it.

    Class: a relative value beside the absolute path of the file it names. Why required: the guard
    against a check that compares strings and refuses a correct bundle. Killed by: counting claims. This
    arm is green on the parent too, as it must be: nothing is lost there.
    """
    world = _world(tmp_path)
    x = _put(world.ad / "x.nc", b"x bytes")
    world = _world(
        tmp_path,
        analysis_over={"storm_tide_boundary_line_gis": "x.nc", "weather_event_summary_csv": str(x)},
    )
    kind, payload = _emit_outcome(world)
    assert kind == "published", f"one file named twice was refused: {payload}"
    assert _zip_bytes(payload, "x.nc") == b"x bytes"


def test_two_renderers_declaring_different_outside_sources_of_one_name_refuse_in_the_harvest(tmp_path):
    """Property: ``_harvest_sources`` itself refuses two DIFFERENT sources under one archive name.

    Class: two renderers each declaring an outside ``f.nc`` of different bytes. Why required: after
    ``entries.setdefault`` the loser is gone from the plan, so no check over the RETURNED plan can see
    it. Killed by: checking only after the harvest. Second correct implementation: raising from the
    caller on a duplicate it inferred (rejected: the information is gone by then).
    """
    ad = tmp_path / "analysis"
    ad.mkdir()
    one = _put(tmp_path / "o1" / "f.nc", b"one")
    two = _put(tmp_path / "o2" / "f.nc", b"two")
    try:
        entries = _emit._harvest_sources({"r1": [one], "r2": [two]}, ad)[0]
    except ConfigurationError as exc:
        for p in (one, two):
            assert str(p.resolve()) in str(exc)
        return
    raise AssertionError(f"the harvest planned {entries}; the source {two} was dropped silently")


def test_a_directory_inputs_member_and_a_file_input_of_one_bundle_name_refuse(tmp_path):
    """Property: a directory input's MEMBER and a file input that reach one bundle name refuse.

    Class: a directory input whose member shares its bundle-relative name with a different file input.
    Why required: copytree merges the directory and copyfile then overwrites a member, so one file is
    lost without a message; the config STRINGS differ ('data' and 'data/a.nc'), so only the per-member
    comparison can see it. Killed by: claiming one name per directory instead of one per member.
    Second correct implementation: planning the members inside _plan_declared_inputs and refusing there.
    """
    base = tmp_path / "w"
    member = _put(base / "system" / "data" / "a.nc", b"directory member bytes")
    file_ = _put(base / "analysis" / "data" / "a.nc", b"file input bytes")
    world = _world(
        tmp_path,
        system_over={"landuse_raster": str(base / "system" / "data")},
        analysis_over={"storm_tide_boundary_line_gis": str(file_)},
    )
    outcome = _emit_outcome(world)
    _assert_emit_refused_naming(outcome, "data/a.nc", {member: b"directory member bytes", file_: b"file input bytes"})


def test_two_different_outside_directories_of_one_name_refuse_per_member(tmp_path):
    """Property: _harvest_sources refuses two DIFFERENT members under one archive name, directory form.

    Class: two renderers each declaring an outside directory of one name whose members share a name but
    not bytes. Why required: the directory-member call site is a separate line from the file site, and
    after setdefault the loser is gone from the plan. Killed by: reverting that one call site.
    Second correct implementation: one shared loop that plans files and members through the same call.
    """
    ad = tmp_path / "analysis"
    ad.mkdir()
    one = _put(tmp_path / "o1" / "store" / "c0", b"one")
    two = _put(tmp_path / "o2" / "store" / "c0", b"two")
    try:
        entries = _emit._harvest_sources({"r1": [one.parent], "r2": [two.parent]}, ad)[0]
    except ConfigurationError as exc:
        for p in (one, two):
            assert str(p.resolve()) in str(exc)
        return
    raise AssertionError(f"the harvest planned {entries}; the member {two} was dropped silently")


def test_one_file_declared_by_two_renderers_is_planned_once(tmp_path):
    """Property: the same file declared by two renderers is ONE entry and is not a collision.

    Class: one outside file under two renderer keys. Why required: the over-fire guard for the arm above.
    Killed by: refusing on a second declaration of the name. Green on the parent by construction.
    """
    ad = tmp_path / "analysis"
    ad.mkdir()
    one = _put(tmp_path / "o1" / "f.nc", b"one")
    entries = _emit._harvest_sources({"r1": [one], "r2": [one]}, ad)[0]
    assert entries == {"external/f.nc": one}


def test_a_config_input_a_renderer_also_declared_publishes_once(tmp_path):
    """Property: one file named by a config field AND declared by a renderer is carried once.

    Class: the overlap ``_emit_bundle_zip`` documents as normal. Why required: a name-only comparison
    would over-fire here. Killed by: comparing names without the files behind them. Green on the parent.
    """
    world = _world(tmp_path)
    shared = _put(world.ad / "shared" / "in.nc", b"shared bytes")
    world = _world(tmp_path, analysis_over={"storm_tide_boundary_line_gis": str(shared)})
    kind, payload = _emit_outcome(world, harvested=("shared/in.nc",))
    assert kind == "published", f"the normal overlap was refused: {payload}"
    assert _zip_names(payload).count("shared/in.nc") == 1
    assert _zip_bytes(payload, "shared/in.nc") == b"shared bytes"


# --------------------------------------------------------------------------------------
# (b) the emit's own names
# --------------------------------------------------------------------------------------


def test_a_declared_input_named_case_yaml_refuses_when_the_manifest_is_a_different_file(tmp_path):
    """Property: ``experiment_cfg_yaml`` naming ``{system}/case.yaml`` refuses when ``case_manifest_yaml`` differs.

    Class: a declared input that takes the name the emit gives the case manifest, from another file.
    Why required: on the parent the generated ``case.yaml`` is replaced by the declared input's bytes.
    The fixture's manifest is a DIFFERENT file with different bytes, because the emit's copy is skipped
    when it is None and a fixture without one cannot show the overwrite. Killed by: exempting the name
    wholesale.
    """
    base = tmp_path / "w"
    manifest = _put(base / "elsewhere" / "orig_case.yaml", b"manifest bytes")
    declared = _put(base / "system" / "case.yaml", b"declared bytes")
    world = _world(tmp_path, analysis_over={"experiment_cfg_yaml": str(declared)}, case_manifest=manifest)
    _assert_emit_refused_naming(
        _emit_outcome(world), "case.yaml", {manifest: b"manifest bytes", declared: b"declared bytes"}
    )


def test_unrelated_declared_names_publish(tmp_path):
    """Property: ``experiment_cfg_yaml`` naming ``{system}/exp.yaml`` and a declared ``weather.nc`` publish.

    Class: declared inputs that take no name the emit writes. Why required: the table must not over-fire.
    Killed by: refusing on any declared name beside a staged one. Green on the parent.
    """
    base = tmp_path / "w"
    exp = _put(base / "system" / "exp.yaml", b"exp bytes")
    weather = _put(base / "analysis" / "weather.nc", b"weather bytes")
    world = _world(
        tmp_path,
        analysis_over={"experiment_cfg_yaml": str(exp), "weather_timeseries": str(weather)},
    )
    kind, payload = _emit_outcome(world)
    assert kind == "published", f"an unrelated name was refused: {payload}"
    assert _zip_bytes(payload, "exp.yaml") == b"exp bytes" and _zip_bytes(payload, "weather.nc") == b"weather bytes"


def test_experiment_cfg_yaml_and_case_manifest_naming_one_file_publish_and_keep_the_bytes(tmp_path):
    """Property: ``experiment_cfg_yaml`` and ``case_manifest_yaml`` naming ONE file publish, and ``case.yaml`` keeps it.

    Class: the exception column's ``case_manifest_yaml`` row. Why required: that file IS what the emit
    copies to ``case.yaml``, so taking the name loses nothing. Killed by: refusing the exception.
    Green on the parent.
    """
    case = _put(tmp_path / "w" / "system" / "case.yaml", b"case bytes")
    world = _world(tmp_path, analysis_over={"experiment_cfg_yaml": str(case)}, case_manifest=case)
    kind, payload = _emit_outcome(world)
    assert kind == "published", f"the exception was refused: {payload}"
    assert _zip_bytes(payload, "case.yaml") == b"case bytes"


def test_static_plot_config_in_the_analysis_plots_folder_publishes_and_a_system_one_refuses(tmp_path):
    """Property: ``{analysis}/plots/style.yaml`` loses nothing; ``{system}/plots/style.yaml`` replaces the emit's copy.

    Class: the ``plots/`` tree row, reached from the same source and from a different file. Why
    required: the emit copies the analysis ``plots/`` tree itself; a DIFFERENT file under the same name
    would replace it on the parent. Killed by: refusing every file under ``plots/``, or none.
    """
    world = _world(tmp_path)
    same = _put(world.ad / "plots" / "style.yaml", b"analysis plots bytes")
    world = _world(tmp_path, analysis_over={"static_plot_configs": [str(same)]})
    kind, payload = _emit_outcome(world)
    assert kind == "published", f"the emit's own copy was refused: {payload}"
    assert _zip_bytes(payload, "plots/style.yaml") == b"analysis plots bytes"

    other_base = tmp_path / "other"
    own = _put(other_base / "w" / "analysis" / "plots" / "style.yaml", b"analysis plots bytes")
    different = _put(other_base / "w" / "system" / "plots" / "style.yaml", b"system plots bytes")
    world2 = _world(other_base, analysis_over={"static_plot_configs": [str(different)]})
    _assert_emit_refused_naming(
        _emit_outcome(world2), "plots/style.yaml", {own: b"analysis plots bytes", different: b"system plots bytes"}
    )


def test_a_declared_input_named_reprex_config_yaml_refuses(tmp_path):
    """Property: a declared input named ``reprex_config.yaml`` refuses -- the AFTER direction.

    Class: a name written AFTER ``_copy_declared_inputs``. Why required: on the parent the generated
    template replaces the declared input, so a file the config names is lost. Killed by: checking only
    the names written before the copy.
    """
    declared = _put(tmp_path / "w" / "system" / "reprex_config.yaml", b"declared bytes")
    world = _world(tmp_path, analysis_over={"experiment_cfg_yaml": str(declared)})
    outcome = _emit_outcome(world)
    kind, payload = outcome
    if kind == "published":
        held = _zip_bytes(payload, "reprex_config.yaml")
        raise AssertionError(
            f"emit PUBLISHED a bundle whose 'reprex_config.yaml' holds the generated template; "
            f"the declared file {declared} (bytes {b'declared bytes'!r}) was dropped silently: {held[:40]!r}"
        )
    assert str(declared.resolve()) in str(payload)


#: The exception column, transcribed as DATA: (destination in the bundle, the file the emit copies to it).
#: ``{analysis}`` is the analysis directory and ``{case}`` the case manifest. ``Snakefile.source`` is absent
#: on purpose: its source ``{analysis}/Snakefile`` cannot be declared at that name.
_EXCEPTION_ROWS = [
    ("case.yaml", "{case}"),
    ("_version.json", "{analysis}/_version.json"),
    ("scenario_status.csv", "{analysis}/scenario_status.csv"),
    ("sensitivity_analysis_definition.csv", "{analysis}/sensitivity_analysis_definition.csv"),
    ("_status/_du.json", "{analysis}/_status/_du.json"),
    ("plots/style.yaml", "{analysis}/plots/style.yaml"),
]


@pytest.mark.parametrize(("dest", "source_template"), _EXCEPTION_ROWS, ids=[r[0] for r in _EXCEPTION_ROWS])
def test_the_exception_column_publishes_its_own_source_and_refuses_a_different_file(tmp_path, dest, source_template):
    """Property: (l) a declared input that IS the exception source publishes; a same-named DIFFERENT file refuses.

    Class: each row of the exception column that names a source. Why required: the emit copies that
    source to the name itself, so declaring it loses nothing, while a different file replaces the emit's
    copy (the parent's behaviour, pasted per row). Killed by: an exception wider than the row, or none.
    Second correct implementation: deriving the allowed source from the table rather than transcribing
    it (checked against the table below).
    """
    analysis_dir_for_template = tmp_path / "w" / "analysis"
    case_path = analysis_dir_for_template / "case.yaml"
    source = Path(source_template.format(analysis=analysis_dir_for_template, case=case_path))
    _put(source, b"source bytes")
    world = _world(
        tmp_path,
        analysis_over={"weather_event_summary_csv": str(source)},
        case_manifest=case_path if dest == "case.yaml" else None,
    )
    kind, payload = _emit_outcome(world)
    assert kind == "published", f"the exception source for {dest!r} was refused: {payload}"
    assert _zip_bytes(payload, dest) == b"source bytes", f"{dest!r} lost the source's bytes"

    other = tmp_path / "different"
    different = _put(other / "w" / "system" / dest, b"different bytes")
    own = _put(
        other / "w" / "analysis" / Path(source).relative_to(analysis_dir_for_template),
        b"source bytes",
    )
    case2 = other / "w" / "analysis" / "case.yaml"
    if dest == "case.yaml":
        _put(case2, b"source bytes")
    world2 = _world(
        other,
        system_over={"landuse_lookup_file": str(different)},
        case_manifest=case2 if dest == "case.yaml" else None,
    )
    outcome = _emit_outcome(world2)
    emit_owned = case2 if dest == "case.yaml" else own
    _assert_emit_refused_naming(outcome, dest, {emit_owned: b"source bytes", different: b"different bytes"})


def test_the_transcribed_exception_rows_equal_the_staged_names_table(tmp_path):
    """Property: the data above IS the table's exception column, so the table cannot drift from the arm.

    Class: every table row with a ``verbatim`` source. Why required: the arm above is parametrized from
    a transcription, and a transcription is a second source until a test ties it to the first.
    Killed by: editing the table without the arm, or the reverse.
    """
    rows = {r.name: r for r in _emit._EMIT_STAGED_NAMES if r.verbatim is not None}
    ad = tmp_path / "analysis"
    case = tmp_path / "case.yaml"
    transcribed = {}
    for dest, template in _EXCEPTION_ROWS:
        top = dest.split("/")[0]
        transcribed[top] = (dest, Path(template.format(analysis=ad, case=case)))
    assert set(transcribed) | {"Snakefile.source"} == set(rows)
    for name, (dest, expected) in transcribed.items():
        rest = dest[len(name) + 1 :] if "/" in dest else ""
        verbatim = rows[name].verbatim
        assert verbatim is not None, f"{name}: the table row carries no verbatim source"
        assert verbatim(ad, case, rest) == expected, f"{name}: the table and the transcription disagree"


# --------------------------------------------------------------------------------------
# (d) the toolkit's own reader recognises the relative form
# --------------------------------------------------------------------------------------


def _write_relative_folder(world, dest: Path) -> None:
    """The two configs in the relative form, written by the rewrite that exists on BOTH trees."""
    for cfg, filename in (
        (world.analysis._system.cfg_system, "cfg_system.yaml"),
        (world.analysis.cfg_analysis, "cfg_analysis.yaml"),
    ):
        result = _emit._rewrite_paths_to_relative(cfg.model_dump(mode="json"), type(cfg), world.ad, world.sd)
        scrubbed = _emit._scrub_user_bucket_fields(result.cfg_dict, cfg_model=type(cfg))
        _put(dest / filename, yaml.safe_dump(scrubbed, sort_keys=False))


def _write_live_folder(world, dest: Path) -> None:
    """The two configs as this repository dumps them: ``model_dump``, absolute paths."""
    _put(dest / "cfg_system.yaml", yaml.safe_dump(world.analysis._system.cfg_system.model_dump(mode="json")))
    _put(dest / "cfg_analysis.yaml", yaml.safe_dump(world.analysis.cfg_analysis.model_dump(mode="json")))


def _load_elsewhere(folder: Path, tmp_path: Path, monkeypatch):
    from hhemt.eda._context import load_eda_context

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir(exist_ok=True)
    monkeypatch.chdir(elsewhere)
    assert Path.cwd().resolve() != folder.resolve(), "the working directory IS the unpacked folder; the arm is vacuous"
    return load_eda_context(folder)


def test_an_unpacked_data_deposit_reads_relative_to_its_own_folder(tmp_path, monkeypatch):
    """Property: an unpacked data deposit's Path fields resolve UNDER the folder, whatever the working directory.

    Class: a folder holding the relative-form configs and no ``bundle_manifest.json``. Why required:
    ``.`` otherwise resolves to the reader's working directory, so a reader standing in a system
    directory holding ``gis/*.gpkg`` silently receives that tree's layers. Killed by: keying only on the
    manifest. The working directory is asserted DIFFERENT from the folder, because a test that changes
    into the folder passes on an unchanged reader.
    """
    world = _world(tmp_path, nested=True)
    folder = tmp_path / "unpacked"
    _write_relative_folder(world, folder)
    ctx = _load_elsewhere(folder, tmp_path, monkeypatch)
    assert Path(ctx.cfg_system.system_directory).resolve() == folder.resolve()
    assert Path(ctx.cfg_system.DEM_fullres).resolve().is_relative_to(folder.resolve()), ctx.cfg_system.DEM_fullres
    assert ctx.is_bundle is False, "is_bundle is derived from bundle_manifest.json alone"


def test_exactly_one_forced_dot_field_reading_dot_is_not_recognised(tmp_path, monkeypatch):
    """Property: only BOTH forced-dot fields reading ``.`` marks a relative-path record.

    Class: a folder whose system config reads ``.`` and whose analysis config carries an absolute
    ``analysis_dir``. Why required: one field alone is how a live config looks when it is edited by
    hand, and rebasing it would repoint a legitimately-relative input. Killed by: recognising on either.
    """
    world = _world(tmp_path, nested=True)
    folder = tmp_path / "half"
    _write_relative_folder(world, folder)
    path = folder / "cfg_analysis.yaml"
    cfg = yaml.safe_load(path.read_text())
    cfg["analysis_dir"] = str(world.ad)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    ctx = _load_elsewhere(folder, tmp_path, monkeypatch)
    assert Path(ctx.cfg_system.system_directory) == Path("."), "an unrecognised folder was rebased"


def test_a_live_folder_with_a_null_analysis_dir_and_an_absolute_system_directory_reads_unchanged(tmp_path, monkeypatch):
    """Property: the dump this repository produces (``analysis_dir: null``, absolute ``system_directory``) is untouched.

    Class: the live ``model_dump`` of a config whose ``analysis_dir`` is unset. Why required: the
    reader serves live analysis directories first; recognising them would repoint real paths.
    """
    world = _world(tmp_path, analysis_over={"analysis_dir": None})
    folder = tmp_path / "live"
    _write_live_folder(world, folder)
    ctx = _load_elsewhere(folder, tmp_path, monkeypatch)
    assert Path(ctx.cfg_system.system_directory) == world.sd


def test_a_live_folder_with_an_absolute_analysis_dir_reads_unchanged(tmp_path, monkeypatch):
    """Property: a live folder whose ``analysis_dir`` is absolute reads exactly as before."""
    world = _world(tmp_path)
    folder = tmp_path / "live2"
    _write_live_folder(world, folder)
    ctx = _load_elsewhere(folder, tmp_path, monkeypatch)
    assert Path(ctx.cfg_system.system_directory) == world.sd and Path(ctx.cfg_analysis.analysis_dir) == world.ad


# --------------------------------------------------------------------------------------
# (f) the bundle manifest spells sources the way the archive names them
# --------------------------------------------------------------------------------------


def test_the_manifest_spells_sources_analysis_relative_member_relative_and_external(tmp_path):
    """Property: absolute sources read ``plots/...``, ``members/member_{N}/...`` and ``external/{filename}``.

    Class: a source inside the analysis directory, inside a member directory, and outside. Why
    required: the producer's absolute layout was deposited in the manifest. Killed by: writing
    ``str(p)``. ``analysis_dir`` is passed only if the function accepts it, so ONE body runs on the parent,
    which writes the absolute strings and fails on the assertion that lists them.
    """
    ad = tmp_path / "analysis"
    sources = {"r": [ad / "plots" / "a.zarr", ad / "members" / "member_3" / "b.nc", tmp_path / "elsewhere" / "c.nc"]}
    staging = tmp_path / "staging"
    staging.mkdir()
    sig = inspect.signature(_emit._write_bundle_manifest)
    kwargs = {"analysis_dir": ad} if "analysis_dir" in sig.parameters else {}
    _emit._write_bundle_manifest(staging, sources_by_renderer=sources, analysis_id="a", git_sha="deadbeef", **kwargs)
    values = json.loads((staging / "bundle_manifest.json").read_text())["source_paths_by_renderer"]["r"]
    absolute = [v for v in values if Path(v).is_absolute()]
    assert not absolute, f"the manifest carries absolute source paths: {absolute}"
    assert values == ["plots/a.zarr", "members/member_3/b.nc", "external/c.nc"]


def test_the_manifest_without_analysis_dir_keeps_relative_sources_and_refuses_an_absolute_one(tmp_path):
    """Property: omitted ``analysis_dir`` leaves relative sources as given and raises on an absolute one.

    Class: the callers that pass relative or empty sources, and one that does not. Why required: the
    keyword is optional so the existing callers are unchanged, and an absolute source with nothing to
    relativize it against could only be written as the producer's own path.
    """
    staging = tmp_path / "s1"
    staging.mkdir()
    _emit._write_bundle_manifest(staging, sources_by_renderer={"r": [Path("eda/a.zarr")]}, analysis_id="a", git_sha="d")
    assert json.loads((staging / "bundle_manifest.json").read_text())["source_paths_by_renderer"]["r"] == ["eda/a.zarr"]
    with pytest.raises(ValueError, match="absolute"):
        _emit._write_bundle_manifest(
            staging, sources_by_renderer={"r": [tmp_path / "x.nc"]}, analysis_id="a", git_sha="d"
        )


def test_an_emit_writes_a_manifest_with_zero_absolute_sources(tmp_path):
    """Property: a real emit's manifest names at least one declared source and none of them is absolute.

    Class: the real ``emit_bundle`` over a tree with one inside and one outside renderer source. Why
    required: an absence assertion over an EMPTY map passes, so the count must be at least 1.
    """
    out = _put(tmp_path / "w" / "outside" / "o.nc", b"o")
    _put(tmp_path / "w" / "analysis" / "i.nc", b"i")
    world = _world(tmp_path)
    kind, payload = _emit_outcome(world, harvested=("i.nc", "../outside/o.nc"))
    assert kind == "published", payload
    manifest = json.loads(_zip_bytes(payload, "bundle_manifest.json"))
    values = [v for paths in manifest["source_paths_by_renderer"].values() for v in paths]
    assert len(values) >= 1, "the declared-source count is zero; an absence assertion would be vacuous"
    assert not [v for v in values if Path(v).is_absolute()], values
    assert sorted(values) == ["external/o.nc", "i.nc"] and out.exists()


# --------------------------------------------------------------------------------------
# (g) the census over the deposit AS UPLOADED
# --------------------------------------------------------------------------------------


def _scan_member_by_member(deposit: list[Path], scratch: Path) -> list[str]:
    """``assert_bundle_zero_user_info`` over each deposited member, because the scanner takes ONE root.

    A directory (the zarr store) is scanned in place; a file is copied into a folder of its own.
    Returns the leak messages.
    """
    from hhemt.bundle._reprex_gate import assert_bundle_zero_user_info
    from hhemt.exceptions import ProcessingError

    leaks: list[str] = []
    for i, member in enumerate(deposit):
        if member.is_dir():
            root = member
        else:
            root = scratch / f"m{i}"
            root.mkdir(parents=True)
            (root / member.name).write_bytes(member.read_bytes())
        try:
            assert_bundle_zero_user_info(root)
        except ProcessingError as exc:
            leaks.append(f"{member.name}: {exc}")
    return leaks


def _census_leaking_files(unpacked: Path, scratch: Path) -> list[str]:
    """The files under ``unpacked`` that leak, each scanned in a scratch folder of its OWN.

    The folder is named from the member's index in the sorted listing, so it is unique by construction:
    zarr chunks share a basename (``0``), and a name built from a count of leaks collides as soon as one
    file has leaked and a second same-named file follows.
    """
    leaking: list[str] = []
    members = sorted(p for p in unpacked.rglob("*") if p.is_file())
    for i, member in enumerate(members):
        if _scan_member_by_member([member], scratch / f"scan_{i}"):
            leaking.append(str(member.relative_to(unpacked)))
    return leaking


def test_the_census_scans_same_named_members_after_a_leak_without_colliding(tmp_path, monkeypatch):
    """Property: every member is scanned, and every leaking one is reported, when same-named files follow a leak.

    Class: three files named ``0`` (zarr chunks) in sorted order leak, clean, leak. Why required: the scratch
    folder was named from the count of leaks so far, so the third file reused the second's folder and the
    census died with FileExistsError, hiding a real leak behind it. Killed by: any folder name that does
    not differ per member. Second correct implementation: one fresh ``tempfile.mkdtemp`` per member.
    """
    carrier = _put(tmp_path / "carrier.txt", "# min-tokens: 1\nacme_private_site\n")
    monkeypatch.setenv("HHEMT_REPREX_BLOCKLIST", str(carrier))
    unpacked = tmp_path / "unpacked"
    _put(unpacked / "x" / "0", "acme_private_site")
    _put(unpacked / "y" / "0", "nothing private here")
    _put(unpacked / "z" / "0", "acme_private_site")
    assert _census_leaking_files(unpacked, tmp_path / "scan") == ["x/0", "z/0"]


def test_a_real_emit_carries_no_producer_token(tmp_path, monkeypatch):
    """Property: the unpacked runnable deposit of a minimal real emit leaks no token of its producer's layout.

    Class: ``emit_bundle`` over a tree whose directories carry ``acme_private_site``, with one declared
    source inside and one outside. Why required: the manifest and the configs both used to carry
    absolute analysis paths. The large-tree census (harvested SWMM ``.inp`` ``FILE`` references and
    ``validation_report.json``) needs a real synth-tier emit and is OWED on Rivanna.
    """
    carrier = _put(tmp_path / "carrier.txt", "# min-tokens: 1\nacme_private_site\n")
    monkeypatch.setenv("HHEMT_REPREX_BLOCKLIST", str(carrier))
    base = tmp_path / "acme_private_site"
    _put(base / "analysis" / "i.nc", b"i")
    _put(base / "outside" / "o.nc", b"o")
    world = _world(tmp_path, base_name="acme_private_site")
    kind, payload = _emit_outcome(world, harvested=("i.nc", "../outside/o.nc"))
    assert kind == "published", payload
    root = tmp_path / "unpacked_bundle"
    with zipfile.ZipFile(payload) as zf:
        zf.extractall(root)
    leaks = _scan_member_by_member([root], tmp_path / "scratch2")
    assert not leaks, f"the unpacked emit leaks the producer's layout: {leaks}"


# --------------------------------------------------------------------------------------
# (i) the planner's refusals carry the keywords ConfigurationError takes
# --------------------------------------------------------------------------------------


def test_an_excluded_input_with_no_coordinates_raises_configuration_error(tmp_path):
    """Property: an excluded input with no coordinate block raises ``ConfigurationError`` naming the field.

    Class: a list field excluded with a block for a DIFFERENT filename. Why required: the parent raised
    ``TypeError: ... unexpected keyword argument 'parameter'``, so the refusal the code documents was never
    delivered. Killed by: the old keyword set. The red on the parent is that ``TypeError`` escaping where
    ``ConfigurationError`` was expected, which is the defect itself and not a call-shape error in this arm.
    """
    from hhemt.config.bundle_exclude import BundleExcludeConfig

    world = _world(tmp_path)
    cfg = _put(world.ad / "p" / "style.yaml", b"style")
    world = _world(tmp_path, analysis_over={"static_plot_configs": [str(cfg)]})
    exclude = BundleExcludeConfig.model_validate(
        {"exclusions": {"static_plot_configs": {"other.yaml": {"citation": "c"}}}}
    )
    with pytest.raises(ConfigurationError) as exc:
        _emit._copy_declared_inputs(world.analysis, tmp_path / "staging_a", exclude)
    assert "static_plot_configs" in str(exc.value)


def test_an_excluded_input_that_is_a_directory_raises_configuration_error(tmp_path):
    """Property: an excluded input that resolves to a DIRECTORY raises ``ConfigurationError`` naming the field.

    Class: a scalar field excluded with coordinates whose file is a directory. Why required: sha256 is
    undefined over a tree. Killed by: the old keyword set, which raised ``TypeError``.
    """
    from hhemt.config.bundle_exclude import BundleExcludeConfig

    folder = tmp_path / "w" / "outside" / "a_dir"
    folder.mkdir(parents=True)
    _put(folder / "inside.txt", b"x")
    world = _world(tmp_path, system_over={"landuse_raster": str(folder)})
    exclude = BundleExcludeConfig.model_validate({"exclusions": {"landuse_raster": {"citation": "c"}}})
    with pytest.raises(ConfigurationError) as exc:
        _emit._copy_declared_inputs(world.analysis, tmp_path / "staging_b", exclude)
    assert "landuse_raster" in str(exc.value)


def test_planning_copies_nothing(tmp_path):
    """Property: ``_plan_declared_inputs`` makes the plan and writes no file, and the executor honours it.

    Class: a world with one carried input. Why required: ``emit_bundle`` plans before it copies, so the
    refusal leaves nothing to unwind. Killed by: a planner that copies.
    """
    base = tmp_path / "w"
    inp = _put(base / "outside" / "w.nc", b"w")
    world = _world(tmp_path, analysis_over={"storm_tide_boundary_line_gis": str(inp)})
    staging = tmp_path / "plan_staging"
    staging.mkdir()
    planned, deposits = _emit._plan_declared_inputs(world.analysis)
    assert list(staging.iterdir()) == [] and deposits == []
    assert ("external/w.nc", inp.resolve()) in [(p.rel, p.src) for p in planned]
    _emit._copy_declared_inputs(world.analysis, staging)
    assert (staging / "external" / "w.nc").read_bytes() == b"w"


# --------------------------------------------------------------------------------------
# (j) the staged-names table is total over the emit's writers
# --------------------------------------------------------------------------------------

#: Functions the totality pass does NOT count: the declared-input writer and its planner (the thing the
#: table polices) and the two reads of the staged crate.
_NOT_COUNTED = {
    "_copy_declared_inputs",
    "_plan_declared_inputs",
    "_crate_license_finding",
    "_upgrade_crate_to_workflow_run_crate",
}
_STAGING_WRITE_PATTERN = re.compile(r"(staging|Path\(staging\)) / ")


class _StagingSites(ast.NodeVisitor):
    """Every ``staging / X`` and ``Path(staging) / X`` in a module, with the names X can take.

    An AST pass and not a text match, because two sites write a name through a loop variable: one binds
    ``filename`` through a TUPLE-UNPACK over a module constant of two-element tuples (indexed, not a single
    name over a tuple) and one binds ``fname`` over a literal tuple. Both resolve here; a text pattern
    sees the variable and not the names.
    """

    def __init__(self, module) -> None:
        self.module = module
        self.funcs: list[str] = []
        self.binds: list[dict[str, set[str]]] = [{}]
        self.sites: list[tuple[str | None, int, set[str] | None]] = []

    def _str(self, node: ast.AST) -> set[str] | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return {node.value}
        if isinstance(node, ast.Name):
            for scope in reversed(self.binds):
                if node.id in scope:
                    return scope[node.id]
            value = getattr(self.module, node.id, None)
            return {value} if isinstance(value, str) else None
        return None

    def _rows(self, node: ast.AST) -> list[tuple[str, ...]] | None:
        if isinstance(node, ast.Name):
            value = getattr(self.module, node.id, None)
            if not isinstance(value, tuple):
                return None
            rows = []
            for r in value:
                # Only rows of strings are path-name rows. A loop over a tuple of other objects (the
                # staged-names table itself, iterated by the matcher) binds nothing.
                if isinstance(r, str):
                    rows.append((r,))
                elif isinstance(r, tuple) and all(isinstance(c, str) for c in r):
                    rows.append(r)
                else:
                    return None
            return rows
        if isinstance(node, ast.Tuple | ast.List):
            rows = []
            for elt in node.elts:
                if isinstance(elt, ast.Tuple):
                    cells = [self._str(c) for c in elt.elts]
                    if any(c is None for c in cells):
                        return None
                    rows.append(tuple(next(iter(c)) for c in cells))
                else:
                    cell = self._str(elt)
                    if cell is None:
                        return None
                    rows.append((next(iter(cell)),))
            return rows
        return None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.funcs.append(node.name)
        self.binds.append({})
        self.generic_visit(node)
        self.binds.pop()
        self.funcs.pop()

    def visit_For(self, node: ast.For) -> None:
        rows = self._rows(node.iter)
        scope: dict[str, set[str]] = {}
        if rows is not None:
            if isinstance(node.target, ast.Name):
                scope[node.target.id] = {r[0] for r in rows}
            elif isinstance(node.target, ast.Tuple):
                for i, elt in enumerate(node.target.elts):
                    if isinstance(elt, ast.Name):
                        scope[elt.id] = {r[i] for r in rows}
        self.binds.append(scope)
        for child in node.body:
            self.visit(child)
        self.binds.pop()
        for child in node.orelse:
            self.visit(child)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        left = node.left
        is_staging = (isinstance(left, ast.Name) and left.id == "staging") or (
            isinstance(left, ast.Call)
            and isinstance(left.func, ast.Name)
            and left.func.id == "Path"
            and len(left.args) == 1
            and isinstance(left.args[0], ast.Name)
            and left.args[0].id == "staging"
        )
        if isinstance(node.op, ast.Div) and is_staging:
            self.sites.append((self.funcs[-1] if self.funcs else None, node.lineno, self._str(node.right)))
        self.generic_visit(node)


def _totality(table_names: set[str]) -> dict:
    source = Path(_emit.__file__).read_text()
    visitor = _StagingSites(_emit)
    visitor.visit(ast.parse(source))
    counted = [(f, ln, s) for f, ln, s in visitor.sites if f not in _NOT_COUNTED]
    written = set().union(*[s for _f, _ln, s in counted if s])
    return {
        "unresolved": [(f, ln) for f, ln, s in counted if s is None],
        "written_not_in_table": written - table_names,
        "table_not_written": table_names - written,
        "site_lines": {ln for _f, ln, _s in visitor.sites},
        "matched_lines": [
            (n, line) for n, line in enumerate(source.splitlines(), 1) if _STAGING_WRITE_PATTERN.search(line)
        ],
    }


def test_the_staged_names_table_is_total_over_the_emit_writers():
    """Property: every name the emit writes into staging is in the table, and every table name has a writer.

    Class: the module's own source, parsed. Why required: check (iii) refuses a declared input that takes
    a name the emit writes, and it can only be as complete as the table; a write site the table does not
    name is a name a declared input can silently replace. Killed by: a table that drifts from the writers
    in either direction. Second correct implementation: deriving the table from the AST at import time
    (rejected: a derived table is the thing under test).
    """
    report = _totality({r.name for r in _emit._EMIT_STAGED_NAMES})
    assert report["unresolved"] == [], f"a staging write whose name does not resolve: {report['unresolved']}"
    assert report["written_not_in_table"] == set(), f"written but not in the table: {report['written_not_in_table']}"
    assert report["table_not_written"] == set(), f"in the table but never written: {report['table_not_written']}"
    assert report["site_lines"] == {n for n, _ in report["matched_lines"]}, (
        "the AST sees different lines from the regenerating command"
    )


@pytest.mark.parametrize("dropped", ["scenario_status.csv", "cfg_system.yaml"])
def test_the_totality_pass_fails_when_a_loop_bound_name_leaves_the_table(dropped):
    """Property: deleting a name only a LOOP-BOUND site writes makes the pass fail -- a text pattern cannot.

    Class: ``scenario_status.csv`` (written through ``fname``) and ``cfg_system.yaml`` (written through the
    tuple-unpack target ``filename``). Why required: the must-fire control for the pass, on the two loop
    shapes. Killed by: a pass that reads only literal right-hand sides.
    """
    names = {r.name for r in _emit._EMIT_STAGED_NAMES} - {dropped}
    report = _totality(names)
    assert report["written_not_in_table"] == {dropped}
    assert all(dropped not in line for _n, line in report["matched_lines"]), (
        "a text match sees the name, so this control would not separate the pass from it"
    )


def test_the_totality_pass_fails_when_the_table_names_a_file_nothing_writes():
    """Property: a table name with no writer is reported -- the opposite direction of the control above."""
    names = {r.name for r in _emit._EMIT_STAGED_NAMES} | {"never_written.txt"}
    assert _totality(names)["table_not_written"] == {"never_written.txt"}


# --------------------------------------------------------------------------------------
# (k) a relative value is rewritten the way an absolute one is (D-199 option 1)
# --------------------------------------------------------------------------------------

_RELATIVE_ROWS = [
    ("sub/../sub/x.nc", "sub/x.nc"),
    ("../../gis/a.gpkg", "gis/a.gpkg"),
    ("../../../o1/w.nc", "external/w.nc"),
    ("./sub/x.nc", "sub/x.nc"),
]


def _pair_through_the_bundle_writer(world, staging: Path) -> list[Path]:
    """The runnable deposit's configs, written by whichever function the tree has."""
    writer = getattr(_emit, "emit_relative_config_pair", None) or _emit._copy_configs_with_relative_paths
    writer(world.analysis, staging)
    return [staging / f for f in CFG_FILES]


@pytest.mark.parametrize(("value", "expected"), _RELATIVE_ROWS, ids=[r[0] for r in _RELATIVE_ROWS])
def test_a_relative_value_is_rewritten_the_way_an_absolute_one_is(tmp_path, value, expected):
    """Property: (k) a relative value is identified as the copy step identifies it, then rewritten.

    Class: relative values with a ``..`` detour, one that reaches the system directory, one that leaves it,
    and a plain ``./`` form, in BOTH configs and BOTH deposits, with the analysis folder inside the system
    folder. Why required: left as written, ``../../../o1/w.nc`` sat in a deposit whose copy step puts that
    file at ``external/w.nc``. The control is the parent, which returns the first three unchanged; the
    FOURTH row is PERMANENTLY GREEN there and is kept as a regression guard only: the config model already
    normalizes ``./sub/x.nc`` to ``sub/x.nc``, so the parent dumps ``sub/x.nc``. The first row pins the tidy
    branch.
    """
    world = _world(
        tmp_path,
        nested=True,
        system_over={"landuse_lookup_file": value},
        analysis_over={"storm_tide_boundary_line_gis": value},
    )
    staging = tmp_path / "deposit_staging"
    staging.mkdir()
    runnable = tmp_path / "runnable_staging"
    runnable.mkdir()
    for label, deposit in (
        ("data deposit", _call_deposit_set(world.analysis, staging)),
        ("runnable deposit", _pair_through_the_bundle_writer(world, runnable)),
    ):
        cfgs = _staged_cfgs(deposit)
        got = (
            cfgs["cfg_system.yaml"]["landuse_lookup_file"],
            cfgs["cfg_analysis.yaml"]["storm_tide_boundary_line_gis"],
        )
        assert got == (expected, expected), f"{label}: {value!r} was written as {got}, expected {expected!r}"


# --------------------------------------------------------------------------------------
# OWED on Rivanna scratch: the synth-tier emit (solver-bearing fixtures, never run on the laptop)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_name", ["rendered_synth_multi_sim", "rendered_synth_sensitivity"])
def test_a_synth_tier_emit_names_declared_sources_relative_and_its_configs_leak_nothing(
    request, tmp_path, monkeypatch, fixture_name
):
    """Property: (f, g) a real synth-tier emit writes a non-empty, wholly relative source map and clean configs.

    Class: a bundle emitted from a completed, rendered synthetic analysis, on the multi-sim and the
    sensitivity arm. Why required: the minimal-tree arms above prove the spelling; this proves it on the
    renderers' real declarations, where the declared-source count is the thing an absence assertion over an
    EMPTY map would otherwise hide (every committed fixture carries an empty map). OWED on Rivanna scratch:
    the fixtures build and run the solver. The fixture is requested INSIDE the body, after the venue check,
    so collecting this module on the laptop builds nothing and a skip there is silent about the solver.

    Run: ``HHEMT_COMPILE_VENUE=toolchain pytest tests/test_publish_deposit_paths.py -k synth_tier_emit -q``.
    The union blocklist carrier is read from ``HHEMT_CENSUS_BLOCKLIST`` when set (the reprex and anonymization
    lists joined), so the producer's real tokens are scanned for without being written into this file.
    """
    from hhemt._compile_venue import venue_is_permitted

    if not venue_is_permitted():
        pytest.skip("OWED on Rivanna scratch: needs HHEMT_COMPILE_VENUE=toolchain (a solver-bearing fixture)")
    analysis = request.getfixturevalue(fixture_name)
    bundle_path = (
        analysis.sensitivity.bundle_report_data(tmp_path / "bundle.zip")
        if getattr(analysis, "sensitivity", None) is not None
        else analysis.bundle_report_data(tmp_path / "bundle.zip")
    )
    manifest = json.loads(_zip_bytes(bundle_path, "bundle_manifest.json"))
    values = [v for paths in manifest["source_paths_by_renderer"].values() for v in paths]
    assert len(values) >= 1, "the declared-source count is zero; an absence assertion would be vacuous"
    assert not [v for v in values if Path(v).is_absolute()], f"absolute source paths: {values}"

    carrier = os.environ.get("HHEMT_CENSUS_BLOCKLIST")
    if carrier:
        monkeypatch.setenv("HHEMT_REPREX_BLOCKLIST", carrier)
        unpacked = tmp_path / "unpacked"
        with zipfile.ZipFile(bundle_path) as zf:
            zf.extractall(unpacked)
        leaking = _census_leaking_files(unpacked, tmp_path / "census_scan")
        print(f"CENSUS leaking files: {leaking}")
        assert not {"cfg_analysis.yaml", "cfg_system.yaml", "bundle_manifest.json"} & set(leaking), leaking
