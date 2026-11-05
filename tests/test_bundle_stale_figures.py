"""``emit_bundle`` refuses a tree holding stale figures by default; ``--allow-stale-figures`` carries them.

A STALE FIGURE is a regenerable figure file under ``plots/sensitivity/per_sim/member-{N}/`` of a
sensitivity analysis whose live member set does not contain ``N``. The live member set is read
from ``df_setup`` on the object ``emit_bundle`` receives, which has two shapes (the sensitivity
wrapper carries it on itself, the sensitivity master at ``.sensitivity.df_setup``); both are
exercised here from ``types.SimpleNamespace`` stand-ins.

Solver-free and ``synth_*``-free: every test writes ``tmp_path`` files only. The Snakefile in the
fixture tree is the sensitivity-master golden at ``495bf92b`` -- the emitter's wildcard template,
never a literal per-instance output -- read with ``git show``. Every refusal arm fails on the parent
commit by API ABSENCE (no accessor, no check, no keyword); the behavioural fact each negates is
recorded by ``test_parent_prune_keeps_stale_figures``.

Stand-in attributes, per arm (the refusal reads exactly two things from the object: the
live-member source and ``cfg_analysis.sensitivity_analysis``): every stand-in below carries every
attribute read on its path; the non-sensitivity stand-in carries
``cfg_analysis.sensitivity_analysis = None`` only.
"""

from __future__ import annotations

import json
import subprocess
import warnings
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from typer.testing import CliRunner

import hhemt.bundle._emit as E
from hhemt.bundle._emit import (
    _check_stale_figures,
    _live_member_ids,
    _prune_undeclared_figures,
    _write_bundle_manifest,
    emit_bundle,
)
from hhemt.cli import app
from hhemt.exceptions import ConfigurationError

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_REL = "tests/fixtures/golden_snakefiles/sensitivity_master.Snakefile.golden"
GOLDEN_PIN = "495bf92b"
FLAG = "--allow-stale-figures"

runner = CliRunner()


def _golden_text() -> str:
    """The sensitivity-master Snakefile golden at the pinned commit (the emitter's wildcard form).

    ``git show`` at the pin means a later recapture of the in-tree golden never moves this
    fixture. A checkout that cannot reach the pin (a shallow clone) falls back to
    the in-tree file, whose plot-rule ``output:`` templates are the same wildcard form.
    """
    shown = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"{GOLDEN_PIN}:{GOLDEN_REL}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if shown.returncode == 0 and shown.stdout:
        return shown.stdout
    return (REPO_ROOT / GOLDEN_REL).read_text()


def _fig(plots: Path, rel: str) -> Path:
    """A figure file plus its stem-named manifest sidecar (``foo.html`` -> ``foo.manifest.json``)."""
    figure = plots / rel
    figure.parent.mkdir(parents=True, exist_ok=True)
    figure.write_text(f"figure {rel}")
    sidecar = figure.parent / f"{figure.stem}.manifest.json"
    if not sidecar.exists():
        sidecar.write_text(json.dumps({"plot_id": figure.stem}))
    return figure


# The four stale member-16 figures: two families, two extensions each (html + svg).
STALE_RELS = [
    "sensitivity/per_sim/member-16/event_index.0/peak_flood_depth__member.16__evt.event_index.0.html",
    "sensitivity/per_sim/member-16/event_index.0/peak_flood_depth__member.16__evt.event_index.0.svg",
    "sensitivity/per_sim/member-16/event_index.0/conduit_flow__member.16__evt.event_index.0.html",
    "sensitivity/per_sim/member-16/event_index.0/conduit_flow__member.16__evt.event_index.0.svg",
]
LIVE_REL = "sensitivity/per_sim/member-22/event_index.0/peak_flood_depth__member.22__evt.event_index.0.html"
UNDECLARED_REL = "retired_family__x.html"


def _tree(tmp_path: Path, *, stale: bool = True, undeclared: bool = False, empty_member16: bool = False) -> Path:
    """An analysis dir: the golden as ``Snakefile``, a live member-22 figure, optional extras."""
    analysis_dir = tmp_path / "analysis"
    plots = analysis_dir / "plots"
    plots.mkdir(parents=True)
    (analysis_dir / "Snakefile").write_text(_golden_text())
    _fig(plots, "system_overview.html")
    _fig(plots, LIVE_REL)
    if stale:
        for rel in STALE_RELS:
            _fig(plots, rel)
    if empty_member16:
        (plots / "sensitivity/per_sim/member-16/event_index.0").mkdir(parents=True)
    if undeclared:
        _fig(plots, UNDECLARED_REL)
    return analysis_dir


def _listing(root: Path) -> dict[str, bytes | None]:
    """Path -> bytes for every file and directory under ``root`` (directories map to None)."""
    return {str(p.relative_to(root)): (None if p.is_dir() else p.read_bytes()) for p in sorted(root.rglob("*"))}


def _df_setup(index) -> pd.DataFrame:
    return pd.DataFrame({"value": [1] * len(index)}, index=pd.Index(index, name="member_id"))


def _standin(shape: str, tmp_path: Path, index, *, analysis_dir: Path | None = None) -> SimpleNamespace:
    """One of the two real shapes (``wrapper`` or ``master``) or the ``plain`` non-sensitivity one."""
    setup_file = tmp_path / "sensitivity_setup.csv"
    setup_file.write_text("member_id\n")
    cfg = SimpleNamespace(
        sensitivity_analysis=None if shape == "plain" else setup_file,
        execution_environment="native",
        analysis_id="aid",
    )
    ns = SimpleNamespace(cfg_analysis=cfg)
    if analysis_dir is not None:
        ns.analysis_paths = SimpleNamespace(analysis_dir=analysis_dir)
    if shape == "wrapper":
        ns.df_setup = _df_setup(index)
    elif shape == "master":
        ns.sensitivity = SimpleNamespace(df_setup=_df_setup(index))
    return ns


SHAPES_AND_INDEXES = [
    pytest.param("wrapper", [5, 22], id="wrapper-int"),
    pytest.param("wrapper", ["5", "22"], id="wrapper-str"),
    pytest.param("master", [5, 22], id="master-int"),
    pytest.param("master", ["5", "22"], id="master-str"),
]


def _caught(callable_, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = callable_(*args, **kwargs)
    return result, [w for w in caught if "stale figure" in str(w.message)]


# --------------------------------------------------------------------------- the accessor


def test_live_member_ids_resolves_both_shapes_as_strings(tmp_path):
    """Property: the live member set is read from `df_setup` on the object itself or from `.sensitivity.df_setup`,
        as strings; neither source, or an empty index, gives the empty set.

    Class: accessor (input shape).
    Why required: emit_bundle receives a wrapper and a master, and an integer index must compare equal to the string
        in a directory name.
    Killed by: dropping either `getattr` branch, or skipping the `str()` conversion.
    Second correct implementation: reading `df_setup.index.astype(str)` into a set instead of a generator expression.
    """
    assert _live_member_ids(_standin("wrapper", tmp_path, [5, 22])) == {"5", "22"}
    assert _live_member_ids(_standin("master", tmp_path, ["5", "22"])) == {"5", "22"}
    # None / empty both mean "no member set".
    assert _live_member_ids(_standin("plain", tmp_path, [])) == frozenset()
    assert _live_member_ids(_standin("wrapper", tmp_path, [])) == frozenset()


# --------------------------------------------------------------------------- four refusal arms


@pytest.mark.parametrize(("shape", "index"), SHAPES_AND_INDEXES)
def test_check_refuses_stale_figures_and_deletes_nothing(tmp_path, shape, index):
    """Property: a tree holding figures under `plots/sensitivity/per_sim/member-{N}/` whose N is not in the live set
        is refused with `ConfigurationError(field="bundle_stale_figures")` naming the setup file, the member, the
        count and both remedies, and the tree is byte-identical afterwards.

    Class: refusal, over both object shapes and both index types.
    Why required: this is the user-visible refusal; a refusal that deletes or omits a remedy defeats it.
    Killed by: returning without raising; deleting the figures; omitting the flag or the render-floor remedy from
        the text; comparing ids as integers so the string index never matches.
    Second correct implementation: building the message from a different template that still carries the same facts.
    """
    analysis_dir = _tree(tmp_path)
    before = _listing(analysis_dir)
    analysis = _standin(shape, tmp_path, index)
    with pytest.raises(ConfigurationError) as raised:
        _check_stale_figures(analysis, analysis_dir, analysis_dir / "plots", False)
    err = raised.value
    assert err.field == "bundle_stale_figures"
    text = str(err)
    assert str(analysis.cfg_analysis.sensitivity_analysis) in text  # the "Config:" line
    assert "member(s) 16" in text
    assert "4 stale figure file(s)" in text
    assert "plots/sensitivity/per_sim/member-16/" in text
    assert "(4 figure files)" in text
    assert "--override-force-rerun" in text
    assert FLAG in text
    assert "member-22" not in text, "the live member is never named as stale"
    assert _listing(analysis_dir) == before, "the refusal deleted or wrote nothing"


def test_parent_prune_keeps_stale_figures(tmp_path):
    """Property: the family-keyed prune `_prune_undeclared_figures` returns `[]` on this tree and leaves all four
        member-16 figures, because their families are still declared.

    Class: behavioural fact the refusal exists for (control).
    Why required: shows the stale figures reach a bundle without the new check; it passes before and after the change.
    Killed by: making the prune key on the member directory (it would then delete them, contradicting the no-
        deletion rule).
    Second correct implementation: an unchanged prune whose `keep` compares families by another normalisation of the
        same name.
    """
    analysis_dir = _tree(tmp_path)
    assert _prune_undeclared_figures(analysis_dir, analysis_dir / "plots") == []
    for rel in STALE_RELS:
        assert (analysis_dir / "plots" / rel).is_file()


# --------------------------------------------------------------------------- four must-not-fire twins


@pytest.mark.parametrize("allow", [False, True])
def test_twin_1_no_stale_figure(tmp_path, allow):
    """Property: when the live set lists every member that has figures, nothing is returned, no warning is issued
        even with the override, and the tree is untouched.

    Class: must-not-fire twin.
    Why required: a check that fires on a clean tree would refuse every emit.
    Killed by: treating any `member-*` figure as stale, or warning unconditionally when the override is on.
    Second correct implementation: returning early when no figure is found before reading the live set.
    """
    analysis_dir = _tree(tmp_path, stale=False)
    before = _listing(analysis_dir)
    analysis = _standin("master", tmp_path, [16, 22])
    result, warned = _caught(_check_stale_figures, analysis, analysis_dir, analysis_dir / "plots", allow)
    assert result == []
    assert warned == [], "no stale figure, no warning -- even with the override"
    assert _listing(analysis_dir) == before


def test_twin_1b_live_member_set_listing_16_and_22_does_not_fire_on_a_stale_looking_tree(tmp_path):
    """Property: member-16 figures are not stale when 16 is in the live set.

    Class: must-fire control varying the live set, not the tree.
    Why required: separates a check keyed on the live set from one keyed on a hard-coded member or on directory age.
    Killed by: comparing against a fixed id, or against the highest id only.
    Second correct implementation: computing the stale set as the difference of directory ids and live ids.
    """
    analysis_dir = _tree(tmp_path)  # member-16 figures present ...
    analysis = _standin("wrapper", tmp_path, [16, 22])  # ... but 16 is live
    assert _check_stale_figures(analysis, analysis_dir, analysis_dir / "plots", False) == []


def test_twin_2_non_sensitivity_standin(tmp_path):
    """Property: an object with neither `df_setup` nor `sensitivity` has no member set and nothing is stale.

    Class: must-not-fire twin (plain analysis).
    Why required: `publish` and `hhemt bundle` of a non-sensitivity analysis pass through the same check.
    Killed by: an `AttributeError` from reading `df_setup` without a default, or treating the empty set as
        'everything stale'.
    Second correct implementation: guarding with `hasattr` instead of `getattr(..., None)`.
    """
    analysis_dir = _tree(tmp_path)
    before = _listing(analysis_dir)
    analysis = _standin("plain", tmp_path, [])
    assert _check_stale_figures(analysis, analysis_dir, analysis_dir / "plots", False) == []
    assert _listing(analysis_dir) == before


def test_twin_3_empty_index(tmp_path):
    """Property: an empty `df_setup` index means no member set, so nothing is stale even with member figures on disk.

    Class: must-not-fire twin (empty index).
    Why required: an empty live set read as a set would mark every member stale.
    Killed by: removing the empty-set early return.
    Second correct implementation: treating a `None` and an empty frame through one shared branch.
    """
    analysis_dir = _tree(tmp_path)
    before = _listing(analysis_dir)
    analysis = _standin("master", tmp_path, [])
    assert _check_stale_figures(analysis, analysis_dir, analysis_dir / "plots", False) == []
    assert _listing(analysis_dir) == before


def test_twin_4_empty_member_directory(tmp_path):
    """Property: a `member-16/` directory holding no figure file holds no stale figure.

    Class: must-not-fire twin (empty directory).
    Why required: a directory left behind after its figures were removed by a render floor must not refuse the emit.
    Killed by: counting member directories instead of figure files.
    Second correct implementation: selecting figures first and grouping by member afterwards.
    """
    analysis_dir = _tree(tmp_path, stale=False, empty_member16=True)
    before = _listing(analysis_dir)
    analysis = _standin("master", tmp_path, [22])
    assert _check_stale_figures(analysis, analysis_dir, analysis_dir / "plots", False) == []
    assert _listing(analysis_dir) == before


def test_undeclared_family_figure_is_not_a_stale_figure(tmp_path):
    """Property: a figure outside any member directory whose family no plot rule declares is not counted by the
        check; the prune handles it.

    Class: scope boundary.
    Why required: keeps the two mechanisms separate: the check concerns removed members only.
    Killed by: counting every figure the prune would delete.
    Second correct implementation: reading the declared families from the Snakefile and skipping those.
    """
    analysis_dir = _tree(tmp_path, stale=False, undeclared=True)
    analysis = _standin("master", tmp_path, [22])
    assert _check_stale_figures(analysis, analysis_dir, analysis_dir / "plots", False) == []


# --------------------------------------------------------------------------- undeclared family under a stale member


def _declared_only_peak_flood_depth_tree(tmp_path: Path, *, with_declared_stale: bool) -> Path:
    """Snakefile declaring only ``peak_flood_depth``; member-16 holds that family and ``retired_family``."""
    analysis_dir = tmp_path / "analysis"
    plots = analysis_dir / "plots"
    plots.mkdir(parents=True)
    (analysis_dir / "Snakefile").write_text(
        "rule plot_peak_flood_depth:\n"
        '    output: "plots/sensitivity/per_sim/member-{member_id}/peak_flood_depth__member.{member_id}.html"\n'
    )
    _fig(plots, "sensitivity/per_sim/member-22/peak_flood_depth__member.22.html")
    _fig(plots, "sensitivity/per_sim/member-16/retired_family__member.16.html")
    if with_declared_stale:
        _fig(plots, "sensitivity/per_sim/member-16/peak_flood_depth__member.16.html")
    return analysis_dir


def test_stale_figure_of_an_undeclared_family_is_not_counted_warned_or_listed(tmp_path):
    """Property: of a declared-family and an undeclared-family figure under member-16, only the declared one is
    counted, warned about and listed under the override, and it is the one that survives the prune.

    Class: stale member figures split by whether their family is declared by the Snakefile.
    Why required: the prune deletes the undeclared one next, so counting it would claim a file is in the bundle
    that is not, and list one path under two manifest keys.
    Killed by: dropping the family test from the keep predicate (count 2, both listed).
    Second correct implementation: counting after the prune and subtracting what it removed.
    """
    analysis_dir = _declared_only_peak_flood_depth_tree(tmp_path, with_declared_stale=True)
    analysis = _standin("master", tmp_path, [22])
    result, warned = _caught(_check_stale_figures, analysis, analysis_dir, analysis_dir / "plots", True)
    declared_rel = "plots/sensitivity/per_sim/member-16/peak_flood_depth__member.16.html"
    assert result == [declared_rel]
    assert len(warned) == 1 and "1 stale figure file(s)" in str(warned[0].message)
    removed = _prune_undeclared_figures(analysis_dir, analysis_dir / "plots")
    assert removed == ["plots/sensitivity/per_sim/member-16/retired_family__member.16.html"]
    assert (analysis_dir / declared_rel).is_file(), "the listed figure is the one that reaches the bundle"


def test_tree_whose_only_stale_figures_are_undeclared_family_does_not_refuse_and_the_prune_deletes_them(tmp_path):
    """Property: when every stale figure belongs to an undeclared family, the check neither refuses nor warns,
    and the prune then deletes those figures exactly as it does without this check.

    Class: must-fire twin of the split test (the same tree without the declared-family stale figure).
    Why required: shows the skip changes what is counted without changing what the prune deletes.
    Killed by: counting undeclared-family figures (refusal here), or removing them in the check (tree changed).
    Second correct implementation: a check that defers to the prune by running it first on a copy.
    """
    analysis_dir = _declared_only_peak_flood_depth_tree(tmp_path, with_declared_stale=False)
    analysis = _standin("wrapper", tmp_path, [22])
    before = _listing(analysis_dir)
    for allow in (False, True):
        result, warned = _caught(_check_stale_figures, analysis, analysis_dir, analysis_dir / "plots", allow)
        assert result == [] and warned == []
    assert _listing(analysis_dir) == before, "the check deleted nothing"
    removed = _prune_undeclared_figures(analysis_dir, analysis_dir / "plots")
    assert removed == ["plots/sensitivity/per_sim/member-16/retired_family__member.16.html"]


# ----------------------------------------------------------------- the override: stale figures carried as they are


def test_override_carries_the_stale_figures_with_one_warning(tmp_path):
    """Property: with `allow_stale_figures=True` the check raises nothing, deletes nothing, issues exactly one
        warning labelled `hhemt bundle emit` at line 0 naming the count and `stale_figures_included`, and returns
        the four analysis-relative paths.

    Class: override.
    Why required: the user has consented; the record of what was carried must be complete and visible once.
    Killed by: one warning per file; a warning carrying a source file and line; returning absolute paths; deleting
        under the override.
    Second correct implementation: issuing the warning through `warnings.warn` with a custom `stacklevel` that still
        reports no source line.
    """
    analysis_dir = _tree(tmp_path)
    before = _listing(analysis_dir)
    analysis = _standin("master", tmp_path, [5, 22])
    result, warned = _caught(_check_stale_figures, analysis, analysis_dir, analysis_dir / "plots", True)
    assert sorted(result) == sorted(f"plots/{rel}" for rel in STALE_RELS)
    assert len(warned) == 1
    message = str(warned[0].message)
    assert "4 stale figure file(s)" in message
    assert "stale_figures_included" in message
    assert warned[0].filename == "hhemt bundle emit" and warned[0].lineno == 0, "no source line can be echoed"
    assert _listing(analysis_dir) == before, "option (a) deletes nothing"


def test_manifest_carries_stale_figures_included_only_when_non_empty(tmp_path):
    """Property: `_write_bundle_manifest` writes `stale_figures_included` with the given list when non-empty and
        writes no such key for `[]` or `None`.

    Class: manifest key.
    Why required: a clean manifest must stay byte-identical to what it was before the key existed.
    Killed by: writing the key unconditionally, or writing an empty list.
    Second correct implementation: writing the key from a differently named keyword with the same non-empty rule.
    """
    staging = tmp_path / "staging"
    staging.mkdir()
    _write_bundle_manifest(
        staging,
        sources_by_renderer={"r": [Path("eda/a.zarr")]},
        analysis_id="aid",
        git_sha="deadbeef",
        stale_figures_included=["plots/sensitivity/per_sim/member-16/x.html"],
    )
    manifest = json.loads((staging / "bundle_manifest.json").read_text())
    assert manifest["stale_figures_included"] == ["plots/sensitivity/per_sim/member-16/x.html"]

    for name, value in (("empty", []), ("none", None)):
        clean_dir = tmp_path / f"staging_{name}"
        clean_dir.mkdir()
        _write_bundle_manifest(
            clean_dir,
            sources_by_renderer={"r": [Path("eda/a.zarr")]},
            analysis_id="aid",
            git_sha="deadbeef",
            stale_figures_included=value,
        )
        clean = json.loads((clean_dir / "bundle_manifest.json").read_text())
        assert "stale_figures_included" not in clean, f"absent, not empty, for {name}"


# --------------------------------------------------------------------------- placement arm, its twin, plumbing


class _PruneReached(Exception):
    """Private sentinel: the emit got as far as the prune."""


def _patch_to_prune(monkeypatch) -> list[int]:
    calls: list[int] = []

    def _recorder(analysis_dir, plots_dir):
        calls.append(1)
        raise _PruneReached

    monkeypatch.setattr(E, "_assert_report_not_older_than_read_model", lambda analysis_dir: None)
    monkeypatch.setattr(E, "_prune_undeclared_figures", _recorder)
    return calls


def _emit_master(tmp_path: Path, analysis_dir: Path, **kwargs):
    analysis = _standin("master", tmp_path, [5, 22], analysis_dir=analysis_dir)
    return emit_bundle(analysis, tmp_path / "out" / "bundle.zip", **kwargs)


def test_placement_arm_refuses_before_the_prune(tmp_path, monkeypatch):
    """Property: run through `emit_bundle` itself, a tree holding a stale figure is refused before
        `_prune_undeclared_figures` is reached (recorder count 0) and nothing on disk changes.

    Class: ordering.
    Why required: the prune deletes from the live tree; a refusal placed after it would delete before refusing.
    Killed by: moving the check after the prune call.
    Second correct implementation: placing the check earlier, before the manifest-presence check, as long as it
        stays ahead of the prune.
    """
    analysis_dir = _tree(tmp_path, undeclared=True)
    calls = _patch_to_prune(monkeypatch)
    before = _listing(analysis_dir)
    with pytest.raises(ConfigurationError) as raised:
        _emit_master(tmp_path, analysis_dir)
    assert raised.value.field == "bundle_stale_figures"
    assert calls == [], "the prune was never reached: a check wired after it would fail here"
    assert _listing(analysis_dir) == before, "the refusal deleted and staged nothing"


def test_placement_twin_reaches_the_prune(tmp_path, monkeypatch):
    """Property: the same emit on a tree with no stale figure reaches the prune (recorder count 1).

    Class: ordering control.
    Why required: proves the arm's zero is the check's doing and not an emit that never gets that far.
    Killed by: a check that refuses on a clean tree.
    Second correct implementation: any placement ahead of the prune.
    """
    analysis_dir = _tree(tmp_path, stale=False, undeclared=True)
    calls = _patch_to_prune(monkeypatch)
    before = _listing(analysis_dir)
    with pytest.raises(_PruneReached):
        _emit_master(tmp_path, analysis_dir)
    assert calls == [1]
    assert _listing(analysis_dir) == before


def test_override_reaches_the_prune_and_hands_the_paths_to_the_manifest(tmp_path, monkeypatch):
    """Property: with the override, `emit_bundle` passes the check, continues, and hands the four stale paths to the
        manifest writer as `stale_figures_included`.

    Class: plumbing (emit to manifest).
    Why required: the manifest arm alone calls the writer directly and cannot see whether `emit_bundle` forwards the
        list.
    Killed by: computing the list and dropping it before the writer call.
    Second correct implementation: passing the paths through a different intermediate variable.
    """
    analysis_dir = _tree(tmp_path)
    handed: dict = {}

    class _ManifestReached(Exception):
        pass

    def _manifest_recorder(staging, **kwargs):
        handed.update(kwargs)
        raise _ManifestReached

    monkeypatch.setattr(E, "_assert_report_not_older_than_read_model", lambda analysis_dir: None)
    monkeypatch.setattr(E, "_prune_undeclared_figures", lambda analysis_dir, plots_dir: [])
    monkeypatch.setattr(E, "_get_toolkit_git_sha", lambda strict=True: "deadbeef")
    monkeypatch.setattr(E, "harvest_source_paths", lambda plots_dir, analysis_dir: {})
    monkeypatch.setattr(E, "harvest_metadata_only_sources", lambda roots, analysis_dir: frozenset())
    monkeypatch.setattr(E, "_harvest_sources", lambda *a, **k: ({}, [], []))
    monkeypatch.setattr(E, "_copy_bundle_baseline", lambda *a, **k: None)
    monkeypatch.setattr(E, "_copy_reference_outputs", lambda *a, **k: None)
    monkeypatch.setattr(E, "emit_relative_config_pair", lambda *a, **k: ({}, {}))
    monkeypatch.setattr(E, "_plan_declared_inputs", lambda *a, **k: ([], []))
    monkeypatch.setattr(E, "_assert_archive_namespace", lambda *a, **k: None)
    monkeypatch.setattr(E, "_assert_no_emit_name_clobber", lambda *a, **k: None)
    monkeypatch.setattr(E, "_emit_resolved_brand_theme", lambda *a, **k: None)
    monkeypatch.setattr(E, "_copy_supporting_files", lambda *a, **k: None)
    monkeypatch.setattr(E, "_emit_hpc_identity", lambda *a, **k: None)
    monkeypatch.setattr(E, "_copy_declared_inputs", lambda *a, **k: [])
    monkeypatch.setattr(E, "_emit_runnable_template_set", lambda *a, **k: None)
    monkeypatch.setattr(E, "_upgrade_crate_to_workflow_run_crate", lambda *a, **k: None)
    monkeypatch.setattr(E, "_annotate_crate_excluded_inputs", lambda *a, **k: None)
    monkeypatch.setattr(E, "_crate_license_finding", lambda *a, **k: None)
    monkeypatch.setattr(E, "_write_bundle_manifest", _manifest_recorder)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(_ManifestReached):
            _emit_master(tmp_path, analysis_dir, allow_stale_figures=True)
    assert sorted(handed["stale_figures_included"]) == sorted(f"plots/{rel}" for rel in STALE_RELS)


# --------------------------------------------------------------------------- CLI arms


def _patch_cli(monkeypatch, tmp_path: Path, emit):
    """``bundle_command`` imports each of these inside its body, so patch them at the source."""
    import hhemt.analysis
    import hhemt.bundle
    import hhemt.system

    monkeypatch.setattr(hhemt.system, "TRITONSWMM_system", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(
        hhemt.analysis,
        "TRITONSWMM_analysis",
        lambda *a, **k: SimpleNamespace(cfg_analysis=SimpleNamespace(toggle_sensitivity_analysis=False)),
    )
    monkeypatch.setattr(hhemt.bundle, "emit_bundle", emit)
    system_cfg = tmp_path / "system.yaml"
    analysis_cfg = tmp_path / "analysis.yaml"
    system_cfg.write_text("x: 1\n")
    analysis_cfg.write_text("x: 1\n")
    return ["bundle", "--system-config", str(system_cfg), "--analysis-config", str(analysis_cfg)]


@pytest.mark.parametrize(("extra", "expected"), [([], False), ([FLAG], True)])
def test_cli_passes_the_flag_as_a_keyword(tmp_path, monkeypatch, extra, expected):
    """Property: `hhemt bundle` passes `allow_stale_figures` as False without `--allow-stale-figures` and True with
        it.

    Class: CLI threading.
    Why required: the flag does nothing unless it reaches `emit_bundle`.
    Killed by: not passing the keyword, or passing it inverted.
    Second correct implementation: reading the option through a wrapper object that yields the same boolean.
    """
    seen: dict = {}

    def _emit(target, output, **kwargs):
        seen.update(kwargs)
        return tmp_path / "bundle.zip"

    args = _patch_cli(monkeypatch, tmp_path, _emit)
    result = runner.invoke(app, args + extra)
    assert result.exit_code == 0, result.output
    assert seen["allow_stale_figures"] is expected


def test_cli_turns_the_refusal_into_a_labelled_exit_2(tmp_path, monkeypatch):
    """Property: a `ConfigurationError` from `emit_bundle` ends as exit code 2 with the `Configuration Error` label
        and the flag name in the output, and is not left as an uncaught exception.

    Class: CLI error handling.
    Why required: before the wrap the verb printed a traceback and exited 1.
    Killed by: removing the `try`/`except` in `bundle_command`.
    Second correct implementation: routing the error through any handler that maps the class to code 2 with the same
        label.
    """

    def _emit(target, output, **kwargs):
        raise ConfigurationError(
            field="bundle_stale_figures",
            message="This sensitivity analysis holds 4 stale figure file(s)",
            fix_hint=f"re-run `hhemt bundle` with {FLAG}",
        )

    args = _patch_cli(monkeypatch, tmp_path, _emit)
    result = runner.invoke(app, args)
    assert result.exit_code == 2, result.output
    assert "Configuration Error" in result.output
    assert FLAG in result.output
    assert not isinstance(result.exception, ConfigurationError), "the wrap turns it into a typer.Exit"


def test_cli_lets_a_typer_exit_through_unreported(tmp_path, monkeypatch):
    """Property: a `typer.Exit` raised inside the body keeps its code and is not reported as an unexpected error.

    Class: CLI error handling (control).
    Why required: `typer.Exit` is an `Exception` subclass and would otherwise be caught by the generic handler.
    Killed by: dropping `except typer.Exit: raise`.
    Second correct implementation: testing for `typer.Exit` by class name inside the generic handler.
    """
    import typer

    def _emit(target, output, **kwargs):
        raise typer.Exit(7)

    args = _patch_cli(monkeypatch, tmp_path, _emit)
    result = runner.invoke(app, args)
    assert result.exit_code == 7
    assert "Unexpected Error" not in result.output


def test_only_the_bundle_verb_lists_the_flag():
    """Property: `bundle --help` lists `--allow-stale-figures`; `publish --help` and `reprex emit --help` do not.

    Class: CLI surface.
    Why required: a deposit gets a DOI and cannot be withdrawn, so the other two verbs must refuse stale figures
        with no override.
    Killed by: adding the option to either other verb.
    Second correct implementation: declaring the option on `bundle` through a shared option object that those verbs
        do not use.
    """
    assert FLAG in runner.invoke(app, ["bundle", "--help"]).output
    assert FLAG not in runner.invoke(app, ["publish", "--help"]).output
    assert FLAG not in runner.invoke(app, ["reprex", "emit", "--help"]).output
