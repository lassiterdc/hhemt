"""Code facts behind the reader-facing text of the combined report's pages.

One file for the combined report's text tests. Each test asserts a fact of the code that a
caption or note states, with a control that changes the input and expects the output to move.
Nothing here compares a whole sentence, compiles or runs a solver.
"""

from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path

from hhemt.du_sentinels import write_du_sentinel
from hhemt.report_renderers import cross_experiment_compatibility as compat
from hhemt.report_renderers import cross_experiment_disk_utilization as disk


def _row(experiment: str, *, solver: str, build: str) -> dict:
    return {
        "experiment_id": experiment,
        "role": "clean",
        "model": "TRITON-SWMM",
        "n_subs": 3,
        "toolkit_sha": "abc",
        "toolkit_version": build,
        "solver_sha": solver,
    }


def _note_text(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html.split("</table>", 1)[1]) if "</table>" in html else ""


def test_the_compatibility_note_states_whether_the_builds_are_the_same_in_every_row() -> None:
    """The caption says a note under the table states whether the hhemt build and the solver
    build are the same in every row. Plausible wrong implementation: a note only when they
    differ, so a reader cannot tell a controlled comparison from a missing note.

    Input class: rows with one solver and one build (both identical), then one row changed in
    each column (each marked).
    """
    same = compat._provenance_table_html(
        [_row("a", solver="11111111aa", build="1.0+g1"), _row("b", solver="11111111aa", build="1.0+g1")]
    )
    split = compat._provenance_table_html(
        [_row("a", solver="11111111aa", build="1.0+g1"), _row("b", solver="22222222bb", build="1.0+g2")]
    )

    assert "Solver sha is identical across every row." in _note_text(same)
    assert "hhemt build is identical across every row" in _note_text(same)
    assert "identical across every row" not in _note_text(split)
    assert "on the Solver sha column marks" in _note_text(split)
    assert "on the hhemt build column marks" in _note_text(split)
    assert split.count(" *<") == 2 or split.count("*</td>") == 2


def test_the_compatibility_table_has_one_row_per_experiment_and_names_none_when_empty() -> None:
    """The caption says one row per combined experiment and that an empty combine says so."""
    three = compat._provenance_table_html([_row(e, solver="s", build="b") for e in ("a", "b", "c")])
    assert three.count("<tr><td>") == 3
    assert "No experiment is recorded for this combine." in compat._provenance_table_html([])
    assert "child crate" not in compat._provenance_table_html([_row("a", solver="s", build="b")]) + three


def _crate(analysis_dir: Path, experiment: str, total: int) -> None:
    write_du_sentinel(
        analysis_dir / "child_crates" / experiment / "_status" / "_du.json",
        scope="analysis",
        disk_utilization_bytes=total,
        sub_path_breakdown={"members": total},
    )


def _render_disk(tmp_path: Path) -> str:
    analysis = types.SimpleNamespace(analysis_paths=types.SimpleNamespace(analysis_dir=tmp_path))
    out = tmp_path / "plots" / "cross_experiment" / "disk_utilization.html"
    out.parent.mkdir(parents=True)
    disk.render(analysis, report_cfg=None, output_path=out)
    return out.read_text()


def test_the_disk_page_has_one_column_per_experiment_labelled_with_its_model(tmp_path: Path) -> None:
    """The caption says one column per EXPERIMENT, labelled with its model. The earlier text said
    one column per model, which is false when two experiments share a model: the second is
    labelled with its experiment id.

    Input class: two experiments with no consolidated tree, so both read "model not recorded"
    (a missing tree is never labelled with a model). Control: a third experiment adds a third
    column.
    """
    for experiment in ("expA", "expB"):
        _crate(tmp_path, experiment, 5 * 1024 * 1024)

    def _columns(html: str) -> set[str]:
        return set(re.findall(r'"title": "([^"]*\(MiB\))"', html))

    two = _columns(_render_disk(tmp_path))
    assert two == {"model not recorded (MiB)", "model not recorded (expB) (MiB)"}

    third = tmp_path.parent / (tmp_path.name + "_three")
    third.mkdir()
    for experiment in ("expA", "expB", "expC"):
        _crate(third, experiment, 1024 * 1024)
    assert len(_columns(_render_disk(third))) == 3


def test_the_disk_page_says_so_when_no_experiment_is_recorded(tmp_path: Path) -> None:
    """The caption says the page says so when no experiment is recorded."""
    html = _render_disk(tmp_path)

    assert "<p class='note'>" in html and "<table" not in html and "tabulator" not in html.lower()
    manifest = json.loads((tmp_path / "plots" / "cross_experiment" / "disk_utilization.manifest.json").read_text())
    assert manifest["source_paths_relative"]


def _swmm_only_experiment(root: Path, name: str) -> Path:
    import xarray as xr

    child = root / name
    child.mkdir(parents=True)
    leaf = xr.Dataset({"depth": ("t", [0.0, 1.0])})
    tree = xr.DataTree.from_dict({"/member_1/swmm_only/swmm_node": leaf})
    tree.to_zarr(str(child / "analysis_datatree.zarr"), mode="w")
    return child


def test_an_experiment_with_no_triton_tier_is_not_labelled_with_a_triton_model(tmp_path: Path) -> None:
    """NEW CAPABILITY (reviewer). Property: a column label that names a model names the model the
    experiment ran, so an experiment whose consolidated tree holds only a SWMM tier is not
    labelled TRITON. The disk caption says each column is "labelled with its model".

    Input class: a consolidated tree whose only member tier is `swmm_only`. Plausible wrong
    implementation: the label falls through to TRITON when neither TRITON tier exists. Reachable:
    the model-toggle fields allow a SWMM-only analysis (the shape's K9 names "SWMM only" and
    "P17's columns"), and the compatibility page's Model column takes the same fall-through.

    Both-states anchor: the assertion turns on the absence of the substring TRITON from the
    label, true of any spelling of the SWMM-only model name. A second correct implementation:
    the label is `SWMM`.
    """
    child = _swmm_only_experiment(tmp_path, "expS")

    label = disk._child_model(child)

    assert label, "an experiment is labelled"
    assert "TRITON" not in label, label
    # Control: the same helper still reads the coupled tier, so the assertion is not vacuous.
    import xarray as xr

    coupled = tmp_path / "expC"
    coupled.mkdir()
    leaf = xr.Dataset({"depth": ("t", [0.0, 1.0])})
    xr.DataTree.from_dict({"/member_1/tritonswmm/triton": leaf}).to_zarr(
        str(coupled / "analysis_datatree.zarr"), mode="w"
    )
    assert disk._child_model(coupled) == "TRITON-SWMM"


def test_a_missing_or_modelless_tree_is_not_labelled_with_a_model_on_either_page(tmp_path: Path) -> None:
    """The captions say an experiment whose consolidated tree is missing or holds no model reads
    "model not recorded", and a SWMM-only tree reads SWMM, on both pages.

    Input class: no tree; a tree with a member but no model tier; a SWMM-only tree. Plausible
    wrong implementation: the former TRITON fall-through, red for all three.
    """
    import xarray as xr

    root = tmp_path / "child_crates"
    (root / "expNone").mkdir(parents=True)
    modelless = root / "expEmpty"
    modelless.mkdir()
    leaf = xr.Dataset({"depth": ("t", [0.0, 1.0])})
    xr.DataTree.from_dict({"/member_1/other/node": leaf}).to_zarr(str(modelless / "analysis_datatree.zarr"), mode="w")
    _swmm_only_experiment(root, "expSwmm")

    disk_labels = {name: disk._child_model(root / name) for name in ("expNone", "expEmpty", "expSwmm")}
    rows = {r["experiment_id"]: r["model"] for r in compat._combine_provenance_rows(tmp_path)}

    assert disk_labels == {"expNone": "model not recorded", "expEmpty": "model not recorded", "expSwmm": "SWMM"}
    assert rows == disk_labels


def test_an_experiment_whose_members_ran_different_models_is_not_labelled_as_one_of_them(tmp_path: Path) -> None:
    """NEW CAPABILITY (reviewer). Property: the label an experiment gets names every model its
    members ran, so an experiment with one coupled member and one SWMM-only member does not read
    the same as an experiment holding the coupled member alone. Both captions say a column or a
    row carries "its model".

    Input class: a consolidated tree whose first member has a `tritonswmm/triton` tier and whose
    second has `swmm_only/swmm_node`. Plausible wrong implementation: the first member with a
    model tier decides, so the mixed experiment is labelled TRITON-SWMM. Reachable: members of a
    sweep can carry their own system configuration (`system_config_yaml` in the setup table), so
    the model toggles can differ between members.

    Both-states anchor: the assertion compares the mixed label with the single-member label, true
    of any spelling that names more than the first member's model.
    """
    import xarray as xr

    leaf = xr.Dataset({"depth": ("t", [0.0, 1.0])})

    def _label(name: str, groups: dict[str, xr.Dataset]) -> str:
        child = tmp_path / name
        child.mkdir()
        xr.DataTree.from_dict(groups).to_zarr(str(child / "analysis_datatree.zarr"), mode="w")
        return disk._child_model(child)

    single = _label("single", {"/member_1/tritonswmm/triton": leaf})
    mixed = _label("mixed", {"/member_1/tritonswmm/triton": leaf, "/member_2/swmm_only/swmm_node": leaf})

    assert single == "TRITON-SWMM"
    assert mixed != single, mixed


def test_a_mixed_experiment_lists_every_model_once_in_a_fixed_order_on_both_pages(tmp_path: Path) -> None:
    """The captions say an experiment whose members ran more than one model lists each, joined
    by "+". Plausible wrong implementation: the first member's model alone. Input class: members
    in an order that differs from the label order (SWMM-only first, then coupled, then a second
    coupled one), so a first-member or a per-member-repeat label is wrong. Control: a one-model
    experiment keeps its single word, with no "+"."""
    import xarray as xr

    root = tmp_path / "child_crates"
    leaf = xr.Dataset({"depth": ("t", [0.0, 1.0])})
    mixed = root / "expMixed"
    mixed.mkdir(parents=True)
    xr.DataTree.from_dict(
        {
            "/member_1/swmm_only/swmm_node": leaf,
            "/member_2/tritonswmm/triton": leaf,
            "/member_3/tritonswmm/triton": leaf,
            "/member_4/triton_only/triton": leaf,
        }
    ).to_zarr(str(mixed / "analysis_datatree.zarr"), mode="w")
    _swmm_only_experiment(root, "expOne")

    assert disk._child_model(mixed) == "TRITON + TRITON-SWMM + SWMM"
    assert disk._child_model(root / "expOne") == "SWMM"
    rows = {r["experiment_id"]: r["model"] for r in compat._combine_provenance_rows(tmp_path)}
    assert rows == {"expMixed": "TRITON + TRITON-SWMM + SWMM", "expOne": "SWMM"}


# ---------------------------------------------------------------------------------------
# The clean-versus-resume pages.
# ---------------------------------------------------------------------------------------
def _intercomparison_page(tmp_path: Path, pairs: list[dict]) -> str:
    from hhemt.report_renderers import cross_experiment_intercomparison as ic

    source = tmp_path / "combined_intercomparison.json"
    source.write_text(json.dumps({"experiments": [], "pairs": pairs}))
    return ic._build_intercomparison_html(source, None)


_PAIR = {
    "config": "run_mode=mpi|n_mpi=2|n_omp=1|n_gpus=0|n_nodes=1|partition=standard",
    "variable": "max_wlevel_m",
    "event_iloc": 0,
    "identical": True,
    "max_abs_diff": 0.0,
    "model": "TRITON-SWMM",
}


def test_the_comparison_page_shows_a_table_with_pairs_and_a_notice_without(tmp_path: Path) -> None:
    """The caption says the page says so when no clean and resume pair is found. Plausible wrong
    implementation: an empty table with no notice. Input class: a read-model with one pair, and
    one with none."""
    with_pair = _intercomparison_page(tmp_path, [_PAIR])
    without = _intercomparison_page(tmp_path, [])

    assert "cross-experiment-intercomparison-table" in with_pair
    assert "<p class='note'>" not in with_pair
    assert "cross-experiment-intercomparison-table" not in without
    assert "<p class='note'>" in without


def test_the_compute_configuration_group_lists_every_field_the_table_splits_out(tmp_path: Path) -> None:
    """The caption says the compute configuration is split into columns, and the group text
    names no count. Plausible wrong implementation: a hand-written "six fields" that goes stale
    when a field is added. Control: the group carries the new heading."""
    from hhemt.report_renderers import cross_experiment_intercomparison as ic

    page = _intercomparison_page(tmp_path, [_PAIR])

    assert "Compute configuration" in page
    for field in ic._CONFIG_FIELDS:
        assert field in page, field
    assert "six" not in page.lower().split("compute configuration", 1)[1][:400]


def test_the_truncation_sentence_states_no_hand_written_count() -> None:
    """The sentence shared by the figure caption and the page caption must not carry a count of
    members or periods, which goes stale when the data changes (brief rule 4). Control: the
    matcher fires on the removed wording."""
    from hhemt.report_renderers import cross_experiment_intercomparison_maps as maps

    digits = re.compile(r"\d+ of \d+|\bN-1\b|\b\d+\b")

    assert not digits.search(maps._TRUNCATION_CAVEAT), maps._TRUNCATION_CAVEAT
    assert digits.search("(measured: 14 of 28 members)")


def test_the_compute_configuration_group_text_lists_as_many_fields_as_the_table_has(tmp_path: Path) -> None:
    """NEW CAPABILITY (reviewer). Property: the sentence under the "Compute configuration" heading
    lists exactly the fields the table splits out, one list item per field, so a field dropped
    from the text (or added to the table) is caught. The existing test checks the raw field names
    against the page, which the table's own data supplies whatever the sentence says.

    Input class: the rendered page for one pair. Plausible wrong implementation: the sentence
    omits a field (for example the partition) while the column remains. Reachable: the sentence
    is hand-written beside `_CONFIG_FIELDS`.

    Both-states anchor: the assertion counts list items of the sentence, true of any wording of
    the items. A second correct implementation: the sentence is built from the field list.
    """
    from hhemt.report_renderers import cross_experiment_intercomparison as ic

    page = _intercomparison_page(tmp_path, [_PAIR])
    match = re.search(r"compute configuration of each pair: ([^.\"<]+)\.", page)

    assert match, "the group text names the compute configuration of each pair"
    items = [part for chunk in match.group(1).split(", ") for part in chunk.split(" and ")]
    assert len(items) == len(ic._CONFIG_FIELDS), items


def test_the_maps_figure_with_no_pairs_does_not_claim_identical_pairs(tmp_path: Path) -> None:
    """With no experiments and no pairs the figure caption must not say every compared pair is
    bit-identical (nothing was compared). Control: with one identical pair it does say so."""
    sys.path.insert(0, str(Path(__file__).parent))
    from test_cross_experiment_diff_maps_render import _EXPERIMENTS, _pair, _write

    from hhemt.report_renderers.cross_experiment_intercomparison_maps import build_cross_experiment_diff_figure

    def _text(root) -> str:
        return " ".join(a.text or "" for a in build_cross_experiment_diff_figure(root).layout.annotations)

    (tmp_path / "empty").mkdir()
    (tmp_path / "one").mkdir()
    empty = _text(_write(tmp_path / "empty", [], []))
    one = _text(_write(tmp_path / "one", [_pair("TRITON", True)], _EXPERIMENTS))

    assert "hold no clean and resume pair to compare" in empty
    assert "every compared pair is bit-identical" not in empty
    assert "every compared pair is bit-identical" in one


def test_the_cross_solver_x_axis_is_titled_compute_configuration() -> None:
    """The caption says the x axis is the compute configuration, so the axis title uses the
    term-table word. Control: the retired spelling is absent from the renderer's axis call."""
    import inspect

    from hhemt.report_renderers import cross_solver_comparison as cs

    src = inspect.getsource(cs)
    assert 'title_text="compute configuration"' in src
    assert 'title_text="compute config"' not in src


def test_the_built_cross_solver_figure_titles_every_panel_x_axis_with_the_term_table_word() -> None:
    """NEW CAPABILITY (reviewer). Property: the figure the renderer BUILDS titles each panel's x
    axis `compute configuration`. The existing test reads the renderer's source text, which a
    constructed string, a variable or a second call site would defeat.

    Input class: a read-model with cells for two model arms (two panels). Plausible wrong
    implementation: one panel's axis keeps the retired spelling. Reachable: the title is set in
    a loop over panels and a second branch could set it.

    Both-states anchor: the assertion reads the built figure's axis titles, true of any code
    path that produces them. A second correct implementation: the title comes from a constant.
    """
    from hhemt.report_renderers import cross_solver_comparison as cs
    from hhemt.report_renderers._provenance import ProvenanceLog

    cell = {"solver": "a" * 40, "model_arm": "tritonswmm", "config": "run_mode=mpi|n_mpi=2", "value": 1.0}
    payload = {
        "model_arms": ["triton", "tritonswmm"],
        "cells": [cell, {**cell, "model_arm": "triton"}],
        "metric": "wall_clock_s",
        "solver_info": {},
    }
    fig, _facts = cs._build_figure(cs._normalize_cells(payload), payload, ProvenanceLog())

    titles = [fig.layout[name].title.text for name in fig.layout if name.startswith("xaxis")]
    assert len(titles) == 2, titles
    assert set(titles) == {"compute configuration"}, titles


def test_the_harvested_caption_names_the_experiment_and_points_to_its_bundle() -> None:
    """The caption keeps one sentence naming the experiment (a template parameter) and one
    pointing to the experiment's bundle under child_crates, with no retired "child bundle"."""
    text = (Path(__file__).resolve().parent.parent / "src/hhemt/report_templates/captions/_harvested.rst").read_text()

    assert "{{ snakemake.params.experiment_id }}" in text
    assert "child_crates/" in text
    assert "child bundle" not in text.lower()


def test_the_combined_description_follows_the_rendered_selections(tmp_path: Path) -> None:
    """Staging writes the landing page from the sections the bundle renders: with one solver and
    no clean and resume pair the cross-solver section and the clean-versus-resume wording are
    absent; with two solvers and a pair both are present (must-fire pair)."""
    from hhemt.bundle import combined_snakefile_generator as g

    def _text(root: Path, solvers: int, pairs: list) -> str:
        root.mkdir()
        (root / "cross_solver_comparison.json").write_text(json.dumps({"distinct_solver_count": solvers}))
        (root / "combined_intercomparison.json").write_text(json.dumps({"pairs": pairs}))
        rendered = {
            t.report_kwargs.get("category") for s in g._active_combined_selections(root) for t in s.rule_spec_template
        }
        return g._combined_workflow_description(rendered, clean_vs_resume=g._has_clean_resume_pairs(root))

    one = _text(tmp_path / "one", 1, [])
    two = _text(tmp_path / "two", 2, [{"config": "c"}])

    assert "Cross-Solver Comparison" not in one and "clean" not in one
    assert "Cross-Solver Comparison" in two and "clean-versus-resume" in two
