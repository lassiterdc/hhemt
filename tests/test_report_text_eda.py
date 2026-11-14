"""Reader-facing text of the EDA pages: the code facts their wording states.

Package 3e rewrites the text of the EDA pages one chunk at a time. Each chunk adds the tests of
its pages here (shape binding B3): a test asserts the code fact the text states, located by a
stable anchor, never by comparing a whole sentence, writes no typed count, and carries a
must-fire control. Nothing here compiles or runs a solver.

Chunk 3e.1, pages P18 to P23 (shape item I38, section 7 rows for P18 to P23). These pages need
consolidated trees and rasters, so none is a stub page kind in `tests/test_caption_page_kinds.py`
(shape B4): the captions are checked here against the code facts and by the reviewer against the
matrix rows.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_caption_page_kinds import CAPTIONS, bold_names

_EDA = Path(__file__).resolve().parent.parent / "src" / "hhemt" / "eda"
_CAPTIONS = {
    "P18": "eda_compute_sensitivity",
    "P19": "eda_dem_resolution_cost_error",
    "P20": "eda_dem_resolution_error_ecdf",
    "P21": "eda_dem_resolution_diff_maps",
    "P22": "eda_dem_resolution_coupling_table",
    "P23": "raw_byte_identity",
}
_MODULES = ("compute_sensitivity", "_config_diff", "cross_sim_identity", "raw_byte_identity")

_EMPHASIS_CAPS = re.compile(r"\b(?:NOT|ALL|ONLY|NEVER|PRESENT|ABSENT|EITHER|ONE)\b")
_HEDGE_OR_WHICH = re.compile(r"\b(?:may|might|typically|usually|generally|likely|which)\b", re.IGNORECASE)
_RETIRED = re.compile(
    r"peak water level|sub-analys[ei]s|sensitivity master|model (?:type|arm)|sentinel|sidecar|child bundle"
    r"|compute-config|\bconfigs?\b|\bsims?\b|\bmasters?\b|primary",
    re.IGNORECASE,
)


def _caption(page: str) -> str:
    return (CAPTIONS / f"{_CAPTIONS[page]}.rst").read_text("utf-8")


def _strings(path: Path) -> list[tuple[int, str]]:
    """String constants of a module that are not docstrings, with their line numbers."""
    tree = ast.parse(path.read_text("utf-8"))
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if ast.get_docstring(node, clean=False) is not None:
                docs.add(id(node.body[0].value))
    return [
        (n.lineno, n.value)
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs
    ]


@pytest.mark.parametrize("page", sorted(_CAPTIONS))
def test_an_eda_caption_has_no_hedge_capital_emphasis_em_dash_or_retired_phrase(page: str) -> None:
    """Each of the six captions follows brief rules 2, 3 and 8, including the capitals the style test misses.

    Input class: the six captions. Required because the style test's floor skips `NOT`, `ALL` and
    `EITHER`, and brief rule 8 bars capitals for emphasis. Killed by restoring any of them. A second
    correct implementation words the sentences differently and passes.
    """
    text = _caption(page)

    assert _EMPHASIS_CAPS.search(text) is None
    assert _HEDGE_OR_WHICH.search(text) is None
    assert _RETIRED.search(text) is None
    assert "—" not in text and " -- " not in text


@pytest.mark.parametrize(
    "old", ["is NOT resume-induced", "EITHER run", "a compute-config", "which shows", "the clean master"]
)
def test_the_eda_prose_matchers_fire_on_the_old_wording(old: str) -> None:
    """Must-fire control for the test above: each matcher fires on a known-bad input."""
    assert _EMPHASIS_CAPS.search(old) or _HEDGE_OR_WHICH.search(old) or _RETIRED.search(old) or "master" in old


_CUT = {
    "P18": ["reporting set", "compute_sensitivity", "experiment_datatree", "tab via"],
    "P19": ["Pareto", "reparameteriz", "compute-hours", "resource-fair"],
    "P20": ["garnish", "prohibition"],
    "P21": ["Resampling", "nearest"],
    "P22": ["node-name", "flat on fixtures", "model-agnostic"],
    "P23": ["clean master", "resume master", "honest-degradation", "BIT4BIT", "sibling masters", "resume count"],
}


@pytest.mark.parametrize("page", sorted(_CUT))
def test_each_eda_caption_cuts_the_sentences_its_shape_row_cuts(page: str) -> None:
    """The caption drops the mechanism, rationale and discussion clauses the P18 to P23 rows cut.

    Input class: the six captions. Required because shape rows P18 to P23 (K1, K16) cut them; the
    reporting-set clause is also false on the byte-identity set. Killed by restoring any of them. A
    second correct implementation words the kept sentences differently and passes.
    """
    text = _caption(page)

    for phrase in _CUT[page]:
        assert phrase not in text


def test_the_cut_check_fires_on_the_old_caption_text() -> None:
    """Must-fire control for the test above: the old text carried each cut phrase."""
    old = "Rendered as a config-selectable tab via the reporting set; deliberately not a Pareto frontier"
    assert "reporting set" in old and "Pareto" in old


def test_the_cost_error_caption_states_the_axes_the_figure_draws() -> None:
    """P19's caption names the same axes the figure sets: compute cost on a log x axis, the p95 depth error on y.

    Input class: the caption and the source of the cost-versus-error figure (read only). Required
    because the old caption described a signed headline on y and compute-hours on x, while the figure
    sets `yaxis_title="p95 depth error ..."` and `xaxis=dict(type="log")`. Killed by restoring the old
    axes description. A second correct implementation words the axes differently while it names the
    95th percentile and the log scale.
    """
    source = (_EDA / "_dem_resolution_plots.py").read_text("utf-8")
    text = _caption("P19")

    assert 'yaxis_title="p95 depth error' in source and 'xaxis=dict(type="log")' in source
    assert "95th-percentile depth error" in text and "log scale" in text
    assert "signed depth-magnitude" not in text


def test_the_nuisance_depth_floor_the_captions_print_is_the_constant_the_dem_figures_apply() -> None:
    """P20 and P21 print the floor as the constant the DEM-resolution figures apply, with no "by default".

    Input class: the two captions, `_dem_resolution_plots._TAU_M` and the figure module's source.
    Required because shape rows P20 and P21 print tau from the value the code applies: the DEM figures
    use the module constant `_TAU_M` (documented as an a-priori floor, never tuned) and read no
    configuration override (`magnitude_tau_m` is read only by `compute_sensitivity._tau`, which these
    figures do not call). Killed by a caption value that differs from the constant or that restores
    "by default", which would promise a setting that does not exist. A second correct implementation
    prints the floor in other units while it matches the constant.
    """
    from hhemt.eda._dem_resolution_plots import _TAU_M

    source = (_EDA / "_dem_resolution_plots.py").read_text("utf-8")

    assert "magnitude_tau_m" not in source and "_tau(" not in source
    for page in ("P20", "P21"):
        assert f"{_TAU_M:g} m," in _caption(page)
        assert "by default" not in _caption(page)


def test_the_floor_check_fires_on_a_different_constant() -> None:
    """Must-fire control: a caption naming 0.05 m, or promising a default, would fail the test above."""
    assert f"{0.03:g} m," not in "tau (0.05 m, the floor)" and "by default" in "tau (0.03 m by default)"


def test_the_byte_identity_caption_names_the_colors_and_the_star_the_figure_draws() -> None:
    """P23 states the cell colors and the starred reference row that the figure draws.

    Input class: the caption and the plotting helper that draws the page (`_plotting.py`, read only:
    identical cells take `_B4B_IDENTICAL_COLOR`, differing cells a white-to-vermillion ramp, and the
    reference label is starred). Required because shape row P23 keeps the families, the cell colors
    and the starred reference. Killed by a caption that names colors the page figure does not use.
    A second correct implementation words the sentence differently while the colors match.
    """
    plotting = (_EDA / "_plotting.py").read_text("utf-8")
    text = _caption("P23")

    assert '_B4B_IDENTICAL_COLOR = "#009E73"' in plotting and '[0.0, "#FFFFFF"], [1.0, "#D55E00"]' in plotting
    assert "★ ref" in plotting
    assert "starred" in text and "green" in text and "vermillion" in text and "crimson" not in text
    assert "**CPU**" in text and "**GPU**" in text


def test_the_byte_identity_caption_bold_names_are_the_families_the_figure_names() -> None:
    """The two bold spans in P23 are the hardware families the verdict names (brief rule 9)."""
    names = set(bold_names(_caption("P23")))
    source = (_EDA / "raw_byte_identity.py").read_text("utf-8")

    assert names == {"CPU", "GPU"}
    assert source.count("CPU") > 0 and source.count("GPU") > 0


@pytest.mark.parametrize("module", _MODULES)
def test_no_figure_string_of_the_compute_and_byte_identity_modules_carries_an_em_dash_or_a_hand_plural(
    module: str,
) -> None:
    """Strings the EDA figures print carry no em dash, no `word(s)` plural and no `compute-config`.

    Input class: every non-docstring string constant of the four modules. Required because brief
    rules 4 and 8 bar the hand plural and the em dash, and rule 2 retires `compute-config`. Killed by
    restoring `pair(s)`, ` — ` or `compute-config` in a verdict summary, a title or a notice. A second
    correct implementation words them differently and passes.
    """
    offenders = [
        (line, s[:60])
        for line, s in _strings(_EDA / f"{module}.py")
        if "—" in s or re.search(r"\w\(s\)", s) or "compute-config" in s
    ]

    assert offenders == []


@pytest.mark.parametrize("old", ["N/A — need both a run", "3 pair(s) dropped", "a compute-config group"])
def test_the_string_check_fires_on_the_old_strings(old: str) -> None:
    """Must-fire control: each old string trips the offender condition."""
    assert "—" in old or re.search(r"\w\(s\)", old) or "compute-config" in old


def test_the_summaries_take_the_noun_form_a_count_takes() -> None:
    """The verdict summaries inflect their nouns by count (brief rule 4).

    Input class: the module-local `_plural` helper of three modules with counts 1 and 2. Killed by
    a helper that returns one form for every count. A second correct implementation inflects through
    any other mechanism while the printed forms are the same.
    """
    from hhemt.eda import compute_sensitivity, cross_sim_identity, raw_byte_identity

    for mod in (compute_sensitivity, cross_sim_identity, raw_byte_identity):
        assert mod._plural(1, "pair", "pairs") == "pair"
        assert mod._plural(2, "pair", "pairs") == "pairs"


# ---------------------------------------------------------------------------------------
# Reviewer additions (round 1, chunk 3e.1).
# ---------------------------------------------------------------------------------------
def test_the_byte_identity_caption_names_the_colors_the_page_figure_draws() -> None:
    """NEW CAPABILITY (reviewer, truth of the reworded caption). The page is drawn by
    `_plotting._b4b_faceted_figure`: identical cells take `_B4B_IDENTICAL_COLOR` (bluish green) and
    differing cells take a white-to-vermillion ramp of the largest absolute difference. The
    `crimson`/`seagreen` scale belongs to `raw_byte_identity.build_binary_timestep_figure`, which
    no renderer calls. Class: the caption and the two figure sources. Must-fire: the old sentence."""
    plotting = (_EDA / "_plotting.py").read_text("utf-8")
    text = _caption("P23")

    assert '_B4B_IDENTICAL_COLOR = "#009E73"' in plotting and '"#D55E00"' in plotting
    assert "crimson" not in text
    assert "crimson" in "Green cells are byte-identical; crimson cells differ."


def test_the_byte_identity_caption_claims_no_cause_the_verdict_does_not_state() -> None:
    """NEW CAPABILITY (reviewer, truth). The verdict summary names the count of differing cells
    and no cause: `check_raw_byte_identity` builds `summary` from `n_diff` alone. Class: the
    caption sentence about the verdict. Must-fire: a sentence that promises a cause."""
    source = (_EDA / "raw_byte_identity.py").read_text("utf-8")
    sentence = next(s for s in re.split(r"(?<=[.])\s+", _caption("P23")) if s.startswith("The verdict"))

    assert "the cause" not in sentence
    assert "cause" not in source.split("def check_raw_byte_identity", 1)[1].split("verdict = CheckResult", 1)[0]
    assert "the cause" in "when it is not, the number of differing cells and the cause"


def test_every_bold_name_of_the_coupling_table_caption_is_a_column_the_table_prints() -> None:
    """NEW CAPABILITY (reviewer, brief rule 9). Bold in a caption names a page element as the page
    prints it. The table prints the headers listed in `_dem_resolution_plots.py`; `Peak depth / flow`
    and the over/under-estimate span are not among them. Class: the caption and the header list.
    Must-fire: a name the table does not print is reported."""
    source = (_EDA / "_dem_resolution_plots.py").read_text("utf-8")
    headers = source.split("    headers = [", 1)[1].split("]", 1)[0]
    printed = set(re.findall(r'"([^"]+)"', headers))
    names = bold_names(_caption("P22"))

    assert [n for n in names if n not in printed] == []
    assert "Peak depth / flow" not in printed


def test_the_compute_and_byte_identity_figures_print_no_ascii_dash_primary_or_b4b() -> None:
    """NEW CAPABILITY (reviewer, brief rules 2, 7 and 8). Class: every reader-facing string of the
    four modules. A ` -- ` dash is the em dash by another spelling; `primary` and `b4b` are internal
    shorthand (the coder's own matcher retires `primary`, and it replaced `this primary` in a
    neighbouring sentence); `b4b (1=identical)` is a colorbar title. Must-fire: the control strings."""
    from tests.test_report_text_style import python_strings

    offenders: list[str] = []
    for module in _MODULES:
        for s in python_strings(ast.parse((_EDA / f"{module}.py").read_text("utf-8"))):
            if " -- " in s or re.search(r"\bprimary\b", s) and len(s.split()) >= 4 or re.match(r"b4b \(", s):
                offenders.append(f"{module}: {s[:60]}")

    assert offenders == []
    assert " -- " in "Config diff maps -- not applicable" and re.match(r"b4b \(", "b4b (1=identical)")


# ---- 3e.2: the DEM-resolution figures, the EDA notebook, the promote notices, the local surface ----

_3E2_MODULES = ("_dem_resolution_plots", "_plotting", "_promote", "_notebook", "_local_surface")
_HELD_NAME_STRING = "## Byte identity across compute configurations"  # the check name
_3E2_BAD = re.compile(
    r"—| -- |\w\(s\)|DRAFT|\bADR-\d|\bOD-\d|reporting-system_|\bb4b\b|\bNOT\b|\bENUMERATED\b|\bINERT\b|\bDISPLAYED\b"
    r"|\bSUMMARY\b|\bRED =|\bEVERY\b|\bANY\b|\bprimary\b|\bmay\b|\bwhich\b|compute-config|compute configs?\b"
)


@pytest.mark.parametrize("module", _3E2_MODULES)
def test_no_printed_string_of_the_eda_figure_notebook_and_notice_modules_carries_a_retired_or_marker_form(
    module: str,
) -> None:
    """Strings the EDA figures, notebook, promote notices and local surface print follow the brief.

    Input class: every printed string of the five modules (the style test's string population), less
    the held check-name cell (Decision 3) and the `deferred:` routing value (a record field, not page
    text). Required because brief rules 1, 2, 3, 4, 7 and 8 bar the draft marker in a title, the em
    dash, capitals for emphasis, planning identifiers (`ADR-`, `OD-`, plan names), the internal
    shorthand `b4b`, `primary`, `compute config`, a hedge and a hand plural. Killed by restoring any
    of them. A second correct implementation words the strings differently and passes.
    """
    from tests.test_report_text_style import python_strings

    offenders = [
        s[:70]
        for s in python_strings(ast.parse((_EDA / f"{module}.py").read_text("utf-8")))
        if not s.startswith((_HELD_NAME_STRING, "deferred:")) and _3E2_BAD.search(s)
    ]

    assert offenders == []


@pytest.mark.parametrize(
    "old",
    [
        "DEM-resolution cost vs error (DRAFT)",
        "Reference 3 m — depth (m)",
        "registered — NOT yet routed (deferred to reporting-system_eda-skill)",
        "# EDA — x; displayed per ADR-14",
        "the b4b calc member",
        "3 EDA figure(s) rendered",
    ],
)
def test_the_3e2_string_check_fires_on_the_old_strings(old: str) -> None:
    """Must-fire control: each old string trips the matcher."""
    assert _3E2_BAD.search(old) is not None


def test_the_three_dem_figure_titles_state_no_draft_marker_and_name_red_in_words() -> None:
    """The cost-error, ECDF, table and difference-map titles print no draft marker; red is a word, not a capital.

    Input class: the title strings of `_dem_resolution_plots.py`. Killed by restoring "(DRAFT)" or
    "RED =". A second correct implementation titles the figures differently while omitting both.
    """
    source = (_EDA / "_dem_resolution_plots.py").read_text("utf-8")

    assert 'title="DEM-resolution cost vs error"' in source
    assert 'title="DEM-resolution depth-error ECDF vs the finest reference"' in source
    assert "red means the coarse run under-estimates" in source
    assert "(DRAFT)" not in source.replace("FIRST DRAFT", "")


def test_the_panel_titles_name_all_compute_configurations_not_all_configs() -> None:
    """The degraded single-panel title reads "All compute configurations" on both title helpers."""
    from hhemt.eda._plotting import _b4b_category_title, _b4b_family_title

    assert _b4b_category_title("all") == _b4b_family_title("all") == "All compute configurations"
    assert _b4b_category_title("cpu") == "CPU" and _b4b_family_title("gpu") == "GPU"


def test_the_seeded_notebook_is_valid_python_names_the_figure_count_by_its_plural_and_keeps_the_held_heading(
    tmp_path: Path,
) -> None:
    """The seeded notebook compiles, inflects its figure count and still carries the held check heading.

    Input class: a notebook emitted for a stub analysis, both as a source analysis and as a bundle.
    Required because the seed cell was edited (lowercase comments, no planning identifier, a plural
    that takes the count's form) and the check heading carries the text "Byte identity across compute
    configurations". Killed by a seed cell that fails to compile, by restoring `figure(s)`, or by dropping the held
    heading. A second correct implementation inflects through other code while it prints "1 EDA figure".
    """
    import nbformat

    from hhemt.eda._notebook import emit_eda_notebook

    cfg = SimpleNamespace(analysis_id="demo")
    eda_cfg = SimpleNamespace(enabled_plots=[])
    path = emit_eda_notebook(tmp_path, cfg_analysis=cfg, eda_cfg=eda_cfg, is_bundle=False)
    nb = nbformat.read(path, as_version=4)
    code = [c.source for c in nb.cells if c.cell_type == "code"]
    markdown = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")

    for src in code:
        compile(src, "<cell>", "exec")
    seed = next(s for s in code if "rendered inline above" in s)
    assert "figure(s)" not in seed and '"figure" if len(_figs) == 1 else "figures"' in seed
    assert "ADR-" not in seed and "—" not in markdown and "NEW notebook" not in markdown
    assert "## Byte identity across compute configurations (source-analysis only)" in markdown


def test_the_local_surface_skeleton_bodies_are_valid_python_without_planning_identifiers() -> None:
    """The two emitted skeleton files parse and carry no planning identifier or em dash."""
    from hhemt.eda._local_surface import _BOOTSTRAP_BODY, _INIT_BODY

    for body in (_INIT_BODY, _BOOTSTRAP_BODY):
        ast.parse(body)
        assert "—" not in body and re.search(r"\bADR-\d|\bOD-\d", body) is None


# ---- 3e.2 round 2: the junction-per-cell claim and the event page strings ------------------------

_EVENT_NOTICE = re.compile(r"^The event page is not shown: [^.]+\.$")


def test_no_printed_text_claims_one_coupling_junction_per_cell() -> None:
    """The coupling-table caption and the figure's printed caption state no junction-per-cell rule.

    Input class: the P22 caption and the printed strings of `_dem_resolution_plots.py`. Required
    because no code line establishes the rule (the count comes from the `[INFLOWS]` entries and
    nothing checks uniqueness per cell), so the sentence turned an unverified hedge into a fact. Killed
    by restoring "holds at most one coupling junction" in either place. A second correct
    implementation sources the constraint and cites it, which would change this test on purpose.
    """
    from tests.test_report_text_style import python_strings

    printed = list(python_strings(ast.parse((_EDA / "_dem_resolution_plots.py").read_text("utf-8"))))

    assert not any("at most one coupling junction" in s for s in printed)
    assert "at most one coupling junction" not in _caption("P22")
    assert any(s.startswith("Coupling junctions exchange water with the TRITON surface cell") for s in printed)


def test_the_event_page_strings_join_with_a_colon_and_state_each_omission_in_the_not_shown_form() -> None:
    """The event page title, section headings and two notices follow brief rules 6 and 8.

    Input class: the printed strings of `report_renderers/per_sim_event_page.py`. Required because the
    P27 template prints these caller strings as the document title, the figure headings and the
    omission notices: a spaced hyphen joined their parts, the title said "Simulation results" beside a
    caption that opens "Results for event", and the notices were not in the "{Element} is not shown:
    {reason}." form ("not yet" also promised later availability). Killed by restoring any of them. A
    second correct implementation words the reasons differently while it keeps the form.
    """
    from tests.test_report_text_style import python_strings

    path = Path(__file__).resolve().parent.parent / "src" / "hhemt" / "report_renderers" / "per_sim_event_page.py"
    printed = list(python_strings(ast.parse(path.read_text("utf-8"))))
    notices = [s for s in printed if "event page" in s]

    assert len(notices) == 2 and all(_EVENT_NOTICE.match(s) for s in notices)
    assert not any(" - " in s or s.startswith("Simulation results") or "not yet" in s for s in printed)
    assert "Results for {}" in printed and "{}: {}" in printed


@pytest.mark.parametrize(
    "old",
    [
        "no per-simulation figures are applicable to this analysis",
        "per-simulation figures are not yet available for this scenario",
        "{} - {}",
        "Simulation results - {}",
    ],
)
def test_the_event_page_form_check_fires_on_the_old_strings(old: str) -> None:
    """Must-fire control: each old string fails the form or carries a dash join, a stale title or 'not yet'."""
    assert _EVENT_NOTICE.match(old) is None and (
        " - " in old or old.startswith("Simulation results") or "not yet" in old or "per-simulation" in old
    )
